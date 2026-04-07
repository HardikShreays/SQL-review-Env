"""
inference.py — Baseline agent for SQL Query Review OpenEnv

Required environment variables:
    API_BASE_URL   — OpenAI-compatible API base URL
    MODEL_NAME     — Model identifier
    HF_TOKEN       — API key

Optional:
    ENV_URL        — Environment server (default: HF Space URL)

Usage:
    python inference.py
"""

import json
import os
import sys
import time
import statistics

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN      = os.environ.get("HF_TOKEN")
ENV_URL      = os.environ.get("ENV_URL", "https://hardikshreyas-sql-query-review.hf.space")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required")

TASKS = [
    "easy_cartesian_product",
    "medium_sql_injection",
    "hard_n_plus_one",
    "medium_missing_null_check",
    "easy_select_star",
]

RUNS_PER_TASK = 3  # for variance reporting
BENCHMARK = "sql_query_review"

# Valid enum values — sanitize LLM output before sending to env
VALID_CATEGORIES = {
    "sql_injection", "missing_index", "n_plus_one",
    "incorrect_logic", "cartesian_product", "missing_null_check", "none"
}
VALID_SEVERITIES = {"critical", "high", "medium", "low"}
MIN_STRICT_SCORE = 0.0001
MAX_STRICT_SCORE = 0.9999


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def hprint(*args, **kwargs) -> None:
    """Human-readable logs go to stderr, not stdout."""
    print(*args, file=sys.stderr, **kwargs)


def clamp_strict_score(value: float) -> float:
    """Clamp scores into strict open interval (0, 1)."""
    return round(min(MAX_STRICT_SCORE, max(MIN_STRICT_SCORE, float(value))), 4)

# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str = None) -> dict:
    r = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id} if task_id else {},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    if r.status_code != 200:
        hprint(f"    ✗ Server response: {r.text[:200]}")
    r.raise_for_status()
    return r.json()


def env_health() -> bool:
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert senior database engineer doing a SQL code review.

You receive a SQL query, its schema, and the developer's intent.

You MUST respond with a single valid JSON object — no markdown, no extra text.

JSON schema:
{
  "action_type": "identify_issue" | "suggest_fix" | "approve" | "reject",
  "issue_category": "sql_injection" | "missing_index" | "n_plus_one" | "incorrect_logic" | "cartesian_product" | "missing_null_check" | "none",
  "severity": "critical" | "high" | "medium" | "low" | null,
  "explanation": "your detailed reasoning here",
  "corrected_sql": "full corrected SQL string or null"
}

STRICT STRATEGY — follow this exact order:
Step 1: Call identify_issue ONCE for the single most critical problem you see.
Step 2: Call suggest_fix with a complete corrected_sql that fixes ALL issues.
Step 3: Call reject to close the review.

RULES:
- Identify ALL issues you find before suggesting a fix (up to 4 identify_issue calls)
- Always call suggest_fix before reject
- corrected_sql must be complete working SQL, not pseudocode
- Never call identify_issue after suggest_fix
- Never approve a query that has problems
- issue_category must be one of the exact values listed above
"""


def build_user_message(obs: dict, history: list) -> str:
    history_str = ""
    if history:
        history_str = "\n\nYour actions so far:\n"
        for i, h in enumerate(history, 1):
            history_str += (
                f"  Step {i}: {h['action_type']} "
                f"[{h.get('issue_category', '')}] — "
                f"{h.get('explanation', '')[:80]}\n"
            )

    feedback = obs.get("feedback", "")
    feedback_str = f"\n\nEnvironment feedback: {feedback}" if feedback else ""

    return f"""## Query Under Review

Developer's intent: {obs['task_description']}

Schema:
```sql
{obs['schema_context']}
```

Submitted query:
```sql
{obs['raw_sql']}
```

Step {obs['step_count']} of {obs['max_steps']}.{feedback_str}{history_str}

Respond with a single JSON action object."""


# ---------------------------------------------------------------------------
# Single episode run
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    """Run one full episode for a task. Returns final grader score."""
    result = env_reset(task_id)
    obs = result["observation"]
    run_started_at = time.time()
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history = []
    conversation = []
    final_score = 0.0
    end_emitted = False
    success = False
    rewards: list[float] = []
    steps_taken = 0

    for _ in range(obs.get("max_steps", 10)):
        if obs.get("done"):
            break

        user_msg = build_user_message(obs, history)
        conversation.append({"role": "user", "content": user_msg})

        # LLM call
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT}
                ] + conversation,
                temperature=0.2,
                max_tokens=2048,  # hard task needs room for corrected SQL
                response_format={"type": "json_object"},
            )
        except Exception as e:
            hprint(f"    ✗ LLM error: {e}")
            break

        raw = response.choices[0].message.content or "{}"
        conversation.append({"role": "assistant", "content": raw})

        try:
            action = json.loads(raw)
        except json.JSONDecodeError as e:
            hprint(f"    ✗ JSON parse error: {e}")
            break

        # Sanitize LLM output
        action.setdefault("issue_category", "none")
        action.setdefault("severity", None)
        action.setdefault("corrected_sql", None)
        action.setdefault("explanation", "No explanation provided.")

        if action.get("issue_category") not in VALID_CATEGORIES:
            action["issue_category"] = "none"
        if action.get("severity") not in VALID_SEVERITIES:
            action["severity"] = None
        if not action.get("explanation") or len(str(action.get("explanation", ""))) < 5:
            action["explanation"] = "No explanation provided by agent."
        if action.get("corrected_sql") is not None and not isinstance(action.get("corrected_sql"), str):
            action["corrected_sql"] = None

        step_number = steps_taken + 1
        hprint(
            f"    step {step_number}: "
            f"{action.get('action_type')} "
            f"[{action.get('issue_category')}]"
        )
        try:
            result = env_step(action)
        except requests.HTTPError as e:
            hprint(f"    ✗ Env error: {e}")
            break

        obs = result["observation"]
        done = result["done"]
        reward = clamp_strict_score(result.get("reward", 0.0))
        final_score = reward
        error = obs.get("last_action_error") if isinstance(obs, dict) else None
        action_str = action.get("action_type", "unknown")
        log_step(step=step_number, action=action_str, reward=reward, done=done, error=error)
        rewards.append(reward)
        steps_taken = step_number
        history.append(action)

        if done:
            final_score = reward
            success = final_score > 0.0
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
            end_emitted = True
            break

    if not end_emitted:
        final_score = clamp_strict_score(final_score)
        success = final_score > 0.0
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return clamp_strict_score(final_score)


# ---------------------------------------------------------------------------
# Task runner with variance reporting
# ---------------------------------------------------------------------------

def run_task(task_id: str, runs: int = RUNS_PER_TASK) -> dict:
    """
    Run multiple episodes for one task.
    Returns mean, std, min, max scores.
    """
    hprint(f"\n{'='*60}")
    hprint(f"Task: {task_id}  ({runs} runs)")
    hprint(f"{'='*60}")

    scores = []
    for run_num in range(runs):
        hprint(f"\n  Run {run_num + 1}/{runs}:")
        score = run_episode(task_id)
        scores.append(score)
        hprint(f"  → Score: {score:.4f}")

    mean  = statistics.mean(scores)
    std   = statistics.stdev(scores) if len(scores) > 1 else 0.0
    best  = max(scores)
    worst = min(scores)

    hprint(f"\n  ── Results for '{task_id}'")
    hprint(f"     mean : {mean:.4f}")
    hprint(f"     std  : {std:.4f}")
    hprint(f"     best : {best:.4f}  worst: {worst:.4f}")

    return {
        "mean": round(mean, 4),
        "std":  round(std, 4),
        "best": round(best, 4),
        "worst": round(worst, 4),
        "runs": scores,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hprint("SQL Query Review — OpenEnv Baseline Inference")
    hprint(f"Model  : {MODEL_NAME}")
    hprint(f"API    : {API_BASE_URL}")
    hprint(f"Env    : {ENV_URL}")
    hprint(f"Tasks  : {len(TASKS)}")
    hprint(f"Runs   : {RUNS_PER_TASK} per task")
    hprint()

    if not env_health():
        hprint(f"✗ Environment not reachable at {ENV_URL}")
        hprint("  Start server: uvicorn server.app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    hprint("✓ Environment is healthy\n")

    all_results = {}
    start = time.time()

    for task_id in TASKS:
        all_results[task_id] = run_task(task_id)

    elapsed = time.time() - start

    # Summary table
    hprint(f"\n{'='*60}")
    hprint("BASELINE RESULTS")
    hprint(f"{'='*60}")
    hprint(f"  {'Task':<35} {'Mean':>6}  {'±Std':>6}  {'Best':>6}")
    hprint(f"  {'-'*55}")

    means = []
    for task_id, r in all_results.items():
        bar = "█" * int(r["mean"] * 20)
        hprint(
            f"  {task_id:<35} {r['mean']:>6.4f}  "
            f"±{r['std']:>5.4f}  {r['best']:>6.4f}  {bar}"
        )
        means.append(r["mean"])

    overall_mean = statistics.mean(means)
    hprint(f"\n  Overall mean : {overall_mean:.4f}")
    hprint(f"  Total time   : {elapsed:.1f}s")

    # Machine-readable output for validators
    output = {
        "model": MODEL_NAME,
        "tasks": all_results,
        "overall_mean": round(overall_mean, 4),
        "elapsed_seconds": round(elapsed, 1),
    }
    hprint("\nJSON:")
    hprint(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()