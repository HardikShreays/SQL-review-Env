"""
inference.py — Baseline agent for SQL Query Review OpenEnv

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed — nothing else on stdout):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL      = os.getenv("ENV_URL", "https://hardikshreyas-sql-query-review.hf.space")
BENCHMARK    = "sql_query_review"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required")

TASKS = [
    "easy_cartesian_product",
    "medium_sql_injection",
    "hard_n_plus_one",
    "medium_missing_null_check",
    "easy_select_star",
]

RUNS_PER_TASK = 8
SUCCESS_SCORE_THRESHOLD = 0.1
TASK_ID = os.getenv("TASK_ID", "easy_cartesian_product")

VALID_CATEGORIES = {
    "sql_injection", "missing_index", "n_plus_one",
    "incorrect_logic", "cartesian_product", "missing_null_check", "none"
}
VALID_SEVERITIES = {"critical", "high", "medium", "low"}

# ---------------------------------------------------------------------------
# Structured stdout loggers — strictly match sample format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def hprint(*args, **kwargs) -> None:
    """Human-readable logs go to stderr only — never pollute stdout."""
    print(*args, file=sys.stderr, **kwargs)

# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

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
Step 1: Call identify_issue for each problem you find (up to 4 calls).
Step 2: Call suggest_fix with a complete corrected_sql that fixes ALL issues.
Step 3: Call reject to close the review.

RULES:
- Always call suggest_fix before reject
- corrected_sql must be complete working SQL, not pseudocode
- Never call identify_issue after suggest_fix
- Never approve a query that has problems
- issue_category must be one of the exact values listed above
- explanation must be at least 5 characters
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
    """Run one full episode. Emits [START], [STEP]..., [END] to stdout."""
    result = env_reset(task_id)
    obs = result["observation"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history = []
    conversation = []
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    episode_done = False

    try:
        for _ in range(obs.get("max_steps", 10)):
            if obs.get("done"):
                break

            step_number = steps_taken + 1

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
                    max_tokens=2048,
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

            # Sanitize
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

            try:
                step_result = env_step(action)
            except requests.HTTPError as e:
                hprint(f"    ✗ Env error: {e}")
                break

            obs = step_result["observation"]
            done = step_result["done"]
            reward = float(step_result.get("reward", 0.0))
            error = obs.get("last_action_error") if isinstance(obs, dict) else None

            # [STEP] to stdout
            action_str = action.get("action_type", "unknown")
            log_step(step=step_number, action=action_str, reward=reward, done=done, error=error)

            rewards.append(reward)
            steps_taken = step_number
            history.append(action)

            if done:
                final_score = reward
                episode_done = True
                break

        if not episode_done and rewards:
            final_score = rewards[-1]
        success = bool(rewards) and (final_score >= SUCCESS_SCORE_THRESHOLD)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return final_score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # hprint("SQL Query Review — OpenEnv Baseline Inference")
    # hprint(f"Model  : {MODEL_NAME}")
    # hprint(f"API    : {API_BASE_URL}")
    # hprint(f"Env    : {ENV_URL}")
    # hprint(f"Tasks  : {len(TASKS)}")
    # hprint()

    # if not env_health():
    #     hprint(f"✗ Environment not reachable at {ENV_URL}")
    #     sys.exit(1)

    # hprint("✓ Environment is healthy\n")

    # Emit exactly one START..END sequence for validator compatibility.
    run_episode(TASK_ID)


if __name__ == "__main__":
    main()