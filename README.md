---
title: SQL Query Review
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SQL Query Review Environment

Last updated: 2026-04-06

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/openenv)
[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/hardikshreyas/sql_query_review)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)

An OpenEnv-compliant reinforcement learning environment where an AI agent acts as a senior database engineer reviewing SQL queries submitted by junior developers. The agent identifies bugs, security vulnerabilities, and performance anti-patterns, then suggests corrected SQL.

- **Hugging Face Space:** <https://huggingface.co/spaces/hardikshreyas/sql_query_review>
- **API Endpoint:** <https://hardikshreyas-sql-query-review.hf.space>

## Why SQL Review?

SQL review is a high-leverage automation target:
- SQL injection is a top OWASP-class security risk.
- N+1 query patterns and missing indexes silently degrade production performance.
- Cartesian products and over-broad query patterns create correctness and data-exposure risks.

This environment trains agents to detect these issues automatically, with direct applicability to database-focused PR review and CI gatekeeping workflows.

## Quick Start

### Install

```bash
pip install openenv-core
git clone <repo>
cd sql_query_review
pip install -e .
```

### Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run baseline inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="meta-llama/llama-4-scout-17b-16e-instruct"
export API_TOKEN="your_groq_key"
python inference.py
```

### Docker

```bash
docker build -t sql-query-review .
docker run -p 7860:7860 sql-query-review
```

## Python Usage Example

```python
from sql_query_review import SqlQueryReviewEnv

with SqlQueryReviewEnv(base_url="https://hardikshreyas-sql-query-review.hf.space") as env:
    result = env.reset(task_id="easy_cartesian_product")
    obs = result.observation
    print(obs.query_id, obs.step_count, obs.max_steps)

    step = env.step({
        "action_type": "identify_issue",
        "issue_category": "cartesian_product",
        "severity": "high",
        "explanation": "Missing JOIN condition creates a Cartesian product.",
        "corrected_sql": None,
    })
    print(step.reward, step.done, step.observation.feedback)
```

## Tasks

The environment includes 5 tasks across 3 difficulty levels:

1. **easy_cartesian_product** (easy)  
   Two tables in `FROM` with no proper join condition.  
   Expected issue: `cartesian_product`, severity: `high`, max steps: `8`.

2. **medium_sql_injection** (medium)  
   Python f-string interpolates user input directly into SQL.  
   Expected issue: `sql_injection`, severity: `critical`, max steps: `8`.

3. **hard_n_plus_one** (hard)  
   One query per product over large tables (50k products, 2M reviews).  
   Expected issues: `n_plus_one`, `missing_index`, `incorrect_logic`, max steps: `12`.

4. **medium_missing_null_check** (medium)  
   `ORDER BY` on nullable data creates subtle null-ordering correctness issues.  
   Expected issue: `missing_null_check`, severity: `medium`, max steps: `6`.

5. **easy_select_star** (easy)  
   `SELECT *` across 3 joined tables exposes sensitive columns (`cost_price`, `internal_notes`).  
   Expected issue: `incorrect_logic`, severity: `medium`, max steps: `6`.

## Action Space

| Field | Type | Allowed values / constraints |
|---|---|---|
| `action_type` | enum | `identify_issue` \| `suggest_fix` \| `approve` \| `reject` |
| `issue_category` | enum | `sql_injection` \| `missing_index` \| `n_plus_one` \| `incorrect_logic` \| `cartesian_product` \| `missing_null_check` \| `none` |
| `severity` | enum/null | `critical` \| `high` \| `medium` \| `low` \| `null` |
| `explanation` | string | required, minimum 5 characters |
| `corrected_sql` | string/null | SQL fix text or `null` |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `query_id` | string | Task identifier |
| `raw_sql` | string | Query under review |
| `schema_context` | string | DDL/schema context for analysis |
| `task_description` | string | Problem framing and expected behavior |
| `feedback` | string | Environment feedback from previous action |
| `step_count` | int | Current step number |
| `max_steps` | int | Episode step budget |
| `done` | bool | Episode termination flag |
| `reward` | float | Current reward signal |

## Reward Function

Shaped per-step reward (not sparse-only terminal reward):

| Component | Value |
|---|---:|
| Correct issue identified | `+0.15` |
| Severity correctly identified as critical | `+0.05` |
| Partial fix keyword match | `+0.25` (max partial component) |
| False positive issue | `-0.05` |
| Duplicate issue | `-0.05` |
| `suggest_fix` without `corrected_sql` | `-0.05` |
| Incorrectly approving buggy query | `-0.20` |
| Rejecting after identifying issues | `+0.05` |
| Step penalty (efficiency pressure) | `-0.01` per step |

Final episode score is deterministically graded in the range **0.0 to 1.0**.

## Baseline Results

Model: `meta-llama/llama-4-scout-17b-16e-instruct` (via Groq), 3 runs per task.

| Task | Mean | Std (±) | Best |
|---|---:|---:|---:|
| `easy_cartesian_product` | 0.8667 | 0.1155 | 1.0000 |
| `medium_sql_injection` | 0.8667 | 0.1155 | 1.0000 |
| `hard_n_plus_one` | 0.9833 | 0.0289 | 1.0000 |
| `medium_missing_null_check` | 0.7333 | 0.0577 | 0.8000 |
| `easy_select_star` | 1.0000 | 0.0000 | 1.0000 |

**Overall mean:** `0.8900`

## API Example

### Reset

```http
POST /reset
Content-Type: application/json

{"task_id": "easy_cartesian_product"}
```

### Step

```http
POST /step
Content-Type: application/json

{
  "action": {
    "action_type": "identify_issue",
    "issue_category": "cartesian_product",
    "severity": "high",
    "explanation": "Missing JOIN condition creates a Cartesian product returning orders × customers rows.",
    "corrected_sql": null
  }
}
```

## Required Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | yes | OpenAI-compatible API base URL |
| `MODEL_NAME` | yes | LLM model identifier |
| `API_TOKEN` | yes | API key/token |
| `ENV_URL` | optional | Environment URL (defaults to HF Space URL) |

## Project Structure

```text
sql_query_review/
├── models.py                             # Typed Pydantic Action/Observation models
├── client.py                             # WebSocket client for the environment
├── inference.py                          # Baseline agent (reproduces scores)
├── openenv.yaml                          # OpenEnv spec metadata
├── pyproject.toml                        # Dependencies
└── server/
    ├── sql_query_review_environment.py   # Core environment logic (reset/step/state)
    ├── app.py                            # FastAPI server
    ├── requirements.txt
    └── Dockerfile
```

## Technical Design Highlights

1. **Session-based state isolation**  
   Each session uses isolated episode state keyed by UUID, preventing cross-agent leakage.

2. **Deterministic graders**  
   Keyword/regex-driven grading avoids LLM-in-the-loop grading variance and improves reproducibility.

3. **Shaped rewards**  
   Agents receive dense per-step feedback plus terminal deterministic scoring for practical RL training.

4. **Input sanitization**  
   Invalid enums, empty/invalid fields, and malformed model outputs are sanitized before env stepping.

5. **Difficulty progression**  
   Five tasks with easy/medium/hard structure and distinct issue categories.

## License

MIT License

---

Built for the Meta-PyTorch OpenEnv Hackathon.
