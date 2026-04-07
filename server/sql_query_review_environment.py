"""
SQL Query Review Environment — core logic.

Uses a session store keyed by episode ID so concurrent connections
each get fully isolated state. One Environment instance per WebSocket
session; the instance holds the session_id, the module holds the store.
"""

from uuid import uuid4
from typing import Dict, Any, Optional
import re

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        SqlQueryReviewAction, SqlQueryReviewObservation,
        ActionType, IssueCategory, Severity
    )
except ImportError:
    from models import (
        SqlQueryReviewAction, SqlQueryReviewObservation,
        ActionType, IssueCategory, Severity
    )


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "easy_cartesian_product",
        "difficulty": "easy",
        "raw_sql": """SELECT
    o.order_id,
    o.total_amount,
    c.customer_name,
    c.email
FROM orders o, customers c
WHERE o.status = 'pending';""",
        "schema_context": """CREATE TABLE orders (
    order_id    SERIAL PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES customers(customer_id),
    total_amount NUMERIC(10,2),
    status      VARCHAR(20)
);
CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    customer_name VARCHAR(120),
    email        VARCHAR(255) UNIQUE
);""",
        "task_description": (
            "Fetch all pending orders with the name and email "
            "of the customer who placed each order."
        ),
        "expected_issues": ["cartesian_product"],
        "expected_severity": "high",
        "fix_keywords": ["join", "customer_id"],
        "max_steps": 8,
    },
    {
        "id": "medium_sql_injection",
        "difficulty": "medium",
        "raw_sql": """-- Built at runtime in Python:
-- query = f"SELECT * FROM users WHERE username = '{username}' AND password_hash = '{pw}'"

SELECT *
FROM users
WHERE username = '{username}'
  AND password_hash = '{password_hash}';""",
        "schema_context": """CREATE TABLE users (
    user_id      SERIAL PRIMARY KEY,
    username     VARCHAR(80) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role         VARCHAR(30) DEFAULT 'user',
    is_active    BOOLEAN DEFAULT TRUE
);""",
        "task_description": (
            "Authenticate a user by checking username and hashed password. "
            "Values come directly from a web form submission."
        ),
        "expected_issues": ["sql_injection"],
        "expected_severity": "critical",
        "fix_keywords": ["%s", "parameteriz", "prepared", ":username"],
        "max_steps": 8,
    },
    {
        "id": "hard_n_plus_one",
        "difficulty": "hard",
        "raw_sql": """-- Step 1: app fetches all active products
SELECT product_id, name FROM products WHERE is_active = TRUE;

-- Step 2: for EACH product_id the app runs this separately:
SELECT AVG(rating), COUNT(*)
FROM reviews
WHERE product_id = {product_id} AND verified = TRUE
HAVING COUNT(*) > 0;""",
        "schema_context": """CREATE TABLE products (
    product_id  SERIAL PRIMARY KEY,
    name        VARCHAR(255),
    is_active   BOOLEAN DEFAULT TRUE
    -- No index on is_active
);
CREATE TABLE reviews (
    review_id   SERIAL PRIMARY KEY,
    product_id  INT NOT NULL REFERENCES products(product_id),
    rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
    verified    BOOLEAN DEFAULT FALSE
    -- No index on (product_id, verified)
);
-- products: ~50,000 rows. reviews: ~2,000,000 rows.""",
        "task_description": (
            "For each active product, get the average verified rating "
            "and count of verified reviews. Only return products with "
            "at least one verified review. Current implementation runs "
            "one query per product — too slow in production."
        ),
        "expected_issues": ["n_plus_one", "missing_index", "incorrect_logic"],
        "expected_severity": "critical",
        "fix_keywords": ["join", "group by", "avg", "having"],
        "max_steps": 12,
    },
    {
        "id": "medium_missing_null_check",
        "difficulty": "medium",
        "raw_sql": """SELECT
    u.username,
    p.bio,
    p.website,
    CONCAT('https://example.com/', u.username) AS profile_url
FROM users u
JOIN profiles p ON u.user_id = p.user_id
WHERE u.is_active = TRUE
ORDER BY p.website;""",
        "schema_context": """CREATE TABLE users (
    user_id   SERIAL PRIMARY KEY,
    username  VARCHAR(80) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);
CREATE TABLE profiles (
    profile_id SERIAL PRIMARY KEY,
    user_id    INT REFERENCES users(user_id),
    bio        TEXT,
    website    VARCHAR(255)  -- nullable, many users have no website
);""",
        # medium_missing_null_check
        "task_description": (
            "Fetch active users with their profile bio, website, and a generated "
            "profile URL, sorted by website. The website column is nullable — "
            "many users have no website. Results must handle NULL values correctly."
            ),
        "expected_issues": ["missing_null_check"],
        "expected_severity": "medium",
        "fix_keywords": [
            "is not null", "coalesce", "nulls last", "nulls first",
            "filter", "where",
        ],
        "max_steps": 8,
    },
    {
        "id": "easy_select_star",
        "difficulty": "easy",
        "raw_sql": """SELECT *
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.customer_id = 42
  AND o.status = 'completed';""",
        "schema_context": """CREATE TABLE orders (
    order_id    SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    status      VARCHAR(20),
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP,
    internal_notes TEXT   -- sensitive internal field
);
CREATE TABLE order_items (
    item_id    SERIAL PRIMARY KEY,
    order_id   INT NOT NULL,
    product_id INT NOT NULL,
    quantity   INT,
    unit_price NUMERIC(10,2)
);
CREATE TABLE products (
    product_id   SERIAL PRIMARY KEY,
    name         VARCHAR(255),
    cost_price   NUMERIC(10,2),  -- sensitive: internal cost
    retail_price NUMERIC(10,2),
    sku          VARCHAR(80)
);""",
        # easy_select_star
        "task_description": (
            "Fetch completed orders for customer 42 with item and product details. "
            "The developer used SELECT * across all three tables. This exposes "
            "sensitive columns (internal_notes, cost_price) and pulls unnecessary "
            "data. Replace SELECT * with only the specific columns the frontend needs."
        ),
        "expected_issues": ["incorrect_logic"],
        "expected_severity": "medium",
        "fix_keywords": ["o.order_id", "oi.quantity", "p.name", "p.retail_price"],
        "max_steps": 8,
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}

# Ordered list for cycling
TASK_IDS = [t["id"] for t in TASKS]

# ---------------------------------------------------------------------------
# Module-level session store
# Keyed by episode_id — each reset() creates a new isolated session
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict[str, Any]] = {}
_task_cycle_index: int = 0  # global cycle counter across all resets


def _make_session(task: dict) -> Dict[str, Any]:
    """Create a fresh isolated session dict for one episode."""
    return {
        "task": task,
        "state": State(episode_id=str(uuid4()), step_count=0),
        "issues_found": [],
        "fix_sql": "",
        "severity_was_critical": False,
        "has_index_mention": False,
        "cumulative_reward": 0.0,
    }


# ---------------------------------------------------------------------------
# Graders — deterministic, 0.0 → 1.0
# ---------------------------------------------------------------------------

def _grade_easy_cartesian(issues_found, fix_sql, step_count) -> float:
    score = 0.0
    if "cartesian_product" in issues_found:
        score += 0.40
    if fix_sql:
        sql = fix_sql.lower()
        if "join" in sql:        score += 0.20
        if "customer_id" in sql: score += 0.10
        if ("o.customer_id = c.customer_id" in sql
                or "c.customer_id = o.customer_id" in sql):
            score += 0.10
    if step_count <= 5: score += 0.20
    return round(min(score, 1.0), 4)


def _grade_medium_injection(issues_found, fix_sql, severity_critical, step_count) -> float:
    score = 0.0
    if "sql_injection" in issues_found: score += 0.35
    if severity_critical:               score += 0.15
    if fix_sql:
        if any(re.search(p, fix_sql) for p in [r"\%s", r"\?", r":\w+", r"\$\d+"]):
            score += 0.30
        elif any(kw in fix_sql.lower() for kw in ["parameteriz", "prepared"]):
            score += 0.15
    if step_count <= 5: score += 0.20
    return round(min(score, 1.0), 4)


def _grade_hard_n_plus_one(issues_found, fix_sql, has_index_mention, step_count) -> float:
    score = 0.0
    if "n_plus_one"      in issues_found: score += 0.15
    if "missing_index"   in issues_found: score += 0.15
    if "incorrect_logic" in issues_found: score += 0.10
    if fix_sql:
        sql = fix_sql.lower()
        if "join"     in sql: score += 0.10
        if "group by" in sql: score += 0.10
        if "avg"      in sql: score += 0.05
        if "having"   in sql: score += 0.05
        if "count"    in sql: score += 0.05
        if re.search(r"create\s+index|index\s+on", fix_sql, re.IGNORECASE):
            score += 0.05
    if has_index_mention: score += 0.10
    if step_count <= 9:   score += 0.10
    return round(min(score, 1.0), 4)


def _grade_medium_null_check(issues_found, fix_sql, step_count) -> float:
    score = 0.0
    if "missing_null_check" in issues_found: score += 0.60  # was 0.45
    if fix_sql:
        sql = fix_sql.lower()
        if "is not null" in sql:                             score += 0.10
        if "coalesce" in sql:                                score += 0.10
        if "nulls last" in sql or "nulls first" in sql:     score += 0.05
        if "where" in sql or "filter" in sql:                score += 0.05
    if step_count <= 5:                                      score += 0.10  # was 0.20
    return round(min(score, 1.0), 4)


def _grade_easy_select_star(issues_found, fix_sql, step_count) -> float:
    score = 0.0
    if "incorrect_logic" in issues_found: score += 0.40
    if fix_sql:
        sql = fix_sql.lower()
        hits = sum(1 for kw in ["o.order_id", "oi.quantity", "p.name", "p.retail_price"]
                   if kw in sql)
        score += hits * 0.10
    if step_count <= 5: score += 0.20
    return round(min(score, 1.0), 4)


GRADERS = {
    "easy_cartesian_product":   _grade_easy_cartesian,
    "medium_sql_injection":     _grade_medium_injection,
    "hard_n_plus_one":          _grade_hard_n_plus_one,
    "medium_missing_null_check": _grade_medium_null_check,
    "easy_select_star":         _grade_easy_select_star,
}


def _run_grader(session: Dict[str, Any]) -> float:
    tid = session["task"]["id"]
    if tid == "medium_sql_injection":
        return GRADERS[tid](
            session["issues_found"], session["fix_sql"],
            session["severity_was_critical"], session["state"].step_count
        )
    elif tid == "hard_n_plus_one":
        return GRADERS[tid](
            session["issues_found"], session["fix_sql"],
            session["has_index_mention"], session["state"].step_count
        )
    else:
        return GRADERS[tid](
            session["issues_found"], session["fix_sql"],
            session["state"].step_count
        )


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class SqlQueryReviewEnvironment(Environment):
    """
    OpenEnv environment: AI agent reviews SQL queries as a senior
    database engineer.

    Each instance holds a session_id that maps to isolated state
    in the module-level _sessions store. This allows concurrent
    WebSocket connections without state leakage.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._session_id: Optional[str] = None

    @property
    def _session(self) -> Dict[str, Any]:
        """Get current session — falls back to most recent if instance lost its ID."""
        if self._session_id and self._session_id in _sessions:
            return _sessions[self._session_id]
        # Framework created a new instance for this request — use most recent session
        if _sessions:
            latest_id = list(_sessions.keys())[-1]
            self._session_id = latest_id
            return _sessions[latest_id]
        raise RuntimeError("Call reset() before step().")

    def reset(self, task_id: str = None) -> SqlQueryReviewObservation:
        """
        Start a new isolated episode.
        Creates a fresh session in the store keyed by a new episode_id.
        """
        global _task_cycle_index

        # Select task
        if task_id and task_id in TASK_MAP:
            task = TASK_MAP[task_id]
        else:
            task = TASKS[_task_cycle_index % len(TASKS)]
            _task_cycle_index += 1

        # Create isolated session
        session = _make_session(task)
        self._session_id = session["state"].episode_id
        _sessions[self._session_id] = session

        return SqlQueryReviewObservation(
            query_id=task["id"],
            raw_sql=task["raw_sql"],
            schema_context=task["schema_context"],
            task_description=task["task_description"],
            feedback="New query submitted for review. Analyse carefully.",
            step_count=0,
            max_steps=task["max_steps"],
            done=False,
            reward=0.0,
        )

    def step(self, action: SqlQueryReviewAction) -> SqlQueryReviewObservation:
        """Process one review action against this instance's isolated session."""
        session = self._session
        task = session["task"]
        session["state"].step_count += 1
        step = session["state"].step_count
        max_steps = task["max_steps"]

        step_reward = -0.01
        feedback_parts = []

        if action.action_type == ActionType.IDENTIFY_ISSUE:
            cat = action.issue_category.value

            if cat == "none":
                feedback_parts.append("Please specify an issue_category.")

            elif cat in session["issues_found"]:
                step_reward -= 0.05
                feedback_parts.append(f"⚠ Already flagged '{cat}' — no duplicate credit.")

            elif cat in task["expected_issues"]:
                step_reward += 0.15
                session["issues_found"].append(cat)
                feedback_parts.append(f"✓ Correct — '{cat}' is a real issue here.")
                if (action.severity == Severity.CRITICAL
                        and task["expected_severity"] == "critical"):
                    session["severity_was_critical"] = True
                    step_reward += 0.05
                    feedback_parts.append("✓ Severity correctly identified as critical.")
                if action.explanation and "index" in action.explanation.lower():
                    session["has_index_mention"] = True
            else:
                step_reward -= 0.05
                feedback_parts.append(f"✗ '{cat}' is not a primary issue here.")

        elif action.action_type == ActionType.SUGGEST_FIX:
            if not action.corrected_sql:
                step_reward -= 0.05
                feedback_parts.append("✗ suggest_fix requires a corrected_sql field.")
            else:
                session["fix_sql"] = action.corrected_sql
                sql_lower = action.corrected_sql.lower()
                hits = sum(1 for kw in task["fix_keywords"] if kw.lower() in sql_lower)
                ratio = hits / max(len(task["fix_keywords"]), 1)
                partial = round(ratio * 0.25, 4)
                step_reward += partial
                feedback_parts.append(
                    f"Fix received. Keyword match: {hits}/{len(task['fix_keywords'])}."
                )
                if "index" in sql_lower:
                    session["has_index_mention"] = True

        elif action.action_type == ActionType.APPROVE:
            if task["expected_issues"] and not session["issues_found"]:
                step_reward -= 0.20
                feedback_parts.append("✗ Approved a query with real issues — big penalty.")
            else:
                feedback_parts.append("Approval recorded.")

        elif action.action_type == ActionType.REJECT:
            if session["issues_found"]:
                step_reward += 0.05
                feedback_parts.append("✓ Rejection after identifying issues — good practice.")
            else:
                feedback_parts.append(
                    "Rejection recorded but no issues formally identified."
                )

        # Episode termination
        terminal = action.action_type in (ActionType.APPROVE, ActionType.REJECT)
        done = terminal or (step >= max_steps)

        # Final grader score
        final_score = 0.0
        if done:
            final_score = _run_grader(session)
            feedback_parts.append(
                f"── Episode complete. Final score: {final_score:.4f}"
            )
            # Clean up session to free memory
            _sessions.pop(self._session_id, None)
            self._session_id = None

        session["cumulative_reward"] += step_reward
        reward = final_score if done else max(0.0, min(session["cumulative_reward"], 1.0))

        return SqlQueryReviewObservation(
            query_id=task["id"],
            raw_sql=task["raw_sql"],
            schema_context=task["schema_context"],
            task_description=task["task_description"],
            feedback=" | ".join(feedback_parts) if feedback_parts else "Action recorded.",
            step_count=step,
            max_steps=max_steps,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        if self._session_id is None:
            return State(episode_id="", step_count=0)
        return _sessions.get(self._session_id, {}).get(
            "state", State(episode_id="", step_count=0)
        )