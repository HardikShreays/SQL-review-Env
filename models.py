# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Sql Query Review Environment.

Defines typed actions and observations for the SQL code-review task.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(str, Enum):
    IDENTIFY_ISSUE = "identify_issue"
    SUGGEST_FIX = "suggest_fix"
    APPROVE = "approve"
    REJECT = "reject"


class IssueCategory(str, Enum):
    SQL_INJECTION = "sql_injection"
    MISSING_INDEX = "missing_index"
    N_PLUS_ONE = "n_plus_one"
    INCORRECT_LOGIC = "incorrect_logic"
    CARTESIAN_PRODUCT = "cartesian_product"
    MISSING_NULL_CHECK = "missing_null_check"   # ← add this
    NONE = "none"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SqlQueryReviewAction(Action):
    """What the agent does each step."""
    action_type: ActionType = Field(..., description="Type of review action")
    issue_category: IssueCategory = Field(
        default=IssueCategory.NONE,
        description="Which issue category was found"
    )
    severity: Optional[Severity] = Field(
        default=None,
        description="How severe is the issue"
    )
    explanation: str = Field(
        ...,
        description="Agent's reasoning in plain English",
        min_length=5
    )
    corrected_sql: Optional[str] = Field(
        default=None,
        description="Fixed SQL (required when action_type is suggest_fix)"
    )


class SqlQueryReviewObservation(Observation):
    """What the agent sees each step."""
    query_id: str = Field(default="", description="Which task is being reviewed")
    raw_sql: str = Field(default="", description="The SQL query under review")
    schema_context: str = Field(default="", description="Relevant table definitions")
    task_description: str = Field(default="", description="What the developer intended")
    feedback: str = Field(default="", description="Environment feedback on last action")
    step_count: int = Field(default=0, description="How many steps taken so far")
    max_steps: int = Field(default=10, description="Max steps before episode ends")