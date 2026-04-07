# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SQL Query Review Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SqlQueryReviewAction, SqlQueryReviewObservation


class SqlQueryReviewEnv(
    EnvClient[SqlQueryReviewAction, SqlQueryReviewObservation, State]
):
    """
    Client for the SQL Query Review Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance gets its own dedicated session.

    Example (sync):
        with SqlQueryReviewEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            print(result.observation.raw_sql)

            result = client.step(SqlQueryReviewAction(
                action_type="identify_issue",
                issue_category="cartesian_product",
                severity="high",
                explanation="Missing JOIN condition."
            ))
            print(result.reward)

    Example (async):
        async with SqlQueryReviewEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(action)
    """

    def _step_payload(self, action: SqlQueryReviewAction) -> Dict:
        """
        Convert SqlQueryReviewAction to JSON payload for the WebSocket step message.
        Every field in the action needs to be included here.
        """
        return {
            "action_type": action.action_type.value,
            "issue_category": action.issue_category.value if action.issue_category else "none",
            "severity": action.severity.value if action.severity else None,
            "explanation": action.explanation,
            "corrected_sql": action.corrected_sql,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SqlQueryReviewObservation]:
        """
        Parse the server's JSON response into a typed StepResult.
        The server sends back an observation dict — we unpack it here.
        """
        obs_data = payload.get("observation", {})

        observation = SqlQueryReviewObservation(
            query_id=obs_data.get("query_id", ""),
            raw_sql=obs_data.get("raw_sql", ""),
            schema_context=obs_data.get("schema_context", ""),
            task_description=obs_data.get("task_description", ""),
            feedback=obs_data.get("feedback", ""),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into a State object."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )