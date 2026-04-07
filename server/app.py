# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Sql Query Review Environment.

This module creates an HTTP server that exposes the SqlQueryReviewEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SqlQueryReviewAction, SqlQueryReviewObservation
    from .sql_query_review_environment import SqlQueryReviewEnvironment
except (ModuleNotFoundError, ImportError):
    from models import SqlQueryReviewAction, SqlQueryReviewObservation
    from server.sql_query_review_environment import SqlQueryReviewEnvironment


# Create the app with web interface and README integration
app = create_app(
    SqlQueryReviewEnvironment,
    SqlQueryReviewAction,
    SqlQueryReviewObservation,
    env_name="sql_query_review",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for local development or single-process runs."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
