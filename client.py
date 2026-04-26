# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Music Rl Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MusicRlAction, MusicRlObservation


class MusicRlEnv(
    EnvClient[MusicRlAction, MusicRlObservation, State]
):
    """
    Client for the Music Rl Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MusicRlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MusicRlAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MusicRlEnv.from_docker_image("music_rl_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MusicRlAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MusicRlAction) -> Dict:
        """
        Convert MusicRlAction to JSON payload for step message.

        Args:
            action: MusicRlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict = {}
        if action.song_id is not None:
            payload["song_id"] = action.song_id
        if action.song_index is not None:
            payload["song_index"] = action.song_index
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[MusicRlObservation]:
        """
        Parse server response into StepResult[MusicRlObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MusicRlObservation
        """
        obs_data = payload.get("observation", {})
        observation = MusicRlObservation(
            step_number=obs_data.get("step_number", 0),
            recommendation_history=obs_data.get("recommendation_history", []),
            liked_song_ids=obs_data.get("liked_song_ids", []),
            disliked_song_ids=obs_data.get("disliked_song_ids", []),
            phase=obs_data.get("phase", 0),
            last_outcome=obs_data.get("last_outcome", "none"),
            recent_song_features=obs_data.get("recent_song_features", [0.0, 0.0, 0.0]),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
