# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the music recommendation RL environment."""

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MusicRlAction(Action):
    """Action containing the recommended song identifier."""

    song_id: Optional[str] = Field(default=None, description="Song ID selected by the agent")
    song_index: Optional[int] = Field(default=None, description="Song index selected by the agent")


class MusicRlObservation(Observation):
    """Observable environment state exposed to the agent."""

    step_number: int = Field(default=0, description="Current step number")
    recommendation_history: list[str] = Field(
        default_factory=list, description="Song IDs recommended so far"
    )
    liked_song_ids: list[str] = Field(
        default_factory=list, description="Song IDs with positive feedback"
    )
    disliked_song_ids: list[str] = Field(
        default_factory=list, description="Song IDs with negative feedback"
    )
    phase: int = Field(default=0, description="0 before shift, 1 after shift")
    last_outcome: str = Field(default="none", description="Outcome of previous action")
    recent_song_features: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Recent song [energy, valence, danceability]",
    )
