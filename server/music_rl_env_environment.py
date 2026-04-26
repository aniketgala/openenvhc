# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Music recommendation RL environment implementation."""

import json
import random
from pathlib import Path
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MusicRlAction, MusicRlObservation
except ImportError:
    from models import MusicRlAction, MusicRlObservation


class MusicRlEnvironment(Environment):
    """Minimal OpenEnv-compatible music preference environment."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    EPISODE_LENGTH: int = 10
    MOOD_SHIFT_STEP: int = 5
    DEFAULT_SEED: int = 42
    _FEATURE_DEBUG_PRINTED: bool = False

    def __init__(self, seed: int = DEFAULT_SEED):
        """Initialize environment state and deterministic catalog RNG."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._seed = seed
        self._rng = random.Random(self._seed)
        self._songs = self._load_songs()
        self._song_index = {song["song_id"]: song for song in self._songs}
        self._genres = sorted({song["genre"] for song in self._songs})
        self._moods = sorted({song["mood"] for song in self._songs})
        self._recommendation_history: list[str] = []
        self._liked_song_ids: list[str] = []
        self._disliked_song_ids: list[str] = []
        self._preferred_genre = ""
        self._preferred_mood = ""
        self._shifted_mood = ""
        self._active_mood = ""
        self._last_outcome = "none"
        self._recent_song_features: list[float] = [0.0, 0.0, 0.0]
        self._init_hidden_preferences()

    def _load_songs(self) -> list[dict[str, str | float]]:
        dataset_path = Path(__file__).resolve().parent.parent / "dataset.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset.csv at: {dataset_path}")

        df = pd.read_csv(dataset_path)
        required = ["energy", "valence", "danceability"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"dataset.csv missing required columns: {missing}")

        songs: list[dict[str, str | float]] = []
        for idx, row in df.iterrows():
            energy = float(row["energy"])
            valence = float(row["valence"])
            danceability = float(row["danceability"])
            genre_raw = str(row["track_genre"]).strip() if "track_genre" in df.columns else "unknown"
            genre = genre_raw if genre_raw else "unknown"
            mood = "positive" if valence >= 0.6 else ("neutral" if valence >= 0.4 else "negative")
            song = {
                "song_id": f"song_{idx}",
                "genre": genre,
                "mood": mood,
                "energy": energy,
                "valence": valence,
                "danceability": danceability,
            }
            songs.append(song)

        if not songs:
            raise ValueError("dataset.csv must contain at least one valid row")

        if not MusicRlEnvironment._FEATURE_DEBUG_PRINTED:
            print("Sample songs after load:")
            for s in songs[:5]:
                print(s)
            energy_vals = [float(s["energy"]) for s in songs]
            valence_vals = [float(s["valence"]) for s in songs]
            dance_vals = [float(s["danceability"]) for s in songs]
            print("Energy range:", min(energy_vals), max(energy_vals))
            print("Valence range:", min(valence_vals), max(valence_vals))
            print("Dance range:", min(dance_vals), max(dance_vals))
            MusicRlEnvironment._FEATURE_DEBUG_PRINTED = True
        return songs

    def _init_hidden_preferences(self) -> None:
        self._preferred_genre = self._rng.choice(self._genres)
        self._preferred_mood = self._rng.choice(self._moods)
        self._shifted_mood = self._rng.choice(
            [mood for mood in self._moods if mood != self._preferred_mood]
        )
        self._active_mood = self._preferred_mood

    def _observation(self) -> MusicRlObservation:
        phase = 0 if self._state.step_count < self.MOOD_SHIFT_STEP else 1
        return MusicRlObservation(
            step_number=self._state.step_count,
            recommendation_history=list(self._recommendation_history),
            liked_song_ids=list(self._liked_song_ids),
            disliked_song_ids=list(self._disliked_song_ids),
            phase=phase,
            last_outcome=self._last_outcome,
            recent_song_features=list(self._recent_song_features),
            done=self._state.step_count >= self.EPISODE_LENGTH,
            reward=0.0,
        )

    def _resolve_song(self, action: MusicRlAction) -> tuple[str | None, dict[str, str] | None]:
        if action.song_index is not None:
            idx = action.song_index
            if 0 <= idx < len(self._songs):
                song = self._songs[idx]
                return song["song_id"], song
            return None, None
        if action.song_id is not None and action.song_id in self._song_index:
            song = self._song_index[action.song_id]
            return action.song_id, song
        return None, None

    def _extract_recent_features(self, song: dict[str, str | float]) -> list[float]:
        return [
            float(song["energy"]),
            float(song["valence"]),
            float(song["danceability"]),
        ]

    def _compute_reward(self, song: dict[str, str], is_repeat: bool) -> dict[str, float]:
        genre_match = int(song["genre"] == self._preferred_genre)
        mood_match = int(song["mood"] == self._active_mood)
        repeat_penalty = -0.5 if is_repeat else 0.0
        adapt_bonus = (
            0.5
            if self._state.step_count >= self.MOOD_SHIFT_STEP and song["mood"] == self._shifted_mood
            else 0.0
        )

        mismatch_penalty = -1.0 if genre_match == 0 and mood_match == 0 else 0.0
        total_reward = genre_match + mood_match + repeat_penalty + adapt_bonus + mismatch_penalty

        return {
            "genre_match": genre_match,
            "mood_match": mood_match,
            "repeat_penalty": repeat_penalty,
            "adapt_bonus": adapt_bonus,
            "total_reward": float(total_reward),
        }

    def reset(self) -> MusicRlObservation:
        """Reset environment and return initial observable state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._recommendation_history = []
        self._liked_song_ids = []
        self._disliked_song_ids = []
        self._last_outcome = "none"
        self._recent_song_features = [0.0, 0.0, 0.0]
        self._init_hidden_preferences()
        return self._observation()

    def step(self, action: MusicRlAction) -> MusicRlObservation:  # type: ignore[override]
        """Apply recommendation action and return next observation."""
        song_id, song = self._resolve_song(action)
        if song_id is None or song is None:
            safe_obs = self._observation()
            safe_obs.done = True
            safe_obs.reward = -2.0
            safe_obs.metadata = {
                "genre_match": 0,
                "mood_match": 0,
                "repeat_penalty": 0.0,
                "adapt_bonus": 0.0,
                "total_reward": -2.0,
            }
            return safe_obs

        if self._state.step_count == self.MOOD_SHIFT_STEP:
            self._active_mood = self._shifted_mood

        is_repeat = song_id in self._recommendation_history
        reward_info = self._compute_reward(song, is_repeat)

        self._recommendation_history.append(song_id)
        if reward_info["genre_match"] or reward_info["mood_match"]:
            self._liked_song_ids.append(song_id)
            self._last_outcome = "positive"
        else:
            self._disliked_song_ids.append(song_id)
            self._last_outcome = "negative" if reward_info["total_reward"] < 0 else "none"
        self._recent_song_features = self._extract_recent_features(song)

        self._state.step_count += 1
        done = self._state.step_count >= self.EPISODE_LENGTH
        obs = self._observation()
        obs.done = done
        obs.reward = reward_info["total_reward"]
        obs.metadata = reward_info
        return obs

    def step_transition(self, action: MusicRlAction) -> tuple[MusicRlObservation, float, bool, dict]:
        """Convenience transition API for local RL trainers."""
        obs = self.step(action)
        return obs, float(obs.reward or 0.0), bool(obs.done), dict(obs.metadata or {})

    def observation_state(self) -> MusicRlObservation:
        """Public observation helper for non-server training loops."""
        return self._observation()

    def run_random_policy_episode(self) -> float:
        """Simulate one episode with random song recommendations."""
        self.reset()
        total_reward = 0.0
        while self._state.step_count < self.EPISODE_LENGTH:
            song_id = self._rng.choice(self._songs)["song_id"]
            obs = self.step(MusicRlAction(song_id=song_id))
            total_reward += obs.reward or 0.0
            if obs.done:
                break
        return total_reward

    def average_random_policy_reward(self, episodes: int = 5) -> float:
        """Compute average episode reward for random baseline."""
        if episodes <= 0:
            return 0.0
        rewards = [self.run_random_policy_episode() for _ in range(episodes)]
        return sum(rewards) / len(rewards)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
