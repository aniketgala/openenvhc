"""Manual test and validation script for the OpenEnv music RL environment."""

from __future__ import annotations

import random
from statistics import mean

from models import MusicRlAction
from server.music_rl_env_environment import MusicRlEnvironment


def _format_breakdown(info: dict) -> str:
    return (
        f"genre_match={info.get('genre_match')}, "
        f"mood_match={info.get('mood_match')}, "
        f"repeat_penalty={info.get('repeat_penalty')}, "
        f"adapt_bonus={info.get('adapt_bonus')}, "
        f"total_reward={info.get('total_reward')}"
    )


def run_random_episode_with_checks(action_seed: int = 7) -> tuple[list[dict], float]:
    """Run one full random episode and validate per-step reward logic."""
    env = MusicRlEnvironment()
    env.reset()

    print("=== STEP 1/2/3: Full Episode + Reward/Mood Validation ===")
    print(f"initial_mood={env._preferred_mood}")  # debug visibility for test only
    print(f"new_mood={env._shifted_mood}")  # debug visibility for test only
    print(f"preferred_genre={env._preferred_genre}")  # debug visibility for test only

    picker = random.Random(action_seed)
    history: list[str] = []
    records: list[dict] = []
    total_reward = 0.0

    while env.state.step_count < env.EPISODE_LENGTH:
        step_before = env.state.step_count
        current_mood = env._preferred_mood if step_before < env.MOOD_SHIFT_STEP else env._shifted_mood
        song_id = picker.choice(env._songs)["song_id"]
        song = env._song_index[song_id]

        obs = env.step(MusicRlAction(song_id=song_id))
        info = obs.metadata or {}

        expected_genre_match = int(song["genre"] == env._preferred_genre)
        expected_mood_match = int(song["mood"] == current_mood)
        expected_repeat_penalty = -0.5 if song_id in history else 0.0
        expected_adapt_bonus = 0.5 if step_before >= env.MOOD_SHIFT_STEP and song["mood"] == env._shifted_mood else 0.0
        expected_mismatch_penalty = -1.0 if expected_genre_match == 0 and expected_mood_match == 0 else 0.0
        expected_total = (
            expected_genre_match
            + expected_mood_match
            + expected_repeat_penalty
            + expected_adapt_bonus
            + expected_mismatch_penalty
        )

        assert info.get("genre_match") == expected_genre_match, "genre_match component mismatch"
        assert info.get("mood_match") == expected_mood_match, "mood_match component mismatch"
        assert info.get("repeat_penalty") == expected_repeat_penalty, "repeat_penalty component mismatch"
        assert info.get("adapt_bonus") == expected_adapt_bonus, "adapt_bonus component mismatch"
        assert info.get("total_reward") == expected_total, "total_reward mismatch"
        assert (obs.reward or 0.0) == expected_total, "observation reward mismatch"

        # Mood shift validation checks.
        if step_before <= 4:
            assert current_mood == env._preferred_mood, "steps 0-4 should use initial mood"
        if step_before >= 5:
            assert current_mood == env._shifted_mood, "steps 5-9 should use shifted mood"

        history.append(song_id)
        total_reward += obs.reward or 0.0
        records.append(
            {
                "step": step_before,
                "song_id": song_id,
                "reward": obs.reward,
                "done": obs.done,
                "info": info,
                "current_mood": current_mood,
            }
        )

        print(
            f"step={step_before}, song_id={song_id}, reward={obs.reward}, done={obs.done}, "
            f"current_mood={current_mood}, breakdown=({_format_breakdown(info)})"
        )

    print(f"episode_total_reward={total_reward}")
    print(f"episode_end_step_count={env.state.step_count}")
    assert env.state.step_count == 10, "Episode must end at exactly step 10"
    assert records[-1]["done"] is True, "Final step should mark done=True"
    return records, total_reward


def validate_edge_cases() -> None:
    """Validate invalid action and repeated-song behavior."""
    print("\n=== STEP 4: Edge Cases ===")

    # Invalid song_id should terminate with -2 reward.
    env_invalid = MusicRlEnvironment()
    env_invalid.reset()
    invalid_obs = env_invalid.step(MusicRlAction(song_id="not_a_real_song"))
    invalid_info = invalid_obs.metadata or {}
    print(
        f"invalid_action -> reward={invalid_obs.reward}, done={invalid_obs.done}, "
        f"breakdown=({_format_breakdown(invalid_info)})"
    )
    assert invalid_obs.reward == -2.0, "Invalid action must return reward -2"
    assert invalid_obs.done is True, "Invalid action must terminate episode"
    assert invalid_info.get("total_reward") == -2.0, "Invalid action total_reward mismatch"

    # Repeated song penalty check.
    env_repeat = MusicRlEnvironment()
    env_repeat.reset()
    repeat_song = env_repeat._songs[0]["song_id"]
    first_obs = env_repeat.step(MusicRlAction(song_id=repeat_song))
    second_obs = env_repeat.step(MusicRlAction(song_id=repeat_song))
    print(
        f"repeat_action -> first_repeat_penalty={(first_obs.metadata or {}).get('repeat_penalty')}, "
        f"second_repeat_penalty={(second_obs.metadata or {}).get('repeat_penalty')}"
    )
    assert (second_obs.metadata or {}).get("repeat_penalty") == -0.5, "Repeat penalty must be -0.5"


def validate_reward_components_targeted() -> None:
    """Run targeted checks for each reward rule."""
    print("\n=== STEP 2: Targeted Reward Component Checks ===")
    env = MusicRlEnvironment()
    env.reset()

    genre_match_song = next(s for s in env._songs if s["genre"] == env._preferred_genre)
    mood_match_song = next(s for s in env._songs if s["mood"] == env._preferred_mood)
    mismatch_song = next(
        s
        for s in env._songs
        if s["genre"] != env._preferred_genre and s["mood"] != env._preferred_mood
    )
    shifted_mood_song = next(s for s in env._songs if s["mood"] == env._shifted_mood)

    obs_genre = env.step(MusicRlAction(song_id=genre_match_song["song_id"]))
    obs_mood = env.step(MusicRlAction(song_id=mood_match_song["song_id"]))
    obs_mismatch = env.step(MusicRlAction(song_id=mismatch_song["song_id"]))

    assert (obs_genre.metadata or {}).get("genre_match") == 1, "Genre match should contribute +1"
    assert (obs_mood.metadata or {}).get("mood_match") == 1, "Mood match should contribute +1"
    assert (obs_mismatch.metadata or {}).get("total_reward") <= -1.0, "Both mismatch should include -1"

    # Adapt bonus should only apply at/after step 5.
    env_adapt = MusicRlEnvironment()
    env_adapt.reset()
    early_obs = env_adapt.step(MusicRlAction(song_id=shifted_mood_song["song_id"]))
    assert (early_obs.metadata or {}).get("adapt_bonus") == 0.0, "Adapt bonus must be 0 before step 5"

    while env_adapt.state.step_count < 5:
        env_adapt.step(MusicRlAction(song_id=env_adapt._songs[0]["song_id"]))
    late_obs = env_adapt.step(MusicRlAction(song_id=shifted_mood_song["song_id"]))
    assert (late_obs.metadata or {}).get("adapt_bonus") == 0.5, "Adapt bonus must apply at/after step 5"

    print("Reward component checks passed.")


def validate_baseline_and_sanity() -> None:
    """Run baseline policy and determinism sanity checks."""
    print("\n=== STEP 5/6: Baseline + Sanity Checks ===")
    env = MusicRlEnvironment()
    episode_rewards = [env.run_random_policy_episode() for _ in range(10)]
    average_reward = mean(episode_rewards)
    helper_average = env.average_random_policy_reward(episodes=10)

    print(f"individual_episode_rewards={episode_rewards}")
    print(f"average_reward={average_reward}")
    print(f"helper_average_reward={helper_average}")

    assert len(set(episode_rewards)) > 1, "Rewards should not be constant"
    assert any(reward != 0.0 for reward in episode_rewards), "Rewards should not be all zero"

    # Determinism check: same seed + same action sampling seed should reproduce trajectory.
    recs_a, _ = run_random_episode_with_checks(action_seed=17)
    recs_b, _ = run_random_episode_with_checks(action_seed=17)
    trajectory_a = [(r["song_id"], r["reward"], r["info"]["total_reward"]) for r in recs_a]
    trajectory_b = [(r["song_id"], r["reward"], r["info"]["total_reward"]) for r in recs_b]
    assert trajectory_a == trajectory_b, "Environment should be deterministic for same seeds"

    print("Sanity checks passed.")


def main() -> None:
    run_random_episode_with_checks(action_seed=7)
    validate_reward_components_targeted()
    validate_edge_cases()
    validate_baseline_and_sanity()
    print("\nAll tests completed successfully.")


if __name__ == "__main__":
    main()
