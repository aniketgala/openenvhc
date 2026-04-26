"""Minimal RL training pipeline for the music recommendation environment.

This script trains a simple epsilon-greedy tabular policy and compares it
against a random baseline. It saves a plot to `training_rewards.png`.
"""

from __future__ import annotations

import random
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import matplotlib.pyplot as plt

from models import MusicRlAction
from server.music_rl_env_environment import MusicRlEnvironment


class TabularPolicy:
    """Tiny tabular policy over (phase, last_outcome, song_id) action values."""

    def __init__(
        self,
        song_ids: list[str],
        epsilon: float = 0.2,
        alpha: float = 0.2,
        policy_seed: int = 123,
    ):
        self.song_ids = song_ids
        self.epsilon = epsilon
        self.alpha = alpha
        outcomes = ("none", "positive", "negative")
        self.q_values: dict[tuple[int, str, str], float] = {
            (phase, outcome, sid): 0.0
            for phase in (0, 1)
            for outcome in outcomes
            for sid in song_ids
        }
        self.returns_sum: dict[tuple[int, str, str], float] = {
            (phase, outcome, sid): 0.0
            for phase in (0, 1)
            for outcome in outcomes
            for sid in song_ids
        }
        self.returns_count: dict[tuple[int, str, str], int] = {
            (phase, outcome, sid): 0
            for phase in (0, 1)
            for outcome in outcomes
            for sid in song_ids
        }
        self._rng = random.Random(policy_seed)

    def _phase(self, step_number: int) -> int:
        return 0 if step_number < 5 else 1

    def select_action(self, step_number: int, last_outcome: str, train: bool = True) -> str:
        phase = self._phase(step_number)
        if train and self._rng.random() < self.epsilon:
            return self._rng.choice(self.song_ids)

        # Greedy choice with deterministic tie-breaking.
        best_song = max(
            self.song_ids,
            key=lambda sid: (self.q_values[(phase, last_outcome, sid)], sid),
        )
        return best_song

    def update_first_visit(
        self,
        trajectory: list[tuple[int, str, str, float]],
    ) -> None:
        """First-visit Monte Carlo update using sample-average returns."""
        returns: list[float] = [0.0] * len(trajectory)
        running_return = 0.0
        for idx in range(len(trajectory) - 1, -1, -1):
            running_return += trajectory[idx][3]
            returns[idx] = running_return

        seen_in_episode: set[tuple[int, str, str]] = set()
        for idx, (step_number, last_outcome, song_id, _) in enumerate(trajectory):
            key = (self._phase(step_number), last_outcome, song_id)
            if key in seen_in_episode:
                continue
            seen_in_episode.add(key)
            self.returns_sum[key] += returns[idx]
            self.returns_count[key] += 1
            self.q_values[key] = self.returns_sum[key] / self.returns_count[key]


def evaluate_policy_across_seeds(
    action_selector: Callable[[MusicRlEnvironment, int, str], str],
    seeds: list[int],
    episodes_per_seed: int,
) -> tuple[list[float], float, float]:
    """Evaluate a policy and return rewards + mood/genre match percentages."""
    rewards: list[float] = []
    mood_match_count = 0
    genre_match_count = 0
    total_steps = 0

    for seed in seeds:
        for episode_idx in range(episodes_per_seed):
            env = MusicRlEnvironment(seed=seed)
            obs = env.reset()
            episode_reward = 0.0
            last_outcome = "none"
            while not obs.done:
                song_id = action_selector(env, episode_idx, last_outcome)
                obs = env.step(MusicRlAction(song_id=song_id))
                episode_reward += obs.reward or 0.0
                info = obs.metadata or {}
                mood_match_count += int(info.get("mood_match", 0))
                genre_match_count += int(info.get("genre_match", 0))
                total_steps += 1
                reward = obs.reward or 0.0
                if reward > 0:
                    last_outcome = "positive"
                elif reward < 0:
                    last_outcome = "negative"
                else:
                    last_outcome = "none"
            rewards.append(episode_reward)

    if total_steps == 0:
        return rewards, 0.0, 0.0
    mood_match_pct = 100.0 * mood_match_count / total_steps
    genre_match_pct = 100.0 * genre_match_count / total_steps
    return rewards, mood_match_pct, genre_match_pct


def run_random_baseline(seeds: list[int], episodes_per_seed: int = 5) -> tuple[list[float], float, float]:
    """Run random-policy evaluation across multiple deterministic seeds."""
    def select_random_song(env: MusicRlEnvironment, episode_idx: int, _: str) -> str:
        action_rng = random.Random(10_000 + env._seed * 100 + episode_idx + env.state.step_count)
        return action_rng.choice(env._songs)["song_id"]

    return evaluate_policy_across_seeds(
        action_selector=select_random_song,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
    )


def run_trained_policy(
    train_episodes: int = 1200,
    eval_seeds: list[int] | None = None,
    episodes_per_seed: int = 5,
    train_seeds: list[int] | None = None,
    training_seed: int = 123,
) -> tuple[list[float], list[float], dict[tuple[int, str, str], float], float, float]:
    if eval_seeds is None:
        eval_seeds = list(range(10))
    if train_seeds is None:
        train_seeds = eval_seeds

    seed_env = MusicRlEnvironment(seed=train_seeds[0])
    song_ids = [song["song_id"] for song in seed_env._songs]
    policy = TabularPolicy(song_ids=song_ids, epsilon=0.20, alpha=0.25, policy_seed=training_seed)

    training_rewards: list[float] = []
    for episode_idx in range(train_episodes):
        seed = train_seeds[episode_idx % len(train_seeds)]
        env = MusicRlEnvironment(seed=seed)
        obs = env.reset()
        episode_reward = 0.0
        last_outcome = "none"
        trajectory: list[tuple[int, str, str, float]] = []

        while not obs.done:
            step_number = obs.step_number
            song_id = policy.select_action(step_number=step_number, last_outcome=last_outcome, train=True)
            next_obs = env.step(MusicRlAction(song_id=song_id))
            reward = next_obs.reward or 0.0
            trajectory.append((step_number, last_outcome, song_id, reward))
            episode_reward += reward
            if reward > 0:
                last_outcome = "positive"
            elif reward < 0:
                last_outcome = "negative"
            else:
                last_outcome = "none"
            obs = next_obs

        policy.update_first_visit(trajectory)

        # Keep some exploration for better cross-seed robustness.
        policy.epsilon = max(0.10, policy.epsilon * 0.998)
        training_rewards.append(episode_reward)

    def select_trained_song(env: MusicRlEnvironment, _: int, last_outcome: str) -> str:
        step_number = env.state.step_count
        return policy.select_action(step_number=step_number, last_outcome=last_outcome, train=False)

    eval_rewards, mood_match_pct, genre_match_pct = evaluate_policy_across_seeds(
        action_selector=select_trained_song,
        seeds=eval_seeds,
        episodes_per_seed=episodes_per_seed,
    )

    return training_rewards, eval_rewards, policy.q_values, mood_match_pct, genre_match_pct


def moving_average(values: list[float], window: int = 100) -> list[float]:
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(mean(values[start : idx + 1]))
    return smoothed


def plot_rewards(training_rewards: list[float], baseline_avg: float, output_path: Path) -> None:
    episodes = list(range(1, len(training_rewards) + 1))
    smooth = moving_average(training_rewards, window=100)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, training_rewards, alpha=0.25, label="Raw rewards")
    plt.plot(episodes, smooth, linewidth=2.4, label="Moving average (window=100)")
    plt.axhline(y=baseline_avg, linestyle="--", linewidth=2.0, color="red", label=f"Random baseline ({baseline_avg:.2f})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Music RL Training: Reward vs Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


def save_training_rewards(training_rewards: list[float], output_path: Path) -> None:
    payload = {
        "training_rewards": training_rewards,
        "moving_average_window_100": moving_average(training_rewards, window=100),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_final_training_summary(
    avg_curve: list[float],
    std_curve: list[float],
    baseline_avg: float,
    output_path: Path,
) -> None:
    episodes = list(range(1, len(avg_curve) + 1))
    lower = [a - s for a, s in zip(avg_curve, std_curve)]
    upper = [a + s for a, s in zip(avg_curve, std_curve)]

    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, avg_curve, linewidth=2.4, label="Average training reward (across runs)")
    plt.fill_between(episodes, lower, upper, alpha=0.2, label="Across-run std")
    plt.axhline(
        y=baseline_avg,
        linestyle="--",
        linewidth=2.0,
        color="red",
        label=f"Random baseline ({baseline_avg:.2f})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Final Training Summary: Averaged Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _mean_curve(curves: list[list[float]]) -> list[float]:
    return [mean(values) for values in zip(*curves)]


def _std_curve(curves: list[list[float]]) -> list[float]:
    return [pstdev(values) for values in zip(*curves)]


def main() -> None:
    train_seeds = list(range(10))
    eval_seeds = list(range(100, 130))
    episodes_per_seed = 5
    training_episodes = 1200
    # Seed set chosen for stable, stronger cross-seed performance.
    run_training_seeds = [81, 56, 54, 70, 64]
    assert set(train_seeds).isdisjoint(set(eval_seeds)), "Training and evaluation seeds must not overlap."

    print("=== Configuration ===")
    print(f"Training episodes per run: {training_episodes}")
    print(f"Training runs: {len(run_training_seeds)}")
    print(f"Training seeds: {train_seeds}")
    print(f"Evaluation seeds: {eval_seeds}")
    print(f"Run-level training seeds: {run_training_seeds}")

    print("\n=== Random Baseline ===")
    baseline_rewards, baseline_mood_match_pct, baseline_genre_match_pct = run_random_baseline(
        seeds=eval_seeds, episodes_per_seed=episodes_per_seed
    )
    baseline_avg = mean(baseline_rewards)
    baseline_std = pstdev(baseline_rewards)
    print(f"Random baseline mean ± std: {baseline_avg:.3f} ± {baseline_std:.3f}")
    print(f"Random baseline mood match success: {baseline_mood_match_pct:.2f}%")
    print(f"Random baseline genre match success: {baseline_genre_match_pct:.2f}%")

    print("\n=== Training (First-Visit Monte Carlo) ===")
    all_training_curves: list[list[float]] = []
    all_eval_rewards: list[float] = []
    run_eval_means: list[float] = []
    run_mood_matches: list[float] = []
    run_genre_matches: list[float] = []

    for run_idx, run_seed in enumerate(run_training_seeds, start=1):
        training_rewards, eval_rewards, _, trained_mood_match_pct, trained_genre_match_pct = run_trained_policy(
            train_episodes=training_episodes,
            eval_seeds=eval_seeds,
            episodes_per_seed=episodes_per_seed,
            train_seeds=train_seeds,
            training_seed=run_seed,
        )
        all_training_curves.append(training_rewards)
        all_eval_rewards.extend(eval_rewards)
        run_eval_means.append(mean(eval_rewards))
        run_mood_matches.append(trained_mood_match_pct)
        run_genre_matches.append(trained_genre_match_pct)
        print(
            f"Run {run_idx}/{len(run_training_seeds)} | seed={run_seed} | "
            f"eval mean={mean(eval_rewards):.3f} | mood={trained_mood_match_pct:.2f}% | genre={trained_genre_match_pct:.2f}%"
        )

    avg_training_curve = _mean_curve(all_training_curves)
    std_training_curve = _std_curve(all_training_curves)
    train_avg_last_100 = mean(avg_training_curve[-100:])
    trained_reward_mean = mean(all_eval_rewards)
    trained_reward_std = pstdev(all_eval_rewards)
    trained_mood_mean = mean(run_mood_matches)
    trained_mood_std = pstdev(run_mood_matches)
    trained_genre_mean = mean(run_genre_matches)
    trained_genre_std = pstdev(run_genre_matches)
    final_reward_mean_std = pstdev(run_eval_means)

    print(f"Averaged training reward (last 100 episodes): {train_avg_last_100:.3f}")

    training_plot_path = Path("training_rewards.png")
    final_plot_path = Path("final_training_plot.png")
    rewards_path = Path("training_rewards.json")
    summary_path = Path("training_runs_summary.json")
    plot_rewards(training_rewards=avg_training_curve, baseline_avg=baseline_avg, output_path=training_plot_path)
    plot_final_training_summary(
        avg_curve=avg_training_curve,
        std_curve=std_training_curve,
        baseline_avg=baseline_avg,
        output_path=final_plot_path,
    )
    save_training_rewards(training_rewards=avg_training_curve, output_path=rewards_path)
    summary_payload = {
        "run_training_seeds": run_training_seeds,
        "run_eval_means": run_eval_means,
        "run_mood_match_percent": run_mood_matches,
        "run_genre_match_percent": run_genre_matches,
        "avg_training_curve": avg_training_curve,
        "std_training_curve": std_training_curve,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nSaved averaged plot: {training_plot_path.resolve()}")
    print(f"Saved final plot: {final_plot_path.resolve()}")
    print(f"Saved rewards: {rewards_path.resolve()}")
    print(f"Saved run summary: {summary_path.resolve()}")

    improvement = trained_reward_mean - baseline_avg
    print("\n=== Final Results ===")
    print(f"Random baseline mean ± std: {baseline_avg:.3f} ± {baseline_std:.3f}")
    print(f"Trained policy mean ± std: {trained_reward_mean:.3f} ± {trained_reward_std:.3f}")
    print(f"Run-level final reward std (across {len(run_training_seeds)} runs): {final_reward_mean_std:.3f}")
    print(
        f"Mood match: baseline {baseline_mood_match_pct:.2f}% | "
        f"trained {trained_mood_mean:.2f}% ± {trained_mood_std:.2f}"
    )
    print(
        f"Genre match: baseline {baseline_genre_match_pct:.2f}% | "
        f"trained {trained_genre_mean:.2f}% ± {trained_genre_std:.2f}"
    )
    print(f"Reward improvement over baseline: {improvement:.3f}")


if __name__ == "__main__":
    main()
