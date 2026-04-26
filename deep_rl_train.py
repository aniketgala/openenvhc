"""DRQN-lite training on a sampled Spotify dataset subset.

Upgrades state-action DQN to recurrent Q(s, a) with a compact LSTM encoder.
Environment simulator behavior is intentionally unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset.csv"
MODEL_PATH = ROOT / "model.pt"
BEST_MODEL_PATH = ROOT / "best_model.pt"
FINAL_TRAIN_PLOT_PATH = ROOT / "final_training_plot.png"
REWARD_DIST_PLOT_PATH = ROOT / "reward_distribution.png"
LOSS_PLOT_PATH = ROOT / "drqn_loss_plot.png"
COMPARISON_BAR_PATH = ROOT / "drqn_comparison_bar.png"
SUMMARY_PATH = ROOT / "summary.json"

FEATURE_COLUMNS = ["energy", "valence", "tempo", "danceability", "acousticness"]
STATE_FEATURE_COLUMNS = ["energy", "valence", "danceability"]
STATE_FEATURE_INDICES = [FEATURE_COLUMNS.index(col) for col in STATE_FEATURE_COLUMNS]
SEQUENCE_LENGTH = 6
STEP_FEATURE_DIM = 5  # [phase, last_outcome, energy, valence, danceability]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def minmax_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        min_val = float(out[col].min())
        max_val = float(out[col].max())
        if max_val - min_val < 1e-9:
            out[col] = 0.5
        else:
            out[col] = (out[col] - min_val) / (max_val - min_val)
    return out


def safe_minmax(value: float, min_val: float, max_val: float) -> float:
    if max_val - min_val < 1e-6:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def normalize_song_features_with_global_stats(
    sampled: pd.DataFrame,
    full_reference: pd.DataFrame,
) -> pd.DataFrame:
    out = sampled.copy()
    energy_vals = full_reference["energy"].astype(float).tolist()
    valence_vals = full_reference["valence"].astype(float).tolist()
    dance_vals = full_reference["danceability"].astype(float).tolist()

    energy_min, energy_max = min(energy_vals), max(energy_vals)
    valence_min, valence_max = min(valence_vals), max(valence_vals)
    dance_min, dance_max = min(dance_vals), max(dance_vals)

    out["energy"] = out["energy"].astype(float).map(lambda x: safe_minmax(float(x), energy_min, energy_max))
    out["valence"] = out["valence"].astype(float).map(lambda x: safe_minmax(float(x), valence_min, valence_max))
    out["danceability"] = out["danceability"].astype(float).map(lambda x: safe_minmax(float(x), dance_min, dance_max))

    # Keep existing behavior for other features.
    out = minmax_normalize(out, ["tempo", "acousticness"])

    print("Energy range:", float(out["energy"].min()), float(out["energy"].max()))
    print("Valence range:", float(out["valence"].min()), float(out["valence"].max()))
    print("Dance range:", float(out["danceability"].min()), float(out["danceability"].max()))
    return out


def load_song_subset(
    dataset_path: Path,
    sample_size: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {dataset_path}. Place Spotify dataset as dataset.csv in project root."
        )
    raw = pd.read_csv(dataset_path)
    missing = [col for col in FEATURE_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"dataset.csv missing required columns: {missing}")

    work = raw[FEATURE_COLUMNS].dropna().reset_index(drop=True)
    if len(work) < 50:
        raise ValueError("dataset.csv has too few usable rows after filtering.")

    capped = min(100, max(50, sample_size))
    n = min(capped, len(work))
    sampled = work.sample(n=n, random_state=seed).reset_index(drop=True)
    sampled = normalize_song_features_with_global_stats(sampled=sampled, full_reference=work)
    features = sampled[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return sampled, features


@dataclass
class Transition:
    seq: np.ndarray
    action_features: np.ndarray
    reward: float
    next_seq: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DRQNLite(nn.Module):
    def __init__(self, feature_dim: int, action_feature_dim: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=32,
            batch_first=True,
        )
        self.net = nn.Sequential(
            nn.Linear(32 + action_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        # seq shape: [batch, K, feature_dim]
        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[:, -1, :]
        x = torch.cat([last_hidden, action_features], dim=1)
        return self.net(x).squeeze(1)


class SpotifyRecEnv:
    """Lightweight recommendation simulator based on normalized song features."""

    def __init__(self, song_features: np.ndarray, seed: int = 42, episode_length: int = 20):
        self.song_features = song_features
        self.n_songs = song_features.shape[0]
        self.seed = seed
        self.episode_length = episode_length
        self.rng = random.Random(seed)
        self.step_count = 0
        self.history: list[int] = []
        self.last_reward = 0.0
        self.genre_like = np.zeros(5, dtype=np.float32)
        self.current_pref = np.zeros(5, dtype=np.float32)
        self.shift_pref = np.zeros(5, dtype=np.float32)
        self.energy_idx = FEATURE_COLUMNS.index("energy")
        self.valence_idx = FEATURE_COLUMNS.index("valence")
        self.dance_idx = FEATURE_COLUMNS.index("danceability")
        self.reset()

    def _sample_preference(self) -> np.ndarray:
        return np.array([self.rng.random() for _ in range(5)], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.history = []
        self.last_reward = 0.0
        self.genre_like = np.zeros(5, dtype=np.float32)
        self.current_pref = self._sample_preference()
        self.shift_pref = self._sample_preference()
        # Mood-like shift explicitly flips energy preference for stronger reward-feature alignment.
        self.shift_pref[self.energy_idx] = 1.0 - self.current_pref[self.energy_idx]
        self.shift_pref[self.valence_idx] = min(1.0, max(0.0, 1.0 - 0.7 * self.current_pref[self.valence_idx]))
        self.shift_pref[self.dance_idx] = min(1.0, max(0.0, 1.0 - 0.7 * self.current_pref[self.dance_idx]))
        return np.zeros(1, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if action < 0 or action >= self.n_songs:
            raise ValueError(f"Invalid action index {action}")
        if self.step_count == self.episode_length // 2:
            self.current_pref = self.shift_pref.copy()

        song = self.song_features[action]
        # Reward uses key observable features to make learning signal predictable.
        song_key = song[STATE_FEATURE_INDICES]
        pref_key = self.current_pref[STATE_FEATURE_INDICES]
        similarity = 1.0 - float(np.mean(np.abs(song_key - pref_key)))  # in [0,1]
        repeat_penalty = -0.2 if action in self.history else 0.0
        reward = 2.0 * similarity - 1.0 + repeat_penalty

        self.history.append(action)
        self.last_reward = reward
        if reward > 0:
            self.genre_like = 0.9 * self.genre_like + 0.1 * song

        self.step_count += 1
        done = self.step_count >= self.episode_length
        info = {"similarity": similarity, "repeat_penalty": repeat_penalty}
        return np.zeros(1, dtype=np.float32), reward, done, info


def encode_step_feature(
    step_count: int,
    episode_length: int,
    last_outcome: str,
    last_song_features: np.ndarray,
) -> np.ndarray:
    """Encode one timestep feature vector."""
    phase = 0.0 if step_count < (episode_length // 2) else 1.0
    outcome_map = {"none": 0.0, "positive": 1.0, "negative": -1.0}
    return np.array(
        [
            phase,
            outcome_map.get(last_outcome, 0.0),
            float(last_song_features[0]),
            float(last_song_features[1]),
            float(last_song_features[2]),
        ],
        dtype=np.float32,
    )


def build_sequence(step_features: list[np.ndarray], k: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Build fixed-length sequence with zero-left-padding."""
    if len(step_features) >= k:
        seq = step_features[-k:]
    else:
        pad_count = k - len(step_features)
        pads = [np.zeros(STEP_FEATURE_DIM, dtype=np.float32) for _ in range(pad_count)]
        seq = pads + step_features
    return np.stack(seq, axis=0).astype(np.float32)


def choose_action(
    model: DRQNLite,
    seq: np.ndarray,
    all_action_features: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    n_actions = all_action_features.shape[0]
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        seq_batch = np.repeat(seq[np.newaxis, :, :], n_actions, axis=0)
        s = torch.tensor(seq_batch, dtype=torch.float32, device=device)
        a = torch.tensor(all_action_features, dtype=torch.float32, device=device)
        q_values = model(s, a)
        return int(torch.argmax(q_values).item())


def max_q_over_actions(
    model: DRQNLite,
    seq_batch: torch.Tensor,
    all_action_features: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Compute max_a Q(seq, a) for each sequence in batch."""
    batch_size = seq_batch.shape[0]
    action_feats = torch.tensor(all_action_features, dtype=torch.float32, device=device)
    n_actions = action_feats.shape[0]
    expanded_seqs = (
        seq_batch.unsqueeze(1)
        .repeat(1, n_actions, 1, 1)
        .reshape(batch_size * n_actions, seq_batch.shape[1], seq_batch.shape[2])
    )
    expanded_actions = action_feats.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * n_actions, -1)
    q_values = model(expanded_seqs, expanded_actions).reshape(batch_size, n_actions)
    return q_values.max(dim=1)[0]


def train_dqn(
    features: np.ndarray,
    episodes: int = 1300,
    batch_size: int = 32,
    gamma: float = 0.95,
    lr: float = 5e-4,
    seed: int = 42,
    best_model_path: Path | None = None,
) -> tuple[DRQNLite, list[float], torch.device, float, list[float]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SpotifyRecEnv(features, seed=seed, episode_length=20)

    model = DRQNLite(feature_dim=STEP_FEATURE_DIM, action_feature_dim=3).to(device)
    target = DRQNLite(feature_dim=STEP_FEATURE_DIM, action_feature_dim=3).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay = ReplayBuffer(capacity=5000)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.997
    episode_rewards: list[float] = []
    recent_losses: deque[float] = deque(maxlen=200)
    loss_history: list[float] = []
    recent_q_values: deque[float] = deque(maxlen=200)
    global_step = 0
    best_avg_reward = float("-inf")

    for ep in range(episodes):
        env.reset()
        last_outcome = "none"
        last_song_features = np.zeros(3, dtype=np.float32)
        step_features: list[np.ndarray] = []
        first_feat = encode_step_feature(
            step_count=env.step_count,
            episode_length=env.episode_length,
            last_outcome=last_outcome,
            last_song_features=last_song_features,
        )
        step_features.append(first_feat)
        seq = build_sequence(step_features)
        total_reward = 0.0
        done = False

        while not done:
            action = choose_action(
                model=model,
                seq=seq,
                all_action_features=env.song_features[:, STATE_FEATURE_INDICES],
                epsilon=epsilon,
                device=device,
            )
            _, reward, done, _ = env.step(action)
            selected_action_features = env.song_features[action][STATE_FEATURE_INDICES].astype(np.float32)
            last_song_features = selected_action_features.copy()
            if reward > 0:
                next_outcome = "positive"
            elif reward < 0:
                next_outcome = "negative"
            else:
                next_outcome = "none"
            next_feat = encode_step_feature(
                step_count=env.step_count,
                episode_length=env.episode_length,
                last_outcome=next_outcome,
                last_song_features=last_song_features,
            )
            next_step_features = step_features + [next_feat]
            next_seq = build_sequence(next_step_features)
            replay.add(
                Transition(
                    seq=seq.copy(),
                    action_features=selected_action_features.copy(),
                    reward=float(reward),
                    next_seq=next_seq.copy(),
                    done=bool(done),
                )
            )
            step_features = next_step_features
            seq = next_seq
            last_outcome = next_outcome
            total_reward += reward
            global_step += 1

            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                seqs = torch.tensor(np.stack([t.seq for t in batch]), dtype=torch.float32, device=device)
                action_features = torch.tensor(
                    np.stack([t.action_features for t in batch]), dtype=torch.float32, device=device
                )
                rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
                next_seqs = torch.tensor(
                    np.stack([t.next_seq for t in batch]), dtype=torch.float32, device=device
                )
                dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                q_values = model(seqs, action_features)
                with torch.no_grad():
                    next_q = max_q_over_actions(
                        model=target,
                        seq_batch=next_seqs,
                        all_action_features=env.song_features[:, STATE_FEATURE_INDICES],
                        device=device,
                    )
                    target_q = rewards + (1.0 - dones) * gamma * next_q

                loss = criterion(q_values, target_q)
                recent_losses.append(float(loss.item()))
                loss_history.append(float(loss.item()))
                recent_q_values.append(float(q_values.mean().item()))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if global_step % 200 == 0:
                target.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if (ep + 1) % 200 == 0:
            tail = episode_rewards[-50:]
            avg_tail = mean(tail)
            if avg_tail > best_avg_reward:
                best_avg_reward = avg_tail
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": FEATURE_COLUMNS,
                    "state_dim": STEP_FEATURE_DIM,
                    "sequence_length": SEQUENCE_LENGTH,
                    "training_seed": seed,
                    "best_avg_reward_last50": best_avg_reward,
                }
                # Required generic best checkpoint path.
                torch.save(checkpoint, BEST_MODEL_PATH)
                # Optional per-seed best checkpoint path.
                if best_model_path is not None:
                    torch.save(checkpoint, best_model_path)
            print(
                f"Episode {ep + 1}/{episodes} | eps={epsilon:.3f} | "
                f"avg_reward(last50)={avg_tail:.3f} | "
                f"best_avg_reward={best_avg_reward:.3f} | "
                f"loss(avg)={(mean(recent_losses) if recent_losses else 0.0):.4f} | "
                f"q(avg)={(mean(recent_q_values) if recent_q_values else 0.0):.4f}"
            )

    return model, episode_rewards, device, best_avg_reward, loss_history


def evaluate_policy(model: DRQNLite, features: np.ndarray, episodes: int = 100, seed: int = 1000) -> list[float]:
    device = next(model.parameters()).device
    rewards: list[float] = []
    for idx in range(episodes):
        env = SpotifyRecEnv(features, seed=seed + idx, episode_length=20)
        env.reset()
        last_outcome = "none"
        last_song_features = np.zeros(3, dtype=np.float32)
        step_features: list[np.ndarray] = []
        first_feat = encode_step_feature(env.step_count, env.episode_length, last_outcome, last_song_features)
        step_features.append(first_feat)
        seq = build_sequence(step_features)
        done = False
        total = 0.0
        while not done:
            action = choose_action(
                model=model,
                seq=seq,
                all_action_features=env.song_features[:, STATE_FEATURE_INDICES],
                epsilon=0.0,
                device=device,
            )
            _, reward, done, _ = env.step(action)
            last_song_features = env.song_features[action][STATE_FEATURE_INDICES].astype(np.float32)
            if reward > 0:
                last_outcome = "positive"
            elif reward < 0:
                last_outcome = "negative"
            else:
                last_outcome = "none"
            next_feat = encode_step_feature(env.step_count, env.episode_length, last_outcome, last_song_features)
            step_features.append(next_feat)
            seq = build_sequence(step_features)
            total += reward
        rewards.append(total)
    return rewards


def evaluate_policy_ensemble(
    models: list[DRQNLite],
    features: np.ndarray,
    episodes: int = 100,
    seed: int = 1000,
) -> list[float]:
    if not models:
        return []
    device = next(models[0].parameters()).device
    rewards: list[float] = []
    for idx in range(episodes):
        env = SpotifyRecEnv(features, seed=seed + idx, episode_length=20)
        env.reset()
        last_outcome = "none"
        last_song_features = np.zeros(3, dtype=np.float32)
        step_features: list[np.ndarray] = []
        first_feat = encode_step_feature(env.step_count, env.episode_length, last_outcome, last_song_features)
        step_features.append(first_feat)
        seq = build_sequence(step_features)
        done = False
        total = 0.0
        while not done:
            n_actions = features.shape[0]
            seq_batch = np.repeat(seq[np.newaxis, :, :], n_actions, axis=0)
            seq_t = torch.tensor(seq_batch, dtype=torch.float32, device=device)
            act_t = torch.tensor(env.song_features[:, STATE_FEATURE_INDICES], dtype=torch.float32, device=device)
            with torch.no_grad():
                q_sum = torch.zeros(n_actions, dtype=torch.float32, device=device)
                for model in models:
                    q_sum += model(seq_t, act_t)
                q_mean = q_sum / len(models)
                action = int(torch.argmax(q_mean).item())

            _, reward, done, _ = env.step(action)
            last_song_features = env.song_features[action][STATE_FEATURE_INDICES].astype(np.float32)
            if reward > 0:
                last_outcome = "positive"
            elif reward < 0:
                last_outcome = "negative"
            else:
                last_outcome = "none"
            next_feat = encode_step_feature(env.step_count, env.episode_length, last_outcome, last_song_features)
            step_features.append(next_feat)
            seq = build_sequence(step_features)
            total += reward
        rewards.append(total)
    return rewards


def evaluate_random(features: np.ndarray, episodes: int = 100, seed: int = 1000) -> list[float]:
    rewards: list[float] = []
    for idx in range(episodes):
        env = SpotifyRecEnv(features, seed=seed + idx, episode_length=20)
        rng = random.Random(seed + idx)
        env.reset()
        done = False
        total = 0.0
        while not done:
            action = rng.randrange(features.shape[0])
            _, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return rewards


def moving_average(values: list[float], window: int = 50) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(mean(values[start : i + 1]))
    return out


def save_plots(
    training_curves: list[list[float]],
    baseline_rewards: list[float],
    trained_rewards: list[float],
    baseline_mean: float,
) -> None:
    episodes = len(training_curves[0]) if training_curves else 0
    x = list(range(1, episodes + 1))
    if episodes > 0:
        avg_curve = np.mean(np.array(training_curves, dtype=np.float32), axis=0)
        std_curve = np.std(np.array(training_curves, dtype=np.float32), axis=0)
        smooth = moving_average(avg_curve.tolist(), window=50)
    else:
        avg_curve = np.array([])
        std_curve = np.array([])
        smooth = []

    plt.figure(figsize=(10, 5))
    if episodes > 0:
        plt.plot(x, avg_curve, alpha=0.35, label="Average reward across seeds")
        plt.plot(x, smooth, linewidth=2, label="Moving average (50)")
        lower = np.maximum(avg_curve - std_curve, -50)
        upper = np.minimum(avg_curve + std_curve, 50)
        plt.fill_between(x, lower, upper, alpha=0.18, label="Variance (±1 std)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Final DRQN Training Curve (Averaged Across Seeds)")
    plt.axhline(baseline_mean, linestyle="--", color="red", label=f"Baseline ({baseline_mean:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_TRAIN_PLOT_PATH, dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(baseline_rewards, bins=24, alpha=0.55, label="Baseline", color="#d62728")
    plt.hist(trained_rewards, bins=24, alpha=0.55, label="Trained", color="#1f77b4")
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title("Reward Distribution: Baseline vs Trained")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REWARD_DIST_PLOT_PATH, dpi=140)
    plt.close()


def save_loss_plot(loss_history: list[float]) -> None:
    if not loss_history:
        return
    plt.figure(figsize=(9, 4.5))
    plt.plot(loss_history, linewidth=1.2)
    plt.title("DRQN Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=140)
    plt.close()


def save_comparison_bar(baseline_mean: float, trained_mean: float) -> None:
    labels = ["Random Baseline", "Ensemble DRQN"]
    values = [baseline_mean, trained_mean]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, values, color=["#d62728", "#1f77b4"])
    plt.title("Baseline vs Ensemble DRQN Performance")
    plt.ylabel("Average Reward")
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        plt.text(x, value, f"{value:.2f}", ha="center", va="bottom")
    plt.text(1, trained_mean, f"+{trained_mean - baseline_mean:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(COMPARISON_BAR_PATH, dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact DQN on Spotify subset.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DATASET_PATH),
        help="Path to dataset.csv",
    )
    parser.add_argument("--sample-size", type=int, default=50, help="Subset size (50-60 recommended).")
    parser.add_argument("--episodes", type=int, default=1400, help="Training episodes (1300-1500 recommended).")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset sampling seed.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    global_seed = 42
    episodes = args.episodes
    batch_size = args.batch_size

    print("Loading dataset and preparing subset...")
    sampled_df, features = load_song_subset(dataset_path, sample_size=args.sample_size, seed=args.seed)
    print(f"Subset size: {len(sampled_df)} songs | action space: {features.shape[0]}")

    train_seeds = [0, 42, 99]
    per_seed_results: list[dict] = []
    all_training_curves: list[list[float]] = []
    all_loss_histories: list[list[float]] = []
    all_trained_rewards: list[float] = []
    device_str = "cpu"

    baseline_rewards = evaluate_random(features, episodes=120, seed=1000)
    baseline_mean = mean(baseline_rewards)
    baseline_std = pstdev(baseline_rewards)

    print("Training DRQN across seeds:", train_seeds)
    for train_seed in train_seeds:
        print(f"\n--- Training seed {train_seed} ---")
        seed_ckpt = ROOT / f"best_model_seed_{train_seed}.pt"
        model, training_rewards, device, best_avg, loss_history = train_dqn(
            features,
            episodes=episodes,
            batch_size=batch_size,
            gamma=0.95,
            lr=5e-4,
            seed=train_seed,
            best_model_path=seed_ckpt,
        )
        device_str = str(device)
        all_training_curves.append(training_rewards)
        all_loss_histories.append(loss_history)

        checkpoint = torch.load(seed_ckpt, map_location=device)
        eval_model = DRQNLite(feature_dim=STEP_FEATURE_DIM, action_feature_dim=3).to(device)
        eval_model.load_state_dict(checkpoint["model_state_dict"])
        eval_model.eval()
        trained_rewards = evaluate_policy(eval_model, features, episodes=120, seed=1000)
        all_trained_rewards.extend(trained_rewards)
        trained_mean = mean(trained_rewards)
        trained_std = pstdev(trained_rewards)
        improvement = trained_mean - baseline_mean
        per_seed_results.append(
            {
                "seed": train_seed,
                "best_avg_reward_last50": best_avg,
                "trained_mean": trained_mean,
                "trained_std": trained_std,
                "improvement": improvement,
                "checkpoint_file": seed_ckpt.name,
            }
        )
        print(f"seed={train_seed} | trained={trained_mean:.3f} ± {trained_std:.3f} | improvement={improvement:.3f}")

    # Optional ensemble of top 3 (or fewer) models by improvement.
    top_models = sorted(per_seed_results, key=lambda x: x["improvement"], reverse=True)[:3]
    ensemble_models: list[DRQNLite] = []
    for item in top_models:
        ckpt = torch.load(ROOT / item["checkpoint_file"], map_location=device_str)
        m = DRQNLite(feature_dim=STEP_FEATURE_DIM, action_feature_dim=3)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        ensemble_models.append(m)
    ensemble_rewards = evaluate_policy_ensemble(ensemble_models, features, episodes=120, seed=1000)
    ensemble_mean = mean(ensemble_rewards) if ensemble_rewards else 0.0
    ensemble_std = pstdev(ensemble_rewards) if len(ensemble_rewards) > 1 else 0.0
    ensemble_improvement = ensemble_mean - baseline_mean

    best_seed_result = max(per_seed_results, key=lambda x: x["trained_mean"])
    mean_trained_reward = mean([r["trained_mean"] for r in per_seed_results])

    # Save the globally best checkpoint as model.pt.
    best_ckpt = torch.load(ROOT / best_seed_result["checkpoint_file"], map_location="cpu")
    torch.save(best_ckpt, MODEL_PATH)

    save_plots(
        training_curves=all_training_curves,
        baseline_rewards=baseline_rewards,
        trained_rewards=all_trained_rewards,
        baseline_mean=baseline_mean,
    )
    merged_loss_history = [loss for per_seed in all_loss_histories for loss in per_seed]
    save_loss_plot(merged_loss_history)
    save_comparison_bar(baseline_mean=baseline_mean, trained_mean=ensemble_mean)

    summary = {
        "dataset_seed": args.seed,
        "train_seeds": train_seeds,
        "episodes": episodes,
        "batch_size": batch_size,
        "subset_size": int(features.shape[0]),
        "device": device_str,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "per_seed_results": per_seed_results,
        "best_trained_mean": best_seed_result["trained_mean"],
        "mean_trained_mean": mean_trained_reward,
        "ensemble_mean": ensemble_mean,
        "ensemble_std": ensemble_std,
        "ensemble_improvement": ensemble_improvement,
        "best_model_seed": best_seed_result["seed"],
        "model_path": MODEL_PATH.name,
        "best_model_path": BEST_MODEL_PATH.name,
        "training_plot": FINAL_TRAIN_PLOT_PATH.name,
        "reward_distribution_plot": REWARD_DIST_PLOT_PATH.name,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINAL DRQN RESULTS ===")
    print(f"Baseline mean ± std: {baseline_mean:.3f} ± {baseline_std:.3f}")
    print("Per-seed improvements:")
    for item in per_seed_results:
        print(f"Seed {item['seed']}: {item['improvement']:.3f}")
    print(f"Best: {best_seed_result['trained_mean']:.3f}")
    print(f"Mean: {mean_trained_reward:.3f}")
    print(f"Ensemble mean ± std: {ensemble_mean:.3f} ± {ensemble_std:.3f}")
    print(f"Ensemble improvement: {ensemble_improvement:.3f}")
    print(f"Baseline: {baseline_mean:.3f}")
    print(f"Trained: {mean_trained_reward:.3f}")
    print(f"Improvement: {mean_trained_reward - baseline_mean:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved training plot: {FINAL_TRAIN_PLOT_PATH}")
    print(f"Saved reward distribution: {REWARD_DIST_PLOT_PATH}")
    print(f"Saved loss plot: {LOSS_PLOT_PATH}")
    print(f"Saved comparison bar: {COMPARISON_BAR_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
