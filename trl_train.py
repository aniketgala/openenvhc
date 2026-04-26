"""TRL PPO training with warm start for MusicRlEnvironment."""

from __future__ import annotations

import os
import json
import locale
import math
import random
import re
from pathlib import Path
from statistics import mean

# This redirects Hugging Face cache to D drive to prevent C drive from filling up.
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.makedirs("D:/hf_cache", exist_ok=True)
os.makedirs("D:/hf_cache/transformers", exist_ok=True)
os.makedirs("D:/hf_cache/datasets", exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Work around Windows default cp1252 decoding issues in some TRL template files.
locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"  # type: ignore[assignment]

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from server.music_rl_env_environment import MusicRlEnvironment
from models import MusicRlAction


ROOT = Path(__file__).resolve().parent
PLOT_PATH = ROOT / "trl_training_plot.png"
SUMMARY_PATH = ROOT / "trl_summary.json"

torch.set_num_threads(2)


def state_to_prompt(obs, max_idx: int) -> str:
    feats = obs.recent_song_features if obs.recent_song_features else [0.0, 0.0, 0.0]
    return (
        f"state p={obs.phase} o={obs.last_outcome} "
        f"f={feats[0]:.2f},{feats[1]:.2f},{feats[2]:.2f}. "
        f"best idx 0-{max_idx}, number only."
    )


def parse_action(text: str, max_idx: int) -> int:
    match = re.search(r"\d+", text)
    if match is None:
        return random.randint(0, max_idx)
    try:
        idx = int(match.group(0))
    except (TypeError, ValueError):
        return random.randint(0, max_idx)
    return max(0, min(max_idx, idx))


def _song_features(song: dict) -> list[float]:
    return [
        float(song.get("energy", 0.5)),
        float(song.get("valence", 0.5)),
        float(song.get("danceability", 0.5)),
    ]


def _heuristic_best_action(obs, songs_subset: list[dict]) -> int:
    feats = obs.recent_song_features if obs.recent_song_features else [0.5, 0.5, 0.5]
    target_energy = 0.8 if obs.phase == 0 else 0.2
    best_idx = 0
    best_score = -1e9
    for idx, song in enumerate(songs_subset):
        energy, valence, danceability = _song_features(song)
        dist = abs(feats[0] - energy) + abs(feats[1] - valence) + abs(feats[2] - danceability)
        phase_bonus = -abs(target_energy - energy)
        outcome_bonus = 0.15 if obs.last_outcome in {"negative", "neutral"} and valence > 0.45 else 0.0
        score = -dist + phase_bonus + outcome_bonus
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def build_warmstart_samples(episodes: int, songs_subset: list[dict]) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    max_idx = len(songs_subset) - 1
    for seed in range(episodes):
        env = MusicRlEnvironment(seed=seed)
        obs = env.reset()
        steps = random.randint(1, 3)
        for _ in range(steps):
            prompt = state_to_prompt(obs, max_idx=max_idx)
            best_action = _heuristic_best_action(obs, songs_subset)
            samples.append((prompt, best_action))
            obs = env.step(MusicRlAction(song_index=random.randint(0, max_idx)))
            if bool(obs.done):
                break
    return samples[:200]


def supervised_warm_start(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    samples: list[tuple[str, int]],
    device: torch.device,
) -> None:
    optimizer = AdamW(model.pretrained_model.parameters(), lr=1e-5)
    model.pretrained_model.train()
    batch_size = 8
    for _ in range(2):
        random.shuffle(samples)
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            prompts = [p for p, _ in batch]
            targets = [str(a) for _, a in batch]
            full_texts = [f"{p} {t}" for p, t in zip(prompts, targets)]

            encoded = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            prompt_encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            labels = encoded["input_ids"].clone()
            labels[:] = -100
            for row in range(labels.shape[0]):
                prompt_len = int(prompt_encoded["attention_mask"][row].sum().item())
                labels[row, prompt_len:] = encoded["input_ids"][row, prompt_len:]

            outputs = model.pretrained_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.pretrained_model.eval()


def _generate_action(
    ppo_trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_idx: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
    model = ppo_trainer.model
    device = model.pretrained_model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    query_tensor = input_ids[0]
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_tensor = outputs[0][-1:]
    text = tokenizer.decode(response_tensor, skip_special_tokens=True)
    action_idx = parse_action(text, max_idx=max_idx)
    return action_idx, query_tensor, response_tensor


def evaluate_model(ppo_trainer: PPOTrainer, tokenizer, env_seed: int, episodes: int, max_idx: int) -> list[float]:
    rewards: list[float] = []
    for ep in range(episodes):
        env = MusicRlEnvironment(seed=env_seed + ep)
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            prompt = state_to_prompt(obs, max_idx=max_idx)
            action_idx, _, _ = _generate_action(ppo_trainer, tokenizer, prompt, max_idx)
            obs = env.step(MusicRlAction(song_index=action_idx))
            total += float(obs.reward or 0.0)
            done = bool(obs.done)
        rewards.append(total)
    return rewards


def evaluate_random(env_seed: int, episodes: int, max_idx: int) -> list[float]:
    rewards: list[float] = []
    for ep in range(episodes):
        env = MusicRlEnvironment(seed=env_seed + ep)
        obs = env.reset()
        total = 0.0
        done = False
        while not done:
            obs = env.step(MusicRlAction(song_index=random.randint(0, max_idx)))
            total += float(obs.reward or 0.0)
            done = bool(obs.done)
        rewards.append(total)
    return rewards


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MusicRlEnvironment(seed=42)
    subset_size = min(15, len(env._songs))
    songs_subset = env._songs[:subset_size]
    max_idx = subset_size - 1

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
    )
    base_model.config.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    model.config.pad_token_id = tokenizer.eos_token_id
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    model.to(device)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        seed=42,
        log_with=None,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    warm_samples = build_warmstart_samples(episodes=120, songs_subset=songs_subset)
    supervised_warm_start(model, tokenizer, warm_samples, device=device)

    episodes = 80
    rewards_per_episode: list[float] = []
    success_list: list[int] = []
    best_last20 = -float("inf")
    print("Starting TRL PPO training...")
    for ep in range(episodes):
        episode = ep + 1
        env_episode = MusicRlEnvironment(seed=1000 + ep)
        obs = env_episode.reset()
        done = False
        episode_reward = 0.0

        while not done:
            prompt = state_to_prompt(obs, max_idx=max_idx)
            action_idx, query_tensor, response_tensor = _generate_action(ppo_trainer, tokenizer, prompt, max_idx)

            obs = env_episode.step(MusicRlAction(song_index=action_idx))
            step_reward = float(obs.reward or 0.0)
            step_reward = max(-3.0, min(3.0, step_reward))
            step_reward = step_reward / 3.0
            done = bool(obs.done)
            episode_reward += step_reward
            success_list.append(1 if step_reward > 0 else 0)
            reward = torch.tensor(step_reward, dtype=torch.float32)
            ppo_trainer.step([query_tensor], [response_tensor], [reward])

        rewards_per_episode.append(episode_reward)
        last_20_mean_reward = mean(rewards_per_episode[-min(20, len(rewards_per_episode)) :])
        best_last20 = max(best_last20, last_20_mean_reward)
        if episode > 40 and last_20_mean_reward < best_last20 * 0.7:
            print("Early stopping: performance dropped from peak")
            break

        if episode % 10 == 0:
            moving_avg = mean(rewards_per_episode[-5:])
            print(f"Episode {episode}/{episodes} | reward={episode_reward:.3f} | moving_avg={moving_avg:.3f}")

    baseline_rewards = evaluate_random(env_seed=2000, episodes=20, max_idx=max_idx)
    baseline_mean = mean(baseline_rewards)
    baseline_std = math.sqrt(mean([(r - baseline_mean) ** 2 for r in baseline_rewards]))
    mean_reward = mean(rewards_per_episode)

    success_rate = (sum(success_list) / len(success_list)) if success_list else 0.0
    rewards_arr = np.asarray(rewards_per_episode, dtype=np.float32)
    window = 10
    if len(rewards_arr) >= window:
        rolling_mean = np.convolve(rewards_arr, np.ones(window, dtype=np.float32) / window, mode="valid")
        rolling_std = np.array([np.std(rewards_arr[i : i + window]) for i in range(len(rewards_arr) - window + 1)])
        rolling_x = np.arange(window, len(rewards_arr) + 1)
    else:
        rolling_mean = rewards_arr.copy()
        rolling_std = np.zeros_like(rewards_arr)
        rolling_x = np.arange(1, len(rewards_arr) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode, alpha=0.3, label="Raw reward")
    plt.plot(rolling_x, rolling_mean, linewidth=2, label="Moving avg (10)")
    plt.fill_between(rolling_x, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2, label="Std band (10)")
    plt.axhline(y=baseline_mean, linestyle="--", color="red", label="Random baseline mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TRL PPO Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140)
    plt.close()

    summary = {
        "episodes": episodes,
        "device": str(device),
        "subset_size": subset_size,
        "mean_reward": mean_reward,
        "last20_mean_reward": mean(rewards_per_episode[-min(20, len(rewards_per_episode)) :]),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "improvement": mean_reward - baseline_mean,
        "success_rate": success_rate,
        "normalized_rewards": True,
        "plot": PLOT_PATH.name,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINAL TRL RESULTS ===")
    print(f"Baseline mean: {baseline_mean:.3f}")
    print(f"TRL mean reward: {summary['mean_reward']:.3f}")
    print(f"Last-20 mean: {summary['last20_mean_reward']:.3f}")
    print(f"Best last-20 mean: {best_last20:.3f}")
    print(f"Training improvement: {summary['improvement']:.3f}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
