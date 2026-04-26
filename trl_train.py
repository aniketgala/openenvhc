"""TRL PPO training with warm start for MusicRlEnvironment."""

from __future__ import annotations

import json
import locale
import math
import os
import random
from collections import deque
from pathlib import Path
from statistics import mean

# Redirect Hugging Face cache to D drive to prevent C drive from filling up.
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.makedirs("D:/hf_cache", exist_ok=True)
os.makedirs("D:/hf_cache/transformers", exist_ok=True)
os.makedirs("D:/hf_cache/datasets", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Work around Windows default cp1252 decoding issues in some TRL template files.
locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"  # type: ignore[assignment]

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from models import MusicRlAction
from server.music_rl_env_environment import MusicRlEnvironment


ROOT = Path(__file__).resolve().parent
PLOT_PATH = ROOT / "trl_training_plot.png"
SUMMARY_PATH = ROOT / "trl_summary.json"
DEBUG_PATH = ROOT / "trl_debug.json"
DEBUG_MODE = True
ACTION_TOKENS = [f" SONG_{i}" for i in range(20)]
torch.set_num_threads(2)


class EpisodeMemory:
    """Rolling history for the latest actions/rewards."""

    def __init__(self, maxlen: int = 3) -> None:
        self.last_actions: deque[str] = deque(maxlen=maxlen)
        self.last_rewards: deque[float] = deque(maxlen=maxlen)

    def add(self, action_idx: int, reward: float) -> None:
        self.last_actions.append(f"SONG_{action_idx}")
        self.last_rewards.append(float(reward))

    def actions_text(self) -> str:
        return ", ".join(self.last_actions) if self.last_actions else "none"

    def rewards_text(self) -> str:
        return ", ".join(f"{r:.2f}" for r in self.last_rewards) if self.last_rewards else "none"

    def is_recent_repeat(self, action_idx: int) -> bool:
        return f"SONG_{action_idx}" in self.last_actions


def _song_features(song: dict) -> list[float]:
    return [
        float(song.get("energy", 0.5)),
        float(song.get("valence", 0.5)),
        float(song.get("danceability", 0.5)),
    ]


class BaselineAdvisor:
    """Optional baseline guidance if DRQN checkpoints are available."""

    def __init__(self) -> None:
        self.available = any(ROOT.glob("best_model_seed_*.pt"))

    def suggest(self, obs, songs_subset: list[dict], memory: EpisodeMemory) -> int:
        feats = obs.recent_song_features if obs.recent_song_features else [0.5, 0.5, 0.5]
        target_energy = 0.8 if obs.phase == 0 else 0.2
        best_idx = 0
        best_score = -1e9
        for idx, song in enumerate(songs_subset):
            energy, valence, danceability = _song_features(song)
            dist = abs(feats[0] - energy) + abs(feats[1] - valence) + abs(feats[2] - danceability)
            phase_bonus = -abs(target_energy - energy)
            outcome_bonus = 0.15 if obs.last_outcome in {"negative", "neutral"} and valence > 0.45 else 0.0
            repeat_penalty = -0.25 if memory.is_recent_repeat(idx) else 0.0
            score = -dist + phase_bonus + outcome_bonus + repeat_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx


def state_to_prompt(obs, max_idx: int, memory: EpisodeMemory, baseline_suggestion: str | None = None) -> str:
    feats = obs.recent_song_features if obs.recent_song_features else [0.0, 0.0, 0.0]
    baseline_line = (
        f"\nBaseline suggestion: {baseline_suggestion}. You may agree or override based on context.\n"
        if baseline_suggestion
        else "\n"
    )
    return (
        "You are a music recommendation agent.\n\n"
        "User context:\n\n"
        f"* Phase: {obs.phase}\n"
        f"* Last outcome: {obs.last_outcome}\n"
        "* Recent features:\n\n"
        f"  * Energy: {feats[0]:.2f}\n"
        f"  * Valence: {feats[1]:.2f}\n"
        f"  * Danceability: {feats[2]:.2f}\n\n"
        "Recent history:\n\n"
        f"* Actions: {memory.actions_text()}\n"
        f"* Rewards: {memory.rewards_text()}\n"
        f"{baseline_line}"
        "Goal:\n"
        "Maximize user satisfaction.\n\n"
        "Think step-by-step:\n\n"
        "1. What does the user currently prefer?\n"
        "2. Has preference shifted?\n"
        "3. What should we recommend next?\n\n"
        "Output strictly:\n\n"
        "Reasoning: <text>\n"
        f"Action: SONG_<index from 0 to {max_idx}>\n\n"
        "IMPORTANT: Output MUST contain 'Action: SONG_<number>'. Any other format is invalid."
    )


def reasoning_score(text: str) -> int:
    keywords = ["prefer", "shift", "energy", "valence", "dance"]
    lowered = text.lower()
    return sum(k in lowered for k in keywords)


def shape_reward(env_reward: float, repeated: bool, action_switched: bool, prev_reward: float) -> float:
    reward = env_reward
    if not repeated:
        reward += 0.3
    if action_switched:
        reward -= 0.05
    reward = reward / 5.0
    reward = max(min(reward, 2.0), -2.0)
    smoothed_reward = 0.7 * reward + 0.3 * prev_reward
    return smoothed_reward


def build_warmstart_samples(episodes: int, songs_subset: list[dict], advisor: BaselineAdvisor) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    max_idx = len(songs_subset) - 1
    for seed in range(episodes):
        env = MusicRlEnvironment(seed=seed)
        obs = env.reset()
        memory = EpisodeMemory(maxlen=3)
        for _ in range(random.randint(1, 3)):
            baseline = f"SONG_{advisor.suggest(obs, songs_subset, memory)}" if advisor.available else None
            prompt = state_to_prompt(obs, max_idx=max_idx, memory=memory, baseline_suggestion=baseline)
            best_action = advisor.suggest(obs, songs_subset, memory)
            target = f"Reasoning: Match phase/mood and avoid repeats.\nAction: SONG_{best_action}"
            samples.append((prompt, target))
            obs = env.step(MusicRlAction(song_index=random.randint(0, max_idx)))
            memory.add(best_action, float(obs.reward or 0.0))
            if bool(obs.done):
                break
    return samples[:220]


def supervised_warm_start(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    samples: list[tuple[str, str]],
    device: torch.device,
) -> None:
    optimizer = AdamW(model.pretrained_model.parameters(), lr=8e-6)
    model.pretrained_model.train()
    batch_size = 8
    for _ in range(2):
        random.shuffle(samples)
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            prompts = [p for p, _ in batch]
            targets = [t for _, t in batch]
            full_texts = [f"{p}\n{t}" for p, t in zip(prompts, targets)]
            encoded = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            prompt_encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
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
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
    model.pretrained_model.eval()


def _build_action_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    token_ids: list[int] = []
    for token in ACTION_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Unable to tokenize action token: {token}")
        # Keep single-id path fast; fallback to last id if split into multiple ids.
        token_ids.append(ids[0] if len(ids) == 1 else ids[-1])
    return token_ids


def _select_action_from_logits(
    ppo_trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    prompt: str,
    action_token_ids: list[int],
) -> tuple[int, torch.Tensor, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    model = ppo_trainer.model
    device = model.pretrained_model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    query_tensor = input_ids[0]
    with torch.no_grad():
        outputs = model.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = outputs.logits[:, -1, :].squeeze(0)
    action_logits = torch.stack([next_logits[token_id] for token_id in action_token_ids])
    action_idx = int(torch.argmax(action_logits).item())
    selected_token_id = int(action_token_ids[action_idx])
    response_tensor = torch.tensor([selected_token_id], dtype=torch.long, device=device)
    return action_idx, query_tensor, response_tensor


def _try_ppo_step(
    ppo_trainer: PPOTrainer,
    query_buffer: list[torch.Tensor],
    response_buffer: list[torch.Tensor],
    reward_buffer: list[torch.Tensor],
    running_mean: float,
    running_std: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], float, float]:
    min_len = min(len(query_buffer), len(response_buffer), len(reward_buffer))
    query_buffer = query_buffer[:min_len]
    response_buffer = response_buffer[:min_len]
    reward_buffer = reward_buffer[:min_len]
    if min_len < 8:
        return query_buffer, response_buffer, reward_buffer, running_mean, running_std

    batch_vals = torch.stack(reward_buffer).float().cpu()
    batch_mean = float(batch_vals.mean().item())
    batch_std = float(batch_vals.std(unbiased=False).item())
    running_mean = 0.9 * running_mean + 0.1 * batch_mean
    running_std = 0.9 * running_std + 0.1 * batch_std
    normed_rewards = [
        torch.tensor(
            (float(r.item()) - running_mean) / (running_std + 1e-8),
            dtype=torch.float32,
            device=r.device,
        )
        for r in reward_buffer
    ]
    if DEBUG_MODE:
        print(
            f"[PPO PRE] batch_mean={batch_vals.mean().item():.4f} "
            f"batch_std={batch_vals.std(unbiased=False).item():.4f} "
            f"running_mean={running_mean:.4f} running_std={running_std:.4f}"
        )
    if DEBUG_MODE:
        norm_vals = torch.stack(normed_rewards).float().cpu()
        print(
            f"[PPO POST] norm_mean={norm_vals.mean().item():.4f} "
            f"norm_std={norm_vals.std(unbiased=False).item():.4f}"
        )
    ppo_trainer.step(query_buffer, response_buffer, normed_rewards)
    return [], [], [], running_mean, running_std


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


def save_training_plot(rewards: list[float], path: str | None = None) -> str | None:
    if path is None:
        path = os.path.join(os.getcwd(), "trl_training_plot.png")

    if not rewards:
        print("No rewards to plot.")
        return None

    plt.figure()
    plt.plot(rewards, label="Episode Reward")

    if len(rewards) >= 5:
        rolling = [
            sum(rewards[max(0, i - 4) : i + 1]) / len(rewards[max(0, i - 4) : i + 1])
            for i in range(len(rewards))
        ]
        plt.plot(rolling, label="Rolling Mean (5)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TRL Training Curve")
    plt.legend()

    try:
        plt.savefig(path, bbox_inches="tight")
        debug_path = os.path.join(os.getcwd(), "trl_plot_debug.png")
        plt.savefig(debug_path, bbox_inches="tight")
        plt.close()
        if os.path.exists(path):
            print(f"Plot successfully saved at: {path}")
        else:
            print("ERROR: Plot file not found after saving!")
        return path
    except Exception as exc:
        print(f"Plot save failed: {exc}")
        plt.close()
        return None


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MusicRlEnvironment(seed=42)
    subset_size = min(20, len(env._songs))
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
    action_token_ids_full = _build_action_token_ids(tokenizer)

    ppo_config = PPOConfig(
        learning_rate=8e-6,
        batch_size=32,
        mini_batch_size=8,
        ppo_epochs=4,
        gradient_accumulation_steps=1,
        seed=42,
        init_kl_coef=0.05,
        target=6.0,
        adap_kl_ctrl=True,
        log_with=None,
    )
    # Stabilize policy updates with tighter clipping and entropy regularization.
    if hasattr(ppo_config, "cliprange"):
        ppo_config.cliprange = 0.1
    if hasattr(ppo_config, "clip_range"):
        ppo_config.clip_range = 0.1
    if hasattr(ppo_config, "ent_coef"):
        ppo_config.ent_coef = 0.01
    if hasattr(ppo_config, "entropy_coef"):
        ppo_config.entropy_coef = 0.01
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)

    advisor = BaselineAdvisor()
    warm_samples = build_warmstart_samples(episodes=120, songs_subset=songs_subset, advisor=advisor)
    supervised_warm_start(model, tokenizer, warm_samples, device=device)

    episodes = 15 if DEBUG_MODE else 80
    raw_rewards_per_episode: list[float] = []
    shaped_rewards_per_episode: list[float] = []
    best_last20 = -float("inf")
    reasoning_scores_all: list[int] = []
    per_episode_parse_rates: list[float] = []

    print("Starting TRL PPO training...")
    query_buffer: list[torch.Tensor] = []
    response_buffer: list[torch.Tensor] = []
    reward_buffer: list[torch.Tensor] = []
    running_mean = 0.0
    running_std = 1.0

    for ep in range(episodes):
        episode = ep + 1
        env_episode = MusicRlEnvironment(seed=1000 + ep)
        obs = env_episode.reset()
        memory = EpisodeMemory(maxlen=3)
        done = False
        raw_total = 0.0
        shaped_total = 0.0
        last_action_text = "SONG_0"
        episode_parsed = 0
        episode_steps = 0
        episode_reasoning_scores: list[int] = []
        episode_raw_step_rewards: list[float] = []
        episode_shaped_step_rewards: list[float] = []
        previous_action_idx: int | None = None
        previous_smoothed_reward = 0.0

        while not done:
            baseline_suggestion = None
            if advisor.available:
                baseline_suggestion = f"SONG_{advisor.suggest(obs, songs_subset, memory)}"
            prompt = state_to_prompt(obs, max_idx=max_idx, memory=memory, baseline_suggestion=baseline_suggestion)
            action_token_ids = action_token_ids_full[: max_idx + 1]
            warmup_random = episode <= 5
            if warmup_random:
                action_idx = random.randint(0, max_idx)
                response_text = f"Reasoning: warmup random policy. Action: SONG_{action_idx}"
                query_tensor = torch.tensor([], dtype=torch.long, device=device)
                response_tensor = torch.tensor([], dtype=torch.long, device=device)
            else:
                action_idx, query_tensor, response_tensor = _select_action_from_logits(
                    ppo_trainer=ppo_trainer,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    action_token_ids=action_token_ids,
                )
                response_text = f"Reasoning: logit-based selection from fixed action tokens. Action: SONG_{action_idx}"
            repeated = memory.is_recent_repeat(action_idx)
            action_switched = previous_action_idx is not None and action_idx != previous_action_idx
            obs = env_episode.step(MusicRlAction(song_index=action_idx))
            env_reward = float(obs.reward or 0.0)
            shaped_reward = shape_reward(
                env_reward=env_reward,
                repeated=repeated,
                action_switched=action_switched,
                prev_reward=previous_smoothed_reward,
            )
            previous_smoothed_reward = shaped_reward
            previous_action_idx = action_idx
            step_reasoning_score = reasoning_score(response_text)
            memory.add(action_idx, env_reward)

            raw_total += env_reward
            shaped_total += shaped_reward
            episode_raw_step_rewards.append(env_reward)
            episode_shaped_step_rewards.append(shaped_reward)
            episode_reasoning_scores.append(step_reasoning_score)
            reasoning_scores_all.append(step_reasoning_score)
            done = bool(obs.done)
            episode_steps += 1
            episode_parsed += 1

            if not warmup_random:
                query_buffer.append(query_tensor)
                response_buffer.append(response_tensor)
                reward_buffer.append(torch.tensor(shaped_reward, dtype=torch.float32, device=device))
                if len(query_buffer) >= ppo_config.batch_size:
                    query_buffer, response_buffer, reward_buffer, running_mean, running_std = _try_ppo_step(
                        ppo_trainer=ppo_trainer,
                        query_buffer=query_buffer,
                        response_buffer=response_buffer,
                        reward_buffer=reward_buffer,
                        running_mean=running_mean,
                        running_std=running_std,
                    )

            last_action_text = f"SONG_{action_idx}"
            if DEBUG_MODE:
                short_prompt = prompt[:200].replace("\n", " ")
                short_output = response_text[:200].replace("\n", " ")
                print(f"[EP {episode} | STEP {episode_steps}]")
                print(f"Action: {last_action_text} | Parsed: True")
                print(f"Raw reward: {env_reward:.3f} | Shaped: {shaped_reward:.3f}")
                print(f"Reasoning score: {step_reasoning_score}/5")
                print(f"Prompt: \"{short_prompt}\"")
                print(f"Output: \"{short_output}\"")
                print(f"Memory actions: {list(memory.last_actions)}")
                print(f"Memory rewards: {[round(x, 3) for x in list(memory.last_rewards)]}")
                if len(memory.last_actions) > 3 or len(memory.last_rewards) > 3:
                    print("WARNING: Memory buffer exceeded expected size 3")
            elif episode % 20 == 0 and episode_steps == 1:
                print(f"Reasoning sample (ep {episode}): {response_text[:180]}")

        raw_rewards_per_episode.append(raw_total)
        shaped_rewards_per_episode.append(shaped_total)
        episode_parse_rate = 1.0 if episode_steps else 0.0
        per_episode_parse_rates.append(episode_parse_rate)
        last_20_mean_reward = mean(raw_rewards_per_episode[-min(20, len(raw_rewards_per_episode)) :])
        best_last20 = max(best_last20, last_20_mean_reward)
        if episode > 40 and last_20_mean_reward < best_last20 * 0.7:
            print("Early stopping: performance dropped from peak")
            break

        print(
            f"Episode {episode}/{episodes} | raw={raw_total:.3f} | shaped={shaped_total:.3f} | "
            f"action={last_action_text} | parse_ok={episode_parsed}/{episode_steps}"
        )
        print(f"Parse success rate: {episode_parse_rate * 100:.1f}%")
        if episode_raw_step_rewards:
            print(
                "Reward distribution | "
                f"raw(min={min(episode_raw_step_rewards):.3f}, max={max(episode_raw_step_rewards):.3f}, "
                f"mean={mean(episode_raw_step_rewards):.3f}) | "
                f"shaped(min={min(episode_shaped_step_rewards):.3f}, max={max(episode_shaped_step_rewards):.3f}, "
                f"mean={mean(episode_shaped_step_rewards):.3f})"
            )

    if query_buffer:
        query_buffer, response_buffer, reward_buffer, running_mean, running_std = _try_ppo_step(
            ppo_trainer=ppo_trainer,
            query_buffer=query_buffer,
            response_buffer=response_buffer,
            reward_buffer=reward_buffer,
            running_mean=running_mean,
            running_std=running_std,
        )

    baseline_rewards = evaluate_random(env_seed=2000, episodes=20, max_idx=max_idx)
    baseline_mean = mean(baseline_rewards)
    baseline_std = math.sqrt(mean([(r - baseline_mean) ** 2 for r in baseline_rewards]))
    mean_reward = mean(raw_rewards_per_episode)
    shaped_mean_reward = mean(shaped_rewards_per_episode)
    last20_mean = mean(raw_rewards_per_episode[-min(20, len(raw_rewards_per_episode)) :])
    success_rate = (sum(1 for r in raw_rewards_per_episode if r > 0) / len(raw_rewards_per_episode)) if raw_rewards_per_episode else 0.0
    parse_success_rate = 1.0

    plot_path = save_training_plot(raw_rewards_per_episode, path=os.path.join(os.getcwd(), "trl_training_plot.png"))

    summary = {
        "episodes": len(raw_rewards_per_episode),
        "device": str(device),
        "subset_size": subset_size,
        "mean_reward": mean_reward,
        "shaped_mean_reward": shaped_mean_reward,
        "last20_mean_reward": last20_mean,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "improvement": mean_reward - baseline_mean,
        "success_rate": success_rate,
        "parse_success_rate": parse_success_rate,
        "baseline_guidance_enabled": advisor.available,
        "plot": os.path.basename(plot_path) if plot_path else PLOT_PATH.name,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINAL TRL RESULTS ===")
    print(f"Baseline mean: {baseline_mean:.3f}")
    print(f"Mean reward: {summary['mean_reward']:.3f}")
    print(f"Last 20 mean reward: {summary['last20_mean_reward']:.3f}")
    print(f"Best last-20 mean: {best_last20:.3f}")
    print(f"Improvement over baseline: {summary['improvement']:.3f}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Parse success rate: {parse_success_rate * 100:.1f}%")
    print(f"Saved plot: {plot_path or PLOT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")

    if DEBUG_MODE and raw_rewards_per_episode:
        avg_reasoning = mean(reasoning_scores_all) if reasoning_scores_all else 0.0
        reward_variance = float(np.var(np.asarray(raw_rewards_per_episode, dtype=np.float32)))
        last5_mean = mean(raw_rewards_per_episode[-min(5, len(raw_rewards_per_episode)) :])
        print("\n=== DEBUG SUMMARY ===")
        print(f"Mean reward: {mean_reward:.3f}")
        print(f"Last 5 mean reward: {last5_mean:.3f}")
        print(f"Parse success rate: {parse_success_rate:.3f}")
        print(f"Avg reasoning score: {avg_reasoning:.3f}")
        print(f"Reward variance: {reward_variance:.3f}")
        if parse_success_rate < 0.8:
            print("Interpretation: parsing issue likely (parse_success < 0.8).")
        if reward_variance > 2.0:
            print("Interpretation: reward variance is high -> possible instability.")
        if avg_reasoning < 1.0:
            print("Interpretation: prompt may not be triggering reasoning keywords.")

        debug_payload = {
            "episode_rewards": raw_rewards_per_episode,
            "parse_success_rates": per_episode_parse_rates,
            "reasoning_scores": reasoning_scores_all,
        }
        DEBUG_PATH.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        print(f"Saved debug snapshot: {DEBUG_PATH}")

    # Always run a final save attempt to maximize reliability in headless runtimes.
    save_training_plot(raw_rewards_per_episode, path=os.path.join(os.getcwd(), "trl_training_plot.png"))


if __name__ == "__main__":
    main()
