# 🎧 Adaptive Music Recommendation with DRQN vs LLM-RL

## 🚀 Overview

We built a sequential music recommendation system under **changing user preferences**, modeled as a non-stationary RL environment.

We compare two fundamentally different approaches:

* **DRQN (Deep RL)** → recurrent Q-learning with memory
* **TRL (LLM-based RL)** → language-model policy optimized via PPO

## 🌟 Why This Project is Unique

Reinforcement learning has been widely explored in:

* games
* robotics
* control systems

However, **sequential recommendation under shifting user preferences** — especially combining **deep RL and LLM-based RL** — is still relatively underexplored.

Most recommendation systems:

* assume static preferences
* rely on supervised learning or bandits

In contrast, this project:

* models **dynamic user behavior**
* introduces **mid-episode preference shifts**
* compares **two fundamentally different learning paradigms**

👉 This makes it closer to real-world user interaction than traditional setups.

## 🧠 Key Idea

User preferences shift mid-episode.

The agent must:

* detect preference change
* adapt recommendations dynamically

## ⚙️ Environment

* State:
  * phase (before/after shift)
  * last outcome
  * recent song features (energy, valence, danceability)
* Action:
  * select a song (index)
* Reward:
  * genre match
  * mood alignment
  * adaptation bonus
  * repeat penalty
  * mismatch penalty

## 🏗️ Methods

### 1. DRQN (Baseline)

* LSTM-based Q-network
* multi-seed training + ensemble
* strong, stable performance

### 2. TRL (LLM-RL)

* TinyLlama / distilgpt2
* PPO optimization
* logit-based action selection (no parsing)
* reward shaping + compression

## 📊 Results

| Model           | Score     |
| --------------- | --------- |
| Random Baseline | ~1.45     |
| TRL (LLM-RL)    | **2.06**  |
| DRQN            | **~7–8+** |

## 🔍 Key Insights

* DRQN is **stable and high-performing**
* TRL is **adaptive but high-variance**
* Even with reward stabilization, TRL shows spikes due to **non-stationary preferences**

## 📈 Interpretation

Reward spikes correspond to:  
👉 successful adaptation after preference shifts

This highlights a key challenge:

> RL in dynamic environments is inherently volatile, especially for language-based policies.

## 🧪 Demo

Run:

```bash
python app.py
```

Choose:

* DRQN → stable behavior
* TRL → adaptive but spiky behavior

## 💡 Takeaway

* Deep RL → optimization
* LLM-RL → reasoning + adaptability

👉 Hybrid systems are promising.

## 🛠️ Build Journey

This wasn’t a smooth ride.

* Multiple RL approaches failed before stabilizing DRQN
* TRL pipeline initially produced invalid outputs (`!!!!`, broken parsing)
* Reward instability required multiple redesigns (scaling, normalization, log compression)

And yes…

> **After ~45 Hugging Face Space pushes, countless crashes, and debugging sessions — it finally worked.**

## 🧠 What We Learned

* LLMs are not naturally good at raw action selection → require structure
* PPO with language models is highly sensitive to reward design
* Non-stationary environments introduce unavoidable variance
* Stability (DRQN) vs adaptability (TRL) is a real trade-off

## ☁️ Running on Google Colab

To avoid dependency conflicts in Colab:

In Colab, some preinstalled packages (like diffusers) may cause dependency conflicts. The provided setup cell removes these and installs a compatible environment.

1. Run setup cell to install compatible versions  
2. Restart runtime  
3. Run training script

> This project started as an experiment — and ended as a comparison of two different ways machines can learn to understand human preferences.
