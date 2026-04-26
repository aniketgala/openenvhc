# 🎧 Adaptive Music Recommendation with DRQN vs LLM-RL

---

## 👤 Human Story — Why This Project Exists

I’ve always loved music.

Not just listening casually, but noticing how preferences change:

* sometimes you want high-energy tracks
* sometimes something calm
* sometimes your mood shifts halfway through a playlist

That made me think:

> Can a system actually *adapt* to changing taste in real time?

Most recommendation systems feel static — they don’t evolve with you in the moment.

So I wanted to explore:

* how learning systems handle changing preferences
* whether AI can adapt like a human would

This project started from that curiosity.

And honestly, it wasn’t easy.

* Early RL attempts failed completely
* The LLM agent produced outputs like `!!!!`
* Training was unstable for a long time

And yes…

> **After ~45 Hugging Face pushes, crashes, and debugging loops — it finally came together.**

---

## 🤖 AI System — Technical Overview

### 🚀 Problem

Sequential music recommendation in a **non-stationary environment**, where user preferences change mid-episode.

---

### ⚙️ Environment

* State:

  * phase (before/after preference shift)
  * last outcome
  * recent song features (energy, valence, danceability)

* Action:

  * select song index

* Reward:

  * genre match
  * mood alignment
  * adaptation bonus
  * repeat penalty
  * mismatch penalty

---

### 🧠 Methods

#### 1. DRQN (Deep RL Baseline)

* LSTM-based Q-network
* Multi-seed training
* Ensemble evaluation

✔ Stable and high-performing

---

#### 2. TRL (LLM-based RL)

* PPO with language model
* Logit-based action selection (no parsing instability)
* Reward shaping + log scaling

✔ Adaptive but higher variance

---

### 🌟 Why This Is Interesting

Reinforcement learning has been widely applied in:

* games
* robotics
* control systems

But **sequential recommendation with dynamic preferences**, especially comparing:

* recurrent RL
* language-model RL

is still relatively underexplored.

This project focuses on:

* real-time preference shifts
* adaptation behavior
* comparing two fundamentally different learning paradigms

---

### 📊 Results

| Model           | Score     |
| --------------- | --------- |
| Random Baseline | ~1.45     |
| TRL (LLM-RL)    | **2.06**  |
| DRQN            | **~7–8+** |

---

### 🔍 Key Insights

* DRQN → stable, consistent optimization
* TRL → adaptive, interpretable, but volatile
* Reward spikes correspond to **preference adaptation**

---

### 📈 Important Observation

Even after stabilizing PPO and compressing rewards:

> Variance remains high due to non-stationary user preferences.

This highlights a real challenge in RL:

👉 Dynamic environments introduce unavoidable instability

---

### 🧠 What We Learned

* LLMs need structured action spaces
* Reward design is critical for PPO
* Adaptability vs stability is a core trade-off
* Hybrid approaches are promising

---

### 🧪 Demo

Run:

```bash
python app.py
```

Compare:

* DRQN → stable behavior
* TRL → adaptive but spiky behavior

---

### 💡 Final Thought

> This project started with a simple idea about music — and turned into an exploration of how machines learn changing human preferences.
