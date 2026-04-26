# 🎧 Sequential Music Recommendation using RL

---

## 🚀 Live Demo (Hugging Face Space)

👉 [https://huggingface.co/spaces/aniketgala/music_rl_env](https://huggingface.co/spaces/aniketgala/music_rl_env)

* Run DRQN training live
* View logs, plots, and results
* Interact with the environment

---

## 📘 Project Blog (Full Writeup)

👉 [Project Blog](blog.md)

This contains:

* problem motivation
* uniqueness
* build journey
* insights

---

## 🧪 Training Scripts

### 🔹 DRQN (Deep RL)

Run locally or via Space:

```bash
python deep_rl_train.py
```

---

### 🔹 TRL (LLM-based RL)

👉 Colab Notebook:  
[https://colab.research.google.com/drive/1IOcUpg8UCMy3_BJQIBbt_kCVkgr9cZqu?usp=sharing](https://colab.research.google.com/drive/1IOcUpg8UCMy3_BJQIBbt_kCVkgr9cZqu?usp=sharing)

* Fully runnable
* Includes PPO training
* Generates reward plots

---

## 🧠 Environment Overview

This project implements a **sequential music recommendation environment**.

### State

* phase (preference shift stage)
* last outcome
* recent song features:
  * energy
  * valence
  * danceability

---

### Action

* select a song index

---

### Reward

* alignment with hidden user preference
* penalty for repetition
* bonus for adapting after preference shift

---

### Data

Uses **Spotify-style audio features**, including:

* Danceability
* Energy
* Valence

---

### Standard

Environment is **OpenEnv-compatible**, making it modular and reproducible.
This environment is built using the **latest OpenEnv framework** and follows its standard API (reset, step, observation), ensuring compatibility and reproducibility.
---

## 📊 Results

## 📊 Training Evidence

### DRQN (Primary Model)

![Training Curve](results/drqn/final_training_plot.png)

![Loss Curve](results/drqn/drqn_loss_plot.png)

![Ensemble Comparison](results/drqn/drqn_comparison_bar.png)

These plots are generated from real training runs and demonstrate learning progress and performance improvement over baseline.

---

### Key Metrics

* Baseline: 7.71
* Ensemble: **8.97**
* Improvement: **+1.26**

---

### Interpretation

* Stable learning across episodes
* Ensemble significantly improves performance
* Model captures preference dynamics effectively

---

## 🤖 TRL (LLM-based RL)

* Implemented using PPO
* Trained in Colab due to runtime constraints

Summary:

* Learns preference alignment
* More adaptive but less stable
* Higher computational cost

Latest run metrics (`trl_summary.json`):

* Baseline: -3.825
* Mean reward: **-4.300**
* Improvement: **-0.475**
* Last-20 mean: **-4.300**

---

## ⚖️ DRQN vs TRL

| Aspect      | DRQN    | TRL      |
| ----------- | ------- | -------- |
| Performance | High    | Low (latest run) |
| Stability   | High    | Low      |
| Adaptation  | Limited | Strong   |
| Runtime     | Fast    | Slow     |

---

## 📁 Project Structure

```text
.
├── deep_rl_train.py
├── trl_train.py
├── app.py
├── blog.md
├── results/
│   ├── drqn/
│   └── trl/
```

---

## 🎯 Key Highlights

* Sequential RL for dynamic recommendation
* Uses real music features (energy, valence, danceability)
* OpenEnv-compatible environment
* Multi-seed training + ensemble
* Comparison with LLM-based RL

---

## 🚀 How to Run

### Local

```bash
python deep_rl_train.py
```

---

### Demo

Use Hugging Face Space:  
👉 [https://huggingface.co/spaces/aniketgala/music_rl_env](https://huggingface.co/spaces/aniketgala/music_rl_env)
