# 🎧 Sequential Music Recommendation using RL

## 🎵 Why This Project Exists

I listen to music a lot — and something always felt off.

Music apps are great at recommending songs…
but not great at adapting when your mood changes.

You might start with energetic tracks, then suddenly want something calm —
but the system keeps pushing the same type of songs.

That made me wonder:

👉 Can a system *adapt in real-time* to changing preferences?

## ❗ The Problem

Most recommendation systems assume your preferences are static.

But in reality:

* mood changes
* context changes
* preferences shift quickly

This project treats recommendation as a **sequential decision problem**, not a static one.

## 🌍 What Makes This Different

There is very little work on applying **reinforcement learning to music recommendation in a sequential setting**.

Instead of predicting what you like once:

👉 this system continuously adapts as your preferences evolve.

## 🎧 Grounded in Real Music Data

This system uses **Spotify-style audio features**, such as:

* Danceability
* Energy
* Valence

These aren’t random numbers — they represent how a track *feels*.

This allows the system to:

* match mood
* adjust to energy levels
* recommend context-aware songs

## ⚙️ How It Works (High-Level)

* Reinforcement Learning (decision making over time)
* Recurrent model (to remember past choices)
* Language model (for exploratory experiments)
* OpenEnv-compatible environment

## 🛠️ What It Took to Build

This project wasn’t straightforward.

There were:

* unstable training runs
* reward function issues
* debugging environment behavior
* many failed experiments

> “After ~40–50 iterations and countless fixes, things finally started working.”

## 💡 Why This Matters to Me

I built this because I genuinely enjoy music.

This wasn’t just about models —
it was about exploring how systems can better understand human preferences.

## 🧠 What I Learned

Adapting to changing preferences is harder than it looks.

* traditional RL gives stability
* LLM-based approaches offer flexibility

The real challenge is balancing both.

## 🚀 Closing Thought

Music is dynamic.  
Recommendation systems should be too.
# 🎧 Sequential Music Recommendation using RL

## 🎵 Why This Project Exists

I spend a lot of time listening to music — and one thing always stood out:

> recommendation systems don’t *adapt fast enough* when your mood changes.

You go from energetic to calm, or from sad to happy — but the system keeps recommending the same kind of songs.

That made me wonder:

👉 **Can we build a system that adapts in real-time to changing user preferences?**

That question became this project.

## 🌍 Why This Is Interesting

Most recommendation systems today rely on:

* collaborative filtering
* static preference modeling

But:

> **Sequential reinforcement learning for music recommendation is still largely unexplored.**

This project treats recommendation as a **dynamic decision-making problem**, where:

* user preference shifts mid-episode
* the agent must continuously adapt

## 🎧 Real Music Data

This system is not trained on synthetic inputs.

It uses **Spotify-style audio features**, including:

* **Danceability**
* **Energy**
* **Valence**

These features allow the agent to reason about:

* how energetic a track feels
* whether it matches the user’s mood
* how suitable it is for the current phase

👉 This makes the recommendation process grounded in **real music characteristics**, not abstract signals.

## 🚀 Try It Yourself

### DRQN (Live Training on Hugging Face)

👉 [https://huggingface.co/spaces/aniketgala/music_rl_env](https://huggingface.co/spaces/aniketgala/music_rl_env)

* Select **DRQN**
* Click **Start Training**
* Observe:
  * Live logs
  * Training curve
  * Final metrics

### TRL (Notebook Training)

👉 [https://colab.research.google.com/drive/1IOcUpg8UCMy3_BJQIBbt_kCVkgr9cZqu?usp=sharing](https://colab.research.google.com/drive/1IOcUpg8UCMy3_BJQIBbt_kCVkgr9cZqu?usp=sharing)

* Open notebook
* Run all cells
* See TRL training behavior

### ⚠️ Note

TRL training is run in Colab due to computational constraints and longer runtime (~20 minutes).

## 🛠️ Build Journey

This wasn’t a straight path.

* multiple training failures
* unstable reward signals
* debugging OpenEnv integration
* and yes… *a lot* of Hugging Face pushes

> “After ~60 iterations, things finally started working.”

The turning point was stabilizing the DRQN training and introducing ensemble evaluation.

## 🧪 Environment Design (OpenEnv Compatible)

The environment follows the **OpenEnv standard**, making it modular and reproducible.

It includes:

* structured state representation:
  * phase (preference shift stage)
  * last outcome
  * recent feature history
* discrete action space:
  * selecting a song index
* reward function:
  * alignment with hidden user preference
  * adaptation bonus
  * penalty for repetition

👉 This ensures the system behaves like a **real sequential decision-making environment**, not a toy simulation.

d## 📈 DRQN Results

![DRQN Training Curve](results/drqn/final_training_plot.png)

![DRQN Loss Plot](results/drqn/drqn_loss_plot.png)

![DRQN Ensemble Comparison](results/drqn/drqn_comparison_bar.png)

DRQN is the **core model of this project**.

It uses:

* recurrent memory (LSTM)
* multi-seed training
* ensemble evaluation

to learn a stable policy in a non-stationary environment.

> Unlike static recommenders, it adapts based on recent interaction history.

Because the model operates on real audio features (energy, valence, danceability), it learns to align recommendations with underlying musical properties.

### 📊 What This Shows

* The model consistently outperforms a random baseline
* Learning stabilizes over time
* Ensemble improves performance significantly (+1.26)

👉 This confirms the agent is **learning meaningful preference patterns**

### 🧠 Key Observation

* Consistent convergence above baseline
* Ensemble significantly boosts performance
* Low variance compared to TRL

## 🤖 TRL (LLM-based RL Exploration)

Alongside DRQN, I explored using a language model with PPO (TRL).

This approach is fundamentally different:

* instead of learning Q-values
* it generates actions through a language model

What I observed:

* it *does* learn to reduce preference mismatch
* but training is slower and more unstable
* performance does not match DRQN

> This highlights a key insight: LLM-based RL is flexible, but not yet as efficient for this type of structured task.

The same environment is used for TRL, ensuring a fair comparison between classical RL and LLM-based approaches.

## ⚖️ DRQN vs TRL

| Aspect      | DRQN    | TRL      |
| ----------- | ------- | -------- |
| Performance | High    | Moderate |
| Stability   | High    | Low      |
| Adaptation  | Limited | Strong   |
| Runtime     | Fast    | Slow     |

## 🧠 Final Insight

This project highlights an important trade-off:

* **DRQN → stable, high-performance learning**
* **TRL → flexible, adaptive but less efficient**

In real-world systems:

> the challenge is not just learning — but learning *under changing conditions*.

## 🚀 Closing Note

This project started from a simple curiosity about music…
and turned into an exploration of how intelligent systems can adapt over time.
