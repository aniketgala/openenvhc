# Music Recommendation RL Project

## 1) Problem

Train an agent to recommend songs that adapt to evolving user preferences over an episode.  
This project includes:

- an OpenEnv-compatible custom environment
- a strong DRQN baseline (recurrent Q-learning)
- an additional TRL-based LLM policy pipeline for hackathon requirements
- Hugging Face Space deployment with a Gradio UI

## 2) Environment

Environment class: `server/music_rl_env_environment.py` (`MusicRlEnvironment`)

- `reset()` returns an observation
- `step(action)` applies an action and returns next observation with reward/done metadata (OpenEnv schema-compatible)
- `observation_state()` provides current observable state helper
- FastAPI server wiring remains in `server/app.py` via `create_app(...)`

Observation includes:

- `phase`
- `last_outcome`
- `recent_song_features` (`energy`, `valence`, `danceability`)

Action supports:

- `song_index` (int) for index-based recommendation
- `song_id` (str) for backward compatibility

Reward logic is unchanged from the existing environment implementation.

## 3) Methods

### DRQN (baseline RL)

File: `deep_rl_train.py`

- recurrent `Q(seq, action_features)` model
- multi-seed training (`[0, 42, 99]`)
- best-checkpoint evaluation per seed (`best_model_seed_<seed>.pt`)
- deterministic evaluation (`epsilon=0`)
- optional ensemble evaluation over top checkpoints

### TRL (LLM-based RL)

File: `trl_train.py`

- lightweight model (`distilgpt2`) with TRL PPO
- state-to-text prompt to predict song index
- environment interaction loop using the same `MusicRlEnvironment`
- small-run training with reward logging and summary outputs

## 4) Results

- DRQN output artifacts:
  - `model.pt`
  - `best_model_seed_0.pt`
  - `best_model_seed_42.pt`
  - `best_model_seed_99.pt`
  - `final_training_plot.png`
  - `reward_distribution.png`
  - `summary.json`
- TRL output artifacts:
  - `trl_training_plot.png`
  - `trl_summary.json`

The DRQN baseline is the primary high-performing method; TRL demonstrates an LLM-based RL training path.

## 5) How To Run

### Install

```bash
pip install -r requirements.txt
```

### DRQN training

```bash
python deep_rl_train.py --dataset-path "dataset.csv" --episodes 1400 --sample-size 50 --batch-size 32 --seed 42
```

### TRL training

```bash
python trl_train.py
```

### Hugging Face Space / local Gradio app

```bash
python app.py
```

In the UI, select:

- `DRQN (Deep RL)` or
- `LLM (TRL)`

and click **Run Training**.

## ☁️ Running on Google Colab

To avoid dependency conflicts in Colab:

In Colab, some preinstalled packages (like diffusers) may cause dependency conflicts. The provided setup cell removes these and installs a compatible environment.

1. Run setup cell to install compatible versions  
2. Restart runtime  
3. Run training script
