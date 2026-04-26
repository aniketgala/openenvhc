"""Hugging Face Space runner with DRQN and TRL modes."""

from __future__ import annotations

import re
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Generator

import gradio as gr


ROOT = Path(__file__).resolve().parent
SCRIPT_CONFIG = {
    "DRQN (Deep RL)": {
        "script": ROOT / "deep_rl_train.py",
        "plot": ROOT / "final_training_plot.png",
        "summary": ROOT / "summary.json",
    },
    "LLM (TRL)": {
        "script": ROOT / "trl_train.py",
        "plot": ROOT / "trl_training_plot.png",
        "summary": ROOT / "trl_summary.json",
    },
}


def _extract_metrics(logs: str) -> str:
    """Extract key final metrics from training logs."""
    patterns = {
        "baseline": r"(?:Random baseline mean ± std|Baseline mean ± std):\s*(.+)",
        "trained": r"(?:Trained policy mean ± std|Ensemble mean ± std):\s*(.+)",
        "mood": r"Mood match:\s*(.+)",
        "genre": r"Genre match:\s*(.+)",
        "improvement": r"(?:Reward improvement over baseline|Ensemble improvement|Improvement over baseline):\s*(.+)",
        "trl_mean": r"Mean reward:\s*(.+)",
        "trl_last": r"Last 20 mean reward:\s*(.+)",
    }

    values: dict[str, str] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, logs)
        if matches:
            values[key] = matches[-1].strip()

    if not values:
        return "Metrics not found in logs."

    lines = ["### Final Metrics"]
    if "baseline" in values:
        lines.append(f"- Baseline mean ± std: `{values['baseline']}`")
    if "trained" in values:
        lines.append(f"- Trained mean ± std: `{values['trained']}`")
    if "mood" in values:
        lines.append(f"- Mood match: `{values['mood']}`")
    if "genre" in values:
        lines.append(f"- Genre match: `{values['genre']}`")
    if "improvement" in values:
        lines.append(f"- Improvement: `{values['improvement']}`")
    if "trl_mean" in values:
        lines.append(f"- TRL mean reward: `{values['trl_mean']}`")
    if "trl_last" in values:
        lines.append(f"- TRL last-20 mean: `{values['trl_last']}`")
    if "trl_mean" in values or "trl_last" in values or "improvement" in values:
        lines.append("")
        lines.append("Observation: TRL shows higher variance due to preference shifts.")
    return "\n".join(lines)


def run_training(mode: str) -> Generator[tuple[str, str | None, str, str | None], None, None]:
    """Run training with streamed status updates for Gradio."""
    logs = ""
    start = time.time()
    cfg = SCRIPT_CONFIG.get(mode, SCRIPT_CONFIG["DRQN (Deep RL)"])
    train_script = cfg["script"]
    plot_path = cfg["plot"]
    summary_path = cfg["summary"]
    yield "🚀 Starting training...\n", None, "Metrics will appear after training completes.", None

    if not train_script.exists():
        yield f"❌ Error: missing `{train_script.name}`", None, "Unable to run training.", None
        return

    yield (
        f"🚀 Starting {mode}...\n⏳ Running training (this may take ~1–2 minutes)...\n",
        None,
        "Training in progress...",
        None,
    )

    cmd = [sys.executable, "-u", str(train_script)]
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONUTF8": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:  # pragma: no cover
        yield (
            f"❌ Failed to launch {mode} process.\n\n{exc}",
            None,
            "No metrics available.",
            None,
        )
        return

    if process.stdout is not None:
        for line in process.stdout:
            logs += line
            yield (
                f"🚀 Starting {mode}...\n⏳ Running training (this may take ~1–2 minutes)...\n\n{logs}",
                None,
                "Training in progress...",
                None,
            )
    return_code = process.wait()

    if return_code != 0:
        yield (
            f"{logs}\n\n❌ Training failed (exit code {return_code}).",
            None,
            "No metrics available due to training failure.",
            None,
        )
        return

    if not plot_path.exists():
        yield (
            f"{logs}\n\n❌ Training finished but `{plot_path.name}` is missing.",
            None,
            "No metrics available because output plot is missing.",
            str(summary_path) if summary_path.exists() else None,
        )
        return

    if not summary_path.exists():
        yield (
            f"{logs}\n\n❌ Training finished but `{summary_path.name}` is missing.",
            str(plot_path),
            "Metrics extracted, but summary JSON is missing.",
            None,
        )
        return

    metrics_md = _extract_metrics(logs)
    duration = time.time() - start
    yield (
        f"{logs}\n\n✅ Training complete in {duration:.2f} seconds.",
        str(plot_path),
        metrics_md,
        str(summary_path),
    )


with gr.Blocks(title="RL Music Recommendation Training") as demo:
    gr.Markdown("# 🎧 RL Music Recommendation Training")
    gr.Markdown("⚠️ Logs update live below. Scroll if needed.")
    gr.Markdown("Run training and review logs, metrics, and final plot.")

    mode_dropdown = gr.Dropdown(
        choices=["DRQN (Deep RL)", "LLM (TRL)"],
        value="DRQN (Deep RL)",
        label="Training Mode",
    )
    run_btn = gr.Button("Run Training", variant="primary")

    log_output = gr.Textbox(label="Logs", lines=24)
    plot_output = gr.Image(label="Final Plot")
    metrics_output = gr.Markdown(label="Metrics")
    summary_file = gr.File(label="Download Summary JSON")

    run_btn.click(
        fn=run_training,
        inputs=[mode_dropdown],
        outputs=[log_output, plot_output, metrics_output, summary_file],
    )


if __name__ == "__main__":
    demo.launch()
