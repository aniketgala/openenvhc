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
        "training_curve": ROOT / "final_training_plot.png",
        "loss_curve": ROOT / "drqn_loss_plot.png",
        "comparison_bar": ROOT / "drqn_comparison_bar.png",
    },
    "LLM (TRL)": {
        "script": ROOT / "trl_train.py",
        "plot": ROOT / "trl_training_plot.png",
        "summary": ROOT / "trl_summary.json",
        "training_curve": ROOT / "trl_training_plot.png",
        "loss_curve": None,
        "comparison_bar": ROOT / "trl_comparison_bar.png",
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


def _path_if_exists(path_obj: Path | None) -> str | None:
    if path_obj is None:
        return None
    return str(path_obj) if os.path.exists(path_obj) else None


def _normalize_mode(mode: str) -> str:
    mode_map = {
        "DRQN": "DRQN (Deep RL)",
        "TRL": "LLM (TRL)",
    }
    return mode_map.get(mode, mode)


def run_training(
    mode: str,
) -> Generator[
    tuple[
        str,
        str | None,
        str,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        float,
    ],
    None,
    None,
]:
    """Run training with streamed status updates for Gradio."""
    mode = _normalize_mode(mode)
    logs = ""
    progress = 0.0
    start = time.time()
    cfg = SCRIPT_CONFIG.get(mode, SCRIPT_CONFIG["DRQN (Deep RL)"])
    train_script = cfg["script"]
    plot_path = cfg["plot"]
    summary_path = cfg["summary"]
    yield (
        "🚀 Starting training...\n",
        None,
        "Metrics will appear after training completes.",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        progress,
    )

    if not train_script.exists():
        yield (
            f"❌ Error: missing `{train_script.name}`",
            None,
            "Unable to run training.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            progress,
        )
        return

    yield (
        f"🚀 Starting {mode}...\n⏳ Running training (this may take ~1–2 minutes)...\n",
        None,
        "Training in progress...",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        progress,
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
            universal_newlines=True,
        )
    except Exception as exc:  # pragma: no cover
        yield (
            f"❌ Failed to launch {mode} process.\n\n{exc}",
            None,
            "No metrics available.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            progress,
        )
        return

    if process.stdout is not None:
        stream_buffer = ""
        last_heartbeat = time.time()
        while True:
            chunk = process.stdout.read(1)
            now = time.time()
            if chunk == "" and process.poll() is not None:
                if stream_buffer:
                    logs += stream_buffer
                    if "Episode" in stream_buffer:
                        progress = min(progress + 2.0, 100.0)
                    yield (
                        f"🚀 Starting {mode}...\n⏳ Running training (this may take ~1–2 minutes)...\n\n{logs}",
                        None,
                        "Training in progress...",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        progress,
                    )
                break
            if chunk:
                stream_buffer += chunk
                # Flush stream frequently to keep logs truly live even without newline.
                if chunk == "\n" or len(stream_buffer) >= 80:
                    logs += stream_buffer
                    if "Episode" in stream_buffer:
                        progress = min(progress + 2.0, 100.0)
                    stream_buffer = ""
                    yield (
                        f"🚀 Starting {mode}...\n⏳ Running training (this may take ~1–2 minutes)...\n\n{logs}",
                        None,
                        "Training in progress...",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        progress,
                    )
                    last_heartbeat = now
            elif now - last_heartbeat >= 3.0:
                heartbeat = f"\n[heartbeat {time.strftime('%H:%M:%S')}] Training still running..."
                logs += heartbeat
                yield (
                    f"🚀 Starting {mode}...\n⏳ Running training (this may take ~1–2 minutes)...\n\n{logs}",
                    None,
                    "Training in progress...",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    progress,
                )
                last_heartbeat = now
    return_code = process.wait()

    if return_code != 0:
        yield (
            f"{logs}\n\n❌ Training failed (exit code {return_code}).",
            None,
            "No metrics available due to training failure.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            progress,
        )
        return

    if not plot_path.exists():
        yield (
            f"{logs}\n\n❌ Training finished but `{plot_path.name}` is missing.",
            None,
            "No metrics available because output plot is missing.",
            str(summary_path) if summary_path.exists() else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            progress,
        )
        return

    if not summary_path.exists():
        yield (
            f"{logs}\n\n❌ Training finished but `{summary_path.name}` is missing.",
            str(plot_path),
            "Metrics extracted, but summary JSON is missing.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            progress,
        )
        return

    metrics_md = _extract_metrics(logs)
    duration = time.time() - start
    training_curve_path = _path_if_exists(cfg.get("training_curve"))
    loss_curve_path = _path_if_exists(cfg.get("loss_curve"))
    comparison_bar_path = _path_if_exists(cfg.get("comparison_bar"))
    summary_download_path = _path_if_exists(summary_path)
    main_plot_path = _path_if_exists(plot_path)
    yield (
        f"{logs}\n\n✅ Training complete in {duration:.2f} seconds.\n✅ Training Complete",
        main_plot_path,
        metrics_md,
        summary_download_path,
        training_curve_path,
        loss_curve_path,
        comparison_bar_path,
        training_curve_path,
        loss_curve_path,
        comparison_bar_path,
        summary_download_path,
        100.0,
    )


with gr.Blocks(title="RL Music Recommendation Training") as demo:
    gr.Markdown("# 🎧 RL Music Recommendation Training")
    gr.Markdown("⚠️ Logs update live below. Scroll if needed.")
    gr.Markdown("Run training and review logs, metrics, and final plot.")

    mode_dropdown = gr.Dropdown(
        choices=["DRQN (Deep RL)", "LLM (TRL)", "DRQN", "TRL"],
        value="DRQN (Deep RL)",
        label="Training Mode",
    )
    run_btn = gr.Button("Run Training", variant="primary")
    progress_bar = gr.Slider(0, 100, value=0, label="Progress", interactive=False)

    log_output = gr.Textbox(label="Logs", lines=24)
    plot_output = gr.Image(label="Final Plot")
    metrics_output = gr.Markdown(label="Metrics")
    summary_file = gr.File(label="Download Summary JSON")
    training_curve_output = gr.Image(label="Training Curve")
    loss_curve_output = gr.Image(label="Loss Curve")
    comparison_bar_output = gr.Image(label="Comparison Bar")
    training_curve_file = gr.File(label="Download Training Curve")
    loss_curve_file = gr.File(label="Download Loss Curve")
    comparison_bar_file = gr.File(label="Download Comparison Bar")
    summary_download_file = gr.File(label="Download Summary")

    run_btn.click(
        fn=run_training,
        inputs=[mode_dropdown],
        outputs=[
            log_output,
            plot_output,
            metrics_output,
            summary_file,
            training_curve_output,
            loss_curve_output,
            comparison_bar_output,
            training_curve_file,
            loss_curve_file,
            comparison_bar_file,
            summary_download_file,
            progress_bar,
        ],
    )


if __name__ == "__main__":
    demo.launch()
