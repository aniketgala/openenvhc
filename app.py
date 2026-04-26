"""Minimal Gradio app with correctly wired training API."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent


def run_training(mode: str) -> tuple[str, str | None]:
    print("RUN TRAINING CALLED", flush=True)
    return f"API WORKING: {mode}", None


with gr.Blocks() as demo:
    mode_dropdown = gr.Dropdown(
        ["DRQN (Deep RL)", "TRL (LLM)"],
        value="DRQN (Deep RL)",
        label="Training Mode",
    )

    run_button = gr.Button("Run Training")

    log_output = gr.Textbox(label="Logs", lines=10)
    plot_output = gr.Image(label="Training Plot")
    print("GRADIO APP INITIALIZED", flush=True)

    run_button.click(
        fn=run_training,
        inputs=[mode_dropdown],
        outputs=[log_output, plot_output],
    )


if __name__ == "__main__":
    demo.queue().launch()