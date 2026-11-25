"""Local Gradio demo for MotifForge running on Ollama."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from ollama_client import chat
from svg_utils import extract_svg, render_svg, validate_svg


def run(prompt, size, temperature, model, base_url, max_new_tokens):
    if not prompt.strip():
        return None, "", None, "Enter a description first."
    try:
        text, seconds, _ = chat(
            base_url, model, prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
    except RuntimeError as exc:
        return None, "", None, str(exc)
    svg = extract_svg(text)
    status = f"Generated in {seconds:.1f}s."
    if not svg:
        return None, text, None, status + " No SVG found in the reply."
    ok, reason = validate_svg(svg)
    if not ok:
        return None, svg, None, status + f" SVG rejected: {reason}"
    image = render_svg(svg, size=int(size))
    if image is None:
        return None, svg, None, status + " SVG could not be rendered."
    path = Path(tempfile.gettempdir()) / "motifforge_last.svg"
    path.write_text(svg, encoding="utf-8")
    return image, svg, str(path), status + " Valid SVG, ready to scale or edit."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="motifforge")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install gradio first: pip install gradio") from exc

    def handler(prompt, size, temperature):
        return run(prompt, size, temperature, args.model, args.ollama, args.max_new_tokens)

    with gr.Blocks(title="MotifForge") as demo:
        gr.Markdown("# MotifForge\nType a motif or icon. Get an editable SVG from a local Qwen3 model.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Description",
                    lines=3,
                    value=(
                        "A minimal three petal flower motif in deep teal and warm sand, "
                        "centered on a soft cream background."
                    ),
                )
                size = gr.Slider(128, 2048, value=512, step=64, label="Preview size in pixels")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                button = gr.Button("Generate SVG", variant="primary")
            with gr.Column():
                image = gr.Image(label="Rendered preview", type="pil")
                svg = gr.Textbox(label="SVG code", lines=10)
                download = gr.File(label="Download SVG")
                status = gr.Markdown("Ready.")
        button.click(handler, [prompt, size, temperature], [image, svg, download, status])
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
