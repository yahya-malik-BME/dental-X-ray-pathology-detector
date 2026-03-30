from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import CLASS_COLORS, visualize_predictions
from src.predict import InferenceConfig, Predictor

# ── Constants ──────────────────────────────────────────────────────────────
WEIGHTS_PATH = "outputs/weights/best.pt"
TITLE = "Dental X-Ray Pathology Detector"
DESCRIPTION = """
Upload a panoramic or periapical dental X-ray to detect pathologies using a
YOLOv8 model trained on the DENTEX dataset.

**Detectable conditions:**
- Caries (cavities)
- Deep caries (near pulp)
- Periapical lesions (root tip infections)
- Impacted teeth

*Built for research collaboration with the Harvard dental research lab.*
"""

EXAMPLES = [
    "data/raw/example_panoramic.jpg",
    "data/raw/example_periapical.jpg",
]


# ── Model loading ───────────────────────────────────────────────────────────
def load_predictor() -> Predictor:
    """Load the predictor once at startup."""
    config = InferenceConfig(
        weights_path=WEIGHTS_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device="cpu",
        save_visualizations=False,
    )
    return Predictor(config)


predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Lazy singleton — only load model once."""
    global predictor
    if predictor is None:
        predictor = load_predictor()
    return predictor


# ── Inference function ──────────────────────────────────────────────────────
def run_inference(image: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Gradio inference function.

    Args:
        image: uploaded image as numpy array (RGB, HxWx3)

    Returns:
        annotated_image: numpy array with bounding boxes drawn
        report_markdown: markdown string with findings summary
    """
    if image is None:
        return None, "Please upload an X-ray image."

    import cv2

    # Save temp file for predictor (predictor expects a file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
        Image.fromarray(image).save(temp_path)

    try:
        pred = get_predictor()
        result = pred.predict_image(temp_path, save_visualization=False)
        detections = result["detections"]
    except FileNotFoundError:
        return image, (
            "**Model weights not found.**\n\n"
            "Please train the model first:\n```bash\npython -m src.train\n```"
        )

    # Draw annotations on image
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(image, cmap="gray")
    ax.axis("off")

    for det in detections:
        if det["class_id"] == 0:  # skip plain tooth detections
            continue
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_id"], "#fff")
        label = f"{det['class_name'].replace('_', ' ')} {det['confidence']:.0%}"
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 6, 0),
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.75, pad=2, edgecolor="none"),
            verticalalignment="bottom",
        )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    annotated = np.array(Image.open(buf))
    plt.close()

    # ── Build markdown report ──
    summary = result["pathology_summary"]
    total = result["num_detections"]

    if total == 0:
        findings = "No pathologies detected above the confidence threshold."
    else:
        findings = "\n".join(
            [
                f"| {name.replace('_', ' ').title()} | {count} |"
                for name, count in summary.items()
                if count > 0
            ]
        )
        findings = "| Condition | Count |\n|---|---|\n" + findings

    report = f"""
### Findings

{findings}

---

**Total pathologies detected:** {total}
**Confidence threshold:** 25%
**Model:** YOLOv8m trained on DENTEX dataset

> This tool is for research purposes only and is not a substitute for clinical diagnosis.
"""

    return annotated, report


# ── Gradio UI ───────────────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload dental X-ray",
                    type="numpy",
                    height=400,
                )
                run_btn = gr.Button("Analyze X-ray", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Detected pathologies",
                    type="numpy",
                    height=400,
                )
                report_output = gr.Markdown(label="Clinical findings")

        run_btn.click(
            fn=run_inference,
            inputs=[input_image],
            outputs=[output_image, report_output],
        )

        input_image.upload(
            fn=run_inference,
            inputs=[input_image],
            outputs=[output_image, report_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            "*Model trained on [DENTEX 2023 dataset](https://github.com/ibrahimethemhamamci/DENTEX). "
            "Built with PyTorch + YOLOv8 + Gradio.*"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
