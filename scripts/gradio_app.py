from __future__ import annotations

import argparse
import tempfile

import gradio as gr
import torch
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensity

from medvlm.evaluation import generate_report
from medvlm.training import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a MedVLM report-generation demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config, _ = load_model_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    transform = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize((config.img_size, config.img_size)),
        ]
    )

    def predict(image):
        if image is None:
            return "Upload a frontal chest X-ray image."
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.convert("L").save(tmp.name)
            tensor = transform(tmp.name).unsqueeze(0)
        return generate_report(model, tokenizer, tensor, min_gen_len=40)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", image_mode="L", label="Frontal chest X-ray"),
        outputs=gr.Textbox(label="Generated report", lines=8),
        title="MedVLM Chest X-Ray Report Generator",
        description="Research demo. Not for clinical use.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
