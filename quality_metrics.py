"""Full-reference and prompt-alignment metrics for one SD experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time

import numpy as np
import torch


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def _as_rgb_array(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _lpips_tensor(image, device: str) -> torch.Tensor:
    array = _as_rgb_array(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def _release_model(model, device: str) -> None:
    if device == "cuda":
        model.to("cpu")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class QualityEvaluator:
    """Reuse LPIPS and CLIP models across multiple image comparisons."""

    def __init__(self, device: str):
        self.device = device
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None

    def __enter__(self):
        import lpips
        from transformers import CLIPModel, CLIPProcessor

        self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
        try:
            self.clip_model = CLIPModel.from_pretrained(
                CLIP_MODEL_ID, local_files_only=True
            ).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL_ID, local_files_only=True
            )
        except OSError:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(
                self.device
            ).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lpips_model is not None:
            _release_model(self.lpips_model, self.device)
        if self.clip_model is not None:
            _release_model(self.clip_model, self.device)
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None

    def evaluate(self, baseline_image, dynamic_image, prompt: str) -> dict:
        if self.lpips_model is None or self.clip_model is None:
            raise RuntimeError("QualityEvaluator must be used as a context manager")

        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        started = time.perf_counter()
        baseline = _as_rgb_array(baseline_image)
        dynamic = _as_rgb_array(dynamic_image)
        if baseline.shape != dynamic.shape:
            raise ValueError(
                "Quality comparison needs equal image sizes: "
                f"{baseline.shape} != {dynamic.shape}"
            )

        psnr = peak_signal_noise_ratio(baseline, dynamic, data_range=1.0)
        ssim = structural_similarity(
            baseline,
            dynamic,
            channel_axis=2,
            data_range=1.0,
        )

        with torch.inference_mode():
            lpips_distance = self.lpips_model(
                _lpips_tensor(baseline_image, self.device),
                _lpips_tensor(dynamic_image, self.device),
            ).item()

        inputs = self.clip_processor(
            text=[prompt],
            images=[baseline_image, dynamic_image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.clip_model(**inputs)
            clip_scores = (
                outputs.image_embeds @ outputs.text_embeds.transpose(0, 1)
            ).squeeze(1) * 100.0
        baseline_clip, dynamic_clip = clip_scores.float().cpu().tolist()

        return {
            "psnr_dynamic_vs_baseline_db": float(psnr),
            "ssim_dynamic_vs_baseline": float(ssim),
            "lpips_dynamic_vs_baseline": float(lpips_distance),
            "clip_score_baseline": float(baseline_clip),
            "clip_score_dynamic": float(dynamic_clip),
            "clip_score_delta_dynamic_minus_baseline": float(
                dynamic_clip - baseline_clip
            ),
            "evaluation_seconds": time.perf_counter() - started,
            "metric_direction": {
                "psnr": "higher is better",
                "ssim": "higher is better",
                "lpips": "lower is better",
                "clip_score": "higher is better",
            },
            "clip_model": CLIP_MODEL_ID,
        }


def evaluate_quality(baseline_image, dynamic_image, prompt: str, device: str) -> dict:
    """Calculate metrics for one pair while preserving the original API."""
    with QualityEvaluator(device) as evaluator:
        return evaluator.evaluate(baseline_image, dynamic_image, prompt)


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser(description="Evaluate two generated images")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dynamic", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    args = parser.parse_args()

    with Image.open(args.baseline) as baseline, Image.open(args.dynamic) as dynamic:
        metrics = evaluate_quality(
            baseline.convert("RGB"), dynamic.convert("RGB"), args.prompt, args.device
        )
    rendered = json.dumps(_json_safe(metrics), indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
