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


def evaluate_quality(baseline_image, dynamic_image, prompt: str, device: str) -> dict:
    """Calculate PSNR, SSIM, LPIPS and CLIP Score after timed generation."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    started = time.perf_counter()
    baseline = _as_rgb_array(baseline_image)
    dynamic = _as_rgb_array(dynamic_image)
    if baseline.shape != dynamic.shape:
        raise ValueError(
            f"Quality comparison needs equal image sizes: {baseline.shape} != {dynamic.shape}"
        )

    psnr = peak_signal_noise_ratio(baseline, dynamic, data_range=1.0)
    ssim = structural_similarity(
        baseline,
        dynamic,
        channel_axis=2,
        data_range=1.0,
    )

    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    with torch.inference_mode():
        lpips_distance = lpips_model(
            _lpips_tensor(baseline_image, device),
            _lpips_tensor(dynamic_image, device),
        ).item()
    _release_model(lpips_model, device)
    del lpips_model

    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    inputs = clip_processor(
        text=[prompt],
        images=[baseline_image, dynamic_image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = clip_model(**inputs)
        clip_scores = (
            outputs.image_embeds @ outputs.text_embeds.transpose(0, 1)
        ).squeeze(1) * 100.0
    baseline_clip, dynamic_clip = clip_scores.float().cpu().tolist()
    _release_model(clip_model, device)
    del clip_model

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
