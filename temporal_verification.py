"""Check no-skip equivalence and counterbalanced timing after a completed study."""

import argparse
import contextlib
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

import combined_speed_test as sd
from controller_profile import load_controller_profile
from pc_skip_controller import PcCosineClient
from pynq_cosine_client import PynqCosineClient
from temporal_evaluation import BASE_PROFILE, VALIDATION, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not (args.root / "completion.json").exists():
        raise ValueError("The main study must complete first")
    root = args.root / "verification"
    root.mkdir(exist_ok=False)
    profile = load_controller_profile(args.root / "selected_profile.json")
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    board = PynqCosineClient(sd.PYNQ_HOST, sd.PYNQ_PORT, timeout=5) if manifest["backend"] == "pynq" else None
    try:
        sd.pipe = sd.create_pipe()
        sd.pipe.set_progress_bar_config(disable=True)
        if sd.warm_up_pipeline() <= 0:
            raise RuntimeError("Warmup failed")
        with settings("a ceramic teapot on a wooden table", 99991, 20, {**BASE_PROFILE, "max_consecutive_skips": 0}):
            with (root / "no_skip.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                _, baseline, _ = sd.run_original_steps(20, root / "no_skip_baseline.png")
                _, executed, skipped, dynamic, _, _ = sd.run_with_dynamic_steps(
                    root / "no_skip_dynamic.png", PcCosineClient(), 2.0, "linear", .5, 1.5)
            delta = np.abs(np.asarray(baseline, dtype=np.int16) - np.asarray(dynamic, dtype=np.int16))
            no_skip = {"executed": executed, "skipped": skipped, "max_pixel_delta": int(delta.max()),
                       "mean_pixel_delta": float(delta.mean()), "identical_pixels": bool(np.all(delta == 0))}
            sd.write_json(root / "no_skip.json", no_skip)
            if executed != 20 or skipped != 0 or not no_skip["identical_pixels"]:
                raise RuntimeError(f"No-skip equivalence failed: {no_skip}")
        results = []
        for prompt_id, prompt in VALIDATION[:3]:
            for repeat in range(2):
                order = ["baseline", "dynamic"] if repeat == 0 else ["dynamic", "baseline"]
                group = root / f"{prompt_id}_{repeat}"
                group.mkdir()
                times, hashes = {}, {}
                with settings(prompt, 202601, 100, profile):
                    for name in order:
                        if name == "dynamic" and board:
                            board.close()
                            board.connect()
                        if sd.DEVICE == "cuda":
                            torch.cuda.empty_cache()
                        with (group / f"{name}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                            if name == "baseline":
                                seconds, image, _ = sd.run_original_steps(100, group / "baseline.png")
                            else:
                                seconds, _, _, image, _, _ = sd.run_with_dynamic_steps(
                                    group / "dynamic.png", board or PcCosineClient(),
                                    profile["distance_threshold"], "linear", profile["predictor_damping"], 1.5)
                        times[name] = seconds
                        hashes[name] = hashlib.sha256(image.tobytes()).hexdigest()
                result = {"prompt_id": prompt_id, "repeat": repeat, "order": order,
                          "times": times, "hashes": hashes,
                          "speedup": times["baseline"] / times["dynamic"]}
                results.append(result)
                sd.write_json(root / "timing_pairs.json", results)
                print(f"{prompt_id} order={order}: {result['speedup']:.3f}x", flush=True)
        summary = {"no_skip": no_skip, "timing_pairs": len(results),
                   "mean_speedup": statistics.mean(r["speedup"] for r in results),
                   "min_speedup": min(r["speedup"] for r in results),
                   "max_speedup": max(r["speedup"] for r in results),
                   "baseline_first_mean": statistics.mean(r["speedup"] for r in results if r["repeat"] == 0),
                   "dynamic_first_mean": statistics.mean(r["speedup"] for r in results if r["repeat"] == 1),
                   "identical_repeat_images": all(results[i]["hashes"] == results[i+1]["hashes"] for i in (0, 2, 4)),
                   "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}}
        sd.write_json(root / "summary.json", summary)
        print(summary, flush=True)
    finally:
        if board:
            board.close()
        sd.pipe = None


if __name__ == "__main__":
    main()
