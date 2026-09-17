"""Generate one ordinary SD1.5 image, defaulting to 30 DPM-Solver steps."""

import argparse
from datetime import datetime
from pathlib import Path
import time

import combined_speed_test as sd
from project_health import check_project, print_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=sd.RUN_SEED)
    parser.add_argument("--prompt", default=sd.PROMPT)
    parser.add_argument("--negative-prompt", default=sd.NEGATIVE_PROMPT)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    if not 2 <= args.steps <= 1000 or not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("steps must be in [2,1000], seed in [0,4294967295]")
    health = check_project("pc")
    print_health(health)
    if health["errors"]:
        raise SystemExit(1)
    root = args.output_root / f"standard_{datetime.now():%Y%m%d_%H%M%S_%f}_seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=False)
    sd.PROMPT, sd.NEGATIVE_PROMPT, sd.RUN_SEED = args.prompt, args.negative_prompt, args.seed
    started = time.perf_counter()
    try:
        sd.pipe = sd.create_pipe()
        sd.pipe.set_progress_bar_config(disable=True)
        sd.sync()
        loading_seconds = time.perf_counter() - started
        warmup_ms = sd.warm_up_pipeline()
        seconds, _, timing = sd.run_original_steps(args.steps, root / "image.png")
        sd.write_json(root / "summary.json", {
            "mode": "ordinary_sd_no_skip", "steps": args.steps, "seed": args.seed,
            "prompt": args.prompt, "negative_prompt": args.negative_prompt,
            "model": sd.MODEL_ID, "device": sd.DEVICE,
            "scheduler": dict(sd.pipe.scheduler.config),
            "generation_seconds": seconds, "model_load_seconds": loading_seconds,
            "warmup_ms": warmup_ms, "timing": timing, "pynq_used": False,
            "speedup_claim": None,
        })
        print(f"普通SD / Standard SD: {args.steps} steps, {seconds:.3f} s")
        print(f"图片 / Image: {(root / 'image.png').resolve()}")
    finally:
        sd.pipe = None


if __name__ == "__main__":
    main()
