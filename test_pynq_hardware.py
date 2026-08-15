import argparse
import math

import numpy as np
import torch

from pynq_cosine_client import PynqCosineClient, quantize_feature


def configure(client, distance_threshold):
    client.configure(
        enabled=True,
        total_steps=100,
        warmup_steps=0,
        max_consecutive_skips=2,
        fixed_threshold=0.999,
        warmup_threshold=0.99995,
        middle_threshold=0.999,
        late_start_ratio=0.7,
        late_margin=0.00035,
        late_min=0.99945,
        late_max=0.99965,
        ema_alpha=0.2,
        distance_threshold=distance_threshold,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(1234)
    first = torch.randn((2, 4, 64, 64), generator=generator)
    first[1] = first[0]
    different = torch.randn((2, 4, 64, 64), generator=generator)
    different[1] = different[0]

    with PynqCosineClient(args.host, args.port) as client:
        client.reset()
        configure(client, 2.0)
        establish = client.decide(first, 0)
        identical = client.decide(first, 1)
        second_skip = client.decide(first, 2)
        forced_unet = client.decide(first, 3)
        changed = client.decide(different, 4)

        near = first.clone()
        near[:, :, :8, :8] += 0.1
        client.reset()
        configure(client, 0.0)
        client.decide(first, 0)
        distance_rejected = client.decide(near, 1)

    print("establish reference:", establish)
    print("identical feature:", identical)
    print("second allowed skip:", second_skip)
    print("forced UNet after max skips:", forced_unet)
    print("different feature:", changed)
    if establish.should_skip:
        raise SystemExit("First feature must establish a reference")
    if not math.isnan(establish.similarity):
        raise SystemExit("First feature has no reference, so similarity must be NaN")
    if not identical.should_skip:
        raise SystemExit("Identical feature should be skipped")
    if not identical.cosine_passed or not identical.distance_passed:
        raise SystemExit("Identical feature did not pass both FPGA tests")
    if abs(identical.similarity - 1.0) > 1e-9:
        raise SystemExit(f"Identical feature similarity is {identical.similarity}, not 1")
    if identical.skip_streak != 1 or not second_skip.should_skip:
        raise SystemExit("FPGA consecutive skip counter did not reach two")
    if second_skip.skip_streak != 2:
        raise SystemExit("FPGA second skip streak is incorrect")
    if forced_unet.should_skip or not forced_unet.threshold_passed:
        raise SystemExit("FPGA did not enforce max_consecutive_skips")
    if forced_unet.skip_streak != 0:
        raise SystemExit("FPGA skip streak did not reset after forced UNet")
    if changed.should_skip:
        raise SystemExit("Different feature must execute UNet")
    if changed.similarity >= 0.999:
        raise SystemExit("Different feature unexpectedly passed the cosine threshold")
    first_q = quantize_feature(first[0]).astype(np.int64)
    different_q = quantize_feature(different[0]).astype(np.int64)
    dot = int(np.dot(first_q, different_q))
    norm_first = int(np.dot(first_q, first_q))
    norm_different = int(np.dot(different_q, different_q))
    expected_similarity = dot / math.sqrt(norm_first * norm_different)
    if abs(changed.similarity - expected_similarity) > 1e-12:
        raise SystemExit(
            "PYNQ similarity differs from the PC compressed int8 reference: "
            f"{changed.similarity} != {expected_similarity}"
        )
    if not distance_rejected.cosine_passed:
        raise SystemExit(
            "Near feature did not pass cosine; hardware distance test is inconclusive"
        )
    if distance_rejected.distance_passed or distance_rejected.should_skip:
        raise SystemExit("FPGA distance gate did not reject the near feature")
    if distance_rejected.normalized_distance <= 0.0:
        raise SystemExit("PYNQ did not report a positive normalized distance")
    print("Real PYNQ hardware protocol test passed")


if __name__ == "__main__":
    main()
