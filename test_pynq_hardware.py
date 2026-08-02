import argparse
import math

import numpy as np
import torch

from pynq_cosine_client import PynqCosineClient, quantize_feature


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
        establish = client.decide(first, 0, 0.999, 0, 2)
        identical = client.decide(first, 1, 0.999, 0, 2)
        second_skip = client.decide(first, 2, 0.999, 0, 2)
        forced_unet = client.decide(first, 3, 0.999, 0, 2)
        changed = client.decide(different, 4, 0.999, 0, 2)

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
    print("Real PYNQ hardware protocol test passed")


if __name__ == "__main__":
    main()
