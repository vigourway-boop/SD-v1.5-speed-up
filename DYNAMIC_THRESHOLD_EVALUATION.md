# Dynamic threshold evaluation

## Calibration

Hardware: RTX 4060 Laptop GPU + PYNQ-Z2, SD 1.5, 100 diffusion steps. Eight CSK3 runs used the fixed `0.999` threshold and different random seeds. All runs used the same horse prompt so that seed variation, rather than prompt variation, was isolated.

The fixed runs produced 800 timestep records. Similarities for successful consecutive skips were:

| Consecutive skip | Samples | 5th percentile | Median | 95th percentile |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0.999750 | 0.999822 | 0.999873 |
| 2 | 196 | 0.999378 | 0.999499 | 0.999675 |
| 3 | 91 | 0.999012 | 0.999218 | 0.999414 |

The clear separation supports a threshold above most third-skip similarities while retaining first and selected second skips.

## Schedule

| Phase | Steps | Rule |
| --- | ---: | --- |
| Warmup | 1-15 | FPGA forces UNet; logged threshold 0.99995 |
| Middle | 16-70 | Fixed strict threshold 0.99960 |
| Late | 71-100 | `adjacent_similarity_ema - 0.00035` |

The late threshold is clamped to `0.99945-0.99965`, and the EMA coefficient is `0.20`. The EMA only learns from comparisons where the previous timestep executed UNet, so its reference is exactly one timestep old. A higher observed adjacent-step cosine raises the threshold, while a lower cosine relaxes it within the safe bounds.

## Paired validation

Each dynamic run reused a seed from a fixed-threshold run. Baseline images are therefore identical within each pair.

| Seed | Fixed PSNR | Dynamic PSNR | Fixed SSIM | Dynamic SSIM | Fixed LPIPS | Dynamic LPIPS | Dynamic UNet/skip | Dynamic speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1287417247 | 19.06 | 20.96 | 0.795 | 0.852 | 0.178 | 0.124 | 51 / 49 | 1.84x |
| 1590782835 | 34.48 | 40.48 | 0.980 | 0.992 | 0.012 | 0.005 | 52 / 48 | 1.80x |
| 2108257338 | 24.20 | 25.01 | 0.866 | 0.888 | 0.087 | 0.075 | 51 / 49 | 1.84x |
| 3669225787 | 30.25 | 34.07 | 0.921 | 0.963 | 0.049 | 0.022 | 50 / 50 | 1.88x |
| 4249141154 | 25.23 | 27.35 | 0.843 | 0.900 | 0.074 | 0.044 | 51 / 49 | 1.84x |

| Five-run median | Fixed threshold | Dynamic threshold |
| --- | ---: | ---: |
| PSNR | 25.23 dB | 27.35 dB |
| SSIM | 0.866 | 0.900 |
| LPIPS | 0.074 | 0.044 |
| Speedup | 2.35x | 1.84x |

All five paired runs improved PSNR, SSIM and LPIPS simultaneously. Their requested late thresholds ranged from `0.999455` to `0.999513`, proving that the adaptive phase changed with observed data. Runtime CSV files also record the Q1.15 encoded and effective FPGA thresholds. None reached a third consecutive skip.

These results validate the schedule for the current horse prompt and hardware configuration. Broader claims require additional prompts and scene types.
