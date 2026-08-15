# Distance threshold evaluation

The CSK4 overlay was deployed to PYNQ-Z2 and tested with the horse prompt,
100 diffusion steps, int8 features and the ARM-side dynamic cosine schedule.

## Reproducibility correction

The first collection batch exposed a command-line seed bug. The script updated
the `RUN_SEED` name imported into `combined_speed_test.py`, while
`config.make_generator()` still read the original random value in the `config`
module. Baseline and dynamic generation inside one process still shared noise,
but separate processes using the same `--seed` did not reproduce the same
baseline. Those early runs remain useful as general quality samples, but are
not valid same-seed threshold comparisons.

`make_generator(seed)` now receives the command-line seed explicitly. The fix
has unit coverage, and all threshold runs for each corrected seed produced one
identical baseline SHA-256 hash.

## Corrected calibration matrix

Three fixed seeds were tested at each candidate distance threshold. Every row
compares dynamic output with the 100-step baseline generated from the same
noise in the same run.

| Seed | Threshold | UNet / skipped | Speedup | PSNR (dB) | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1777169545 | 0.00035 | 54 / 46 | 1.738x | 39.932 | 0.9913 | 0.0062 |
| 1777169545 | 0.00040 | 52 / 48 | 1.799x | 39.216 | 0.9906 | 0.0068 |
| 1777169545 | 0.00045 | 50 / 50 | 1.866x | 36.443 | 0.9796 | 0.0145 |
| 2499984135 | 0.00035 | 53 / 47 | 1.774x | 23.747 | 0.8581 | 0.0591 |
| 2499984135 | 0.00040 | 50 / 50 | 1.872x | 23.935 | 0.8550 | 0.0612 |
| 2499984135 | 0.00045 | 49 / 51 | 2.036x | 23.535 | 0.8499 | 0.0642 |
| 2710708788 | 0.00035 | 56 / 44 | 1.677x | 35.897 | 0.9869 | 0.0054 |
| 2710708788 | 0.00040 | 53 / 47 | 1.779x | 34.747 | 0.9811 | 0.0081 |
| 2710708788 | 0.00045 | 52 / 48 | 1.805x | 34.335 | 0.9817 | 0.0081 |

| Threshold | Average skipped | Average speedup | Average PSNR | Minimum PSNR | Average SSIM | Average LPIPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00035 | 45.67 | 1.729x | 33.192 dB | 23.747 dB | 0.9454 | 0.0235 |
| 0.00040 | 48.33 | 1.816x | 32.633 dB | 23.935 dB | 0.9422 | 0.0253 |
| 0.00045 | 49.67 | 1.902x | 31.438 dB | 23.535 dB | 0.9371 | 0.0290 |

`0.00035` is the best of these three speed/quality tradeoffs, but its worst
case is not acceptable as a quality-safe default.

## Strict-threshold diagnosis

The worst seed was also tested with stricter distance gates:

| Threshold | UNet / skipped | Speedup | PSNR (dB) | SSIM | LPIPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00015 | 79 / 21 | 1.22x | 24.455 | 0.8845 | 0.0462 |
| 0.00025 | 57 / 43 | 1.67x | 23.662 | 0.8640 | 0.0560 |
| 0.00030 | 57 / 43 | 1.67x | 23.662 | 0.8640 | 0.0560 |

Even 21 skipped steps did not restore high full-reference quality. The failure
is therefore not solved by fine-tuning one global distance threshold. Reusing
the previous UNet output can change the diffusion trajectory irreversibly for
some seeds.

## Decision

`config.py` keeps `DISTANCE_THRESHOLD=2.0` as a permissive collection and
legacy-compatibility default. It is not a claim that the current skip policy is
quality-safe. No calibrated candidate is promoted to the default.

Future quality work should change the skip policy itself, for example by using
phase-specific skip budgets, forbidding skips at sensitive timesteps, or
predicting the missing noise output instead of copying the previous output.
Threshold-only scans should not resume until that controller change is made.
