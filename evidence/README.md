# Experiment Evidence

This directory contains a curated, reviewable copy of the experiment evidence
used to evaluate the SD 1.5 temporal-similarity acceleration project. It was
added retrospectively after the final report was prepared. The original,
larger `experiments/` directory remains excluded from Git because it contains
duplicated runs, intermediate diagnostics, and large latent traces.

## Final validation

`final_validation_20260914_v2/` contains the final 100-step validation:

- 8 prompts x 2 seeds (16 paired baseline/accelerated cases)
- mean measured speedup: 2.506074x
- measured speedup range: 2.408832x to 2.812352x
- runs above 2x: 16/16
- mean real UNet executions: 36.625 per 100-step run
- mean LPIPS against the paired baseline: 0.069418
- maximum LPIPS: 0.327170 (1/16 exceeds the 0.25 diagnostic threshold)

Each case under `validation/` includes the baseline and selected accelerated
image, run metadata, quality measurements, and per-step controller records.
The top-level CSV, JSON, and figures contain aggregate results and parity
checks. These speedups are relative to the original 100-step SD 1.5 baseline;
they must not be interpreted as universal results for shorter schedules,
different prompts, resolutions, schedulers, models, or hardware.

## Predictor experiment

`predictor_30run_20260815_2005/` contains the aggregate outputs from 30 dynamic
runs (15 fixed seeds, each tested with `linear` and `reuse` prediction). The
median measured speedups were 1.841x and 1.836x respectively. Detailed results
are available in `results.csv`, `batch_summary.json`, and `batch_report.html`.

## FPGA scope

The PYNQ-Z2 implements feature reduction/quantization, similarity and distance
evaluation, and dynamic skip-control decisions. In the recorded parity tests,
908 replayed steps produced zero software/FPGA decision mismatches, and 6/6
paired PC/PYNQ runs produced identical decision sequences and image hashes.
The Stable Diffusion UNet still runs on the GPU, so the end-to-end speedup comes
primarily from avoiding UNet executions rather than from faster vector math on
the FPGA.

## Limitations

- The final validation covers 16 cases and is not a universal quality claim.
- One final-validation sample exceeded the LPIPS 0.25 diagnostic threshold.
- The reported result uses a 100-step baseline; shorter common schedules show
  less opportunity for temporal skipping.
- Whole-system energy consumption was not measured.
- The full SD 1.5 model is not deployed on the PYNQ-Z2.
