# PYNQ cosine runtime

The PYNQ-Z2 runs `pynq_cosine_server.py` as a persistent TCP service. The PC
downsamples each CFG-reduced latent to `1 x 4 x 32 x 32`, quantizes it to int8,
and sends a 4096-byte vector for every diffusion timestep. The FPGA keeps the
reference vector in BRAM and applies cosine threshold, warmup, consecutive-skip
limit and reference-update logic. The ARM returns the FPGA decision, skip streak
and measured cosine similarity. The `CSK3` magic deliberately rejects an old
client/server protocol combination. Deploy the bitstream, HWH and server from
the same artifact set.

## One-time board deployment

From the Windows `speedup` folder:

```bat
deploy_pynq_server.cmd 192.168.2.99
```

The default PYNQ username is `xilinx`. To use another username:

```bat
deploy_pynq_server.cmd 192.168.2.99 your_username
```

The command copies the bitstream, HWH and server, then installs and starts the
`pynq-cosine` systemd service. SSH and sudo may ask for the board password.

Check the service on PYNQ:

```sh
sudo systemctl status pynq-cosine
journalctl -u pynq-cosine -f
```

## Generate an image

Install the quality-metric packages once in the `sd_accel` environment if they
are not already present:

```bat
D:\lenovo\download\conda\envs\sd_accel\python.exe -m pip install -r requirements_quality.txt
```

After the one-time deployment, run this single command on Windows:

```bat
run_pynq_speedup.cmd 192.168.2.99
```

The command generates both the 100-step baseline and the PYNQ dynamic image.
It also creates a timestamped folder under `experiments` containing:

- `baseline.png` and `pynq_dynamic.png`
- `step_metrics.csv` with one row per diffusion timestep
- `summary.json` with timing and speedup
- `quality_metrics.json` with PSNR, SSIM, LPIPS and CLIP Score

Quality evaluation starts after generation timing, so it is not included in the
reported dynamic time or speedup. A connection failure stops the run; the PC
never silently replaces the FPGA decision with PyTorch.

In `step_metrics.csv`, `prepare_ms` is PC quantization/copy time,
`round_trip_ms` includes network transfer and the board response, and
`kernel_ms` is the board-side FPGA invocation time, including ARM MMIO setup,
polling and result reads. The HLS core latency is smaller. `pynq_total_ms` is
`prepare_ms + round_trip_ms`. The first similarity is `NaN` because no previous
feature reference exists yet.

## Dynamic threshold schedule

The PC sends a threshold with every CSK3 step request. By default it uses:

- steps 1-15: FPGA warmup forces UNet;
- steps 16-70: strict threshold 0.99960;
- steps 71-100: adjacent-step cosine EMA minus 0.00035, clamped to
  0.99945-0.99965.

The FPGA still performs the comparison and complete skip control. No bitstream
change is required. Set `SD_DYNAMIC_THRESHOLD=0` before launching to use the
legacy fixed threshold, or set `SD_SEED` to reproduce a particular run.
