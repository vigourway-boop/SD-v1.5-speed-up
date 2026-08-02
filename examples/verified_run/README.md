# Verified run

该目录保存一次端到端验证结果，随机种子为 `2366136021`。

- Baseline：10.88 s
- PYNQ dynamic：4.57 s
- 加速比：2.38x
- UNet 执行 38 步，跳过 62 步
- PSNR：29.199 dB
- SSIM：0.951994
- LPIPS：0.025454
- CLIP Score：32.409（baseline），32.466（dynamic）

`step_metrics.csv` 包含全部 100 个时间步的通信、FPGA、UNet 和 Scheduler 数据。
