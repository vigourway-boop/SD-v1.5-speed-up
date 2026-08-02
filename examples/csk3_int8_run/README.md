# CSK3 int8 verified run

该目录保存一次 CSK3 端到端真实硬件验证结果，随机种子为 `3669225787`。

- Baseline：11.10 s
- PYNQ dynamic：4.70 s
- 加速比：2.36x
- UNet 执行 39 步，跳过 61 步
- 每步特征载荷：4096 B int8
- 总特征流量：400 KiB
- 平均板端 FPGA 调用时间（含 ARM MMIO）：0.573 ms
- 平均网络往返时间：1.486 ms
- PSNR：30.246 dB
- SSIM：0.921491
- LPIPS：0.048996
- CLIP Score：30.502（baseline），30.382（dynamic）

`step_metrics.csv` 包含全部 100 个时间步的下采样/量化、通信、FPGA、UNet 和 Scheduler 数据。`summary.json` 和 `quality_metrics.json` 可供程序直接解析，生成时间包含特征准备、网络传输和 FPGA 判断，质量评估时间不计入加速比。
