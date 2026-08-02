# SD-v1.5 Speed-up with PYNQ-Z2

基于时间步特征余弦相似度的 Stable Diffusion 1.5 动态跳步加速项目。

Stable Diffusion 的 CLIP、UNet、Scheduler 和 VAE 在 PC GPU 上运行。PC 每个扩散时间步先去除重复 CFG batch，执行 `2x2` 平均池化，再将 latent 特征量化为 int8 并通过以太网发送给 PYNQ-Z2。FPGA 使用片上 BRAM 保存参考特征，并完成点积、范数、余弦阈值、warmup、连续跳步限制和参考更新。PYNQ ARM 只负责通信以及将 FPGA 统计量换算为具体余弦值。

```mermaid
flowchart LR
    A["PC GPU: SD 1.5"] --> B["2x downsample + int8 feature"]
    B -->|"TCP Ethernet"| C["PYNQ ARM"]
    C --> D["FPGA cosine IP"]
    D --> C
    C -->|"similarity + skip decision"| A
    A --> E["Scheduler + VAE image"]
```

## 稳定版本

原始 CSK2/int16 稳定版本永久保存在 Git 标签 [`v1.0-csk2`](https://github.com/vigourway-boop/SD-v1.5-speed-up/tree/v1.0-csk2)。CSK3/int8 的开发记录保存在 `feature/int8-skip-controller` 分支，旧版本不会被覆盖。

## v1.0-csk2 已验证结果

硬件平台：RTX 4060 Laptop GPU + PYNQ-Z2，SD 1.5，100 个基础扩散时间步。

| 指标 | 结果 |
| --- | ---: |
| Baseline 时间 | 10.88 s |
| PYNQ 动态时间 | 4.57 s |
| 加速比 | 2.38x |
| UNet 执行/跳过 | 38 / 62 |
| 平均 PYNQ 往返 | 2.174 ms |
| PSNR | 29.199 dB |
| SSIM | 0.951994 |
| LPIPS | 0.025454 |
| CLIP Score（baseline / dynamic） | 32.409 / 32.466 |

动态时间包含 PC 特征量化、以太网传输、PYNQ 服务、FPGA 判断和扩散过程。PSNR、SSIM、LPIPS 和 CLIP Score 在生成计时结束后计算，不计入加速比。

验证图片和完整 CSV/JSON 数据位于 [`examples/verified_run`](examples/verified_run)。

## CSK3/int8 已验证结果

硬件平台：RTX 4060 Laptop GPU + PYNQ-Z2，SD 1.5，100 个基础扩散时间步，随机种子 `3669225787`。

| 指标 | 结果 |
| --- | ---: |
| Baseline 时间 | 11.10 s |
| PYNQ 动态时间 | 4.70 s |
| 加速比 | 2.36x |
| UNet 执行/跳过 | 39 / 61 |
| 每步特征载荷 | 4096 B |
| 平均板端 FPGA 调用时间 | 0.573 ms |
| 平均网络往返 | 1.486 ms |
| PSNR | 30.246 dB |
| SSIM | 0.921491 |
| LPIPS | 0.048996 |
| CLIP Score（baseline / dynamic） | 30.502 / 30.382 |

CSK3 将每步传输量从 CSK2 的 32768 B 降至 4096 B，并把 warmup、余弦阈值、最大连续跳步、跳步计数和参考更新都放入 FPGA。动态时间包含特征下采样、int8 量化、传输、FPGA 判断和扩散过程。表中的板端 FPGA 调用时间还包含 ARM 侧 MMIO 寄存器读写；HLS 核心综合最大延迟为 82.29 us。

本次验证的图片和完整 100 步 CSV/JSON 数据位于 [`examples/csk3_int8_run`](examples/csk3_int8_run)。

## 目录

```text
combined_speed_test.py       一键 baseline + PYNQ 动态生成与评估
config.py                    模型、提示词、随机种子和跳步参数
pynq_cosine_client.py        PC 端 CSK3 二进制协议客户端
quality_metrics.py           PSNR、SSIM、LPIPS、CLIP Score
run_pynq_speedup.cmd         Windows 一键运行入口
deploy_pynq_server.cmd       部署 bit/hwh 和板端服务
test_pynq_protocol.py        本地协议测试
test_pynq_hardware.py        真实 PYNQ/FPGA 测试
pynq_cosine_overlay/
  hls/src/                   HLS C++ IP 源码和测试台
  vivado/                    Vivado Block Design 构建脚本
  artifacts/                 bit、hwh、IP 导出包、Notebook 和板端服务
```

## 环境要求

- Windows 10/11
- Python 3.10 和 Conda
- NVIDIA CUDA GPU，当前验证环境为 CUDA 11.8
- PYNQ-Z2，默认地址 `192.168.2.99`
- Vivado/Vitis HLS 2022.2，仅重新生成 FPGA 产物时需要

Stable Diffusion 模型权重不会存放在本仓库中，首次运行会从 Hugging Face 下载 `runwayml/stable-diffusion-v1-5`。

## PC 安装

```bat
conda create -n sd_accel python=3.10 -y
conda activate sd_accel
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

如果 Conda 或 Python 安装位置不同，请修改 `run_pynq_speedup.cmd` 中的 `SD_PYTHON`，或者先激活 `sd_accel` 环境。

## 部署 PYNQ 服务

确保 PC 可以连接 PYNQ 的 SSH 端口，然后在项目根目录运行：

```bat
deploy_pynq_server.cmd 192.168.2.99 xilinx
```

该命令会上传 `cosine_overlay.bit/.hwh` 和板端 Python 服务，安装并启动 `pynq-cosine.service`。SSH 和 sudo 可能要求输入 PYNQ 密码。不要将密码、Token 或 SSH 私钥提交到仓库。

## 一键生成与评估

```bat
run_pynq_speedup.cmd 192.168.2.99
```

每次运行都会使用一个新的随机种子；同一次运行中的 baseline 和 dynamic 使用相同种子，以保证比较公平。输出目录格式为：

```text
experiments/YYYYMMDD_HHMMSS_seed_<seed>/
  baseline.png
  pynq_dynamic.png
  step_metrics.csv
  summary.json
  quality_metrics.json
```

`step_metrics.csv` 每个扩散时间步一行，包含余弦相似度、跳步结果、量化时间、网络往返时间、FPGA kernel 时间、UNet 时间和 Scheduler 时间。第一步还没有参考向量，因此 similarity 为 `NaN`。

## 测试

本地协议测试：

```bat
python -m unittest -v test_pynq_protocol.py
```

真实 FPGA 测试：

```bat
python test_pynq_hardware.py --host 192.168.2.99
```

硬件测试会验证：首次无参考向量、相同向量余弦为 1、最大连续跳步限制、不同向量不通过阈值，以及 FPGA 结果与 PC int8 参考计算一致。

## 重新生成 FPGA 产物

```bat
D:\path\to\Vitis_HLS\2022.2\bin\vitis_hls.bat -f pynq_cosine_overlay\hls\run_hls.tcl
D:\path\to\Vivado\2022.2\bin\vivado.bat -mode batch -source pynq_cosine_overlay\vivado\create_overlay.tcl
```

已生成的 PYNQ-Z2 文件位于 [`pynq_cosine_overlay/artifacts`](pynq_cosine_overlay/artifacts)。构建状态和寄存器地址见 [`pynq_cosine_overlay/BUILD_STATUS.txt`](pynq_cosine_overlay/BUILD_STATUS.txt)。

## 限制

PYNQ-Z2 只负责压缩特征的统计和完整跳步控制。完整 CLIP、UNet 和 VAE 的参数量及带宽需求远超 Zynq-7020 的资源，因此仍由 PC GPU 执行。当前结果是一组验证实验，正式性能结论应继续使用多提示词、多随机种子统计。
