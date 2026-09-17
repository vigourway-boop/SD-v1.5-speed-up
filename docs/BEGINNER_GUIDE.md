# 从开机到生成图片

## 平常怎么用

**只想生成图片，先试普通30步：**

```powershell
.\run_standard_sd.cmd
```

这个入口完全不需要PYNQ，采用普通DPM-Solver 30步，不跳过UNet。结果在 `experiments/standard_日期时间_seed_种子/`，包含 `image.png` 和 `summary.json`。也可以使用 `--steps 20` 或 `--steps 50`。日常生成没有必要固定100步。

**想运行本项目的原始/动态对照实验：**

打开项目目录的终端，执行：

```powershell
.\run_project.cmd
```

程序先显示模式菜单：输入 `1` 使用默认加速，输入 `2` 使用快速策略，直接回车选择 `1`。随后检查Python依赖、CUDA、本地模型文件、参数文件和PYNQ服务，再调用原有 `combined_speed_test.py`。两种模式都生成一张原始图和一张动态图。省略种子时每次换一个种子，同一次的两张图使用相同种子。快速策略仅对本次运行生效，不覆盖默认配置。

也可以使用 `.\run_project.cmd --mode 1` 或 `.\run_project.cmd --mode 2` 直接选择模式、不显示菜单。`--check-only` 和明确指定 `--controller-profile` 的命令也不会显示菜单；`--mode` 与 `--controller-profile` 不能同时指定。

只检查环境、不生成图：

```powershell
.\run_project.cmd --check-only
```

检查通过不代表画质一定好；PING只能核对CSK4服务响应，不能证明板上bitstream与本地文件相同。真实运算检查使用 `python test_pynq_hardware.py`，应在没有生成任务时单独运行。

使用已完成速度测试的快速配置：

```powershell
.\run_project.cmd --controller-profile profiles/temporal_fast_100.json
```

这个配置在已有100步实验中约2.5倍，但个别图片偏差较大。不同步数、提示词和电脑不保证相同加速比。

使用更保守的配置：

```powershell
.\run_project.cmd --controller-profile profiles/temporal_conservative_100.json
```

保守配置在本轮12个独立样本中均比快速配置更接近100步原始图，但平均约1.95倍，未达到2倍，不保证其他场景也改善。两套参数均为显式选择，不覆盖旧默认配置。

严格使用PYNQ做实验：

```powershell
.\run_project.cmd --decision-backend pynq --controller-profile profiles/temporal_fast_100.json --generation-retries 0
```

没有板卡时：

```powershell
.\run_project.cmd --decision-backend pc
```

`pc`表示判断在电脑CPU、主要生成仍在GPU，不代表把整个模型放到CPU。`auto`先连接板卡，无法连接时允许电脑后备。正式FPGA实验用 `pynq`，避免把后备运行误算成板端结果。

修改本次提示词、步数和种子：

```powershell
.\run_project.cmd --decision-backend pc --steps 50 --seed 12345 --prompt "a photograph of a red sports car" --negative-prompt "blurry, low quality, distorted, watermark"
```

更换题材时同时修改负面提示词：默认负面词针对马，不一定适合其他对象。指定种子会使重复生成可复现；要每次不同就省略 `--seed`，并确保没有设置 `SD_SEED` 环境变量。

## 图片和时间在哪里

每次结果放在 `experiments/日期时间_seed_种子/`：

| 文件 | 内容 |
|---|---|
| `baseline.png` | 原始扩散图片 |
| `dynamic.png` | 动态跳步图片 |
| `report.html` | 双击查看的离线图片、时间和指标报告 |
| `summary.json` | 实际后端、参数、时间、显存及缓存等信息 |
| `step_metrics.csv` | 每一步的判断和耗时 |
| `timing_summary.csv` | 中英双语耗时明细 |
| `quality_metrics.json` | PSNR、SSIM、LPIPS、CLIP指标 |

加速比是原始生成秒数除以动态生成秒数。模型加载、预热、保存PNG和质量评价另行计时。动态时间包含特征压缩、网络往返、板端计算、扩散和VAE；若断线重跑，失败尝试和等待也计入恢复开销。初次连接耗时不包含在生成秒数内。

## PYNQ-Z2重新开机

1. 插入已部署PYNQ系统的SD卡，正确供电，用网线连接电脑。
2. 电脑以太网保持与板卡同一子网，例如电脑 `192.168.2.1`、板卡 `192.168.2.99`。不要改Wi-Fi连接来代替配置以太网。
3. 等待板卡系统启动，执行 `.\run_project.cmd --check-only --decision-backend pynq`。
4. 若服务不可达，先检查网线、网卡地址和板卡电源，再在终端执行 `ssh xilinx@192.168.2.99`。
5. SSH密码输入时屏幕不显示字符是正常行为。登录后检查：

```sh
systemctl status pynq-cosine.service --no-pager
journalctl -u pynq-cosine.service -n 40 --no-pager
```

若需要重新启动服务：

```sh
sudo systemctl restart pynq-cosine.service
```

服务原来安装在 `/home/xilinx/pynq_cosine`，负责加载 `cosine_overlay.bit/.hwh` 并监听9000端口。SD卡保存这些文件；断电后FPGA逻辑消失，下次服务启动会重新加载bitstream。

Jupyter是交互式笔记本工具，不是这条运行链路必需的编程语言或编译器。这里通过终端和TCP服务运行，无需打开浏览器里的Jupyter页面。

## 换一张SD卡或重新部署

在电脑项目目录执行现有部署命令：

```powershell
.\deploy_pynq_server.cmd 192.168.2.99 xilinx
```

命令上传bit/hwh、Python服务、阈值控制器、协议和启动脚本，并安装开机自启。需要板卡系统中已有PYNQ Python运行环境。不需要每次生成都重新部署；参数配置由电脑在每轮生成前发送。

同一时间只运行一个生成或硬件测试任务。不要在生成过程中重启服务、重新加载overlay，或运行另一项会重置控制器的测试。

## 换一台电脑

项目的CMD脚本优先使用本机的 `sd_accel` Python路径；其他电脑可先激活自己的环境，直接运行 `python launch_speedup.py`。也可设置 `SD_PYTHON` 为自己的Python完整路径。

PyTorch/CUDA安装方式应匹配显卡驱动，随后安装 `requirements.txt`；生成实验图表还需 `requirements_evaluation.txt`。模型权重未包含在项目交付包里，首次使用需下载或准备本地缓存。不要把模型目录、账号密码和GitHub令牌混入源码提交。
