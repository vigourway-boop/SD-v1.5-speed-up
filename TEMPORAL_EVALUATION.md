# 时间步相似性实验 / Temporal Similarity Evaluation

本轮以《基于时间步相似性的生成模型推理加速》的项目简介为依据：利用时间步相似性、量化和差异统计，复用已有计算，验证至少一种平台的两倍加速。评价同时报告画质、通信及存储开销。不会把GPU少执行UNet的全部收益归因于FPGA。

## 运行 / Run

使用本轮选定的参数生成一对原始/加速图片：

```powershell
.\run_evaluated_speedup.cmd
```

这条命令保留原有默认配置，通过 `profiles/temporal_fast_100.json` 显式加载新参数。没有板卡时沿用已有的电脑后备功能；如果要严格测PYNQ，执行 `.\run_evaluated_speedup.cmd --decision-backend pynq`。

也可以在激活环境后使用Python命令：

```powershell
python combined_speed_test.py --with-baseline --controller-profile profiles/temporal_fast_100.json
```

`--seed` 固定本次种子；`--prompt` 和 `--negative-prompt` 修改本次提示词；`--steps` 修改总步数。省略种子时每次生成使用新种子，同一次实验的两张图仍使用同一种子。加速比以本次实测为准，不承诺每个提示词都相同。

在项目目录执行：

```powershell
conda activate sd_accel
python temporal_evaluation.py --backend pynq
```

板卡必须开机，CSK4服务需监听 `192.168.2.99:9000`。通过 `--host`、`--port` 可以指定其他地址。实验采用严格板端模式，生成中途断线会报错，不会把电脑后备结果混入FPGA数据。

不使用板卡时：

```powershell
python temporal_evaluation.py --backend pc
```

电脑模式可以评价GPU动态跳步算法；不构成FPGA测试。每次运行都会创建独立的 `experiments/temporal_<日期时间>` 文件夹，不覆盖之前的数据。

生成图表和报告：

```powershell
python temporal_report.py experiments/<本次输出目录>
```

完成主批次后，可执行 `python temporal_verification.py experiments/<本次输出目录>`，核对关闭跳步后是否与原始pipeline完全一致，并补充交换原始/动态执行顺序的6组计时。随后重新生成报告即可包含这些结果。

如果一批实验意外中止，代码和参数没有改变时，可以使用 `--resume --output-root <原目录>` 跳过完整完成的分组，重新运行未完成的分组。程序校验源码哈希和实验划分；修改代码后应当使用新目录。失败记录保留在 `failure.json`，成功完成以 `completion.json` 为准。

## 实验划分 / Study Design

| 部分 / Stage | 提示词 / Prompts | 种子 / Seeds | 对比 / Comparison |
|---|---:|---:|---|
| 特征采集 / Feature traces | 3 | 1 | 原始100步的相邻latent、池化INT8差值及量化误差 |
| 参数筛选 / Calibration | 3 | 2 | 原始100步、现有参数、balanced、fast |
| 独立验证 / Held-out validation | 8 | 2 | 原始100步、现有动态、选定动态、原始50步 |
| 电脑与板卡 / PC versus PYNQ | 验证集前3种 | 2 | 同参数、同种子的电脑与真实PYNQ完整生成 |
| 不同步数 / Step sweep | 验证集前3种 | 1 | 20、30、50、75步分别比较原始与动态 |

候选参数和筛选规则先写入 `manifest.json`。距离门限来自调参集真实轨迹中候选跳步距离的90%分位数；这个规则是实验设计，不是已经证明最优的门限。

候选筛选条件为平均LPIPS不超过0.10、最差LPIPS不超过0.25、平均CLIP原始余弦下降不超过0.01；在符合条件的候选中选平均速度最快的。若全部不满足，选择平均LPIPS最低者继续验证，并明确记录失败。以上只是工程诊断条件，不是项目简介额外规定的验收标准。

## 测量边界 / Measurement Scope

- 比较使用同模型、分辨率、采样器、提示词、种子和精度。每组方案顺序由种子打乱，统一预热。CPU线程数设为4。
- 生成总时间包括CLIP文本编码、扩散循环、VAE解码和图像转换；动态方案额外包含控制器重置/配置、特征压缩、每步网络往返和板端计算。建立TCP连接在组前完成。
- 模型加载、预热、PNG保存、质量评价、单独的诊断轨迹采集不计入生成时间。质量模型在CPU上运行，不占用被测CUDA显存。
- PyTorch分配/保留显存峰值不等于整机显存。噪声预测缓存按张量字节计数；不会据此宣称模型权重减少。
- 特征网络载荷按4096字节/步统计，不包括TCP/IP头。对比单份FP16 latent为8倍压缩，对比重复CFG输入为16倍。
- 板端 `kernel_ms` 是ARM调用IP的时间，包含MMIO和轮询，不能替代纯FPGA周期数。
- CLIP旧指标为余弦乘100；本轮同时提供原始余弦变化，避免尺度混淆。

## 文件 / Files

- `manifest.json`：环境、划分、种子、候选、筛选规则及源码SHA256。
- `diagnostics/`：原始轨迹、相似性/差异/量化统计和诊断图片。
- `parity.csv`：同输入、同配置的电脑与FPGA逐步核对及耗时。
- `calibrated_candidates.json`、`selected_profile.json`：距离标定结果及最终选定参数。
- `results.csv`：中英双语逐组结果；`results.json`、`aggregate.json`：机器可读结果及聚合统计。
- 各分组目录：图像、`run.json`、逐步CSV、画质JSON和日志。
- `REPORT.md`、`speed_quality.png`、`feature_curves.png`、`comparison_gallery.jpg`：汇总报告与图表。

INT8特征逐步独立缩放，会丢失整体幅值信息。因此这里的INT8归一化距离不等于原始latent距离，量化差值为零也不代表UNet内部卷积可以直接跳过。
