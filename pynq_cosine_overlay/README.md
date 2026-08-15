# PYNQ cosine-skip overlay

该目录包含用于动态 Stable Diffusion 跳步判断的 PYNQ-Z2 overlay。

FPGA IP 从 DDR 读取一个 4096 元素的 int8 当前向量，并在片上 BRAM 保存参考向量。硬件完成：

- 点积 `dot`
- 两个范数平方 `norm_x`、`norm_y`
- 固定点余弦阈值判断
- Q20 归一化特征距离阈值判断
- warmup 限制
- 最大连续跳步限制和计数
- 非跳步时更新参考向量

PYNQ ARM 使用 FPGA 返回的统计量计算具体余弦值：

```text
similarity = dot / sqrt(norm_x * norm_y)
distance = (norm_x + norm_y - 2 * dot) / (norm_x + norm_y)
```

跳步阈值比较在 FPGA 内避免了开方和除法：

```text
dot^2 > threshold^2 * norm_x * norm_y
distance_num * 2^20 <= distance_threshold_q20 * (norm_x + norm_y)
```

PC 会去除 classifier-free guidance 中重复的 batch，将 `4 * 64 * 64` 特征做 `2x2` 平均池化，并只发送 `4 * 32 * 32 = 4096` 个 int8 元素，即每步 4096 字节。相比 v1.0-csk2 的 32768 字节减少 8 倍。

CSK4 启动时由 PC 一次性发送控制参数。PYNQ ARM 根据时间步和真实相邻步余弦 EMA 计算三阶段动态阈值；FPGA 负责余弦、距离、warmup、连续跳步限制和参考更新，最终规则为 `cosine_passed && distance_passed`。

## 构建顺序

1. 使用 Vitis HLS 2022.2 运行 `hls/run_hls.tcl`。
2. 使用 Vivado 2022.2 运行 `vivado/create_overlay.tcl`。
3. 将 `artifacts` 中的 bit、hwh 和板端服务部署到 PYNQ-Z2。

已验证产物：

- `artifacts/cosine_overlay.bit`
- `artifacts/cosine_overlay.hwh`
- `artifacts/cosine_skip_hls_ip_export.zip`
- `artifacts/cosine_skip_test.ipynb`

详细构建与验证状态见 [`BUILD_STATUS.txt`](BUILD_STATUS.txt)，运行说明见 [`artifacts/PYNQ_RUNTIME_README.md`](artifacts/PYNQ_RUNTIME_README.md)。
