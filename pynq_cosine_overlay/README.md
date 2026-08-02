# PYNQ cosine-skip overlay

该目录包含用于动态 Stable Diffusion 跳步判断的 PYNQ-Z2 overlay。

FPGA IP 从 DDR 读取两个 int16 向量，计算：

- 点积 `dot`
- 两个范数平方 `norm_x`、`norm_y`
- 固定点余弦阈值判断

PYNQ ARM 使用 FPGA 返回的统计量计算具体余弦值：

```text
similarity = dot / sqrt(norm_x * norm_y)
```

跳步阈值比较在 FPGA 内避免了开方和除法：

```text
dot^2 > threshold^2 * norm_x * norm_y
```

PC 会去除 classifier-free guidance 中重复的 batch，只发送 `1 * 4 * 64 * 64 = 16384` 个 int16 元素，即每步 32768 字节。

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
