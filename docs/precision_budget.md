# 数值误差预算表 (Precision Budget)

| Stage | Operation | Precision | Max Rel. Error Budget | RMS Error Budget | Note |
|---|---|---|---|---|---|
| Forward | QK^T | FP16/BF16 -> FP32 | 1e-5 | 1e-6 | Accumulate in FP32 |
| Forward | Softmax | FP32 | 1e-6 | 1e-7 | Online Softmax with scaling |
| Forward | PV | FP16/BF16 -> FP32 | 1e-5 | 1e-6 | Accumulate in FP32 |
| Backward | dO * V^T | FP16/BF16 -> FP32 | 2e-5 | 2e-6 | Accumulate in FP32 |
| Backward | dS * K | FP16/BF16 -> FP32 | 2e-5 | 2e-6 | Accumulate in FP32 |
| Backward | dS * Q | FP16/BF16 -> FP32 | 2e-5 | 2e-6 | Accumulate in FP32 |
| Softmax | Exp/Sum | FP32 | 1e-6 | 1e-7 | Use `__expf` carefully |

## Implementation Requirements
- **Accumulator**: Must be at least 48-bit (simulated via double float or Kahan summation if necessary, but standard FP32 is 23+8+1=32. Wait, FP32 has 24 bits significand. Double has 53. The requirement ">= 48 bit" implies using `double` or `float2` for accumulation).
- **Softmax Scale**: Must be kept in FP32.
- **NVCC Flags**: `-prec-div=true -prec-sqrt=true -fmad=false`.
