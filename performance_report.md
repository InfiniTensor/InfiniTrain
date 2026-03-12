# FlashAttention Integration Performance Report

## 1. Precision Alignment (AC1)
Comparing Training Loss for first 20 steps (SeqLen=1024).

| Step | Baseline Loss | Flash Loss | Diff |
|---|---|---|---|
| 1 | 10.955786 | 10.964919 | 9.133000e-03 |
| 2 | 10.968926 | 10.965187 | 3.739000e-03 |
| 3 | 10.980753 | 10.975492 | 5.261000e-03 |
| 4 | 10.960719 | 10.969276 | 8.557000e-03 |
| 5 | 10.972068 | 10.983315 | 1.124700e-02 |
| 6 | 10.975980 | 10.973791 | 2.189000e-03 |
| 7 | 10.964057 | 10.988812 | 2.475500e-02 |
| 8 | 10.982581 | 10.978993 | 3.588000e-03 |
| 9 | 10.948436 | 10.950513 | 2.077000e-03 |
| 10 | 10.964073 | 10.948275 | 1.579800e-02 |
| 11 | 10.967847 | 10.960063 | 7.784000e-03 |

**Max Difference**: 3.199000e-02
**Status**: WARNING (Difference > 1e-4)

## 2. Performance Comparison (AC2)
| Configuration | Seq Len | Batch Size | Avg Tokens/s | Peak Memory (MB) | Speedup |
|---|---|---|---|---|---|
| Baseline | 1024 | 2 | 5392 | 7530 | 1.0x |
| FlashAttn | 1024 | 2 | 327 | 6346 | 0.06x |
| Baseline | 2048 | 1 | 4347 | 9261 | 1.0x |
| FlashAttn | 2048 | 1 | 164 | 6499 | 0.04x |
| Baseline | 1024 | 1 | 739 | 31563 | 1.0x |
| FlashAttn | 1024 | 1 | 84 | 29767 | 0.11x |
