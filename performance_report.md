# FlashAttention Integration Performance Report

## 1. Precision Alignment (AC1)
Comparing Training Loss for first 2 steps (SeqLen=1024).

| Step | Baseline Loss | Flash Loss | Diff |
|---|---|---|---|
| 1 | 11.009614 | 10.983067 | 2.654700e-02 |
| 2 | 11.007620 | 10.993398 | 1.422200e-02 |

**Max Difference**: 2.654700e-02
**Status**: WARNING (Difference > 1e-4)

## 2. Performance Comparison (AC2)
| Configuration | Seq Len | Batch Size | Avg Tokens/s | Peak Memory (MB) | Speedup |
|---|---|---|---|---|---|
| Baseline | 1024 | 2 | 12320 | 7530 | 1.0x |
| FlashAttn | 1024 | 2 | 0 | 6346 | 0.00x |
| Baseline | 2048 | 1 | 9968 | 9261 | 1.0x |
| FlashAttn | 2048 | 1 | 0 | 6499 | 0.00x |
| Baseline | 1024 | 1 | 1666 | 31563 | 1.0x |
| FlashAttn | 1024 | 1 | 0 | 29767 | 0.00x |
