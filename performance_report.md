# FlashAttention Integration Performance Report

## 1. Precision Alignment (AC1)
Comparing Training Loss for first 1 steps (SeqLen=1024).

| Step | Baseline Loss | Flash Loss | Diff |
|---|---|---|---|
| 1 | 10.988086 | 10.957374 | 3.071200e-02 |

**Max Difference**: 3.071200e-02
**Status**: WARNING (Difference > 1e-4)

## 2. Performance Comparison (AC2)
| Configuration | Seq Len | Batch Size | Avg Tokens/s | Peak Memory (MB) | Speedup |
|---|---|---|---|---|---|
| Baseline | 1024 | 2 | 12271 | 7530 | 1.0x |
| FlashAttn | 1024 | 2 | 0 | 6346 | 0.00x |
| Baseline | 2048 | 1 | 9932 | 9261 | 1.0x |
| FlashAttn | 2048 | 1 | 0 | 6499 | 0.00x |
| Baseline | 1024 | 1 | 1670 | 31563 | 1.0x |
| FlashAttn | 1024 | 1 | 0 | 0 | 0.00x |
