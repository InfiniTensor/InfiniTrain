# FlashAttention Integration Performance Report

## 1. Precision Alignment (AC1)
Comparing Training Loss for first 20 steps (SeqLen=1024).

| Step | Baseline Loss | Flash Loss | Diff |
|---|---|---|---|
| 1 | 11.062558 | 10.977810 | 8.474800e-02 |
| 2 | 11.046689 | 10.997502 | 4.918700e-02 |
| 3 | 11.033809 | 10.995351 | 3.845800e-02 |
| 4 | 11.038082 | 10.986852 | 5.123000e-02 |
| 5 | 11.051686 | 10.995112 | 5.657400e-02 |
| 6 | 11.035001 | 10.979043 | 5.595800e-02 |
| 7 | 11.048413 | 10.980491 | 6.792200e-02 |
| 8 | 11.050271 | 10.980883 | 6.938800e-02 |
| 9 | 11.031726 | 11.003664 | 2.806200e-02 |
| 10 | 11.026350 | 10.976301 | 5.004900e-02 |
| 11 | 11.030182 | 10.992303 | 3.787900e-02 |

**Max Difference**: 8.474800e-02
**Status**: WARNING (Difference > 1e-4)

## 2. Performance Comparison (AC2)
| Configuration | Seq Len | Batch Size | Avg Tokens/s | Peak Memory (MB) | Speedup |
|---|---|---|---|---|---|
| Baseline | 1024 | 2 | 5381 | 7530 | 1.0x |
| FlashAttn | 1024 | 2 | 285 | 6346 | 0.05x |
| Baseline | 2048 | 1 | 4354 | 9261 | 1.0x |
| FlashAttn | 2048 | 1 | 128 | 6499 | 0.03x |
| Baseline | 1024 | 1 | 735 | 31563 | 1.0x |
| FlashAttn | 1024 | 1 | 79 | 29767 | 0.11x |
