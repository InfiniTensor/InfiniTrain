# FlashAttention Integration Performance Report

## 1. Precision Alignment (AC1)
Comparing Training Loss for first 1 steps (SeqLen=1024).

| Step | Baseline Loss | Flash Loss | Diff |
|---|---|---|---|
| 1 | 11.007487 | 11.013704 | 6.217000e-03 |

**Max Difference**: 6.217000e-03
**Status**: WARNING (Difference > 1e-4)

## 2. Performance Comparison (AC2)
| Configuration | Seq Len | Batch Size | Avg Tokens/s | Peak Memory (MB) | Speedup |
|---|---|---|---|---|---|
| Baseline | 1024 | 2 | 12282 | 7530 | 1.0x |
| FlashAttn | 1024 | 2 | 7814 | 6346 | 0.64x |
| Baseline | 2048 | 1 | 9933 | 9261 | 1.0x |
| FlashAttn | 2048 | 1 | N/A (Crash) | N/A | N/A |

**Notes:**
- **Flash-1024**: Achieving 7814 TPS (0.64x of Baseline).
  - Improvement from Story 7 (~800 TPS) to Story 8 (~7800 TPS) due to Tiled Backward Kernel (removing atomicAdd bottleneck).
  - Still slower than Baseline, likely due to unoptimized Forward Kernel (WMMA pipeline stalls or bank conflicts) and lack of async memory copy.
- **Memory**: FlashAttention saves memory (6346 MB vs 7530 MB), confirming linear memory advantage.
- **LLaMA-3**: Baseline OOM on T4 (expected). FlashAttention crashes/fails (investigation needed).
