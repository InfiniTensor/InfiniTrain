# Distributed gradient norm and clipping

The public entry point is:

    auto total_norm = optimizer->ClipGradNorm_(parameters, max_norm, norm_type, error_if_nonfinite, std::nullopt);

It returns a CPU FP32 scalar containing the pre-clipping norm and scales the selected gradients in place. Parameters without gradients are ignored and duplicate parameter pointers are counted once. Supported norm types are positive finite p and positive infinity. The coefficient is min(max_norm / (total_norm + 1e-6), 1); max_norm == 0 therefore zeros gradients rather than disabling clipping.

foreach=true is intentionally rejected because a fused multi-tensor kernel is not part of this implementation yet. The CPU and CUDA ScaleInplace kernels write through the existing gradient view, preserving flat-buffer and ZeRO-shard aliases. CUDA norm inspection synchronizes only to obtain scalar statistics; scaling remains on the CUDA stream.

DistributedOptimizer::ClipGradNorm_ first finishes pending gradient collectives, computes the norm over the shard parameters consumed by its base optimizer, reduces statistics over the DP process group, and applies the same coefficient to every local shard. When PP is enabled, statistics are also reduced over the PP process group. Pipeline scheduling calls ClipGradNormConfigured after all micro-batches complete.

The GPT-2 and LLaMA-3 examples expose:

- --clip_grad_norm=-1 (negative disables clipping)
- --grad_norm_type=2
- --clip_grad_error_if_nonfinite=true
- --clip_grad_foreach=auto|true|false

A total_grad_norm log line is emitted when clipping is enabled.

Current limitation: tensor-parallel replicated-parameter ownership metadata is not yet represented by Tensor/Optimizer, so TP/SP duplicate filtering needs follow-up integration before claiming full mixed-parallel equivalence.
