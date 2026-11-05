# Tiny model fixtures

`generate.py --output PATH` generates all four models and token streams offline,
using the Python standard library. `manifest.json` is the committed version and
SHA-256 contract. The runner verifies every generated file against it. Changing
fixture values requires an intentional manifest review; recording a new run does
not update this contract. The tiny files total less than 1.5 MB and are build artifacts, not source.

All models use their production Transformer implementation and LLMC loader:

| Model | Layers | Hidden / heads / KV heads | Vocabulary | FFN | Other |
|---|---:|---|---:|---:|---|
| GPT-2 | 4 | 32 / 4 / 4 | 128 | 128 | Learned positions, GELU, LayerNorm, shared initial embedding values |
| Llama3 | 4 | 32 / 4 / 4 | 128 | 64 | Interleaved RoPE, SwiGLU, RMSNorm |
| Qwen3 | 4 | 32 / 4 / 4 | 128 | 64 | Split-half RoPE, Q/K RMSNorm, SwiGLU |
| Mixtral | 4 | 32 / 4 / 4 | 128 | 64 per expert | Four experts, top-2 routing |

Context length is 64; scenarios normally use 16. Four layers support PP2×VPP2,
and dimensions support TP2. Mixtral's application currently supports single-rank
training only. These fixtures measure correctness; their throughput does not
represent production-model performance.

Each token file has 16,384 deterministic tokens. Successive batches differ, so
resume tests exercise the consumed-sample cursor. Weights use a specified 32-bit
integer recurrence and exactly representable binary float scales; generation does
not depend on NumPy, platform transcendental math or a Python PRNG version.

The 32-element bias/norm tensors deliberately exercise the distributed
optimizer's 64-element alignment padding. This exposed a real framework defect:
`ParamAndGradBuffer::BuildBuckets` allocated aligned parameter starts, while
`ParamAndGradBucket` rebuilt its range map by adding unpadded parameter lengths.
ZeRO optimizer views consequently updated padding or the wrong parameter region.
The fix carries the actual offsets into each bucket. The focused
`OptimizerBucketPaddingTest` validates both ZeRO stages' ranges and independent
expected SGD updates across two shard boundaries. Production tensors with sizes
already divisible by 64 follow the same layout as before.

## Independent numerical reference

`reference.py` is an optional **PyTorch + NumPy** oracle used by `tiny_oracle`
in `--alignment deep` mode. Default history mode runs the training cases without
the oracle or tensor snapshots.
It independently decodes LLMC tensors, verifies the framework's complete initial
parameter inventory and values, and implements dense attention, normalizations,
rotary positions, dense expert routing and autograd using PyTorch operations.
It uses the production entrypoints’ SGD for GPT-2 and Adam for the other models.
GPT-2 currently retains tied parameter storage on CPU, while `Module::To`
materializes separate CUDA parameters. The oracle preserves the explicit
`alias_of` snapshot metadata; this known backend difference is not a claim that
the two initialization paths have identical weight-tying behavior.
It compares loss, complete gradients and updated parameters through the normal
snapshot comparison path. The suite uses **operation composition**: independently
compute forward/backward from the preceding verified candidate parameters, check
the gradients, then independently replay SGD/Adam using the captured candidate
gradients as optimizer inputs. Optimizer moments persist in the reference. The
preceding independent update must pass before its candidate value can be used as
the next forward input. Expected final parameters are never copied from the
candidate. This avoids amplifying harmless FP32 gradient cancellation near
Adam's epsilon into misleading update failures, without relaxing gradient or
parameter tolerances. Omitting `--candidate-snapshots` runs a fully independent
trajectory instead; this can be numerically stricter around tiny Adam gradients. It deliberately supports this fixed FP32 architecture,
single rank, no LoRA, and a constant learning rate. Other configurations use
equivalence, scheduler, checkpoint and historical-log checks.

`--micro-batch-size` preserves gradient accumulation and the loader's complete
microbatch boundaries. In the 200-step stability case (microbatch two, GAS four,
sequence length 16), step 128 crosses the token fixture's epoch boundary. The
independent oracle handles that transition and drops the incomplete epoch tail;
`test_reference_batching.py` asserts concrete expected token and label sequences
at the boundary without depending on framework code or PyTorch.

```bash
python3 tests/regression/fixtures/reference.py \
  --initial-snapshot artifacts/candidate/snapshots/step_0/rank_0/manifest.json \
  --tokens .cache/regression/fixtures/gpt2_tokens.bin \
  --candidate-snapshots artifacts/candidate/snapshots \
  --output artifacts/reference --steps 4 --batch-size 2 --sequence-length 16 \
  --learning-rate 0.0001
```

The matching `<model>_weights.bin` must be beside the token file. Reference
metrics identify `oracle: pytorch_operations` or `pytorch_trajectory`; their time measures the CPU reference
computation and GPU memory is zero. They must never be used as an InfiniTrain
performance reference. The reference snapshots contain parameters and
gradients, not framework optimizer serialization; checkpoint checks cover that
separately.

## Runtime artifact contract

All four model executables accept `--metrics_file PREFIX` and
`--regression_snapshot_dir DIRECTORY`; empty defaults disable their work.
Metrics go to `PREFIX.rank_R.jsonl`, one stream per rank. Each optimizer step
records full-precision JSON numbers, data progress and topology; each successful
rank finishes with a `type: complete` record. Loss is DP-averaged and defined only
on the last pipeline stage; other pipeline stages emit `loss: null`. Step time
includes optimizer completion via device synchronization. Distributed timing is
the maximum rank duration, calculated by the checker. Memory uses the existing
allocator API with 1 MiB resolution, explicitly recorded as
`memory_resolution_bytes: 1048576`. A sub-MiB tiny footprint can report zero;
this cannot establish the absence of small leaks.

Snapshots live under `DIRECTORY/step_N/rank_R/manifest.json`; each rank writes
one `tensors.bin` bundle of NPY records. Manifest byte offsets and lengths avoid
thousands of separate filesystem opens on network storage. Readers retain
support for individual NPY files from earlier artifacts. Step zero contains
initial parameters; subsequent steps contain parameters, available gradients and
optimizer state. PP local layer names map to global layer names. TP metadata
describes local shapes, the split axis and packed Q/K/V or gate/up blocks. ZeRO-1/2
reduced gradient shards are gathered only when snapshots are requested, without modifying
training's gradient bindings. Optimizer names include rank because they describe
the rank-local serialized state. NPY widens BF16/FP16 tensors to FP32 but preserves integer counters and FP64
state exactly; training dtype remains in the manifest. Snapshot I/O is outside step timing, and performance cases disable
snapshots entirely.
