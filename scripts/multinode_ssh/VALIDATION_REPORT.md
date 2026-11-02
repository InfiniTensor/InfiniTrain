# NVIDIA Two-Node Validation Report

## Code and Environment

- Date: 2026-11-02
- Base: `origin/feat/ssh-multinode-launcher` at
  `924d21256f3baf66bccb7458aa77395f0fbf5841`
- Observed master: `5a7be78e9b9d42c2f26efe33984dd3167dc60dc9`
- Development branch: `codex/ssh-multinode-nvidia-validation`
- Nodes: two private training hosts (identifiers intentionally omitted)
- GPUs: 8 x NVIDIA A100-SXM4-80GB per node; driver 535.161.08
- SSH identity, host addresses, container name, worktree path, and interface:
  intentionally omitted; configure them as described in `README.md`

The branch remains based directly on the requested feature branch. Master had
broad unrelated changes, so it was not merged or rebased.

## Results

| Area | Result | Evidence |
|---|---|---|
| Parser, quoting, hostfile, invalid config | Passed | local launcher regression |
| Simulated success, node failure, timeout, SIGTERM cleanup | Passed | exits 0, 17, 124, 130; job PID checked |
| Live controller output | Passed | `--follow-output` observed start and final lines |
| TCP rendezvous PUT/GET | Passed | local protocol test against node-0 server |
| C++ build and rank tests | Passed | full build; rank CTest 2/2 |
| Two-node 1 process per node | Passed | real SSH smoke |
| Two-node 8 processes per node | Passed | 16 unique global ranks; local devices 0-7 |
| GPT2 TP4 x PP2 x DP2, 20 steps | Passed | final loss 4.468029 |
| Qwen3 TP4 x PP2 x DP2, 20 steps, BF16 | Passed | final loss 2.951596 |
| Llama3 TP4 x PP2 x DP2, 20 steps, BF16 | Passed | final loss 3.553573 |
| NCCL communication path | Passed for observed run | NCCL log reported `NET/IBext_v10` and GDRDMA |
| Real-node bad SSH / bad path | Partially verified | preflight/path failures verified; full matrix not retained |
| Real GPU rank failure / initialization timeout / Ctrl-C | Not verified | only simulated failure paths completed |
| Performance baseline | Not verified | no throughput target; correctness was prioritized |

## Numerical Comparison

Single-node references used TP4 x PP2 x DP1 with gradient accumulation adjusted
to preserve the global batch. Two-node runs used TP4 x PP2 x DP2. Initial
weights, data order, optimizer, global batch, and 20-step count were held fixed.

| Model | Single-node final | Two-node final | Maximum absolute loss difference |
|---|---:|---:|---:|
| GPT2 | 4.468028 | 4.468029 | 0.000001 |
| Qwen3 BF16 | 2.950893 | 2.951596 | 0.004184 |
| Llama3 BF16 | 3.553813 | 3.553573 | 0.002242 |

Maximum relative differences were 0.000022% (GPT2), 0.146558% (Qwen3), and
0.069305% (Llama3). Qwen3 and Llama3 use BF16 and different reduction order /
gradient accumulation between DP1 and DP2; these figures are recorded as
observations, not as a newly invented project tolerance.

## Logs and Known Limits

Control and rank log roots are operator-configured; their concrete validation
locations are intentionally omitted. The exact command is retained
in `controller-command.txt`, and runtime/version information is in metadata and
preflight logs.

- PP2 text generation with `--tokenizer_bin` currently aborts when generation
  is triggered because that path requires pipeline parallel size 1. Validated
  PP2 commands omit tokenizer-based generation.
- The rendezvous service binds the selected node-0 port on all interfaces and
  assumes a trusted training network; it is not an authenticated service.
- GPU preflight checks visibility/count, not exclusivity or free memory.
- More-node scaling still needs rendezvous load tests, larger failure matrices,
  and a cluster log-retention policy.
