# SSH Multinode Launcher

This directory contains:

- `launch_multinode_ssh.sh`: starts InfiniTrain on all nodes from one control node.
- `test_launch_multinode_ssh.sh`: tests launcher argument handling, rank assignment,
  shell quoting, Docker command generation, and node startup delay with a fake
  `infini_run`. It does not connect to a real cluster or use GPUs.

## Prerequisites

- The repository and training assets are available at the same paths in every
  training container.
- The control node can use non-interactive SSH authentication for remote hosts.
- The SSH account can run `docker exec` on the remote host when
  `--remote-docker-container` is used.
- `build/infini_run` and the selected training executable have already been built.
- `NCCL_SOCKET_IFNAME` is set to the physical interface that routes to the other
  training nodes. Do not allow NCCL to select Docker `veth` interfaces.

Check the interface with:

```bash
ip route get 192.168.162.63
```

## Two-node DP example

The following example starts one GPU process on each node and runs one GPT-2
training iteration. Update the variables for the target environment.

```bash
CONTROL_HOST=192.168.162.62
REMOTE_HOST=192.168.162.63
REMOTE_CONTAINER=infinitrain
WORKDIR="$HOME/code/InfiniTrain"
SSH_KEY="$HOME/.ssh/id_ed25519"
NCCL_IFNAME=ens22f0

scripts/multinode_ssh/launch_multinode_ssh.sh \
  --hosts "$CONTROL_HOST,$REMOTE_HOST" \
  --local-host "$CONTROL_HOST" \
  --remote-docker-container "$REMOTE_CONTAINER" \
  --ssh-option "IdentityFile=$SSH_KEY" \
  --ssh-option IdentitiesOnly=yes \
  --workdir "$WORKDIR" \
  --nproc-per-node 1 \
  --master-addr "$CONTROL_HOST" \
  --node-start-delay 5 \
  --env "NCCL_SOCKET_IFNAME=$NCCL_IFNAME" \
  --env NCCL_IB_DISABLE=1 \
  -- ./build/gpt2 \
    --device cuda \
    --tensor_parallel 1 \
    --pipeline_parallel 1 \
    --batch_size 1 \
    --sequence_length 64 \
    --num_iteration 1 \
    --total_batch_size 128 \
    --input_bin /data1/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin \
    --tokenizer_bin /data1/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_tokenizer.bin \
    --llmc_filepath /data1/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_124M.bin
```

For pure data parallelism:

```text
DP size = number of nodes x processes per node
TP size = 1
PP size = 1
```

`total_batch_size` must be divisible by:

```text
batch_size x sequence_length x DP size
```

Increase `--nproc-per-node` to use more GPUs on each node. For two nodes with
four processes per node and the same batch and sequence lengths, use
`--total_batch_size 512` for one microbatch per optimizer step.

The launcher generates a unique rendezvous ID when `--rdzv-id` is omitted.
Per-node logs are written to `logs/<rdzv-id>/` unless `--log-dir` is supplied.

## Inspect commands without launching

```bash
scripts/multinode_ssh/launch_multinode_ssh.sh \
  --hosts 192.168.162.62,192.168.162.63 \
  --local-host 192.168.162.62 \
  --workdir "$HOME/code/InfiniTrain" \
  --dry-run \
  -- ./build/gpt2 --device cuda
```

## Launcher regression test

```bash
scripts/multinode_ssh/test_launch_multinode_ssh.sh
```

This is a local simulated test. A passing result does not verify SSH, Docker,
NCCL, GPUs, model files, or training on the real cluster.
