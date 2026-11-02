# NVIDIA SSH Multinode Launcher

`launch_multinode_ssh.sh` starts one `infini_run` per node from a control host.
A two-node, eight-GPU-per-node job creates 16 rank processes, records separate
rank logs, and returns nonzero if either node fails.

## Preparation

- Use the same repository commit and matching binaries on every node.
- Ensure non-interactive SSH and known-host verification work before launch.
- The remote account must be able to run the build, and optionally `docker
  exec`, without relying on an interactive shell profile.
- Select an unused TCP port reachable from every node. Node rank 0 hosts the
  rendezvous service; multi-node CCL unique-ID exchange does not require NFS.
- Set `NCCL_SOCKET_IFNAME` from the actual route. Confirm RDMA from NCCL logs;
  successful communication alone is not proof that RDMA was used.

A hostfile is plain text in node-rank order; blank lines and `#` comments are
ignored:

```text
# node rank order
TRAIN_NODE_0
TRAIN_NODE_1
```

For a short list, `--hosts TRAIN_NODE_0,TRAIN_NODE_1` avoids maintaining a
separate file.

## Two Nodes, Eight GPUs Each

Run only on the control node. The second node needs no manual command. Set the
following environment-specific values first. `CONTROL_HOST` must be reachable by
every training node, and `NCCL_IFNAME` must be the interface selected by the route
to the peer.

```bash
CONTROL_HOST=TRAIN_NODE_0
REMOTE_HOST=TRAIN_NODE_1
SSH_USER=TRAIN_USER
SSH_KEY=/path/to/private_key
REMOTE_CONTAINER=TRAINING_CONTAINER
WORKDIR=/path/to/InfiniTrain
NCCL_IFNAME=TRAINING_INTERFACE
LOG_ROOT=/path/to/controller/logs
REMOTE_LOG_ROOT=/path/to/node/logs


scripts/multinode_ssh/launch_multinode_ssh.sh \
  --hosts "$CONTROL_HOST,$REMOTE_HOST" \
  --local-host "$CONTROL_HOST" \
  --ssh-user "$SSH_USER" \
  --ssh-option "IdentityFile=$SSH_KEY" \
  --remote-docker-container "$REMOTE_CONTAINER" \
  --workdir "$WORKDIR" \
  --infini-run ./build/infini_run \
  --nproc-per-node 8 \
  --master-addr "$CONTROL_HOST" \
  --master-port 29500 \
  --rdzv-timeout 180 \
  --job-timeout 1800 \
  --log-root "$LOG_ROOT" \
  --remote-log-root "$REMOTE_LOG_ROOT" \
  --env "NCCL_SOCKET_IFNAME=$NCCL_IFNAME" \
  --env NCCL_DEBUG=INFO \
  --follow-output \
  -- ./build/gpt2 \
    --device cuda \
    --tensor_parallel 4 \
    --pipeline_parallel 2 \
    --batch_size 1 \
    --sequence_length 64 \
    --num_iteration 20 \
    --total_batch_size 16 \
    --input_bin /path/to/gpt2_train.bin \
    --llmc_filepath /path/to/gpt2_124M.bin
```

The tested PP2 command intentionally omits `--tokenizer_bin`: current text
sampling asserts that pipeline parallel size is 1 when generation runs.

`--remote-docker-container` applies to SSH nodes. A host selected by
`--local-host` runs in the controller's current environment, so launch the
controller from the correct container when node 0 also uses a container.

## Logs and Failure Handling

Preflight checks the work directory, executable files, Git SHA, requested data
paths, binary SHA-256, `nvidia-smi`, and visible GPU count. The control log
contains metadata, the complete invocation, preflight output, and one SSH
wrapper log per node. Rank output is stored at:

```text
REMOTE_LOG_ROOT/JOB_ID/node-N/rank-R.stdout.log
REMOTE_LOG_ROOT/JOB_ID/node-N/rank-R.stderr.log
```

`--follow-output` tails the final global rank's stderr on the controller while
preserving the original file. Training loss is normally emitted on stderr.
Every rank first prints hostname, global rank, local rank, world size, device,
and job ID.

The launcher does not restart failed jobs. On node failure, SSH loss, timeout,
SIGINT, or SIGTERM, it reads this job's PID file, verifies the command contains
the job ID, and terminates only that process group. It never uses broad `pkill`.

## Validation

Run the local regression first:

```bash
scripts/multinode_ssh/test_launch_multinode_ssh.sh
```

Then expand in order: single-node 8 ranks, two-node 1 rank each, two-node 8
ranks each, short training, numerical comparison, and failure injection. Keep
`NCCL_DEBUG=INFO` for communication-path evidence and retain generated job
folders. Local simulation must not be reported as GPU or multi-node validation.
