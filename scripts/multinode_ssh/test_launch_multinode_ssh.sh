#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
launcher="$script_dir/launch_multinode_ssh.sh"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/fake_infini_run" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*"
EOF
chmod +x "$tmpdir/fake_infini_run"

output=$(
  "$launcher" \
    --hosts local.test,remote.test \
    --local-host local.test \
    --workdir "$tmpdir" \
    --infini-run ./fake_infini_run \
    --nproc-per-node 2 \
    --master-addr 10.0.0.1 \
    --rdzv-id test-job \
    --env NCCL_DEBUG=INFO \
    --remote-docker-container test-container \
    --dry-run \
    -- ./train --flag 'value with spaces'
)

grep -q -- '--node_rank=0' <<<"$output"
grep -q -- '--node_rank=1' <<<"$output"
grep -q -- '--nnodes=2' <<<"$output"
grep -q -- '--nproc_per_node=2' <<<"$output"
grep -q -- '--rdzv_endpoint=10.0.0.1:29500' <<<"$output"
grep -q -- '--rdzv_id=test-job' <<<"$output"
grep -q -- 'local.test' <<<"$output"
grep -q -- 'remote.test' <<<"$output"
grep -q -- 'docker.*exec.*test-container' <<<"$output"

"$launcher" \
  --hosts local.test \
  --local-host local.test \
  --workdir "$tmpdir" \
  --infini-run ./fake_infini_run \
  --rdzv-id local-test \
  --log-dir "$tmpdir/logs" \
  -- ./train --flag value

grep -q -- '--node_rank=0' "$tmpdir/logs/node-0-local.test.log"
grep -q -- './train --flag value' "$tmpdir/logs/node-0-local.test.log"

SECONDS=0
"$launcher" \
  --hosts local.test,local.test \
  --local-host local.test \
  --workdir "$tmpdir" \
  --infini-run ./fake_infini_run \
  --rdzv-id delayed-local-test \
  --node-start-delay 1 \
  --log-dir "$tmpdir/delayed-logs" \
  -- ./train
((SECONDS >= 1))

if "$launcher" --hosts local.test --node-start-delay invalid -- ./train \
    >"$tmpdir/invalid-delay.out" 2>&1; then
  echo "invalid node start delay unexpectedly succeeded" >&2
  exit 1
fi
grep -q -- '--node-start-delay must be non-negative' "$tmpdir/invalid-delay.out"

echo "launch_multinode_ssh tests passed"
