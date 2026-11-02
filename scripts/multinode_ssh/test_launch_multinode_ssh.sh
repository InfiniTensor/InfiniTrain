#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
launcher="$script_dir/launch_multinode_ssh.sh"
repo=$(cd "$script_dir/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/fake_infini_run" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@"
printf '\n'
log_dir=
node_rank=0
nproc=1
for arg in "$@"; do
  case "$arg" in
    --log_dir=*) log_dir=${arg#*=} ;;
    --node_rank=*) node_rank=${arg#*=} ;;
    --nproc_per_node=*) nproc=${arg#*=} ;;
  esac
done
case "${FAKE_MODE:-ok}" in
  fail) exit 17 ;;
  sleep)
    trap 'exit 143' TERM INT
    while :; do sleep 1; done
    ;;
  output)
    mkdir -p "$log_dir"
    rank=$(((node_rank + 1) * nproc - 1))
    printf 'live-start rank=%d\n' "$rank" >"$log_dir/rank-$rank.stderr.log"
    sleep 1
    printf 'live-end rank=%d\n' "$rank" >>"$log_dir/rank-$rank.stderr.log"
    ;;
esac
RUN
chmod +x "$tmp/fake_infini_run"

output=$("$launcher" --hosts alpha,beta --workdir "$repo" \
  --infini-run "$tmp/fake_infini_run" --nproc-per-node 8 \
  --master-addr 10.0.0.1 --rdzv-id dry-run-test \
  --log-root "$tmp/logs" --remote-log-root "$tmp/remote logs" \
  --remote-docker-container test-container --skip-gpu-check --dry-run \
  -- /bin/echo 'value with spaces' 'semi;colon')
grep -q -- 'node_rank=0' <<<"$output"
grep -q -- 'node_rank=1' <<<"$output"
grep -q -- 'nproc_per_node=8' <<<"$output"
grep -q -- '10.0.0.1:29500' <<<"$output"

printf '# rank order\nlocal.test\n' >"$tmp/hosts"
"$launcher" --hostfile "$tmp/hosts" --local-host local.test --workdir "$repo" \
  --infini-run "$tmp/fake_infini_run" --rdzv-id success-test \
  --log-root "$tmp/logs" --remote-log-root "$tmp/remote" --skip-gpu-check \
  --env 'SAFE_VALUE=value with spaces;still-data' -- /bin/echo 'arg with spaces'
grep -q 'exit_code=0' "$tmp/logs/success-test/metadata.txt"
grep -q 'arg\\ with\\ spaces' "$tmp/logs/success-test/node-0-controller.log"
grep -q -- '--hostfile' "$tmp/logs/success-test/controller-command.txt"

FAKE_MODE=output "$launcher" --hosts local.test --local-host local.test \
  --workdir "$repo" --infini-run "$tmp/fake_infini_run" --nproc-per-node 2 \
  --rdzv-id output-test --log-root "$tmp/logs" \
  --remote-log-root "$tmp/remote" --skip-gpu-check --follow-output \
  -- /bin/true >"$tmp/follow-output.log"
grep -q 'live-start rank=1' "$tmp/follow-output.log"
grep -q 'live-end rank=1' "$tmp/follow-output.log"

if FAKE_MODE=fail "$launcher" --hosts local.test --local-host local.test \
  --workdir "$repo" --infini-run "$tmp/fake_infini_run" \
  --rdzv-id failure-test --log-root "$tmp/logs" \
  --remote-log-root "$tmp/remote" --skip-gpu-check -- /bin/true; then
  echo "non-zero node unexpectedly succeeded" >&2
  exit 1
fi
grep -q 'exit_code=17' "$tmp/logs/failure-test/metadata.txt"

FAKE_MODE=sleep "$launcher" --hosts local.test --local-host local.test \
  --workdir "$repo" --infini-run "$tmp/fake_infini_run" \
  --rdzv-id signal-test --log-root "$tmp/logs" \
  --remote-log-root "$tmp/remote" --skip-gpu-check -- /bin/true &
launcher_pid=$!
pid_file="$tmp/remote/signal-test/node-0/launcher.pid"
for _ in {1..150}; do
  [[ -r $pid_file ]] && break
  sleep 0.1
done
[[ -r $pid_file ]] || { echo "signal test never launched remote job" >&2; exit 1; }
kill -TERM "$launcher_pid"
if wait "$launcher_pid"; then
  echo "signalled launcher unexpectedly succeeded" >&2
  exit 1
fi
sleep 1
if [[ -r $pid_file ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "remote job process remained after signal" >&2
  exit 1
fi

start=$SECONDS
if FAKE_MODE=sleep "$launcher" --hosts local.test --local-host local.test \
  --workdir "$repo" --infini-run "$tmp/fake_infini_run" \
  --rdzv-id timeout-test --job-timeout 2 --log-root "$tmp/logs" \
  --remote-log-root "$tmp/remote" --skip-gpu-check -- /bin/true; then
  echo "timed job unexpectedly succeeded" >&2
  exit 1
else
  rc=$?
  [[ $rc == 124 ]] || { echo "timeout returned $rc" >&2; exit 1; }
fi
((SECONDS - start < 20))
grep -q 'exit_code=124' "$tmp/logs/timeout-test/metadata.txt"
pid_file="$tmp/remote/timeout-test/node-0/launcher.pid"
if [[ -r $pid_file ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "remote job process remained after timeout" >&2
  exit 1
fi

if "$launcher" --hosts duplicate,duplicate --skip-gpu-check -- /bin/true \
  >"$tmp/invalid" 2>&1; then
  echo "duplicate hosts unexpectedly accepted" >&2
  exit 1
fi
grep -q 'duplicate host' "$tmp/invalid"

if "$launcher" --hosts local.test --local-host local.test --workdir "$repo" \
  --infini-run "$tmp/fake_infini_run" --rdzv-id missing-path-test \
  --log-root "$tmp/logs" --remote-log-root "$tmp/remote" --skip-gpu-check \
  --check-path "$tmp/does-not-exist" -- /bin/true; then
  echo "missing check path unexpectedly accepted" >&2
  exit 1
fi

echo "launch_multinode_ssh tests passed"
