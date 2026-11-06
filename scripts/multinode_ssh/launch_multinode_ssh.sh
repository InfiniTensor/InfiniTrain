#!/usr/bin/env bash
set -euo pipefail

original_argv=("$@")

usage() { sed -n '2,38p' "$0" | sed 's/^# //'; }
# Launch InfiniTrain on multiple nodes from one control node.
# Usage: launch_multinode_ssh.sh [options] -- TRAIN_PROGRAM [ARGS...]
#   --hosts HOSTS | --hostfile PATH
#   --workdir PATH                 Repository path on every node.
#   --nproc-per-node N             Default: 1.
#   --master-addr HOST             Default: first host.
#   --master-port PORT             Default: 29500.
#   --rdzv-id ID                   Default: generated unique ID.
#   --rdzv-timeout SEC             Default: 120.
#   --job-timeout SEC              Default: 0 (disabled).
#   --infini-run PATH              Default: ./build/infini_run.
#   --ssh-user USER --ssh-port PORT
#   --ssh-option OPTION            Repeatable; host checking remains enabled.
#   --local-host HOST              Execute matching host locally.
#   --remote-docker-container NAME
#   --log-root PATH                Default: ./logs.
#   --remote-log-root PATH         Default: WORKDIR/logs.
#   --env NAME=VALUE               Repeatable.
#   --check-path PATH              Repeatable readable data/weight check.
#   --skip-gpu-check               For CPU/simulated tests.
#   --follow-output                 Stream the last global rank stderr.
#   --dry-run

die() { echo "error: $*" >&2; exit 2; }
quote_command() { local result= arg; for arg in "$@"; do printf -v arg '%q' "$arg"; result+="${result:+ }$arg"; done; printf '%s' "$result"; }

hosts_csv= hostfile= workdir=$PWD nproc=1 master_addr= master_port=29500
rdzv_id= rdzv_timeout=120 job_timeout=0 infini_run=./build/infini_run
ssh_user= ssh_port=22 container= local_host= log_root="$PWD/logs" remote_log_root=
dry_run=0 skip_gpu=0 follow_output=0
declare -a ssh_options=() exports=() check_paths=()
while (($#)); do
  case "$1" in
    --hosts) hosts_csv=${2:?}; shift 2;; --hostfile) hostfile=${2:?}; shift 2;;
    --workdir) workdir=${2:?}; shift 2;; --nproc-per-node) nproc=${2:?}; shift 2;;
    --master-addr) master_addr=${2:?}; shift 2;; --master-port) master_port=${2:?}; shift 2;;
    --rdzv-id) rdzv_id=${2:?}; shift 2;; --rdzv-timeout) rdzv_timeout=${2:?}; shift 2;;
    --job-timeout) job_timeout=${2:?}; shift 2;; --infini-run) infini_run=${2:?}; shift 2;;
    --ssh-user) ssh_user=${2:?}; shift 2;; --ssh-port) ssh_port=${2:?}; shift 2;;
    --ssh-option) ssh_options+=("${2:?}"); shift 2;;
    --remote-docker-container) container=${2:?}; shift 2;; --local-host) local_host=${2:?}; shift 2;;
    --log-root) log_root=${2:?}; shift 2;; --remote-log-root) remote_log_root=${2:?}; shift 2;;
    --env) exports+=("${2:?}"); shift 2;; --check-path) check_paths+=("${2:?}"); shift 2;; --skip-gpu-check) skip_gpu=1; shift;;
    --follow-output) follow_output=1; shift;; --dry-run) dry_run=1; shift;; -h|--help) usage; exit 0;; --) shift; break;;
    *) die "unknown launcher option: $1";;
  esac
done
[[ $# -gt 0 ]] || die "missing training program after --"
[[ -z $hosts_csv || -z $hostfile ]] || die "use only one of --hosts and --hostfile"
[[ -n $hosts_csv || -n $hostfile ]] || die "--hosts or --hostfile is required"
for value in "$nproc" "$master_port" "$ssh_port" "$rdzv_timeout"; do [[ $value =~ ^[1-9][0-9]*$ ]] || die "positive integer required: $value"; done
[[ $job_timeout =~ ^[0-9]+$ ]] || die "--job-timeout must be non-negative"
for entry in "${exports[@]}"; do [[ $entry == [A-Za-z_]*=* ]] || die "invalid --env: $entry"; done

declare -a hosts=()
if [[ -n $hosts_csv ]]; then IFS=',' read -r -a hosts <<<"$hosts_csv"; else
  [[ -r $hostfile ]] || die "cannot read hostfile: $hostfile"
  while IFS= read -r line || [[ -n $line ]]; do line=${line%%#*}; read -r line <<<"$line"; [[ -n $line ]] && hosts+=("$line"); done <"$hostfile"
fi
[[ ${#hosts[@]} -gt 0 ]] || die "host list is empty"
declare -A seen=(); for host in "${hosts[@]}"; do [[ -n $host ]] || die "empty host"; [[ -z ${seen[$host]:-} ]] || die "duplicate host: $host"; seen[$host]=1; done
master_addr=${master_addr:-${hosts[0]}}
rdzv_id=${rdzv_id:-"infinitrain-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"}
[[ $rdzv_id =~ ^[A-Za-z0-9_.-]+$ ]] || die "rdzv-id may contain only letters, digits, dot, underscore, dash"
remote_log_root=${remote_log_root:-"$workdir/logs"}
job_dir="$log_root/$rdzv_id"; mkdir -p "$job_dir"
train=("$@"); nnodes=${#hosts[@]}; expected_sha= reference_fingerprint=
if [[ -d $workdir/.git ]]; then expected_sha=$(git -c safe.directory="$workdir" -C "$workdir" rev-parse HEAD 2>/dev/null || true); fi

ssh_base=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$ssh_port")
for option in "${ssh_options[@]}"; do ssh_base+=(-o "$option"); done
node_shell() {
  local host=$1 command=$2 destination=$host
  if [[ -n $local_host && $host == "$local_host" ]]; then bash -c "$command"
  else
    [[ -z $ssh_user ]] || destination="$ssh_user@$host"
    if [[ -n $container ]]; then
      local wrapped; wrapped=$(quote_command docker exec "$container" bash -c "$command")
      "${ssh_base[@]}" "$destination" "$wrapped"
    else "${ssh_base[@]}" "$destination" "$command"; fi
  fi
}

follow_node_shell() {
  local host=$1 command=$2 destination=$host
  if [[ -n $local_host && $host == "$local_host" ]]; then
    exec bash -c "$command"
  fi
  [[ -z $ssh_user ]] || destination="$ssh_user@$host"
  if [[ -n $container ]]; then
    local wrapped
    wrapped=$(quote_command docker exec "$container" bash -c "$command")
    exec "${ssh_base[@]}" "$destination" "$wrapped"
  fi
  exec "${ssh_base[@]}" "$destination" "$command"
}

remote_job_dir="$remote_log_root/$rdzv_id"
code_sha=${expected_sha:-unknown}
code_dirty=unknown
if [[ -d $workdir/.git ]]; then
  if [[ -z $(git -c safe.directory="$workdir" -C "$workdir" status --porcelain --untracked-files=normal 2>/dev/null) ]]; then code_dirty=false; else code_dirty=true; fi
fi
printf 'job_id=%s\ncode_sha=%s\ncode_dirty=%s\nbase_sha=%s\nmaster_sha=%s\nstart_utc=%s\ncontroller=%s\nmaster=%s:%s\nnnodes=%d\nnproc_per_node=%d\n' \
  "$rdzv_id" "$code_sha" "$code_dirty" "924d21256f3baf66bccb7458aa77395f0fbf5841" \
  "5a7be78e9b9d42c2f26efe33984dd3167dc60dc9" "$(date -u +%FT%TZ)" "$(hostname)" \
  "$master_addr" "$master_port" "$nnodes" "$nproc" >"$job_dir/metadata.txt"
printf '%q ' "$0" "${original_argv[@]}" >"$job_dir/controller-command.txt"
printf '\n' >>"$job_dir/controller-command.txt"

for rank in "${!hosts[@]}"; do
  host=${hosts[$rank]}
  printf -v preflight 'set -e; cd %q; test -x %q; test -x %q; command -v git >/dev/null; command -v sha256sum >/dev/null; printf "hostname=%%s\nsha=%%s\ninfini_run_sha256=%%s\ntrain_sha256=%%s\n" "$(hostname)" "$(git -c safe.directory=%q rev-parse HEAD)" "$(sha256sum %q)" "$(sha256sum %q)"' "$workdir" "$infini_run" "${train[0]}" "$workdir" "$infini_run" "${train[0]}"
  for path in "${check_paths[@]}"; do printf -v qpath "%q" "$path"; preflight+="; test -r $qpath"; done
  preflight+="; { command -v nvcc >/dev/null && nvcc --version || true; }; { command -v ldconfig >/dev/null && ldconfig -p 2>/dev/null | grep -i nccl || true; }"
  if ((skip_gpu == 0)); then preflight+="; command -v nvidia-smi >/dev/null; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader; nvidia-smi --query-gpu=index --format=csv,noheader | awk 'END { exit(NR < $nproc) }'"; fi
  if ((dry_run)); then echo "$preflight"; continue; fi
  if ! node_shell "$host" "$preflight" >"$job_dir/preflight-$rank.log" 2>&1; then
    echo "FAILED preflight node=$rank host=$host log=$job_dir/preflight-$rank.log" >&2; exit 1
  fi
  node_sha=$(sed -n 's/^sha=//p' "$job_dir/preflight-$rank.log" | head -1)
  [[ -z $expected_sha || $node_sha == "$expected_sha" ]] || die "code SHA mismatch on $host: $node_sha != $expected_sha"
  fingerprint=$(sed -n '/^infini_run_sha256=/p; /^train_sha256=/p' "$job_dir/preflight-$rank.log")
  if [[ -z $reference_fingerprint ]]; then reference_fingerprint=$fingerprint; else [[ $fingerprint == "$reference_fingerprint" ]] || die "binary SHA-256 mismatch on $host"; fi
done
declare -a pids=() node_for_pid=()
follow_pid=

stopping=0
cleanup_nodes() {
  ((stopping == 0)) || return; stopping=1
  local -a cleanup_pids=()
  for rank in "${!hosts[@]}"; do
    host=${hosts[$rank]}; printf -v qdir '%q' "$remote_job_dir"; printf -v qid '%q' "$rdzv_id"
    cleanup=$(quote_command bash -c 'p=$1; id=$2; if test -r "$p"; then pid=$(cat "$p"); case "$pid" in ""|*[!0-9]*) exit 0;; esac; if test -r /proc/$pid/cmdline && tr "\0" " " </proc/$pid/cmdline | grep -F -- "$id" >/dev/null; then kill -TERM -- -$pid 2>/dev/null || true; sleep 1; kill -KILL -- -$pid 2>/dev/null || true; fi; fi' _ "$remote_job_dir/node-$rank/launcher.pid" "$rdzv_id")
    node_shell "$host" "$cleanup" >/dev/null 2>&1 & cleanup_pids+=("$!")
  done
  for pid in "${cleanup_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
stop_follower() { if [[ -n ${follow_pid:-} ]]; then kill "$follow_pid" 2>/dev/null || true; wait "$follow_pid" 2>/dev/null || true; follow_pid=; fi; }
watchdog_pid=
abort_job() { local rc=$1 reason=$2; cleanup_nodes; stop_follower; printf 'end_utc=%s\nexit_code=%d\nfirst_error=%s\n' "$(date -u +%FT%TZ)" "$rc" "$reason" >>"$job_dir/metadata.txt"; echo "FAILED job=$rdzv_id exit=$rc error=$reason logs=$job_dir remote_logs=$remote_job_dir" >&2; exit "$rc"; }
if ((job_timeout > 0)); then (sleep "$job_timeout"; kill -USR1 $$) & watchdog_pid=$!; fi
trap 'abort_job 124 "job timeout after ${job_timeout}s"' USR1
trap 'abort_job 130 "controller interrupted"' INT TERM

for rank in "${!hosts[@]}"; do
  host=${hosts[$rank]}; node_log_dir="$remote_job_dir/node-$rank"
  node_cmd=("$infini_run" "--nnodes=$nnodes" "--nproc_per_node=$nproc" "--node_rank=$rank"
    "--rdzv_endpoint=$master_addr:$master_port" "--rdzv_id=$rdzv_id" "--rdzv_timeout=$rdzv_timeout"
    "--log_dir=$node_log_dir" "${train[@]}")
  printf -v qwork '%q' "$workdir"; printf -v qdir '%q' "$remote_job_dir"
  command="set -e; mkdir -p $qdir/node-$rank; cd $qwork; "
  for entry in "${exports[@]}"; do printf -v qentry '%q' "$entry"; command+="export $qentry; "; done
  command+="setsid $(quote_command "${node_cmd[@]}") & pid=\$!; echo \$pid >$qdir/node-$rank/launcher.pid; trap 'kill -TERM -- -\$pid 2>/dev/null || true' INT TERM HUP EXIT; wait \$pid"
  echo "launch node=$rank host=$host log=$job_dir/node-$rank-controller.log"
  if ((dry_run)); then echo "$command"; continue; fi
  node_shell "$host" "$command" >"$job_dir/node-$rank-controller.log" 2>&1 &
  pids+=("$!"); node_for_pid+=("$rank:$host")
done
if ((follow_output && !dry_run)); then
  follow_node_rank=$((nnodes - 1)); follow_rank=$((nnodes * nproc - 1)); follow_host=${hosts[$follow_node_rank]}
  follow_file="$remote_job_dir/node-$follow_node_rank/rank-$follow_rank.stderr.log"
  printf -v qfollow '%q' "$follow_file"
  follow_command="while ! test -f $qfollow; do sleep 0.2; done; exec tail -s 0.1 -n +1 -F $qfollow"
  echo "follow global_rank=$follow_rank host=$follow_host file=$follow_file"
  follow_node_shell "$follow_host" "$follow_command" & follow_pid=$!
fi
((dry_run)) && exit 0

status=0 first_error= deadline=0
((job_timeout > 0)) && deadline=$((SECONDS + job_timeout))
declare -A running=(); for i in "${!pids[@]}"; do running[${pids[$i]}]=${node_for_pid[$i]}; done
while ((${#running[@]})); do
  if ((deadline > 0 && SECONDS >= deadline)); then status=124; first_error="job timeout after ${job_timeout}s"; cleanup_nodes; break; fi
  finished=; if wait -n -p finished; then rc=0; else rc=$?; fi
  [[ -n ${finished:-} ]] || { sleep 1; continue; }
  node=${running[$finished]}; unset 'running[$finished]'
  if ((rc != 0)); then status=$rc; first_error="node $node exited with status $rc"; cleanup_nodes; break; fi
done
if ((follow_output)); then sleep 0.3; fi
stop_follower
for pid in "${!running[@]}"; do wait "$pid" 2>/dev/null || true; done
[[ -z $watchdog_pid ]] || kill "$watchdog_pid" 2>/dev/null || true
trap - INT TERM USR1
printf 'end_utc=%s\nexit_code=%d\nfirst_error=%s\n' "$(date -u +%FT%TZ)" "$status" "$first_error" >>"$job_dir/metadata.txt"
if ((status == 0)); then echo "SUCCESS job=$rdzv_id logs=$job_dir remote_logs=$remote_job_dir"; else echo "FAILED job=$rdzv_id exit=$status error=$first_error logs=$job_dir remote_logs=$remote_job_dir" >&2; fi
exit "$status"
