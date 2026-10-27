#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Launch InfiniTrain on multiple nodes from one control node.

Usage:
  scripts/multinode_ssh/launch_multinode_ssh.sh [launcher options] -- TRAIN_PROGRAM [ARGS...]

Launcher options:
  --hosts HOSTS              Comma-separated hosts in node-rank order.
  --hostfile PATH            One host per line; blank lines and # comments ignored.
  --workdir PATH             Repository path on every node (default: current directory).
  --nproc-per-node N         Processes started by infini_run on each node (default: 1).
  --master-addr HOST         Rendezvous address (default: first host).
  --master-port PORT         Rendezvous port (default: 29500).
  --rdzv-id ID               Shared job ID (default: generated).
  --infini-run PATH          infini_run path relative to workdir (default: ./build/infini_run).
  --ssh-user USER            SSH user for remote nodes.
  --ssh-port PORT            SSH port (default: 22).
  --remote-docker-container NAME
                             Run remote commands through docker exec NAME.
  --ssh-option OPTION        Additional ssh option; may be repeated.
  --log-dir PATH             Per-node log directory (default: ./logs/<rdzv-id>).
  --local-host HOST          Host entry to execute locally instead of through SSH.
  --node-start-delay SEC     Seconds to wait between starting adjacent nodes (default: 0).
  --env NAME=VALUE           Export an environment variable on every node; may be repeated.
  --dry-run                  Print commands without starting them.
  -h, --help                 Show this help.

Example:
  scripts/multinode_ssh/launch_multinode_ssh.sh \
    --hosts 192.168.162.62,192.168.162.63 \
    --local-host 192.168.162.62 \
    --workdir "$HOME/code/InfiniTrain" \
    --nproc-per-node 4 \
    --master-addr 192.168.162.62 \
    --env NCCL_SOCKET_IFNAME=ens22f0 \
    --env NCCL_IB_DISABLE=1 \
    -- ./build/gpt2 --device cuda --tensor_parallel 2 --pipeline_parallel 2
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

quote_command() {
  local quoted=()
  local arg
  for arg in "$@"; do
    printf -v arg '%q' "$arg"
    quoted+=("$arg")
  done
  local IFS=' '
  printf '%s' "${quoted[*]}"
}

hosts_csv=""
hostfile=""
workdir="$PWD"
nproc_per_node=1
master_addr=""
master_port=29500
rdzv_id=""
infini_run="./build/infini_run"
ssh_user=""
ssh_port=22
remote_docker_container=""
log_dir=""
local_host=""
node_start_delay=0
dry_run=0
declare -a ssh_options=()
declare -a exports=()

while (($#)); do
  case "$1" in
    --hosts) [[ $# -ge 2 ]] || die "--hosts requires a value"; hosts_csv=$2; shift 2 ;;
    --hostfile) [[ $# -ge 2 ]] || die "--hostfile requires a value"; hostfile=$2; shift 2 ;;
    --workdir) [[ $# -ge 2 ]] || die "--workdir requires a value"; workdir=$2; shift 2 ;;
    --nproc-per-node) [[ $# -ge 2 ]] || die "--nproc-per-node requires a value"; nproc_per_node=$2; shift 2 ;;
    --master-addr) [[ $# -ge 2 ]] || die "--master-addr requires a value"; master_addr=$2; shift 2 ;;
    --master-port) [[ $# -ge 2 ]] || die "--master-port requires a value"; master_port=$2; shift 2 ;;
    --rdzv-id) [[ $# -ge 2 ]] || die "--rdzv-id requires a value"; rdzv_id=$2; shift 2 ;;
    --infini-run) [[ $# -ge 2 ]] || die "--infini-run requires a value"; infini_run=$2; shift 2 ;;
    --ssh-user) [[ $# -ge 2 ]] || die "--ssh-user requires a value"; ssh_user=$2; shift 2 ;;
    --ssh-port) [[ $# -ge 2 ]] || die "--ssh-port requires a value"; ssh_port=$2; shift 2 ;;
    --remote-docker-container) [[ $# -ge 2 ]] || die "--remote-docker-container requires a value"; remote_docker_container=$2; shift 2 ;;
    --ssh-option) [[ $# -ge 2 ]] || die "--ssh-option requires a value"; ssh_options+=("$2"); shift 2 ;;
    --log-dir) [[ $# -ge 2 ]] || die "--log-dir requires a value"; log_dir=$2; shift 2 ;;
    --local-host) [[ $# -ge 2 ]] || die "--local-host requires a value"; local_host=$2; shift 2 ;;
    --node-start-delay) [[ $# -ge 2 ]] || die "--node-start-delay requires a value"; node_start_delay=$2; shift 2 ;;
    --env) [[ $# -ge 2 ]] || die "--env requires NAME=VALUE"; exports+=("$2"); shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) die "unknown launcher option: $1" ;;
  esac
done

[[ $# -gt 0 ]] || die "missing training program after --"
[[ -z $hosts_csv || -z $hostfile ]] || die "use only one of --hosts and --hostfile"
[[ -n $hosts_csv || -n $hostfile ]] || die "--hosts or --hostfile is required"
[[ $nproc_per_node =~ ^[1-9][0-9]*$ ]] || die "--nproc-per-node must be positive"
[[ $master_port =~ ^[1-9][0-9]*$ ]] || die "--master-port must be positive"
[[ $ssh_port =~ ^[1-9][0-9]*$ ]] || die "--ssh-port must be positive"
[[ $node_start_delay =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--node-start-delay must be non-negative"

declare -a hosts=()
if [[ -n $hosts_csv ]]; then
  IFS=',' read -r -a hosts <<<"$hosts_csv"
else
  [[ -r $hostfile ]] || die "cannot read hostfile: $hostfile"
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}
    read -r line <<<"$line"
    [[ -n $line ]] && hosts+=("$line")
  done <"$hostfile"
fi
[[ ${#hosts[@]} -gt 0 ]] || die "host list is empty"

for entry in "${exports[@]}"; do
  [[ $entry == *=* && $entry != =* ]] || die "invalid --env value: $entry"
done

master_addr=${master_addr:-${hosts[0]}}
rdzv_id=${rdzv_id:-"infinitrain-$(date +%Y%m%d-%H%M%S)-$$"}
log_dir=${log_dir:-"$PWD/logs/$rdzv_id"}
mkdir -p "$log_dir"

train_command=("$@")
nnodes=${#hosts[@]}
declare -a child_pids=()
declare -a child_nodes=()
stopping=0

stop_children() {
  ((stopping == 0)) || return
  stopping=1
  local pid
  for pid in "${child_pids[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap 'stop_children; exit 130' INT TERM

for node_rank in "${!hosts[@]}"; do
  host=${hosts[$node_rank]}
  node_command=(
    "$infini_run"
    "--nnodes=$nnodes"
    "--nproc_per_node=$nproc_per_node"
    "--node_rank=$node_rank"
    "--rdzv_endpoint=$master_addr:$master_port"
    "--rdzv_id=$rdzv_id"
    "${train_command[@]}"
  )

  printf -v quoted_workdir '%q' "$workdir"
  command_text="cd $quoted_workdir && "
  for entry in "${exports[@]}"; do
    printf -v quoted_export '%q' "$entry"
    command_text+="export $quoted_export; "
  done
  command_text+="exec $(quote_command "${node_command[@]}")"

  log_file="$log_dir/node-${node_rank}-${host//[^a-zA-Z0-9_.-]/_}.log"
  if [[ -n $local_host && $host == "$local_host" ]]; then
    launch=(bash -lc "$command_text")
  else
    destination=$host
    [[ -z $ssh_user ]] || destination="$ssh_user@$host"
    launch=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$ssh_port")
    for option in "${ssh_options[@]}"; do
      launch+=(-o "$option")
    done
    if [[ -n $remote_docker_container ]]; then
      docker_command=(docker exec "$remote_docker_container" bash -lc "$command_text")
      launch+=("$destination" "$(quote_command "${docker_command[@]}")")
    else
      launch+=("$destination" "$command_text")
    fi
  fi

  launch=(setsid "${launch[@]}")

  printf '[node %d %s] ' "$node_rank" "$host"
  quote_command "${launch[@]}"
  printf ' > %q 2>&1\n' "$log_file"
  if ((dry_run)); then
    continue
  fi

  "${launch[@]}" >"$log_file" 2>&1 &
  child_pids+=("$!")
  child_nodes+=("$host")

  if ((node_rank + 1 < nnodes)) && [[ $node_start_delay != 0 ]]; then
    sleep "$node_start_delay"
  fi
done

((dry_run)) && exit 0

status=0
declare -A node_by_pid=()
declare -A running_pids=()
for index in "${!child_pids[@]}"; do
  pid=${child_pids[$index]}
  node_by_pid[$pid]=${child_nodes[$index]}
  running_pids[$pid]=1
done

while ((${#running_pids[@]} > 0)); do
  finished_pid=""
  if wait -n -p finished_pid; then
    rc=0
  else
    rc=$?
  fi
  [[ -n ${finished_pid:-} ]] || die "failed to identify completed node launcher"
  unset 'running_pids[$finished_pid]'
  if ((rc != 0)); then
    echo "node ${node_by_pid[$finished_pid]} failed (launcher status $rc)" >&2
    status=1
    stop_children
    for pid in "${!running_pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
    break
  fi
done

trap - INT TERM
exit "$status"
