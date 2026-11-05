#!/usr/bin/env bash
# Compatibility entry for classic groups; all execution uses one runner.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="$repo_root/scripts/test_config.json"
only_run=""
only_set=false
forward=()
usage() {
    cat <<'EOF'
Usage: run_models_and_profile.bash [--test-config PATH] [--only-run TAGS] [runner options]

Runs migrated classic tests through scripts/run_all_tests.sh --suite nightly.
--only-run basic,zero,... selects the UNION of legacy groups.
--test-config accepts only the repository's original scripts/test_config.json.
New/custom configurations belong in tests/regression/scenarios/*.json.
All other options are passed to run_all_tests.sh (for example --list, --dry-run,
--model llama3, --data-root PATH, --build-dir PATH, --no-build).

Builds and checkpoints are retained. Profiling is no longer run automatically.
See tests/regression/README.md for suites, fixture paths and log-history alignment.
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-config|--only-run)
            [[ $# -ge 2 && "$2" != --* && -n "$2" ]] || { echo "Error: $1 requires a value." >&2; exit 2; }
            if [[ "$1" == --test-config ]]; then config="$2"; else only_run="$2"; only_set=true; fi
            shift 2 ;;
        --test-config=*) config="${1#*=}"; shift ;;
        --only-run=*) only_run="${1#*=}"; only_set=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) forward+=("$1"); shift ;;
    esac
done
# Historically the default was relative to scripts/, even when no file was read.
if [[ "$config" == test_config.json ]]; then config="$repo_root/scripts/test_config.json"; fi
if [[ ! -f "$config" ]] || [[ "$(realpath -- "$config")" != "$(realpath -- "$repo_root/scripts/test_config.json")" ]]; then
    echo "Error: custom --test-config files are not interpreted by the new runner." >&2
    echo "Migrate the configuration to tests/regression/scenarios/*.json; see tests/regression/schema/README.md." >&2
    exit 2
fi
selection=()
if [[ "$only_set" == true ]]; then
    [[ -n "${only_run//[[:space:],]/}" ]] || { echo 'Error: --only-run must contain at least one group.' >&2; exit 2; }
    IFS=',' read -r -a tags <<< "$only_run"
    for tag in "${tags[@]}"; do
        tag="${tag//[[:space:]]/}"
        case "$tag" in
            recompute)
                selection+=(--case "classic_recompute*/*") ;;
            basic|zero|lora|checkpoint|lr_scheduler|8_proc|qwen3|qwen3_8_proc|moe)
                selection+=(--case "classic_${tag}/*") ;;
            *) echo "Error: unknown legacy group '$tag'." >&2; exit 2 ;;
        esac
    done
else
    selection+=(--case 'classic_*')
fi
printf '%s\n' '[migration] Using the regression runner; configure machine paths with --data-root or tests.local.json.' >&2
exec "$repo_root/scripts/run_all_tests.sh" --suite nightly "${selection[@]}" "${forward[@]}"
