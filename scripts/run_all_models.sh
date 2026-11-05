#!/usr/bin/env bash
# Familiar all-model entry; explicit suite selection remains available.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$repo_root/scripts/run_all_tests.sh" --suite nightly "$@"
