#!/usr/bin/env bash
# Submit Phase-A replay as a Heimdall job.
# Run from inside the repo on node1: bash heimdall_jobs/submit_phase_a.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/athuser/gnome_home/concept_search}"
exec python3 "$REPO_DIR/heimdall_jobs/submit_phase_a.py" "$@"
