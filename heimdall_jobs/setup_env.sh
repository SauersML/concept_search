#!/usr/bin/env bash
# Create the concept_search conda env and install deps. Idempotent.
# Run once on each node where Phase-A jobs will land. CPU-only, ~5 min.
set -euo pipefail

ENV_NAME="${ENV_NAME:-concept_search}"
REPO_DIR="${REPO_DIR:-/home/athuser/gnome_home/concept_search}"

source ~/miniconda/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists"
else
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

# CPU torch is fine for Phase-A (no GPU work). Use the standard wheel; if the
# node already has CUDA torch from another env it doesn't matter — concept_search
# never touches a GPU in Phase-A.
python -m pip install -U pip
python -m pip install \
  "torch>=2.1" \
  "gpytorch>=1.11" \
  "botorch>=0.10" \
  "numpy" "scipy" "pandas" "httpx" \
  "matplotlib" "pytest"

cd "$REPO_DIR"
python -m pip install -e .

python - <<'PY'
import torch, gpytorch, botorch, numpy, scipy, pandas
print("torch       ", torch.__version__)
print("gpytorch    ", gpytorch.__version__)
print("botorch     ", botorch.__version__)
print("numpy       ", numpy.__version__)
print("scipy       ", scipy.__version__)
print("pandas      ", pandas.__version__)
PY

echo "env '$ENV_NAME' ready"
