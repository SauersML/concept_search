# concept_search

Find SAE features corresponding to a target concept on a new SAE / new model, without text-activation prefiltering or manual calibration scaffolding. Scales to abstract concepts where naturalistic corpus search is unreliable.

## Approach

GP-BO over the SAE feature index, with the kernel induced by **coactivation similarity** on a real corpus:

- `f(idx)` = a 0–100 concept-relatedness rating from an agentic eval in which the model itself runs under steering and reports on its own state.
- Kernel: `K(i, j) = exp(-theta_ij^2 / (2 * lengthscale^2))` where `theta_ij = arccos(cos_ij)` and `cos_ij` is the cosine similarity of the pre-TopK ReLU activations of features `i` and `j` on a fixed corpus sample (~100k tokens of layer-40 residual stream).
- UCB acquisition over the candidate pool. Random search is reported as a baseline.

We dropped the angular-RBF-on-decoder-direction kernel that was the original plan: TopK SAE training pushes decoder columns near-orthogonal independently of concept similarity (mean off-diagonal angle ≈ 90° on this SAE), so geometric proximity in decoder space encodes almost no concept information. Coactivation measures concept similarity directly: features that fire (or want to fire) on the same tokens are concept-related.

We use *pre-TopK* ReLU rather than post-TopK because TopK competition systematically suppresses joint firing of similar features — they fight for the same top-k slots on the same tokens. Pre-TopK is what each encoder *wants* to do at every token, before the competition.

## Phases

- **Phase-A** (this repo, kernel validation): replay GP-BO against the existing 1000-label dataset (`results/sae_self_eval_top500.tsv` + `next500.tsv`). Treat scores as ground truth `f(i)`. Budget=100. Compare UCB / Thompson / random.
- **Phase-B** (next): live agentic eval with model-under-steering. Liquid → misaligned AI.

## Layout

```
src/concept_search/
  data.py            load SAE encoder/decoder, persistence cache, label TSVs
  coactivation.py    encode tokens through SAE, build pairwise angle matrix
  kernel.py          angular-RBF kernel on a precomputed angle matrix
  gp.py              BoTorch SingleTaskGP wrapper (homoscedastic + fixed-noise)
  acquisition.py     UCB / Thompson / random
  bo_loop.py         seed → loop[acquire, observe, update] → posterior
  metrics.py         best-observed curve, recall@K, posterior-top-K mean
scripts/
  build_coactivation.py   one-off: build the angle matrix
  phase_a_replay.py       BO loop + report
  phase_a_plot.py         best-observed curves + posterior scatter
heimdall_jobs/
  setup_env.sh                    install conda env + deps
  submit_build_coactivation.py    Heimdall submission for the matrix builder
  submit_phase_a.py               Heimdall submission for Phase-A replay
```

## Running

```bash
# 1. one-off env setup on each node
bash heimdall_jobs/setup_env.sh

# 2. build the coactivation angle matrix (~minutes, CPU)
python heimdall_jobs/submit_build_coactivation.py

# 3. run Phase-A replay (seconds, CPU)
python heimdall_jobs/submit_phase_a.py

# 4. plot
PYTHONPATH=src python scripts/phase_a_plot.py results/phase_a/<job_id>
```

Default paths assume node1's `/models/sae/k25-145M-16x-k64.pt` SAE, `/models/k25_tokens/emotions/activations.npy` cache, and `assistant-axis-exp/results/persistence_final/k25/sae_persistence_arrays.npz`.
