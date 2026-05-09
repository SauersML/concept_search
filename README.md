# concept_search

Find SAE features corresponding to a target concept on a new SAE / new model, without text-activation prefiltering or manual calibration scaffolding. Scales to abstract concepts where naturalistic corpus search is unreliable.

## Approach

GP Bayesian optimization over the SAE feature index, with the kernel induced by decoder-direction similarity:

- `f(idx)` = a 0–100 concept-relatedness rating produced by an agentic eval in which the model itself runs under steering and reports on its own state.
- `K(i, j) = exp(-theta_ij^2 / 2 * l^2)` where `theta_ij` is the angle between `W_dec[:, i]` and `W_dec[:, j]`. Decoder columns are unit-norm at training time, so this is `arccos(W_dec.T @ W_dec)` followed by an RBF.
- Heteroscedastic Gaussian noise: variance per observation comes from re-runs of the same feature under different prompts.
- UCB acquisition (with optional Thompson-batched variant) over active features (those with nonzero fire counts on a held-out corpus).

## Phases

- **Phase-A (kernel/acquisition validation)**: Replay against existing 1000 labeled features (`results/sae_self_eval_top500.tsv` + `next500.tsv`). Treat the existing self-eval scores as ground truth `f(i)`. Budget=100. Measure recall@K of top-20 and Spearman vs full ranking. No live model.
- **Phase-B (live)**: Liquid (concrete sanity check) → misaligned AI (abstract target). Live agentic eval with faithful per-segment K,V tracking.

## Layout

```
src/concept_search/
  data.py          load W_dec, persistence cache, labeled TSVs
  kernel.py        angular-RBF kernel on decoder columns
  gp.py            BoTorch FixedNoiseGP wrapper
  acquisition.py   UCB / Thompson
  bo_loop.py       acquire -> observe -> update
scripts/
  phase_a_replay.py    Phase-A entry point
  fit_noise.py         re-run subset of labeled features for sigma estimate
heimdall_jobs/
  submit_phase_a.sh
  submit_fit_noise.sh
```

## Running

All compute runs through Heimdall.

```bash
# Phase-A replay (CPU only)
bash heimdall_jobs/submit_phase_a.sh
```
