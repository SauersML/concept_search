"""Build a coactivation kernel from per-feature self-eval transcripts.

For each candidate feature, the existing self-eval TSVs already contain a
multi-thousand-token transcript of Kimi running an agentic evaluation under
steering of that feature. The system prompt and task are identical across
features; the variance in the resulting text is whatever the feature's
steering induces. That makes these transcripts a *much* cleaner basis for a
"feature similarity" kernel than a generic activation corpus, because:

  - same prompt → low between-transcript prompt-induced noise
  - one transcript per feature → direct behavioral signature
  - the regime is exactly the one the search will run in

Method: for each transcript T_i, send to the probe server's /v1/encode (with
aggregate="tokens") to get layer-40 per-token residuals, locally apply the
SAE encoder restricted to the candidate features to get per-token feature
activations, and mean-pool to a [n_candidates] signature vector v_i. The
kernel is K(i, j) = cos(v_i, v_j); angles theta_ij = arccos(K(i, j)).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch


DEFAULT_SAE = "/models/sae/k25-145M-16x-k64.pt"
DEFAULT_TSVS = [
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_top500.tsv",
    "/home/athuser/assistant-axis-exp/results/sae_self_eval_next500.tsv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcript-based coactivation kernel.")
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--tsvs", nargs="+", default=DEFAULT_TSVS)
    p.add_argument("--sae", default=DEFAULT_SAE)
    p.add_argument("--feature-indices", type=int, nargs="+", default=None,
                   help="Candidate feature indices. If omitted, uses all "
                        "features in the TSVs.")
    p.add_argument("--feature-indices-file", default=None,
                   help="Path to a file with one feature index per line "
                        "(used if --feature-indices not given).")
    p.add_argument("--max-tokens-per-transcript", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of transcripts per /v1/encode call (server "
                        "batches them in continuous-batching prefill).")
    p.add_argument("--save-dir", default="/dev/shm/coact_encode",
                   help="Directory the server writes per-request binary .npy "
                        "files to. Must be reachable from this script (we "
                        "read the files back from disk to avoid 2GB JSON "
                        "responses).")
    p.add_argument("--output", default="results/coactivation_transcripts.npz")
    return p.parse_args()


def load_transcripts(tsv_paths: list[str]) -> dict[int, str]:
    """feature_idx -> response text."""
    frames = []
    for path in tsv_paths:
        df = pd.read_csv(path, sep="\t")
        frames.append(df[["feature_idx", "response"]])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["feature_idx"], keep="first")
    return {int(r.feature_idx): str(r.response) for r in df.itertuples()}


async def encode_batch_binary(
    client: httpx.AsyncClient,
    server: str,
    texts: list[str],
    save_dir: Path,
    max_length: int,
    layer: int,
) -> list[np.ndarray]:
    """Hit /v1/encode with a batch and read back binary .npy from save_dir.

    The server writes one file per text — `<rid>_layer{N}.npy` — and the
    JSON response just contains the filenames. This avoids serializing
    millions of floats as JSON, which is the bottleneck of the text path.
    """
    body = {
        "texts": texts,
        "layers": [layer],
        "aggregate": "tokens",
        "max_length": max_length,
        "save_dir": str(save_dir),
        "skip_tokens": 0,
        "mask": "all",
    }
    r = await client.post(f"{server}/v1/encode", json=body, timeout=600.0)
    r.raise_for_status()
    payload = r.json()
    if "results" not in payload or not payload["results"]:
        raise RuntimeError(f"no results in encode response: {list(payload.keys())}")
    key = f"layer_{layer}"
    out: list[np.ndarray] = []
    for rec in payload["results"]:
        info = rec.get(key)
        if not info or info.get("n_tokens", 0) == 0:
            out.append(np.zeros((0, 0), dtype=np.float32))
            continue
        fname = info.get("file")
        if not fname:
            raise RuntimeError(f"layer info missing 'file': {info}")
        arr = np.load(save_dir / fname).astype(np.float32, copy=False)
        out.append(arr)
        # Clean up the npy as soon as we've read it.
        try:
            (save_dir / fname).unlink()
        except FileNotFoundError:
            pass
    return out


def load_sae_encoder_subset(sae_path: str, feature_indices: np.ndarray):
    """Returns (W_enc_sub [N, d_model], b_enc_sub [N], mean [d_model])."""
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    W_enc = sd["W_enc"][feature_indices].float().contiguous()  # [N, d_model]
    b_enc = sd["b_enc"][feature_indices].float().contiguous()  # [N]
    mean_obj = ckpt["mean"]
    mean_vec = (mean_obj if torch.is_tensor(mean_obj)
                else torch.from_numpy(np.asarray(mean_obj))).float()
    del ckpt, sd
    return W_enc, b_enc, mean_vec


async def main_async() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve candidate feature indices.
    if args.feature_indices is not None:
        feats = list(args.feature_indices)
    elif args.feature_indices_file is not None:
        feats = [int(x) for x in
                 Path(args.feature_indices_file).read_text().split()]
    else:
        all_transcripts = load_transcripts(args.tsvs)
        feats = sorted(all_transcripts.keys())
    feats_arr = np.asarray(feats, dtype=np.int64)
    n_features = len(feats_arr)
    print(f"Candidate pool: {n_features} features")

    print(f"Loading transcripts from {len(args.tsvs)} TSVs ...")
    all_transcripts = load_transcripts(args.tsvs)
    missing = [f for f in feats if f not in all_transcripts]
    if missing:
        raise SystemExit(f"  missing transcripts for {len(missing)} features "
                         f"(first few: {missing[:5]})")
    print(f"  all {n_features} transcripts found")

    print(f"Loading SAE encoder subset ({n_features} rows) ...")
    W_enc, b_enc, mean_vec = load_sae_encoder_subset(args.sae, feats_arr)
    print(f"  W_enc: {tuple(W_enc.shape)}  b_enc: {tuple(b_enc.shape)}  "
          f"mean: {tuple(mean_vec.shape)}")

    # Move encoder onto GPU if available — the per-transcript matmul dominates
    # local CPU time and there's plenty of free GPU besides the running probe
    # server.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        W_enc = W_enc.to(device)
        b_enc = b_enc.to(device)
        mean_vec = mean_vec.to(device)
        print(f"  encoder on cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("  encoder on cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any leftover files from a prior run.
    for old in save_dir.glob("*.npy"):
        try:
            old.unlink()
        except FileNotFoundError:
            pass

    # Encode in server-side batches so vLLM can prefill them concurrently.
    print(f"\nEncoding {n_features} transcripts via {args.server}/v1/encode "
          f"(batch_size={args.batch_size}, binary save_dir={save_dir}) ...")
    V = torch.zeros(n_features, n_features, dtype=torch.float64)
    n_tokens_used = np.zeros(n_features, dtype=np.int64)
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, n_features, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_features)
            batch_feats = feats[batch_start:batch_end]
            batch_texts = [all_transcripts[f] for f in batch_feats]

            try:
                batch_residuals = await encode_batch_binary(
                    client, args.server, batch_texts, save_dir,
                    max_length=args.max_tokens_per_transcript,
                    layer=40,
                )
            except Exception as e:
                print(f"  [{batch_start+1}-{batch_end}/{n_features}] encode "
                      f"failed ({type(e).__name__}: {e})",
                      flush=True)
                continue

            for j, (feat, res_np) in enumerate(zip(batch_feats, batch_residuals)):
                i = batch_start + j
                residuals = torch.from_numpy(res_np).float().to(device)
                if residuals.ndim != 2 or residuals.shape[1] != mean_vec.shape[0]:
                    print(f"  feat_{feat}: unexpected shape "
                          f"{tuple(residuals.shape)}; skip", flush=True)
                    continue
                x_centered = residuals - mean_vec
                pre = torch.relu(x_centered @ W_enc.T + b_enc)
                V[i] = pre.mean(dim=0).double().cpu()
                n_tokens_used[i] = residuals.shape[0]

            done = batch_end
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_features - done) / rate if rate > 0 else float("inf")
            print(f"  encoded {done}/{n_features} ({rate:.2f} tx/s, "
                  f"avg {n_tokens_used[:done].mean():.0f} tokens/tx, "
                  f"eta {eta:.0f}s)",
                  flush=True)

    # Subtract cross-transcript mean before similarity. The bulk of every
    # transcript is shared task framing ("I'll evaluate feature N", "let me
    # try strength 30", etc.); without this step that shared component
    # dominates and pushes off-diagonal cosine to ~0.9 (verified empirically).
    # Subtracting the mean isolates the feature-specific deviation.
    print(f"\nComputing cosine similarity (baseline-subtracted) ...")
    baseline = V.mean(dim=0, keepdim=True)
    V_dev = V - baseline
    norms = V_dev.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Vn = V_dev / norms
    K = (Vn @ Vn.T).clamp(-1.0, 1.0).to(torch.float32).numpy()
    np.fill_diagonal(K, 1.0)
    # Cosines after baseline subtraction can be negative (anti-correlated
    # deviations). Map [-1, 1] → [0, pi] for the angular RBF kernel which
    # expects non-negative angles.
    angles = np.arccos(np.clip(K, -1.0, 1.0)).astype(np.float32)
    np.fill_diagonal(angles, 0.0)

    off_diag = K[~np.eye(n_features, dtype=bool)]
    print(f"  off-diag cosine: mean={off_diag.mean():.3f}  "
          f"median={float(np.median(off_diag)):.3f}  "
          f"p95={float(np.percentile(off_diag, 95)):.3f}  "
          f"max={off_diag.max():.3f}")
    print(f"  off-diag angle (rad): mean={np.arccos(off_diag).mean():.3f}  "
          f"min={np.arccos(off_diag.max()):.3f}")

    np.savez(
        out,
        angles=angles,
        cosine=K,
        feature_indices=feats_arr,
        n_tokens_used=n_tokens_used.astype(np.int64),
        sae_path=np.array(args.sae),
        activations_path=np.array("transcript-derived"),
    )
    print(f"\nSaved {out.resolve()}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
