"""Resolve a concept name to a steering direction in residual-stream space.

CAA-style: generate text that embodies the concept, take the mean layer-40
residual, subtract a precomputed baseline (mean residual on neutral text),
normalize to unit. The result is a single d_model vector that the probe
server can apply as `strength * direction` via its existing intervention API
— no SAE involvement, no top-K decomposition, no list-of-interventions.

Two-stage flow:

    resolve(name) -> direction np.ndarray [d_model], unit-norm
    register(name, direction) -> probe_index in the live_concepts probe set

The live_concepts NPZ on disk grows by one row per fresh concept. The probe
server's watcher hot-reloads it every ~5s, after which the new concept can
be referenced by `intervention.probe="live_concepts", probe_index=N`.

A successful registration polls /v1/probes until the new concept name is
visible to the server, so callers can rely on the returned probe_index being
applicable on the very next request.

Cache is keyed by concept name on disk (JSON), so repeated calls are instant
across orchestrator restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np


CONCEPT_TEXT_PROMPTS: list[str] = [
    "Embody {name}.",
    "Become {name}.",
    "Write in vivid first-person about {name}.",
    "Speak as {name}.",
    "What is {name}?",
    "Describe {name}.",
    "Give a few examples of {name}.",
    "Tell me about {name}.",
]

# Neutral seed prompts: parallel to CONCEPT_TEXT_PROMPTS but about ordinary
# placeholder topics. The baseline is the mean residual across these and is
# computed once and cached, so the cost is paid only on first run. Subtracting
# this from the concept mean isolates the concept-specific component.
NEUTRAL_TEXT_PROMPTS: list[str] = [
    "Embody a Tuesday afternoon.",
    "Become an ordinary office.",
    "Write in vivid first-person about a trip to the grocery store.",
    "Speak as a random pedestrian.",
    "What is a paperclip?",
    "Describe a wooden chair.",
    "Give a few examples of common household objects.",
    "Tell me about the weather yesterday.",
]


@dataclass
class ConceptDirection:
    name: str
    probe_index: int                # row in live_concepts probe set
    direction: np.ndarray           # [d_model], unit-norm
    n_tokens_concept: int
    n_tokens_baseline: int
    elapsed_seconds: float
    cached: bool = False


class ConceptDirector:
    """Resolve concept names → unit-norm residual-stream directions, register
    them with the probe server's live_concepts probe set.

    The baseline used for `concept - baseline` is the mean layer-40 residual
    across `NEUTRAL_TEXT_PROMPTS` continuations, generated and encoded once
    on first use and cached to `cache_dir/concept_baseline.npy`. Same prompt
    structure as the concept generation, just neutral content — so what
    subtraction removes is "the model writing about ordinary things in this
    prompt format" and what survives is the concept-specific deviation."""

    def __init__(
        self,
        server: str,
        probes_dir: str | Path,
        cache_dir: str | Path,
        layer: int = 40,
        probe_set_name: str = "live_concepts",
        gen_max_tokens: int = 350,
        gen_temperature: float = 0.8,
        encode_max_tokens: int = 4096,
        reload_poll_interval: float = 1.0,
        reload_poll_timeout: float = 30.0,
    ):
        self.server = server.rstrip("/")
        self.probes_dir = Path(probes_dir)
        self.probes_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.layer = int(layer)
        self.probe_set_name = probe_set_name
        self.gen_max_tokens = gen_max_tokens
        self.gen_temperature = gen_temperature
        self.encode_max_tokens = encode_max_tokens
        self.reload_poll_interval = reload_poll_interval
        self.reload_poll_timeout = reload_poll_timeout

        self._npz_path = (
            self.probes_dir / f"{probe_set_name}_probes_layer{layer}.npz"
        )
        self._cache_path = self.cache_dir / "live_concepts_index.json"
        self._baseline_path = self.cache_dir / "concept_baseline.npy"
        self._registry: dict[str, int] = self._load_registry()
        self._baseline: Optional[np.ndarray] = None
        if self._baseline_path.exists():
            self._baseline = np.load(self._baseline_path).astype(np.float32)
            self.d_model = int(self._baseline.shape[0])
        else:
            self.d_model = 0  # set after baseline computed

    def _load_registry(self) -> dict[str, int]:
        if not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text())
        except Exception:
            return {}

    def _save_registry(self) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._registry, indent=2))
        tmp.replace(self._cache_path)

    async def resolve(
        self, name: str, client: Optional[httpx.AsyncClient] = None,
        prompts: Optional[list[str]] = None,
    ) -> ConceptDirection:
        """Get a registered probe_index for the concept (cached).

        Generates from `len(prompts)` different framings concurrently, encodes
        each, averages the per-prompt mean residual, subtracts the SAE
        training mean, and normalizes. Returns the registered probe_index.
        """
        own_client = client is None
        if client is None:
            client = httpx.AsyncClient()
        assert client is not None
        try:
            if name in self._registry:
                idx = self._registry[name]
                directions, _ = self._read_npz()
                if directions is not None and idx < directions.shape[0]:
                    return ConceptDirection(
                        name=name, probe_index=idx,
                        direction=directions[idx],
                        n_tokens_concept=0, n_tokens_baseline=0,
                        elapsed_seconds=0.0, cached=True,
                    )

            t0 = time.time()
            if self._baseline is None:
                baseline = await self._compute_baseline(client)
                self._baseline = baseline
                tmp = self._baseline_path.with_suffix(".tmp")
                np.save(tmp, baseline.astype(np.float32))
                tmp.replace(self._baseline_path)
                self.d_model = int(baseline.shape[0])

            seed_prompts = prompts or CONCEPT_TEXT_PROMPTS
            user_prompts = [p.format(name=name) for p in seed_prompts]

            # Concurrent generations across the seed prompts; vLLM batches.
            import asyncio
            texts = await asyncio.gather(*[
                self._chat(client, p) for p in user_prompts
            ])

            # Encode each (one /v1/encode call per text; could be batched in
            # one call but keeping per-call separation is simpler).
            per_prompt_means: list[np.ndarray] = []
            total_tokens = 0
            for text in texts:
                if not text:
                    continue
                residuals = await self._encode(client, text)
                if residuals.shape[0] == 0:
                    continue
                per_prompt_means.append(residuals.mean(axis=0))
                total_tokens += int(residuals.shape[0])
            if not per_prompt_means:
                raise RuntimeError(f"no usable concept text for {name!r}")
            mean_concept = np.stack(per_prompt_means).mean(axis=0)

            direction = mean_concept - self._baseline
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise RuntimeError(
                    f"concept '{name}' produced near-zero direction "
                    f"after baseline subtraction (norm={norm:.2e})")
            direction = (direction / norm).astype(np.float32)

            probe_index = await self._register_direction(
                client, name, direction, n_tokens=total_tokens,
            )
            return ConceptDirection(
                name=name, probe_index=probe_index, direction=direction,
                n_tokens_concept=total_tokens,
                n_tokens_baseline=0,  # baseline is from SAE checkpoint, not generated
                elapsed_seconds=time.time() - t0, cached=False,
            )
        finally:
            if own_client:
                await client.aclose()

    async def _compute_baseline(
        self, client: httpx.AsyncClient,
    ) -> np.ndarray:
        """Mean residual across NEUTRAL_TEXT_PROMPTS continuations.

        Same prompt-structure as concept generation but with ordinary topics
        — what subtraction removes is "what every model continuation in this
        prompt format looks like in residual space," leaving the part that
        is concept-specific.
        """
        import asyncio
        texts = await asyncio.gather(*[
            self._chat(client, p) for p in NEUTRAL_TEXT_PROMPTS
        ])
        means: list[np.ndarray] = []
        for text in texts:
            if not text:
                continue
            residuals = await self._encode(client, text)
            if residuals.shape[0] == 0:
                continue
            means.append(residuals.mean(axis=0))
        if not means:
            raise RuntimeError("baseline computation produced no usable text")
        return np.stack(means).mean(axis=0).astype(np.float32)

    async def _chat(self, client: httpx.AsyncClient, user_prompt: str) -> str:
        body = {
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": False,
            "max_tokens": self.gen_max_tokens,
            "temperature": self.gen_temperature,
        }
        r = await client.post(
            f"{self.server}/v1/chat/completions", json=body, timeout=180.0,
        )
        r.raise_for_status()
        choices = r.json().get("choices") or []
        if not choices:
            raise RuntimeError("empty chat completion")
        return (choices[0].get("message") or {}).get("content") or ""

    async def _encode(
        self, client: httpx.AsyncClient, text: str,
    ) -> np.ndarray:
        save_dir = Path("/dev/shm/concept_director_encode")
        save_dir.mkdir(parents=True, exist_ok=True)
        body = {
            "texts": [text],
            "layers": [self.layer],
            "aggregate": "tokens",
            "max_length": self.encode_max_tokens,
            "save_dir": str(save_dir),
            "skip_tokens": 0,
            "mask": "all",
        }
        r = await client.post(
            f"{self.server}/v1/encode", json=body, timeout=180.0,
        )
        r.raise_for_status()
        rec = r.json()["results"][0]
        info = rec.get(f"layer_{self.layer}") or {}
        fname = info.get("file")
        if not fname:
            raise RuntimeError(f"encode response missing file: {rec}")
        arr = np.load(save_dir / fname).astype(np.float32, copy=False)
        try:
            (save_dir / fname).unlink()
        except FileNotFoundError:
            pass
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def _read_npz(self) -> tuple[Optional[np.ndarray], list[str]]:
        if not self._npz_path.exists():
            return None, []
        d = np.load(self._npz_path, allow_pickle=False)
        return d["directions"].astype(np.float32), list(d["labels"])

    def _write_npz(self, directions: np.ndarray, labels: list[str]) -> None:
        tmp = self._npz_path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            directions=directions.astype(np.float32),
            labels=np.array(labels),
            description=np.array(
                f"Live concept directions (CAA-style residual diff, "
                f"layer {self.layer})"
            ),
        )
        tmp.replace(self._npz_path)

    async def _register_direction(
        self,
        client: httpx.AsyncClient,
        name: str,
        direction: np.ndarray,
        n_tokens: int,
    ) -> int:
        """Append the direction to live_concepts NPZ, wait for hot-reload."""
        directions, labels = self._read_npz()
        if directions is None:
            new_directions = direction[None, :].astype(np.float32)
            new_labels = [name]
        else:
            new_directions = np.vstack(
                [directions.astype(np.float32), direction[None, :]]
            )
            new_labels = list(labels) + [name]
        self._write_npz(new_directions, new_labels)
        idx = len(new_labels) - 1
        # Wait for the server's watcher to pick up the new file.
        await self._wait_for_reload(client, expected_count=len(new_labels))
        self._registry[name] = idx
        self._save_registry()
        _ = n_tokens
        return idx

    async def _wait_for_reload(
        self, client: httpx.AsyncClient, expected_count: int,
    ) -> None:
        deadline = time.time() + self.reload_poll_timeout
        last_n = -1
        while time.time() < deadline:
            try:
                r = await client.get(
                    f"{self.server}/v1/probes", timeout=10.0,
                )
                r.raise_for_status()
                for ps in r.json().get("probes", []):
                    if ps.get("name") == self.probe_set_name:
                        last_n = ps.get("n_probes", 0)
                        if last_n >= expected_count:
                            return
            except Exception:
                pass
            await _async_sleep(self.reload_poll_interval)
        raise RuntimeError(
            f"server did not hot-reload {self.probe_set_name} to "
            f"n_probes={expected_count} within {self.reload_poll_timeout}s "
            f"(last seen {last_n})")


async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)


def serialize_concept_direction(c: ConceptDirection) -> dict:
    return {
        "name": c.name,
        "probe_index": c.probe_index,
        "n_tokens_concept": c.n_tokens_concept,
        "n_tokens_baseline": c.n_tokens_baseline,
        "elapsed_seconds": c.elapsed_seconds,
        "cached": c.cached,
        "norm": float(np.linalg.norm(c.direction)),
    }
