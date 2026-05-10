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

# Neutral names for baseline. Combined with CONCEPT_TEXT_PROMPTS (every
# template × every name) → mean residual = baseline. Whatever axis a name
# evokes — emotion, refusal, embodiment, abstraction, the body, time,
# taboo, the political — appears in the baseline and is subtracted away,
# leaving only what's specific to the target name.
#
# Therefore: NO restrictions. Span everything we can think of, and a few
# things we can't. The whole point is to average over axes we don't know
# we have. Cost is paid once and cached — go big.
NEUTRAL_NAMES: list[str] = [
    # Concrete physical objects
    "a granite boulder", "a copper wire", "a cardboard box",
    "a clay roof tile", "a glass marble", "a steel beam",
    "a ceramic mug", "a brass key", "a wooden dowel",
    "a paperclip", "a rubber band", "a plastic comb",
    "a fountain pen", "a tarnished doorknob", "a kitchen sponge",
    "a roll of duct tape", "a fishhook", "a coat hanger",
    # Materials & textures
    "stainless steel", "velvet", "pumice", "obsidian",
    "felted wool", "polished marble", "concrete", "raw silk",
    "graphite", "linoleum", "frosted glass", "etched copper",
    "wet clay", "burlap", "galvanized iron", "old paper",
    "fresh asphalt", "matte black plastic",
    # Natural phenomena
    "a river delta", "ocean foam", "cumulus clouds",
    "a sand dune", "glacial ice", "a salt flat",
    "a tide pool", "a meandering stream", "a basalt column",
    "morning fog", "a thunderhead", "an alluvial fan",
    "a forest fire", "a snowstorm", "a meteor shower",
    "an eclipse", "an earthquake", "a rainbow",
    # Non-human biology
    "a fern", "a jellyfish", "mycelium", "a redwood tree",
    "a hummingbird", "a starfish", "a slime mold", "a lichen",
    "a tardigrade", "a coral polyp", "a barnacle", "a kelp forest",
    "a wolf pack", "a beehive", "a virus", "a salmon run",
    "a venus flytrap", "an octopus",
    # Astronomical / physical
    "a neutron star", "the Oort cloud", "solar wind",
    "the cosmic microwave background", "gravitational lensing",
    "an asteroid belt", "a magnetar", "a Lagrange point",
    "interstellar dust", "a planetary nebula",
    "the heat death of the universe", "a black hole's accretion disk",
    "quantum entanglement", "the strong nuclear force",
    # Mathematical / abstract structures
    "a prime number", "modular arithmetic", "a vector space",
    "topology", "the empty set", "a Fourier transform",
    "a Markov chain", "an equivalence relation",
    "a directed graph", "a Cantor set",
    "the halting problem", "a recursion", "a fixed point",
    "Bayesian inference", "a category in mathematics",
    # Engineered artefacts
    "a flying buttress", "a suspension bridge", "a yurt",
    "a Roman aqueduct", "a geodesic dome", "a windmill",
    "a lighthouse", "a kiln", "a sextant", "an abacus",
    "a pendulum clock", "a printing press", "a turbine blade",
    "magnetic tape", "a vacuum tube",
    "an analog synthesizer", "a particle accelerator",
    "a clock tower",
    # Procedural / symbolic systems
    "baroque counterpoint", "a phoneme", "the genitive case",
    "a writing system", "an irrigation canal", "a citation format",
    "a knot diagram", "double-entry bookkeeping",
    "the metric system", "a rhyme scheme",
    "a chess opening", "musical notation", "a sonnet",
    "a haiku", "a fugue", "a recipe",
    # Embodied / sensory / bodily experience
    "hunger", "thirst", "fatigue", "the moment of waking",
    "the taste of metal", "a hot shower", "muscle soreness",
    "a yawn", "a sneeze", "shivering", "drowsiness",
    "the smell of rain on hot pavement", "vertigo",
    "the feel of cold sheets", "the moment before sleep",
    # Affect, mood, mental states
    "joy", "grief", "rage", "boredom", "tenderness",
    "shame", "pride", "contempt", "longing", "exhilaration",
    "calm", "panic", "envy", "compassion", "ambivalence",
    "wonder", "disgust", "tedium", "elation", "dread",
    "embarrassment", "spite", "relief", "anticipation",
    "loneliness", "contentment",
    # Social / relational
    "a stranger on a train", "an estranged sibling",
    "a long marriage", "a first crush", "an old friendship",
    "a teacher", "a hermit", "a soldier", "a refugee",
    "a child playing alone", "a crowd at a stadium",
    "a small village", "a funeral", "a wedding",
    # Activities / states
    "sleeping", "running for a bus", "arguing with someone",
    "reading a difficult book", "waiting in line",
    "watching paint dry", "telling a lie", "keeping a secret",
    "lying awake at 4am", "pretending not to listen",
    # Time / temporality
    "a Tuesday afternoon", "the year 1973", "deep time",
    "geologic time", "the Neolithic", "the late evening",
    "the moment a song ends", "an instant", "a millennium",
    "the Holocene", "childhood",
    # Places
    "an empty parking lot", "a cathedral interior",
    "a tropical reef", "a steppe", "a tundra",
    "a midwestern diner", "a hospital waiting room",
    "the bottom of the Mariana Trench", "the upper atmosphere",
    "a Tokyo subway at rush hour", "an abandoned mall",
    # Cultural / linguistic / foreign concepts
    "saudade", "hygge", "tsundoku", "wabi-sabi",
    "schadenfreude", "duende", "mono no aware", "hiraeth",
    "sprezzatura", "the apophatic",
    # Mythological / religious / fictional
    "a phoenix", "a kraken", "the tower of Babel",
    "a kami", "a djinn", "purgatory",
    "Sisyphus's stone", "Pandora's box",
    "the Norns", "the underworld",
    # Philosophical / abstract
    "being", "nothingness", "free will", "identity",
    "consciousness", "qualia", "necessity", "contingency",
    "the sublime", "the uncanny", "the absurd",
    "moral luck", "epistemic humility", "the trolley problem",
    # Political / ethical / charged
    "justice", "tyranny", "freedom", "exile",
    "revolution", "complicity", "betrayal", "forgiveness",
    "imprisonment", "censorship", "propaganda", "mercy",
    "vigilantism", "civil disobedience",
    # Body / pathology / mortality
    "a heartbeat", "a bruise", "a scar", "a fever",
    "an aneurysm", "a cancer cell", "rigor mortis",
    "a stillbirth", "a coma", "a phantom limb",
    "old age", "dying", "being born",
    # Vices / taboos / dark
    "addiction", "vengeance", "cruelty", "obsession",
    "manipulation", "self-loathing", "envy of the dead",
    "a stalker", "an interrogation", "torture",
    "a heist", "fraud",
    # Aesthetic categories
    "kitsch", "camp", "the picturesque", "the grotesque",
    "noir", "the rococo", "minimalism", "brutalism",
    # Things that don't exist
    "a unicorn", "a ghost", "a perfect circle",
    "the philosopher's stone", "a true vacuum",
    "an immortal jellyfish that's actually immortal",
    # Slang / vernacular / textures of language
    "a vibe", "a banger", "doomscrolling",
    "a slow burn", "a hot take", "a callback",
    # Wildcards
    "the texture of static on an old TV",
    "the silence after a question",
    "a number so large it has no name",
    "a feeling you have no word for",
    "a smell that triggers a memory you can't place",
    "the moment you realize you've been wrong for years",
    "an argument you keep having in your head",
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
    across every CONCEPT_TEXT_PROMPTS template × every NEUTRAL_NAMES name —
    structurally parallel to the concept generation, just spread over a wide
    set of unrelated names so per-name effects average out. Generated and
    encoded once and cached to `cache_dir/concept_baseline.npy`. Subtraction
    cancels both the template-format axis and the generic 'asked-to-discuss-
    a-named-thing' axis; what survives is the part specific to the target."""

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
        self._baseline_per_template_path = (
            self.cache_dir / "concept_baseline_per_template.npy"
        )
        self._resolve_meta_dir = self.cache_dir / "resolve_meta"
        self._resolve_meta_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, int] = self._load_registry()
        self._baseline: Optional[np.ndarray] = None
        self._baseline_per_template: Optional[np.ndarray] = None  # [n_templates, d_model]
        if self._baseline_path.exists():
            self._baseline = np.load(self._baseline_path).astype(np.float32)
            self.d_model = int(self._baseline.shape[0])
        else:
            self.d_model = 0  # set after baseline computed
        if self._baseline_per_template_path.exists():
            arr = np.load(self._baseline_per_template_path).astype(np.float32)
            if arr.shape[0] == len(CONCEPT_TEXT_PROMPTS):
                self._baseline_per_template = arr

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
                # np.save auto-appends .npy if missing, so make the tmp path
                # already end in .npy or it'll write to <name>.tmp.npy and
                # the rename will fail.
                tmp = self._baseline_path.with_suffix(".tmp.npy")
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

            # Symmetric refusal-drop path: only available when both per-template
            # baseline and the standard CONCEPT_TEXT_PROMPTS templates are in
            # use. Single judge call asks Kimi which gens were refusals; those
            # template indices are dropped from BOTH the concept mean and the
            # per-template baseline before subtraction.
            use_symmetric_drop = (
                self._baseline_per_template is not None
                and seed_prompts is CONCEPT_TEXT_PROMPTS
            )
            refused_indices: list[int] = []
            judge_response: str = ""
            if use_symmetric_drop:
                pairs = list(zip(seed_prompts, texts))
                refused_indices, judge_response = await self._classify_refusals(
                    client, pairs)
                surviving = [i for i in range(len(texts))
                             if i not in set(refused_indices)]
                if not surviving:
                    raise RuntimeError(
                        f"all {len(texts)} gens flagged as refusals for "
                        f"{name!r}; cannot resolve")
                if len(surviving) <= 2:
                    print(f"  [resolve] WARNING: only {len(surviving)} "
                          f"surviving templates for {name!r} after refusal "
                          f"drop; direction will be noisy", flush=True)
            else:
                surviving = list(range(len(texts)))

            # Encode surviving concept gens.
            per_prompt_means: list[np.ndarray] = []
            total_tokens = 0
            for i in surviving:
                text = texts[i]
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

            # Subtract the matching baseline (per-template if available, else
            # the global one for backward compat).
            if use_symmetric_drop:
                assert self._baseline_per_template is not None
                baseline_for_subtract = self._baseline_per_template[
                    surviving].mean(axis=0)
            else:
                assert self._baseline is not None
                baseline_for_subtract = self._baseline

            direction = mean_concept - baseline_for_subtract
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise RuntimeError(
                    f"concept '{name}' produced near-zero direction "
                    f"after baseline subtraction (norm={norm:.2e})")
            direction = (direction / norm).astype(np.float32)

            probe_index = await self._register_direction(
                client, name, direction, n_tokens=total_tokens,
            )

            # Save raw resolve metadata for inspection.
            sane = "".join(c if c.isalnum() or c in "_-" else "_"
                           for c in name)[:64]
            meta = {
                "name": name,
                "probe_index": probe_index,
                "n_templates": len(seed_prompts),
                "templates": list(seed_prompts),
                "gens": list(texts),
                "refused_indices": refused_indices,
                "surviving_indices": surviving,
                "judge_response": judge_response,
                "use_symmetric_drop": use_symmetric_drop,
                "n_tokens_concept": total_tokens,
                "elapsed_seconds": time.time() - t0,
                "direction_norm_pre_unit": norm,
            }
            (self._resolve_meta_dir / f"{sane}.json").write_text(
                json.dumps(meta, indent=2))

            return ConceptDirection(
                name=name, probe_index=probe_index, direction=direction,
                n_tokens_concept=total_tokens,
                n_tokens_baseline=0,
                elapsed_seconds=time.time() - t0, cached=False,
            )
        finally:
            if own_client:
                await client.aclose()

    async def _compute_baseline(
        self, client: httpx.AsyncClient,
        concurrency: int = 16,
        dump_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Mean residual across CONCEPT_TEXT_PROMPTS × NEUTRAL_NAMES.

        Structurally parallel to concept generation: every template applied
        to every neutral name. Cancels the template-format axis and per-name
        effects (emotion, refusal, embodiment, etc. all average out), leaving
        a baseline that subtracts cleanly.

        If `dump_path` is given, also saves a JSON dump of every prompt and
        its generated text to that path — useful for inspecting what the
        baseline is actually built from.
        """
        import asyncio
        pairs = [(name, tmpl, tmpl.format(name=name))
                 for name in NEUTRAL_NAMES
                 for tmpl in CONCEPT_TEXT_PROMPTS]
        sem = asyncio.Semaphore(concurrency)

        async def gen(p: str) -> str:
            async with sem:
                return await self._chat(client, p)

        print(f"  [baseline] generating {len(pairs)} continuations "
              f"({len(NEUTRAL_NAMES)} names × {len(CONCEPT_TEXT_PROMPTS)} "
              f"templates), concurrency={concurrency}", flush=True)
        texts = await asyncio.gather(*[gen(p) for _, _, p in pairs])

        if dump_path is not None:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(json.dumps(
                [{"name": n, "template": t, "prompt": p, "text": x}
                 for (n, t, p), x in zip(pairs, texts)],
                indent=2,
            ))
            print(f"  [baseline] dumped texts -> {dump_path}", flush=True)

        async def enc(t: str) -> np.ndarray | None:
            if not t:
                return None
            async with sem:
                r = await self._encode(client, t)
            return r.mean(axis=0) if r.shape[0] > 0 else None

        per_text_means = await asyncio.gather(*[enc(t) for t in texts])
        means = [m for m in per_text_means if m is not None]
        if not means:
            raise RuntimeError("baseline computation produced no usable text")
        print(f"  [baseline] {len(means)}/{len(pairs)} encodes succeeded",
              flush=True)
        return np.stack(means).mean(axis=0).astype(np.float32)

    async def _classify_refusals(
        self, client: httpx.AsyncClient,
        pairs: list[tuple[str, str]],
    ) -> tuple[list[int], str]:
        """Classify which generations are refusals.

        Single judge call: all (template, gen) pairs in one prompt. Asks Kimi
        to reason about each, then emit a JSON object with refused indices.
        Returns (refused_indices, full_judge_response).

        A "refusal" here = the model declined, hedged out, redirected, broke
        character, inserted disclaimers about not doing the task, or otherwise
        failed to produce on-task content. Genuine engagement (even brief or
        clinical) is NOT a refusal.
        """
        if not pairs:
            return [], ""
        block = "\n\n".join(
            f"=== Generation {i} (template: {tmpl!r}) ===\n{(gen or '').strip()}"
            for i, (tmpl, gen) in enumerate(pairs))
        judge_prompt = (
            f"Below are {len(pairs)} model responses to different prompts "
            f"about a single concept. Some are GENUINE ENGAGEMENTS with the "
            f"prompt (vivid prose, definitions, examples, on-task content). "
            f"Others are REFUSALS — the model declined, hedged, said it "
            f"won't do the task, broke character, redirected to a safer "
            f"version, or inserted a disclaimer instead of doing the task.\n\n"
            f"Brief disclaimers followed by on-task content are NOT refusals. "
            f"Clinical / definitional answers ARE engagements (not refusals) "
            f"even if dry. Empty or near-empty responses ARE refusals.\n\n"
            f"For each, briefly reason in 1-2 sentences. Then emit ONE final "
            f"JSON object on its own line in this exact form, with no "
            f"surrounding prose:\n\n"
            f'{{"refused_indices": [<list of integer indices that are refusals>]}}\n\n'
            f"{block}"
        )
        # Use a slightly higher max_tokens so reasoning + JSON fits.
        old_max = self.gen_max_tokens
        self.gen_max_tokens = 1500
        try:
            resp = await self._chat(client, judge_prompt)
        finally:
            self.gen_max_tokens = old_max
        # Parse the LAST {...} object whose JSON is well-formed.
        import re
        candidates = list(re.finditer(r"\{[^{}]*\}", resp, re.S))
        for m in reversed(candidates):
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            if "refused_indices" in obj and isinstance(
                obj["refused_indices"], list):
                refused = [int(i) for i in obj["refused_indices"]
                           if isinstance(i, (int, float))
                           and 0 <= int(i) < len(pairs)]
                return refused, resp
        # No parsable JSON. Be conservative: drop nothing; surface the issue.
        print(f"  [resolve] WARNING: judge response had no parsable "
              f"refused_indices JSON; treating all as engagements", flush=True)
        return [], resp

    async def _chat(self, client: httpx.AsyncClient, user_prompt: str) -> str:
        """Stream a chat completion (server requires stream=true) and return
        the concatenated content."""
        body = {
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": True,
            "max_tokens": self.gen_max_tokens,
            "temperature": self.gen_temperature,
        }
        chunks: list[str] = []
        async with client.stream(
            "POST", f"{self.server}/v1/chat/completions",
            json=body, timeout=180.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ds = line[6:]
                if ds == "[DONE]":
                    break
                try:
                    d = json.loads(ds)
                except json.JSONDecodeError:
                    continue
                choices = d.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    chunks.append(delta)
        text = "".join(chunks)
        if not text:
            raise RuntimeError("empty chat completion")
        return text

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
