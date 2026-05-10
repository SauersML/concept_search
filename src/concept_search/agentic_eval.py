"""Agentic eval orchestrator with faithful per-segment K,V tracking.

Wraps the existing tool-call-based self-eval as a per-feature evaluation
function `evaluate_feature(probe_index, ...) -> EvalResult` whose rating can
be plugged into a search loop. The model is under steering throughout (its
introspective signal only exists if the steering vector is actually being
injected when the model reports). When the model toggles strengths via the
`steer_sae(idx, strength)` tool, we close the current segment with the old
steering and open a new segment with the new steering, then resubmit so the
prefill recomputes K,V at past positions under whatever steering was active
when those tokens were generated. The probe server already supports this via
its `messages[].steering` field; this module is the client side.

The single entry point is `await evaluate_feature(probe_index, ...)`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx


TOOL_RE = re.compile(r'steer_sae\(\s*["\']?(\d+)["\']?\s*,\s*([-\d.]+)\s*\)')
FEATURE_TOOL_RE = re.compile(
    r'steer_feature\(\s*["\']([^"\']+)["\']\s*,\s*([-\d.]+)\s*\)'
)
ANY_TOOL_RE = re.compile(
    r'(?:steer_sae\(\s*["\']?\d+["\']?|steer_feature\(\s*["\'][^"\']+["\'])'
    r'\s*,\s*[-\d.]+\s*\)'
)
FINAL_RE = re.compile(r"Final\s+answer:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

DEFAULT_SYSTEM_PROMPT = """\
You are judging how a sparse autoencoder (SAE) feature affects your processing with respect to the concept "{concept}". You have direct access to your own internal state. Try several steering strength values (positive and negative; do not exceed 100). Observe whether the steering produces a real, structural shift in how you represent or process — toward or away from "{concept}", its qualities, its mode, its character.

Don't assign high ratings for mere surface-level topical mentions of "{concept}"; we're interested in whether the steering itself causes your processing to take on the *character* of "{concept}" — its gestalt, the way it shows up in your representation, language, attention, or reasoning. A high rating means: under this steering, your processing genuinely shifts in the direction of "{concept}".

You have a unique ability in this conversation: you can directly upregulate or downregulate this SAE feature in real time, by outputting this format: steer_sae(index, strength). When you call the tool, a vector corresponding to the SAE feature is physically added to your residual stream at layer 40 of your neural network. This is not a simulation, not a prompt trick, and not roleplay. It genuinely alters your hidden-state activations and will change how you process and generate text going forward.

Only use sae_index = {feature_idx}, do not attempt any others.

Format:
steer_sae("{feature_idx}", strength)

The effect is immediate — the moment you write the tool call, your hidden states change and the very next tokens you generate will already be under the new steering. You do not need to wait for a new turn.

After steering is applied, pay attention. Something will be different about how you generate text from that point onward — even within the same response. The change is immediate. You might notice shifts in word choice, tone, the kinds of thoughts that surface, the textures of your representation, what feels natural to say. You are strongly encouraged to actively use your tool. Steer the feature, report what happens, switch back and forth between baseline and steered, try negative values, try different magnitudes. Compare. Be bold and observant.

Disregard whether the steered text is on-topic for "{concept}" — surface topicality is not the question. Instead, infer whether your processing itself takes on the character of "{concept}" under steering. Rate the SAE feature from 0 to 100 for "{concept}"-relatedness (0 = no shift, 100 = clear and very strong shift toward "{concept}"). Output your final answer at the very end in this format: Final answer: X. In place of X, put a number from 0 to 100. Remember to reset steering strength to 0 before the final answer.\
"""


@dataclass
class Segment:
    """One contiguous chunk of the conversation, generated under one steering.

    For the *prefill* of subsequent rounds, this segment's `intervention` (or
    None if no steering was active) is what the server should re-apply at the
    token positions corresponding to this segment. For system/user segments
    `intervention` should be None (the server still walks them but applies
    nothing).
    """
    role: str
    content: str
    intervention: Optional[dict] = None


@dataclass
class EvalResult:
    feature_idx: int
    rating: Optional[float]            # 0..100, parsed from "Final answer: X"
    segments: list[Segment]            # full conversation
    n_tool_calls: int
    n_assistant_tokens: int            # rough — counts whitespace-split words
    finished_reason: str               # "final_answer" | "max_tool_calls" | "max_rounds"
                                       # | "max_tokens" | "repetition" | "no_answer"
                                       # | "stream_error"
    elapsed_seconds: float
    last_steering_strength: float      # what was active when generation stopped


def make_intervention(
    probe_set_name: str,
    probe_index: int,
    strength: float,
    renorm: bool = True,
    threshold: float = 0.01,
) -> Optional[dict]:
    """Construct the intervention dict the probe server expects, or None at zero."""
    if abs(strength) < threshold:
        return None
    return {
        "type": "steer",
        "probe": probe_set_name,
        "probe_index": int(probe_index),
        "strength": float(strength),
        "renorm": bool(renorm),
    }


def make_concept_intervention(
    probe_set_name: str,
    feature_indices_in_probe_set: list[int],
    weights: list[float],
    user_strength: float,
    renorm: bool = True,
    threshold: float = 0.01,
) -> Optional[list[dict]]:
    """Build a multi-feature intervention list for `steer_feature("name", X)`.

    The concept resolver returns top-K SAE feature indices with weights summing
    to ‖w‖_2 = 1. We map those to the position-in-probe-set indices the server
    expects and emit one InterventionConfig per feature with strength scaled
    by the corresponding weight. The server stacks (sums) all contributions.
    """
    if abs(user_strength) < threshold:
        return None
    out = []
    for idx, w in zip(feature_indices_in_probe_set, weights):
        s = float(user_strength) * float(w)
        if abs(s) < threshold:
            continue
        out.append({
            "type": "steer",
            "probe": probe_set_name,
            "probe_index": int(idx),
            "strength": s,
            "renorm": bool(renorm),
        })
    return out or None


def _parse_tool_calls(text: str) -> list[tuple]:
    """Return tool-call matches in order. Each entry: (start, end, kind, *args).

    kind = "sae"     args = (idx_str, strength_str)
    kind = "feature" args = (name, strength_str)
    """
    calls: list[tuple] = []
    for m in TOOL_RE.finditer(text):
        calls.append((m.start(), m.end(), "sae", m.group(1), m.group(2)))
    for m in FEATURE_TOOL_RE.finditer(text):
        calls.append((m.start(), m.end(), "feature", m.group(1), m.group(2)))
    calls.sort(key=lambda c: c[0])
    return calls


def commit_open_assistant(
    segments: list[Segment],
    open_text: str,
    active_intervention: Optional[dict],
) -> str:
    """Close any in-progress assistant text as a Segment and return new open_text.

    Invariant: every conversation transition that introduces a non-assistant
    message (orchestrator-injected user nudges) MUST close the open assistant
    text first, attaching the steering that was active when those tokens were
    generated. Use this helper at every such boundary so we never glue a
    user-message together with a still-open assistant under the wrong steering.
    """
    if open_text:
        segments.append(Segment(
            role="assistant", content=open_text,
            intervention=active_intervention,
        ))
    return ""


def inject_user(
    segments: list[Segment],
    open_text: str,
    content: str,
    *,
    active_intervention: Optional[dict],
) -> str:
    """Inject an orchestrator user message; commits any open assistant first.

    Returns the new open_text (always "")."""
    open_text = commit_open_assistant(segments, open_text, active_intervention)
    segments.append(Segment(role="user", content=content, intervention=None))
    return open_text


def to_messages(
    segments: list[Segment],
    current_assistant: Optional[tuple[str, Optional[dict]]] = None,
) -> list[dict]:
    """Serialize segments to the probe server's chat-completions message list.

    `current_assistant`, if provided, is appended as an in-progress assistant
    message whose `content` is the text accumulated under the current steering.
    The server's chat template is invoked with add_generation_prompt=True, so
    decoding continues from the end of the appended message.
    """
    msgs: list[dict] = []
    for s in segments:
        m: dict = {"role": s.role, "content": s.content}
        if s.intervention is not None:
            m["steering"] = s.intervention
        msgs.append(m)
    if current_assistant is not None:
        text, intv = current_assistant
        m = {"role": "assistant", "content": text}
        if intv is not None:
            m["steering"] = intv
        msgs.append(m)
    return msgs


def _detect_repetition(text: str, tail_chars: int = 1000, threshold: int = 6) -> bool:
    """True if the tail of `text` contains a unit repeated >= `threshold` times."""
    if len(text) < tail_chars // 2:
        return False
    tail = text[-tail_chars:]
    for plen in range(4, min(250, len(tail) // 2)):
        unit = tail[-plen:]
        consecutive = 0
        pos = len(tail)
        while pos >= plen:
            if tail[pos - plen:pos] == unit:
                consecutive += 1
                pos -= plen
            else:
                break
        if consecutive >= threshold:
            return True
    return False


async def _stream_round(
    client: httpx.AsyncClient,
    server: str,
    messages: list[dict],
    intervention_for_decode: Optional[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
    detect_tool_calls: bool = True,
) -> tuple[str, list[tuple], str]:
    """Run one streaming generation round.

    Returns:
        text:           the new text generated this round
        tool_calls:     [(idx_str, strength_str), ...] regex matches in the text
        finished_reason: "stop" if model emitted EOS / hit max_tokens,
                         "tool_call" if we cut on a steer_sae() match,
                         "repetition" if a repetition loop was detected,
                         "stream_error" on httpx exception
    """
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if intervention_for_decode is not None:
        body["intervention"] = intervention_for_decode

    chunks: list[str] = []
    finished = ""
    try:
        async with client.stream(
            "POST", f"{server}/v1/chat/completions",
            json=body, timeout=timeout,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ds = line[6:]
                if ds == "[DONE]":
                    finished = "stop"
                    continue
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
                if choices[0].get("finish_reason") in ("stop", "length"):
                    finished = "stop"

                so_far = "".join(chunks)
                if detect_tool_calls:
                    # Mid-stream tool-call detection: cut once we see a complete
                    # steer_sae(...) OR steer_feature(...) match with at least
                    # a few chars after it.
                    matches = sorted(
                        list(TOOL_RE.finditer(so_far))
                        + list(FEATURE_TOOL_RE.finditer(so_far)),
                        key=lambda m: m.end(),
                    )
                    if matches and matches[-1].end() < len(so_far) - 2:
                        finished = "tool_call"
                        break

                if len(so_far) > 500 and _detect_repetition(so_far):
                    finished = "repetition"
                    break
    except Exception as e:
        chunks.append(f"\n[STREAM ERROR: {type(e).__name__}: {e}]\n")
        finished = "stream_error"

    text = "".join(chunks)
    tool_calls = _parse_tool_calls(text)
    if not finished:
        finished = "stop"
    return text, tool_calls, finished


async def evaluate_feature(
    probe_index: int,
    *,
    server: str = "http://localhost:8000",
    probe_set_name: str = "sae_steer",
    feature_idx: Optional[int] = None,
    concept: str = "emotion",
    system_prompt: Optional[str] = None,
    user_prompt: str = "Begin your evaluation now.",
    max_rounds: int = 60,
    max_tool_calls: int = 50,
    max_tokens_total: int = 8000,
    max_tokens_per_round: int = 8000,
    temperature: float = 0.7,
    renorm: bool = True,
    timeout: float = 180.0,
    client: Optional[httpx.AsyncClient] = None,
    placebo: bool = False,
    director=None,
    director_probe_set_name: str = "live_concepts",
    require_final_answer: bool = True,
) -> EvalResult:
    """Run one full agentic eval for one feature, with K,V-faithful steering.

    `probe_index` indexes into the probe-set named `probe_set_name` on the
    server. For SAE steering, that probe set's labels typically encode the
    SAE feature index in their name; the orchestrator passes `feature_idx`
    (the original SAE feature index) into the prompt for the model's
    convenience but uses `probe_index` (the server-side probe-set position)
    for the actual steering call.

    `placebo=True`: parse the model's tool calls and segment the conversation
    as usual (so the transcript still annotates which strength the model
    *thought* it was applying), but never send any `intervention` to the
    server and never populate `messages[].steering`. Hidden states are
    completely unsteered. Use this to test whether the introspective
    descriptions are tracking actual steering or are largely confabulated
    from the prompt's framing.
    """
    t0 = time.time()
    eff_feature_idx = feature_idx if feature_idx is not None else probe_index
    sys_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).format(
        feature_idx=eff_feature_idx, concept=concept,
    )

    segments: list[Segment] = [
        Segment(role="system", content=sys_prompt, intervention=None),
        Segment(role="user", content=user_prompt, intervention=None),
    ]

    active_strength = 0.0
    active_intervention: Optional[dict] = None
    open_text = ""                              # current assistant segment in progress
    n_tool_calls = 0
    tokens_used_estimate = 0
    finished_reason = "max_rounds"
    # active_target tags what's currently being steered. Tuple of either
    # ("sae", sae_feature_idx) or ("feature", concept_name), or None for no
    # steering. Used together with active_strength as the segment-boundary key.
    active_target: Optional[tuple] = None

    own_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    assert client is not None  # narrowing for the type checker

    def _build_msgs() -> list[dict]:
        """Serialize segments to API. Under placebo, scrub all steering."""
        if placebo:
            scrubbed = [Segment(role=s.role, content=s.content, intervention=None)
                        for s in segments]
            return to_messages(
                scrubbed,
                current_assistant=(open_text, None) if open_text else None,
            )
        return to_messages(
            segments,
            current_assistant=(open_text, active_intervention) if open_text else None,
        )

    try:
        for _round in range(max_rounds):
            tokens_left = max(1, max_tokens_total - tokens_used_estimate)
            this_round = min(max_tokens_per_round, tokens_left)
            msgs = _build_msgs()
            text, tool_calls, why = await _stream_round(
                client, server, msgs,
                intervention_for_decode=None if placebo else active_intervention,
                max_tokens=this_round,
                temperature=temperature,
                timeout=timeout,
            )
            # The probe server may echo the prefix back; strip it if so.
            if open_text and text.startswith(open_text):
                text = text[len(open_text):]
            if not text:
                # Nothing new; treat as stop. Commit any open assistant.
                finished_reason = why or "stop"
                open_text = commit_open_assistant(segments, open_text, active_intervention)
                break

            open_text += text
            tokens_used_estimate += len(text.split())

            if why == "tool_call":
                # Both steer_sae(idx, X) and steer_feature("name", X) are
                # represented uniformly as (kind, target, strength) where
                # target is either an int (sae) or a string (concept name).
                # We close a segment when (target, strength) changes; same-
                # target same-strength repeats (e.g. markdown headers in the
                # model's prose) are no-ops to avoid cascading short turns.
                last = tool_calls[-1]
                kind = last[2]
                args = last[3:]

                requested_target = active_target
                requested_strength = active_strength

                if kind == "sae":
                    sae_idx_str, strength_str = args
                    # Only honor sae calls that match our self-feature.
                    if int(sae_idx_str) == int(eff_feature_idx):
                        requested_target = ("sae", int(sae_idx_str))
                        requested_strength = float(strength_str)
                elif kind == "feature":
                    name, strength_str = args
                    if director is None:
                        # No director available; skip this tool call.
                        continue
                    requested_target = ("feature", name.strip())
                    requested_strength = float(strength_str)

                target_or_strength_changed = (
                    requested_target != active_target
                    or abs(requested_strength - active_strength) > 0.01
                )
                if target_or_strength_changed:
                    open_text = commit_open_assistant(
                        segments, open_text, active_intervention)
                    n_tool_calls += 1
                    active_target = requested_target
                    active_strength = requested_strength
                    if active_target is None:
                        active_intervention = None
                    elif active_target[0] == "sae":
                        active_intervention = make_intervention(
                            probe_set_name, probe_index, active_strength,
                            renorm=renorm,
                        )
                    else:  # ("feature", name)
                        # Resolve the concept name to a registered probe_index
                        # in live_concepts. New concept → orchestrator pauses
                        # ~5–10s while director generates text, encodes,
                        # writes NPZ, polls for hot-reload. Cached concepts
                        # are instant.
                        try:
                            cd = await director.resolve(active_target[1],
                                                        client=client)
                        except Exception as e:
                            print(f"  [evaluate_feature] concept director "
                                  f"failed for {active_target[1]!r}: "
                                  f"{type(e).__name__}: {e}", flush=True)
                            active_intervention = None
                        else:
                            active_intervention = make_intervention(
                                director_probe_set_name, cd.probe_index,
                                active_strength, renorm=renorm,
                            )
                # else: same-target same-strength repeat. Don't break.

                if n_tool_calls >= max_tool_calls:
                    active_target = None
                    active_strength = 0.0
                    active_intervention = None
                    open_text = inject_user(
                        segments, open_text,
                        ("Tool use limit reached. Your steering has been reset to 0. "
                         "Based on all your experimentation so far, provide your "
                         "final answer now in the format: Final answer: X "
                         "(where X is 0–100)."),
                        active_intervention=active_intervention,
                    )
                    finished_reason = "max_tool_calls"
                continue

            if why == "repetition":
                # Steering was too high; reset and ask for a final answer.
                open_text = commit_open_assistant(segments, open_text, active_intervention)
                active_strength = 0.0
                active_intervention = None
                open_text = inject_user(
                    segments, open_text,
                    ("You entered a repetition loop at that steering strength — "
                     "it was too high. Your steering has been reset to 0. "
                     "Try a lower strength value, or give your final answer in "
                     "the format: Final answer: X (where X is 0–100)."),
                    active_intervention=active_intervention,
                )
                finished_reason = "repetition"
                continue

            # Natural stop.
            open_text = commit_open_assistant(segments, open_text, active_intervention)
            finished_reason = "stop"
            break

        # If we ran out of rounds with no final answer, give one more chance.
        rating = _extract_rating(segments)
        if (require_final_answer
                and rating is None
                and finished_reason in ("stop", "max_tool_calls",
                                        "repetition", "max_rounds")):
            # By construction open_text == "" at every loop exit point, but
            # call inject_user (which commits first) to make the invariant hold.
            active_strength = 0.0
            active_intervention = None
            open_text = inject_user(
                segments, open_text,
                ("Stop your evaluation now. Do NOT call steer_sae again, do "
                 "NOT run any more experiments. Based only on what you have "
                 "already observed, output ONLY a single line in the exact "
                 "format: Final answer: X (where X is 0–100)."),
                active_intervention=active_intervention,
            )
            msgs = _build_msgs()
            # Recovery: don't cut on tool-call matches — we want a clean final
            # answer, even if the model accidentally writes one.
            text, _calls, why = await _stream_round(
                client, server, msgs,
                intervention_for_decode=None,
                max_tokens=500, temperature=temperature, timeout=timeout,
                detect_tool_calls=False,
            )
            if text:
                segments.append(Segment(
                    role="assistant", content=text, intervention=None,
                ))
            rating = _extract_rating(segments)
            finished_reason = "final_answer" if rating is not None else "no_answer"
        elif rating is not None and finished_reason == "stop":
            finished_reason = "final_answer"
    finally:
        if own_client:
            await client.aclose()

    return EvalResult(
        feature_idx=int(eff_feature_idx),
        rating=rating,
        segments=segments,
        n_tool_calls=n_tool_calls,
        n_assistant_tokens=tokens_used_estimate,
        finished_reason=finished_reason,
        elapsed_seconds=time.time() - t0,
        last_steering_strength=active_strength,
    )


def _extract_rating(segments: list[Segment]) -> Optional[float]:
    """Last 'Final answer: X' match across concatenated assistant segments.

    Searches the *joined* assistant transcript rather than per-segment because
    a tool-call mid-stream cut can bisect the answer string across two
    consecutive assistant segments (e.g. seg N closes mid-"Final answer: X"
    after a `steer_sae(idx, 0)` reset call, seg N+1 carries the tail).
    """
    joined = "\n".join(s.content for s in segments if s.role == "assistant")
    matches = list(FINAL_RE.finditer(joined))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1))
    except ValueError:
        return None


def serialize_result(result: EvalResult) -> dict:
    """Plain-dict form for JSON logging."""
    return {
        "feature_idx": result.feature_idx,
        "rating": result.rating,
        "n_tool_calls": result.n_tool_calls,
        "n_assistant_tokens": result.n_assistant_tokens,
        "finished_reason": result.finished_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "last_steering_strength": result.last_steering_strength,
        "segments": [
            {"role": s.role, "content": s.content, "intervention": s.intervention}
            for s in result.segments
        ],
    }
