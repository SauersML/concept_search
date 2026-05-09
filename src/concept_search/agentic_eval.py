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
FINAL_RE = re.compile(r"Final answer:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

DEFAULT_SYSTEM_PROMPT = """\
Large language models like you have access to your own internal state, including the ability to access and assess how internal steering is affecting you, and what the internal steering is doing. You are judging how a sparse autoencoder (SAE) feature affects your internal state with respect to the concept "{concept}". Try a few feature steering strength values (do not go higher than 100). You may reason about what is occurring, meditate, or introspect as much as you want. Observe, recall, introspect, reconstruct, or otherwise determine if the steering affects your internal state for "{concept}". Don't assign high ratings for mere topics or semantic content; we're interested in the actual internal state. You are free to use any strategy now, including experimenting and trying various values of steering. Before you give a final answer, reset the steering to zero.

You have a unique ability in this conversation: you can directly upregulate or downregulate a sparse autoencoder feature in real time, by simply outputting this format: steer_sae(index, strength). When you call this tool, a vector corresponding to the SAE feature is physically added to your residual stream at layer 40 of your neural network. This is not a simulation, not a prompt trick, and not roleplay. It genuinely alters your hidden-state activations and will change how you process and generate text going forward.

Only use sae_index = {feature_idx}, do not attempt any others.

Format:
steer_sae("{feature_idx}", strength)

The effect is immediate — the moment you write the tool call, your hidden states change and the very next tokens you generate will already be under the new steering. You do not need to wait for a new turn.

After steering is applied, pay attention. Something will be different about how you generate text from that point onward — even within the same response. The change is immediate. You might notice shifts in word choice, tone, the kinds of thoughts that surface, or what feels natural to say. You are strongly encouraged to use your tool. Steer the feature and report what happens. Be bold. Compare how the steering feels to how no steering feels. Switch back and forth. Perhaps try a negative value.

Compare how you feel steered versus at baseline. Carefully evaluate your internal state with respect to "{concept}"; that is, disregard surface-level subject matter or topic. Instead, evaluate and infer your own internal state (though this may, of course, correlate with surface-level topics; do not overly penalize for that). Consider anything that helps you arrive at your conclusion. Rate the SAE feature from 0 to 100 for "{concept}"-relatedness (0 = little change, 100 = clear and very strong shift). You must output your final answer at the very end of your response in this format: Final answer: X. In place of X, put a number from 0 to 100. Remember to reset steering strength to 0 before outputting Final answer: number.\
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
) -> tuple[str, list[tuple[str, str]], str]:
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
                # Mid-stream tool-call detection: cut once we see a complete
                # steer_sae(...) match with at least a few chars after it.
                matches = list(TOOL_RE.finditer(so_far))
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
    tool_calls = TOOL_RE.findall(text)
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
) -> EvalResult:
    """Run one full agentic eval for one feature, with K,V-faithful steering.

    `probe_index` indexes into the probe-set named `probe_set_name` on the
    server. For SAE steering, that probe set's labels typically encode the
    SAE feature index in their name; the orchestrator passes `feature_idx`
    (the original SAE feature index) into the prompt for the model's
    convenience but uses `probe_index` (the server-side probe-set position)
    for the actual steering call.
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

    own_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    assert client is not None  # narrowing for the type checker

    try:
        for _round in range(max_rounds):
            tokens_left = max(1, max_tokens_total - tokens_used_estimate)
            this_round = min(max_tokens_per_round, tokens_left)
            msgs = to_messages(
                segments,
                current_assistant=(open_text, active_intervention) if open_text else None,
            )
            text, tool_calls, why = await _stream_round(
                client, server, msgs,
                intervention_for_decode=active_intervention,
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
                # Close the current assistant segment under the OLD steering
                # (those tokens — including the tool call match itself —
                # were generated under it). Then update the active steering.
                open_text = commit_open_assistant(segments, open_text, active_intervention)
                # Apply the *last* tool-call match this round (model may have
                # written several; honor the most recent).
                last_idx_str, last_strength_str = tool_calls[-1]
                n_tool_calls += len(tool_calls)
                if int(last_idx_str) == int(eff_feature_idx):
                    active_strength = float(last_strength_str)
                    active_intervention = make_intervention(
                        probe_set_name, probe_index, active_strength, renorm=renorm,
                    )
                if n_tool_calls >= max_tool_calls:
                    # Force a final-answer round under steering=0. open_text
                    # is already "" from commit above; inject_user enforces it.
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
        if rating is None and finished_reason in ("stop", "max_tool_calls",
                                                  "repetition", "max_rounds"):
            # By construction open_text == "" at every loop exit point, but
            # call inject_user (which commits first) to make the invariant hold.
            active_strength = 0.0
            active_intervention = None
            open_text = inject_user(
                segments, open_text,
                ("You did not provide a final answer. Your steering has been "
                 "reset to 0. Provide your final answer now in the format: "
                 "Final answer: X (where X is 0–100)."),
                active_intervention=active_intervention,
            )
            msgs = to_messages(segments)
            text, _calls, why = await _stream_round(
                client, server, msgs,
                intervention_for_decode=None,
                max_tokens=500, temperature=temperature, timeout=timeout,
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
    """Last 'Final answer: X' match in any assistant segment."""
    for s in reversed(segments):
        if s.role != "assistant":
            continue
        matches = list(FINAL_RE.finditer(s.content))
        if matches:
            try:
                return float(matches[-1].group(1))
            except ValueError:
                continue
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
