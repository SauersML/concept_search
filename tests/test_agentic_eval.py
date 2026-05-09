"""Structural tests for agentic_eval — message serialization, parsing, repetition.

These do not hit a live probe server; they verify that the orchestrator's
*payloads* are exactly what the server's compute_steering_ranges expects, so
historical steering replays correctly during prefill.
"""

from __future__ import annotations

from concept_search.agentic_eval import (
    FINAL_RE,
    TOOL_RE,
    Segment,
    _detect_repetition,
    _extract_rating,
    commit_open_assistant,
    inject_user,
    make_intervention,
    to_messages,
)


def test_make_intervention_threshold():
    assert make_intervention("sae_steer", 0, 0.0) is None
    assert make_intervention("sae_steer", 0, 0.005) is None
    iv = make_intervention("sae_steer", 5, 50.0, renorm=True)
    assert iv == {
        "type": "steer", "probe": "sae_steer", "probe_index": 5,
        "strength": 50.0, "renorm": True,
    }


def test_to_messages_omits_steering_when_none():
    segs = [
        Segment(role="system", content="sys"),
        Segment(role="user", content="usr"),
        Segment(role="assistant", content="part1"),
    ]
    msgs = to_messages(segs)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "usr"}
    assert msgs[2] == {"role": "assistant", "content": "part1"}
    assert "steering" not in msgs[2]


def test_to_messages_per_segment_steering():
    iv50 = make_intervention("sae_steer", 5, 50.0)
    iv_neg = make_intervention("sae_steer", 5, -25.0)
    segs = [
        Segment(role="system", content="sys"),
        Segment(role="user", content="usr"),
        Segment(role="assistant", content="zero", intervention=None),
        Segment(role="assistant", content="up", intervention=iv50),
        Segment(role="assistant", content="down", intervention=iv_neg),
    ]
    msgs = to_messages(segs)
    assert msgs[2].get("steering") is None or "steering" not in msgs[2]
    assert msgs[3]["steering"]["strength"] == 50.0
    assert msgs[4]["steering"]["strength"] == -25.0


def test_to_messages_open_assistant_attaches_current_steering():
    iv = make_intervention("sae_steer", 5, 75.0)
    segs = [
        Segment(role="system", content="sys"),
        Segment(role="user", content="usr"),
        Segment(role="assistant", content="closed1", intervention=None),
    ]
    msgs = to_messages(segs, current_assistant=("partial", iv))
    assert len(msgs) == 4
    assert msgs[-1] == {"role": "assistant", "content": "partial",
                        "steering": iv}


def test_to_messages_open_assistant_no_steering_passed_through():
    segs = [Segment(role="system", content="sys"),
            Segment(role="user", content="usr")]
    msgs = to_messages(segs, current_assistant=("hello", None))
    assert msgs[-1] == {"role": "assistant", "content": "hello"}
    assert "steering" not in msgs[-1]


def test_tool_re_finds_calls():
    text = "I'll try steer_sae(123, 50) and then steer_sae(\"123\", -10.5)"
    matches = TOOL_RE.findall(text)
    assert matches == [("123", "50"), ("123", "-10.5")]


def test_final_re_finds_rating():
    text = "...stuff...\n\nFinal answer: 42\n"
    m = FINAL_RE.search(text)
    assert m is not None
    assert float(m.group(1)) == 42.0


def test_detect_repetition_positive():
    text = "different prelude text " + ("loop loop loop loop loop loop loop " * 3) * 5
    assert _detect_repetition(text)


def test_detect_repetition_negative():
    text = "the cat sat on the mat. " * 2 + "varied content about other things. " * 5
    assert not _detect_repetition(text)


def test_extract_rating_picks_last():
    segs = [
        Segment(role="system", content="…"),
        Segment(role="user", content="…"),
        Segment(role="assistant", content="…Final answer: 30"),
        Segment(role="user", content="please answer again"),
        Segment(role="assistant", content="…Final answer: 75"),
    ]
    assert _extract_rating(segs) == 75.0


def test_commit_open_assistant_empty_is_noop():
    segs = [Segment(role="system", content="s")]
    new_open = commit_open_assistant(segs, "", None)
    assert new_open == ""
    assert len(segs) == 1


def test_commit_open_assistant_attaches_intervention():
    iv = make_intervention("sae_steer", 5, 50.0)
    segs: list = []
    new_open = commit_open_assistant(segs, "partial under +50", iv)
    assert new_open == ""
    assert len(segs) == 1
    assert segs[0].role == "assistant"
    assert segs[0].content == "partial under +50"
    assert segs[0].intervention == iv


def test_inject_user_commits_open_assistant_first():
    """Boundary invariant: when the orchestrator injects a user message after
    an in-progress assistant has accumulated text under steering, the open
    text MUST close as a segment under that steering before the user message
    appears in the conversation."""
    iv = make_intervention("sae_steer", 5, 50.0)
    segs = [
        Segment(role="system", content="sys"),
        Segment(role="user", content="usr"),
    ]
    open_text = "I notice X under +50"
    open_text = inject_user(segs, open_text, "Tool limit reached.",
                            active_intervention=iv)
    assert open_text == ""
    assert len(segs) == 4
    assert segs[2].role == "assistant"
    assert segs[2].content == "I notice X under +50"
    assert segs[2].intervention == iv         # closed under OLD steering
    assert segs[3].role == "user"
    assert segs[3].content == "Tool limit reached."
    assert segs[3].intervention is None       # user message has no steering


def test_inject_user_with_no_open_text_just_appends_user():
    segs = [
        Segment(role="system", content="sys"),
        Segment(role="user", content="usr"),
        Segment(role="assistant", content="closed", intervention=None),
    ]
    open_text = inject_user(segs, "", "next nudge", active_intervention=None)
    assert open_text == ""
    assert len(segs) == 4
    assert segs[3].role == "user"
    assert segs[3].content == "next nudge"


def test_extract_rating_none_when_missing():
    segs = [
        Segment(role="system", content="…"),
        Segment(role="user", content="…"),
        Segment(role="assistant", content="no answer here"),
    ]
    assert _extract_rating(segs) is None


def test_segment_round_trip_three_rounds():
    """Trace a 3-round conversation; verify per-message steering ends up correct.

    This is the central K,V-faithfulness check: the server's
    compute_steering_ranges walks `messages` linearly, so the steering attached
    to each message is exactly what gets applied to its token range during
    prefill. Confirm we attach the right thing per round.
    """
    iv50 = make_intervention("sae_steer", 5, 50.0)

    # Round 1: model writes under steering=0, ends with a tool call.
    r1 = [
        Segment(role="system", content="sys"),
        Segment(role="user", content="usr"),
        Segment(role="assistant", content="trying: steer_sae(5, 50)",
                intervention=None),
    ]
    msgs1 = to_messages(r1)
    # Last message has no steering field (since intervention=None).
    assert "steering" not in msgs1[-1]

    # Round 2 setup: append second assistant segment (under +50), with new tool reset.
    r2 = r1 + [
        Segment(role="assistant", content="I notice X. steer_sae(5, 0)",
                intervention=iv50),
    ]
    msgs2 = to_messages(r2)
    assert msgs2[-1]["steering"] == iv50
    # Prior assistant still has no steering.
    assert "steering" not in msgs2[-2]

    # Round 3 setup: ask for final answer under steering=0.
    r3 = r2 + [
        Segment(role="user",
                content="provide your final answer now.",
                intervention=None),
    ]
    msgs3 = to_messages(r3, current_assistant=("Final answer: ", None))
    # Three message-level steering states distinct: None, +50, None, None, None
    assert "steering" not in msgs3[2]   # closed seg under 0
    assert msgs3[3]["steering"] == iv50  # closed seg under +50
    assert "steering" not in msgs3[4]   # user nudge
    assert "steering" not in msgs3[5]   # open final answer
