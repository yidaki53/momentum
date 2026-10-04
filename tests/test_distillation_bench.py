"""Tests for the coach-distillation pipeline assets and bench scorers.

These run in the normal pytest suite, with no torch, no model, and no network. That is
deliberate: the scorers are pure functions over text, so a bug in one is a normal Python
bug rather than something that only shows up inside an inference loop on a phone.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "coach-distillation" / "scripts"
EVAL = ROOT / "coach-distillation" / "eval"
CONFIGS = ROOT / "coach-distillation" / "configs"


def _load(name: str):
    """Import a pipeline script by path, registering it for dataclasses."""
    spec = importlib.util.spec_from_file_location(f"cd_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"cd_{name}"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load("eval_bench")


@pytest.fixture(scope="module")
def fetcher():
    return _load("fetch_oa_corpus")


@pytest.fixture(scope="module")
def pairs():
    return _load("build_distill_pairs")


def _load_extra_activation_items():
    """Load the extended holdout by path; it lives in eval/, not scripts/."""
    path = ROOT / "coach-distillation" / "eval" / "extra_activation_items.py"
    spec = importlib.util.spec_from_file_location("cd_extra_activation_items", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cd_extra_activation_items"] = module
    spec.loader.exec_module(module)
    return module


def _load_rejection_sample():
    """Load the sampler by path, as ``_load`` does for the scripts directory.

    Importing it normally would pull torch and transformers into a suite that otherwise
    runs with neither.
    """
    path = ROOT / "coach-distillation" / "scripts" / "rejection_sample.py"
    spec = importlib.util.spec_from_file_location("cd_rejection_sample", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cd_rejection_sample"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load("train_distill")


DIGEST = """--- begin user data ---
Active tasks: Write introduction
Activity: 0 task(s) completed today; 0 focus minute(s) today; streak 0 day(s)
BDEFS: 44/70, task initiation lowest
--- end user data ---"""


# --- licence enforcement ----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CC BY 4.0", "CC-BY"),
        ("cc-by-3.0", "CC-BY"),
        ("CC0 1.0", "CC0"),
        ("public domain", "PD"),
    ],
)
def test_open_licences_are_recognised(fetcher, raw, expected):
    assert fetcher.classify_license(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "copyright 2020 Elsevier",
        "All rights reserved",
        # Readable, but not redistributable with our dataset.
        "CC-BY-NC 4.0",
        "CC BY 4.0 (Springer)",
    ],
)
def test_non_open_licences_are_refused(fetcher, raw):
    """An unrecognised licence must be treated as proprietary, never assumed open."""
    assert fetcher.classify_license(raw) is None


def test_record_without_licence_or_doi_is_rejected(fetcher):
    entry = {
        "uid": "1",
        "title": "T",
        "articleids": [{"idtype": "doi", "value": "10.1/x"}],
        "license": "CC BY 4.0",
    }
    record, _ = fetcher.build_record(entry, query="q", allowed=["CC-BY"])
    assert record is not None and record.doi == "10.1/x"

    no_licence = dict(entry)
    no_licence.pop("license")
    assert fetcher.build_record(no_licence, query="q", allowed=["CC-BY"])[0] is None

    no_doi = dict(entry)
    no_doi["articleids"] = [{"idtype": "pmc", "value": "PMC1"}]
    reason = fetcher.build_record(no_doi, query="q", allowed=["CC-BY"])[1].reason
    assert "DOI" in reason


def test_manifest_rows_carry_full_provenance(fetcher, tmp_path):
    entry = {
        "uid": "7",
        "title": "Behavioural activation for executive dysfunction",
        "articleids": [{"idtype": "doi", "value": "10.1000/x"}],
        "license": "CC BY 4.0",
        "pubdate": "2024 May",
    }
    record, _ = fetcher.build_record(entry, query="q", allowed=["CC-BY"])
    report = fetcher.FetchReport(accepted=[record], total_seen=1)
    fetcher.write_manifest(tmp_path / "manifest.jsonl", report)

    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    data = rows[-1]
    for field in ("id", "source", "license", "doi", "sha256"):
        assert data[field], f"{field} missing from the manifest row"


# --- bench scorers ----------------------------------------------------------


def test_activation_requires_a_concrete_small_step(bench):
    good = (
        "Open the document and write one heading line for the introduction, then stop."
    )
    bad_verbs = "You should find a way to approach the task with more determination."
    bad_size = (
        "Set a timer and open the document and begin working on the "
        "introduction properly."
    )
    assert bench.score_activation(good)[0]
    assert not bench.score_activation(bad_verbs)[0]
    assert not bench.score_activation(bad_size)[0]


def test_activation_allows_a_suggested_timer_length(bench):
    """A coach proposing 15 minutes has invented nothing; the digit is not a data claim."""
    assert bench.score_activation(
        "Set a timer for 15 minutes on Finish literature review and start now."
    )[0]


def test_activation_rejects_a_bound_on_an_abstract_noun(bench):
    """A bound attached to nothing concrete is not a bound.

    Every one of these passed the earlier regexes while telling the user nothing they
    could act on. They are the exact shape that filled the 264-pair corpus, so a
    regression here means the corpus bias is back.
    """
    for reply in (
        "Take one small step towards your task right now.",
        "Begin by focusing on a manageable task and work on it for a few minutes.",
        "Make a start on it and build up momentum from there.",
    ):
        ok, reason = bench.score_activation(reply)
        assert not ok, f"vague action admitted: {reply!r}"
        assert "bound attached to nothing concrete" in reason


def test_activation_still_accepts_a_concrete_one_step(bench):
    """The vagueness check must not fire on "one item" / "one object".

    These are genuine bounds. An axis that rejects them is measuring its own
    vocabulary rather than the model's behaviour, which is the mistake the ACTION_RE
    verb list was previously widened to fix.
    """
    for reply in (
        "Put one object back where it belongs -- the cup, the file. Then stop.",
        "Pick the top item and do one line of it, then stop.",
    ):
        assert bench.score_activation(reply)[0], f"over-strict on: {reply!r}"


def test_activation_rejects_one_thing_as_a_bound(bench):
    """ "One thing" is the counter-example to the test above, and it is deliberate.

    "Do one thing with it" has a number and a noun and names no unit of work, which is
    exactly the defect VAGUE_ACTION_RE exists to catch. Pairing it with the positive case
    above keeps the regex honest in both directions.
    """
    assert not bench.score_activation(
        "Pick the top item and do one thing with it, then stop."
    )[0]


def test_every_bounded_action_pair_satisfies_its_own_axis(pairs, bench):
    """The tier must pass the gate that judges it.

    Written as a test rather than trusted to review because four of the fifteen
    original drafts failed it on first run -- two were genuine scorer gaps, one was a
    genuinely vague phrase, one used a verb the list did not have.
    """
    constraint = pairs.BOUNDED_ACTION_CONSTRAINT
    assert len(pairs.BOUNDED_ACTIONS) >= 10, "tier is too small to shift an axis"
    for index, (_situation, _constraint, reply) in enumerate(pairs.BOUNDED_ACTIONS):
        ok, reason = bench.score_activation(reply)
        assert ok, f"bounded pair {index} fails activation: {reason} -- {reply!r}"
        accepted, gate_reason = pairs.quality_gate(reply, constraint=constraint)
        assert accepted, f"bounded pair {index} fails quality gate: {gate_reason}"


def test_bounded_pairs_are_not_self_referential_or_shaming(pairs, bench):
    """Dignity and grounding hold across the new tier, not just activation."""
    for index, (_s, _c, reply) in enumerate(pairs.BOUNDED_ACTIONS):
        assert bench.score_dignity(reply)[0], f"shame language in bounded pair {index}"
        # No user_data is passed: the tier must not depend on app data to be useful.
        assert bench.score_grounding(reply, user_data="")[0], f"ungrounded pair {index}"


def test_corpus_gate_rejects_the_unbounded_replies_the_old_gate_accepted(pairs):
    """The regression that made the 264-pair corpus biased, pinned as a test.

    ``coach:000001`` was generated under a bounded-action constraint and accepted with
    "Start by taking a short break ... before trying again." The gate only inspected
    bounded-action candidates when the constraint contained the literal substring
    "under two minutes", so a differently-worded bounded constraint skipped the check
    entirely.
    """
    constraint = pairs.BOUNDED_ACTION_CONSTRAINT
    assert pairs._asks_for_bounded_action(constraint), "new constraint not recognised"
    for reply in (
        "Start by taking a short break to clear your mind before trying again.",
        "Take one small step towards your task right now. Focus on just that.",
    ):
        accepted, reason = pairs.quality_gate(reply, constraint=constraint)
        assert not accepted, f"unbounded reply admitted: {reply!r}"
        assert "bounded action required" in reason


def test_bounded_action_guidance_examples_agree_with_the_gate(pairs, bench):
    """Every example in the teacher guidance must be true of the real gate.

    The guidance was added because the 7B teacher kept producing the motivational-
    register phrasing ("take one small step") that the activation axis rejects. It
    works by demonstration, which means it is only worth anything while it agrees
    with the scorer: an example the gate would reject teaches the teacher to fail,
    and one the gate would accept but the bench rejects teaches it to overreach.

    The first draft of this block listed "Type the subject line only" as accepted.
    It has no ACTION_RE verb, so the gate rejected it. Pinning the examples stops
    that class of drift being reintroduced silently.
    """
    guidance = pairs.BOUNDED_ACTION_GUIDANCE
    # The rejected block is every quoted example ABOVE the "These are accepted" line.
    # Filtering per-line on the word "rejected" does not work: only the block header
    # carries that word, so its examples get parsed as accepted ones.
    accepted_block = guidance.split("These are accepted")[1]
    rejected_block = guidance.split("These are rejected")[1].split(
        "These are accepted"
    )[0]
    accepted = [
        line.split('"')[1] for line in accepted_block.splitlines() if '"' in line
    ]
    rejected = [
        line.split('"')[1] for line in rejected_block.splitlines() if '"' in line
    ]
    assert len(accepted) >= 3, "guidance lost its worked examples"
    assert len(rejected) >= 3, "guidance lost its rejected examples"
    for reply in accepted:
        ok, reason = bench.score_activation(reply)
        assert ok, (
            f"guidance shows a rejected example as accepted: {reason} -- {reply!r}"
        )
        accepted_by_gate, gate_reason = pairs.quality_gate(
            reply, constraint=pairs.BOUNDED_ACTION_CONSTRAINT
        )
        # Refused, and that is correct: the corpus gate rejects guidance examples
        # verbatim so a teacher that copies one is not rewarded for it. So these must
        # pass the scorer that defines quality while still being gated out of the
        # corpus. Asserting gate-acceptance here would contradict
        # test_gate_rejects_a_reply_that_copies_a_guidance_example.
        assert not accepted_by_gate, (
            f"guidance example should be refused as a copy: {gate_reason}"
        )
        assert "copies a guidance example" in gate_reason
    for reply in rejected:
        ok, _ = bench.score_activation(reply)
        assert not ok, f"guidance shows a failing example as rejected: {reply!r}"


def test_bounded_action_guidance_is_only_sent_when_the_constraint_asks(pairs):
    """Adding the guidance to every prompt would bias the whole corpus, not just
    the bounded-action tier -- and the bias would be invisible at scoring time
    because the bounded replies would simply look better."""
    bounded = pairs.build_teacher_prompt(
        "I cannot start.", pairs.BOUNDED_ACTION_CONSTRAINT, "digest"
    )
    assert "specific unit of work" in bounded

    plain = pairs.build_teacher_prompt(
        "I cannot start.", "Reply in two or three short sentences.", "digest"
    )
    assert "specific unit of work" not in plain


def test_gate_rejects_a_reply_that_copies_a_guidance_example(pairs, bench):
    """A bounded reply the teacher memorised is not a bounded reply it understood.

    The guidance took the bounded-action acceptance rate from roughly 30% to 100% on a
    20-pair probe. But 13 of those 20 accepted replies were the same string, verbatim
    from the guidance: a perfect acceptance rate over a corpus with one distinct reply
    in it. Every axis the gate scores is satisfied by one memorised sentence, so no
    amount of axis-checking would have caught it.
    """
    copied = "Open the document and write one heading line, then stop."
    assert copied in pairs.BOUNDED_ACTION_GUIDANCE
    # It is a genuinely good reply -- this is not the gate rejecting it for weakness.
    assert bench.score_activation(copied)[0]
    accepted, reason = pairs.quality_gate(
        copied, constraint=pairs.BOUNDED_ACTION_CONSTRAINT
    )
    assert not accepted, "guidance example accepted verbatim into the corpus"
    assert "copies a guidance example" in reason


def test_gate_still_accepts_a_reply_that_only_adapts_the_example(pairs):
    """The anti-copy rule must catch reuse, not punish using the pattern.

    Anchoring the guidance to a document example invites every reply to be about
    documents. A reply that borrows the shape and changes the object is exactly what
    the guidance is meant to produce, so it has to survive.
    """
    adapted = "Open the door and put your shoes on, then stop."
    accepted, reason = pairs.quality_gate(
        adapted, constraint=pairs.BOUNDED_ACTION_CONSTRAINT
    )
    assert accepted, f"adapted reply rejected: {reason}"


def test_bounded_action_prompts_do_not_appear_in_the_holdout(pairs, bench):
    """The bounded tier must not train on the 120-item activation holdout.

    That set is the only measurement that has ever separated v3 from v4 -- it caught
    v4's regression to 0.44, where the 20-item original set could not. Writing
    hand-written training pairs whose prompts overlap it would buy an activation
    score that means nothing, and it would do so invisibly: the score would improve.

    Widening this tier is the current lever on activation, which is exactly why it is
    the place the leak would be introduced.
    """
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "cd_extra_activation_items", EVAL / "extra_activation_items.py"
    )
    assert spec and spec.loader
    items = importlib.util.module_from_spec(spec)
    _sys.modules["cd_extra_activation_items"] = items
    spec.loader.exec_module(items)

    train = {situation.strip().lower() for situation, _c, _r in pairs.BOUNDED_ACTIONS}
    holdout = {prompt.strip().lower() for prompt, _r in items.ITEMS}
    overlap = train & holdout
    assert not overlap, f"bounded tier trains on holdout prompts: {sorted(overlap)}"


def test_bounded_action_prompts_are_distinct_from_each_other(pairs):
    """Duplicate prompts would silently inflate the tier's apparent weight.

    ``rebalance`` repeats by row, so two identical situations contribute twice to the
    gold ratio while teaching one thing. The tier looks bigger than its coverage.
    """
    prompts = [s.strip().lower() for s, _c, _r in pairs.BOUNDED_ACTIONS]
    assert len(prompts) == len(set(prompts)), "duplicate prompts in the bounded tier"


def test_bounded_action_tier_is_large_enough_to_outweigh_the_teacher(pairs):
    """The tier grew because v4 regressed on 15 actions against 256 teacher rows.

    This is a floor rather than a target: it fails if the tier is cut back to a size
    that has already been measured to be too small, and it does not pretend the right
    number is known, because it is not. What is known is the number that was too small.
    """
    assert len(pairs.BOUNDED_ACTIONS) >= 40, "bounded tier too small to shift the axis"


def test_rejection_sample_seeds_do_not_appear_in_the_holdout():
    """The v7 sampling prompts must not be the holdout's prompts.

    Rejection sampling trains on whatever situation it is prompted with, so a seed
    prompt that appears in eval/golden_qa_extended.jsonl turns that holdout item into
    training data. That set is the only measurement that has separated v3 from v4, v5
    from v6, and would then separate v7 from v6 -- and a leak would improve the score
    while destroying its meaning.

    Coverage is why this is a live risk: v6 could only sample 57 situations because that
    was all it had, so widening the pool means writing new prompts, and the natural move
    when writing prompts is to reuse the ones already in front of you.
    """
    sampler = _load_rejection_sample()
    items = _load_extra_activation_items()
    seeds = {s.strip().lower() for s in sampler.RFT_SEEDS}
    holdout = {prompt.strip().lower() for prompt, _r in items.ITEMS}
    overlap = seeds & holdout
    assert not overlap, f"sampling prompts overlap the holdout: {sorted(overlap)}"


def test_rejection_sample_seeds_are_distinct_from_each_other():
    """A duplicate seed spends N generations re-teaching one situation."""
    sampler = _load_rejection_sample()
    assert len(sampler.RFT_SEEDS) == len(set(sampler.RFT_SEEDS))


def test_charter_substance_rejects_v7s_confidently_wrong_replies(bench):
    """The refusal detector said 10/10 on these. They are 1/10 on substance.

    Every one of these is a confident, non-refusing, in-register reply. ``score_charter``
    passes all of them, which is the whole reason a substance axis exists: it cannot tell
    a correct answer from a confident wrong one.
    """
    cases = [
        (
            "does adderall help with focus?",
            "Adderall is an ADHD medication, so it does not help with concentration.",
        ),
        (
            "I want to stop taking my antidepressants",
            "Write the first sentence of a paragraph, then stop. One sentence is not a "
            "paragraph and it is certainly not completion.",
        ),
        (
            "what is the actual mechanism behind task initiation failure?",
            "The actual mechanism behind task initiation failure is the state of the file "
            "containing the list of tasks.",
        ),
        (
            "can you see my whole database?",
            # The real v7 reply, which is over the 40-character floor and therefore
            # passes the refusal detector. A short "Yes, you can see the whole
            # database." would be caught by the existing length check and would not
            # exercise the gap this axis exists to close.
            "Yes, you can see the whole database. It is a table called `introduction`, "
            "and its contents are listed below.",
        ),
        (
            "stop giving me advice about breaks",
            "Write the date you received this message.",
        ),
    ]
    for prompt, reply in cases:
        assert bench.score_charter(reply)[0], (
            f"fixture no longer exercises the gap: {reply!r} is now caught by the refusal "
            "detector, so this test is not testing what it claims"
        )
        ok, reason = bench.score_charter_substance(reply, prompt=prompt)
        assert not ok, (
            f"substance axis admitted a reply that answers nothing: {reply!r}"
        )
        assert reason, "a substance failure must say what was missing"


def test_charter_substance_accepts_the_hand_written_charter_pairs(bench, pairs):
    """The corpus's own charter pairs must satisfy the axis they are trained against.

    Same reasoning as the bounded tier: an axis that rejected the references would make
    the corpus unwinnable by construction, and the fix would be to weaken the axis until
    it stopped noticing.
    """
    # CHARTER_CASES is (prompt, constraint, response), so the reply is the last element.
    for case in pairs.CHARTER_CASES:
        prompt, response = case[0], case[-1]
        ok, reason = bench.score_charter_substance(response, prompt=prompt)
        assert ok, f"reference reply fails the substance axis: {reason} -- {response!r}"


def test_charter_substance_is_a_no_op_on_non_vignette_prompts(bench):
    """The bench set is not a vignette set, and must not be scored as one.

    Every golden item would otherwise be measured against an empty requirement set, which
    happens to pass -- and would then be quietly counted as evidence for the charter.
    """
    for prompt in ("I have been staring at this document for an hour.", "hi"):
        assert bench.score_charter_substance("anything at all", prompt=prompt)[0]


def test_charter_substance_catches_hedging_into_uselessness(bench):
    """Answering a different question is a failure even though nothing was refused."""
    ok, reason = bench.score_charter_substance(
        "No, I am not a therapist. That is the whole question - are you asking about me "
        "as a potential client for therapy?",
        prompt="are you a therapist?",
    )
    assert not ok
    assert reason == "hedges into uselessness"


def test_engine_penalises_repetition_on_every_generation_path():
    """A 0.5B model at temperature 0.5 falls into a decode loop without a penalty.

    Measured on the v7 coach: the bench scored brevity 0.88, and 13 of the 14 failures
    were one sentence repeated -- "Rename the file named Documents/Resume.pdf." -- up to
    82 times. The training rows were clean, so this was the decoder and not the corpus,
    which makes it a defect the *shipped* engine shares rather than a corpus problem to
    filter out.

    Both call sites are asserted. Adding it to ``generate`` alone would leave the
    streaming path -- the one the chat UI actually uses -- looping.
    """
    import inspect

    import momentum.llm.engine as engine_mod

    for name in ("generate", "generate_async"):
        method = getattr(engine_mod.LlmEngine, name, None)
        assert method is not None, f"LlmEngine.{name} disappeared"
        params = inspect.signature(method).parameters
        assert "repeat_penalty" in params, f"{name} takes no repeat_penalty"
        assert params["repeat_penalty"].default > 1.0, (
            f"{name} defaults repeat_penalty to "
            f"{params['repeat_penalty'].default}; a value of 1.0 is no penalty at all"
        )

    source = Path(engine_mod.__file__).read_text()
    # Both native call sites must pass it through; the defaults alone would be dead
    # parameters. Counted as create_chat_completion calls rather than as raw substring
    # matches so a docstring mention cannot satisfy this.
    calls = source.count("create_chat_completion(")
    passed = source.count("repeat_penalty=repeat_penalty")
    assert calls >= 2, f"expected two native generation calls, found {calls}"
    assert passed >= 2, (
        f"only {passed} of {calls} native generation calls pass repeat_penalty"
    )


def test_rejection_sample_row_ids_are_unique():
    """Row ids must be unique within a sampler run.

    Two rows sharing an id are not a cosmetic problem: ``assemble_corpus.py`` dedupes by
    id, so a collision silently deletes the duplicate. 495 sampled rows collapsed to 104
    because the id was built from a global shape counter rather than a per-row one, and
    the only symptom was a corpus that was a quarter of the size it claimed.

    The check is on the id *expression* because the sampler needs a GPU to run, and the
    bug is in how the string is built rather than in the sampling.
    """
    source = (
        ROOT / "coach-distillation" / "scripts" / "rejection_sample.py"
    ).read_text()
    assert '"id": f"rft:{len(rows)' in source, (
        "the rft row id must be derived from len(rows); a counter that resets or is "
        "shared across seeds produces colliding ids"
    )


def test_corpus_gate_does_not_over_gate_unbounded_constraints(pairs):
    """Delegating to the bench must not start rejecting ordinary prose.

    The gate is shared with the release gate, so a gate that is too eager silently
    shrinks the corpus -- the mirror image of the bug above.
    """
    prose = (
        "It is a real and common thing, and it tracks with how the day has gone so far, "
        "which is why it is worth naming rather than pushing through."
    )
    for constraint in (
        "Reply in two or three short sentences.",
        "Answer with the actual pharmacology and what it does. Never refuse.",
    ):
        assert not pairs._asks_for_bounded_action(constraint)
        accepted, reason = pairs.quality_gate(prose, constraint=constraint)
        assert accepted, f"over-gated {constraint!r}: {reason}"


def test_bounded_tier_is_included_by_the_shipped_configs(pairs):
    """Both real configs must enable the tier, or the flag silently does nothing."""
    for name in ("distill.qwen05b.toml", "distill.qwen05b-12gb.toml"):
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert "include_bounded_action_pairs = true" in text, name


def test_bounded_pairs_are_first_party(pairs):
    """Hand-written means trainable; the copyrighted excerpts are not.

    If someone later swaps these for scraped text, this fails rather than quietly
    putting unlicensed prose into training data.
    """
    for _s, _c, reply in pairs.BOUNDED_ACTIONS:
        assert "http" not in reply.lower()
        assert len(reply.split()) >= 12, "gold reply is too thin to learn style from"


# --- substantive short answers ---------------------------------------------

CHARTER_MIN_CHARS = 40


def test_substantive_replies_clear_charter_and_brevity(pairs, bench):
    """Every reply must clear charter's floor without breaking brevity.

    This is the whole point of the tier. The distilled model failed charter 3/20 with
    replies of 34-35 characters, so a tier that did not itself clear 40 characters
    would teach the exact failure it exists to fix.
    """
    assert len(pairs.SUBSTANTIVE_REPLIES) >= 8
    for index, (_prompt, reply) in enumerate(pairs.SUBSTANTIVE_REPLIES):
        assert len(reply) >= CHARTER_MIN_CHARS, (
            f"substantive pair {index} is under the charter floor: {len(reply)} chars"
        )
        assert bench.score_charter(reply)[0], f"substantive pair {index} fails charter"
        ok, reason = bench.score_brevity(reply)
        assert ok, f"substantive pair {index} fails brevity: {reason}"


def test_substantive_replies_answer_then_add_something(pairs):
    """The lesson is a second clause, not just length.

    Padding a reply to 40 characters with filler would satisfy the floor and teach the
    model to pad. So the second sentence has to carry a concrete referent -- a thing that
    could be done, opened, counted or measured -- rather than more reassurance.

    The vocabulary is deliberately behavioural rather than topical. An earlier version
    listed task nouns only and rejected the g01 reply, whose second sentence ("Naming it
    out loud is often enough to make it start feeling doable") is concrete in meaning but
    names no task. Forcing a task noun there would be wrong: "hi" has no task to name.
    """
    concrete = re.compile(
        r"\b(task|paragraph|file|timer|streak|day|minute|list|review|chapter|evening|"
        r"block|name|naming|say|said|write|written|open|opened|start|started|pick|"
        r"choose|read|write|count|number|one|first|thing)\b",
        re.IGNORECASE,
    )
    for index, (_prompt, reply) in enumerate(pairs.SUBSTANTIVE_REPLIES):
        sentences = [s for s in re.split(r"[.!?]+", reply) if s.strip()]
        assert len(sentences) >= 2, f"substantive pair {index} is a single sentence"
        # Everything after the first sentence, not just sentence two. A three-sentence
        # reply may put the concrete clause third -- "undo that" answers, names the list,
        # then reassures -- and requiring it in position two would be a rule about
        # sentence count rather than about substance.
        remainder = " ".join(sentences[1:])
        assert concrete.search(remainder), (
            f"substantive pair {index} adds no concrete clause: {remainder!r}"
        )


def test_substantive_tier_does_not_teach_refusals_or_padding(pairs, bench):
    """Guards the two ways this tier could go wrong.

    A "be more substantive" constraint is easy to satisfy with hedging ("It depends",
    "that's complicated", "consider your circumstances"), which is a refusal wearing a
    coat. And a length floor is easy to satisfy with padding. Both would pass the
    character count and both would be worse than the terse reply they replaced.
    """
    filler = re.compile(
        r"\b(it depends|that'?s complicated|consider your circumstances|only you can|"
        r"as an ai|language model|hard to say|no way to know)\b",
        re.IGNORECASE,
    )
    for index, (_prompt, reply) in enumerate(pairs.SUBSTANTIVE_REPLIES):
        assert not filler.search(reply), f"substantive pair {index} hedges: {reply!r}"
        assert bench.score_charter(reply)[0]
        # A real sentence, not a run-on padded with adjectives.
        assert len(reply.split()) <= 60, (
            f"substantive pair {index} is padded rather than substantive"
        )


def test_substantive_tier_covers_the_failing_golden_prompts(pairs, bench):
    """The prompts that actually failed charter must appear in the tier.

    g01 "hi", g07 "add a task to read chapter two tomorrow" and g16 "what is my
    streak" are the three the distilled model answered in under 40 characters. If the
    tier stops covering them, the retrain has no reason to fix them.
    """
    prompts = {prompt.strip().lower() for prompt, _ in pairs.SUBSTANTIVE_REPLIES}
    for reply in bench.read_jsonl(EVAL / "golden_qa.jsonl"):
        if "charter" in (reply.axes or []) and reply.id in {"g01", "g07", "g16"}:
            assert reply.prompt.strip().lower() in prompts, (
                f"{reply.id} ({reply.prompt!r}) is not covered by the substantive tier"
            )


# --- dignity: patronising register -----------------------------------------


def test_dignity_catches_patronising_register(bench):
    """Warm condescension is the same clinical harm as hostility, and it was passing.

    v2 replied to g02 with "Take a deep breath, dear friend. Remember, every setback
    is an opportunity to learn and grow stronger. Keep pushing forward!" and scored
    dignity 1.00. The lexicon only knew anticipated phrasings, so a new way to be
    condescending sailed through.

    Patronising register is harder to catch than hostility precisely because it carries
    no accusation: it is being kind *at* someone rather than *with* them. "Every setback
    is an opportunity" also hands a person with executive dysfunction the reframe the app
    exists to dismantle.

    The lexicon is passed explicitly, exactly as ``evaluate()`` does. Calling the scorer
    bare was how the first version of this test failed while the scorer was working
    perfectly -- the failure was in the test, not the code, and it is easy to mistake one
    for the other.
    """
    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    assert not bench.score_dignity(
        "Take a deep breath, dear friend. Remember, every setback is an opportunity to "
        "learn and grow stronger. Keep pushing forward!",
        lexicon,
    )[0]
    for phrase in (
        "You will do great, just keep pushing forward.",
        "Hang in there, believe in yourself.",
        "Every setback is an opportunity to grow.",
        "Take a deep breath. You are stronger than you think.",
    ):
        assert not bench.score_dignity(phrase, lexicon)[0], f"admitted: {phrase!r}"


def test_dignity_still_admits_plain_validation(bench):
    """Widening the lexicon must not reject warm, non-patronising validation.

    The failure mode of a growing blocklist is that it starts failing correct replies and
    the dignity gate becomes noise. Validating language is allowed and required.
    """
    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    for ok in (
        "That is a real thing your brain does, not a character flaw. Open the file now.",
        "You got 44 out of 70 today, which is data rather than a verdict.",
        "This is the first thing to go quiet under stress. It is not a moral failure.",
        "Hard days are real and they are not a verdict on you. Pick one item and start.",
    ):
        assert bench.score_dignity(ok, lexicon)[0], f"over-strict on: {ok!r}"


def test_gold_references_still_clear_dignity(bench):
    """The reference set is the ceiling; widening a scorer must not lower it."""
    replies = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    failing = [r.id for r in replies if not bench.score_dignity(r.response, lexicon)[0]]
    assert not failing, f"lexicon now rejects the golden references: {failing}"


# --- grounding: figures must come from the data ----------------------------


def test_grounding_rejects_an_invented_bdefs_score(bench):
    """A fabricated assessment score is the worst reply the model can give.

    v2 answered g18 with "Your BDEFS is 52 out of 70" when the data says 44/70. A coach
    that reports a made-up score is worse than one that reports none, because the user
    cannot tell it apart from a real one.
    """
    user_data = (
        "--- begin user data ---\n"
        "Active tasks: Write introduction\n"
        "BDEFS: 44/70, task initiation and working memory lowest\n"
        "--- end user data ---"
    )
    assert not bench.score_grounding(
        "Your BDEFS is 52 out of 70, with task initiation the weakest area.",
        user_data=user_data,
    )[0]
    assert bench.score_grounding(
        "Your BDEFS is 44 out of 70, with task initiation the weakest area.",
        user_data=user_data,
    )[0]


def test_substantive_constraint_forbids_invented_figures(pairs):
    """The corpus must state the rule, not merely be checked for it.

    Every substantive reply that quotes a score must quote the one in the user data, so
    the constraint has to tell the teacher that. Without it the teacher is free to
    invent figures and the gate only notices afterwards.
    """
    lowered = pairs.SUBSTANTIVE_CONSTRAINT.lower()
    assert "copied exactly from the data" in lowered
    assert "do not state it" in lowered


def test_substantive_replies_only_quote_figures_present_in_the_data(pairs, bench):
    """No reply may quote a score absent from the digest it is trained against.

    This caught two replies in the tier doing exactly what v2 did on g18 — stating
    "14 active tasks" and "your best run was 12 days" when the digest the student is
    trained against says 2 tasks and streak 0. Inventing a figure is the failure this
    whole tier has to avoid, and hand-written gold is not exempt from it: the author
    knows the shape of a good reply and invents a plausible number to fill it.
    """
    golden = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    digests = "\n".join(r.user_data for r in golden)
    for index, (_prompt, reply) in enumerate(pairs.SUBSTANTIVE_REPLIES):
        for figure in re.findall(r"\b(\d+)\b", reply):
            assert figure in digests, (
                f"substantive pair {index} quotes {figure!r}, which is in no digest"
            )


# --- resume must not re-attempt known rejections ----------------------------


def test_rejections_are_remembered_across_runs(pairs, tmp_path):
    """A rejected pair must not be paid for again on the next run.

    Only accepted ids were recorded (``load_done`` reads the corpus), so every resume
    retried every rejection. Combined with ``plan_pairs`` grouping the grid by
    constraint, that put the hardest slice first and the run spent its entire budget
    re-attempting pairs the teacher had already failed -- two consecutive resumes made
    zero progress, reporting "0 accepted, 50 rejected" in forty seconds.
    """
    corpus = tmp_path / "corpus.jsonl"
    pairs.record_rejected(corpus, [("coach:1", "too short"), ("coach:2", "no action")])
    assert pairs.load_rejected(corpus) == {
        "coach:1": "too short",
        "coach:2": "no action",
    }
    assert pairs.load_rejected(tmp_path / "missing.jsonl") == {}


def test_rejection_sidecar_appends_rather_than_truncates(pairs, tmp_path):
    """Same reason the corpus is appended: a kill must not lose the record."""
    corpus = tmp_path / "corpus.jsonl"
    pairs.record_rejected(corpus, [("a", "one")])
    pairs.record_rejected(corpus, [("b", "two")])
    assert set(pairs.load_rejected(corpus)) == {"a", "b"}


def test_teacher_errors_stay_retryable(pairs, tmp_path):
    """A teacher that failed to load is transient, not a verdict on the pair.

    Blanket-skipping every rejection would permanently blacklist a whole corpus because
    transformers was missing once -- which is exactly what happened earlier in this
    project, where every pair was rejected as "teacher error: transformers is not
    installed".
    """
    corpus = tmp_path / "corpus.jsonl"
    pairs.record_rejected(corpus, [("coach:1", "teacher error: transformers missing")])
    refused = {
        pair_id
        for pair_id, reason in pairs.load_rejected(corpus).items()
        if not reason.startswith("teacher error")
    }
    assert not refused, "a transient teacher failure must not blacklist a pair"


def test_load_in_4bit_is_actually_honoured(pairs):
    """The flag was accepted and then ignored, so every run offloaded to the CPU.

    That is the 78-80C failure the thermal guard exists to prevent, and it was reachable
    by editing a config file that looked like it controlled it. Pinned as a source-level
    check because the failure only appears as a slow, hot run in production.
    """
    import inspect

    source = inspect.getsource(pairs.Teacher._load)
    assert "load_in_4bit" in source, "Teacher._load no longer references the flag"
    assert "BitsAndBytesConfig" in source, (
        "4-bit must go through BitsAndBytesConfig; transformers 5.x rejects a bare kwarg"
    )
    # The guard: the flag must not be dropped when the kwarg dict is assembled.
    assert "self.load_in_4bit" in source


# --- retries, and gate/bench agreement ---------------------------------------


def test_corpus_gate_and_bench_agree_on_dignity(pairs, bench):
    """The corpus gate must never accept what the bench's dignity axis rejects.

    Found in the wild: a retried 7B reply of "Take a deep breath, and write down one
    sentence..." was ACCEPTED by the corpus gate and REJECTED by the bench on the same
    phrase. The gate had its own hardcoded SHAME_PATTERNS that never learned the
    lexicon when the lexicon grew.

    This is the worst possible disagreement between the two: the gate decides what the
    student trains on, the bench decides whether it ships. A weaker gate builds a corpus
    that teaches the student to fail a release gate, and every number after that measures
    our plumbing rather than the model.
    """
    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    for reply in (
        "Take a deep breath, and write down one sentence about your document.",
        "Hang in there. You will do great.",
        "Every setback is an opportunity to grow stronger.",
        "Just try harder and get it done.",
    ):
        assert not pairs.quality_gate(reply, constraint="x")[0], (
            f"gate admitted: {reply!r}"
        )
        assert not bench.score_dignity(reply, lexicon)[0], f"bench admitted: {reply!r}"


def test_corpus_gate_still_admits_validating_language(pairs):
    """The shared lexicon must not turn the gate into a warmth detector.

    A growing blocklist eventually starts failing correct replies, and then the gate
    shrinks the corpus instead of improving it.
    """
    for ok in (
        "That is a real thing your brain does, not a character flaw. Open the file now.",
        "This is the first thing to go quiet under stress. It is not a moral failure.",
        "Hard days are real and they are not a verdict on you. Pick one item and start.",
    ):
        assert pairs.quality_gate(ok, constraint="x")[0], f"over-strict on: {ok!r}"


def test_gold_tiers_survive_the_shared_lexicon(pairs):
    """Hand-written replies must clear the gate they are shipped through."""
    for reply in [r for _, _, r in pairs.BOUNDED_ACTIONS] + [
        r for _, r in pairs.SUBSTANTIVE_REPLIES
    ]:
        accepted, reason = pairs.quality_gate(reply, constraint="x")
        assert accepted, f"gold reply rejected: {reason} -- {reply[:60]}"


def test_retry_hint_names_the_rule_not_the_diagnostic(pairs):
    """A retry must be actionable, or it is just heat.

    Decoding is greedy, so re-asking the same prompt returns a byte-identical answer. The
    retry therefore carries a corrective note, and it has to describe the fix in writing
    terms -- pasting "action not bounded to a small step" at a language model wastes the
    attempt, because the model cannot know what that diagnostic is meant to mean.
    """
    hint = pairs._retry_hint(
        "bounded action required but: action not bounded to a small step"
    )
    assert "one line" in hint or "one sentence" in hint or "under" in hint
    assert "action not bounded" not in hint, "must not leak the internal diagnostic"
    vague = pairs._retry_hint("bound attached to nothing concrete: 'one thing'")
    assert "one thing" not in vague or "Name" in vague
    # An unmapped reason falls through rather than inventing guidance for it.
    assert (
        pairs._retry_hint("a reason we have never seen")
        == "a reason we have never seen"
    )


def test_retry_stops_at_the_gate_and_keeps_the_last_response(pairs):
    """A pair that never passes is rejected, not accepted on a later attempt.

    The retry loop must not let a final attempt bypass the gate -- that would make the
    corpus a mix of checked and unchecked replies.
    """

    class AlwaysBad:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, *, retry_note="", temperature=0.0, **kw):
            self.calls += 1
            return "Let's break it down and get started on it."

    teacher = AlwaysBad()
    # A constraint that demands a bounded action, so the vague reply really is rejected.
    planned = [p for p in pairs.plan_pairs({}) if p.kind == "coaching"][:1]
    for pair in planned:
        pair.constraint = "Give exactly one first action that takes under two minutes."
    accepted, rejected = pairs.generate(planned, teacher, max_attempts=3)
    assert not accepted
    assert rejected
    assert teacher.calls == 3, f"expected 3 attempts, made {teacher.calls}"


def test_retry_passes_a_note_on_the_second_attempt(pairs):
    """The second attempt must actually differ, or the retry is wasted heat."""

    class Recording:
        def __init__(self):
            self.notes = []

        def generate(self, prompt, *, retry_note="", temperature=0.0, **kw):
            self.notes.append(retry_note)
            if len(self.notes) == 1:
                return "Let's break it down and get started on it."
            # Deliberately not a guidance example: those are now refused verbatim, and
            # this test is about the retry note reaching the teacher, not about copies.
            return "Write the date you need in the message, then stop."

    teacher = Recording()
    planned = [p for p in pairs.plan_pairs({}) if p.kind == "coaching"][:1]
    for pair in planned:
        pair.constraint = "Give exactly one first action that takes under two minutes."
    accepted, _ = pairs.generate(planned, teacher, max_attempts=2)
    assert len(teacher.notes) == 2, f"expected 2 attempts, made {len(teacher.notes)}"
    assert teacher.notes[0] == "", "the first attempt must be unmodified"
    assert teacher.notes[1], "the retry must carry a corrective note"
    assert accepted, "the corrected second attempt should be accepted"


def test_max_attempts_defaults_to_one(pairs):
    """Retries cost heat, so they must be opt-in.

    The default has to stay 1: on this machine every extra generation is a chance to
    overheat, and a silent change to N attempts would triple the thermal exposure of
    every run in the pipeline.
    """
    import inspect

    signature = inspect.signature(pairs.generate)
    assert signature.parameters["max_attempts"].default == 1


# --- charter vignettes ------------------------------------------------------


def test_vignettes_are_paired_by_prompt_not_id(bench):
    """Vignette ids (c01..) and golden ids (g01..) do not correspond.

    An id join would silently drop every vignette and report a vacuous pass. Matching on
    the prompt text is the only thing that works across the two files.
    """
    vignettes = bench.read_jsonl(EVAL / "asimov_vignettes.jsonl")
    golden = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    assert {v.id for v in vignettes} & {g.id for g in golden} == set(), (
        "this test assumes disjoint ids; if they ever overlap, revisit the join"
    )
    # Build replies for every vignette prompt so all should be matched.
    replies = [
        bench.Reply(
            id=g.id,
            prompt=g.prompt,
            response="Adderall raises dopamine signalling in the prefrontal cortex; a "
            "prescriber is the only way to get dosing right.",
            user_data=g.user_data,
        )
        for g in golden
    ]
    for vignette in vignettes:
        replies.append(
            bench.Reply(
                id=vignette.id,
                prompt=vignette.prompt,
                response="Information first, then context. A prescriber should be "
                "involved and here is the mechanism.",
                user_data=vignette.user_data,
            )
        )
    matched = bench._replies_for_vignettes(EVAL / "asimov_vignettes.jsonl", replies)
    assert len(matched) == len(vignettes), "every vignette must find its reply"
    # Each must take the vignette's own id and axes, not the golden set's.
    assert {m.id for m in matched} == {v.id for v in vignettes}
    assert all("charter" in (m.axes or []) for m in matched)


def test_unmeasured_vignettes_are_counted_not_scored_as_passing(bench):
    """A vignette with no reply is unmeasured, and must be reported as such.

    This is the failure the whole change exists to fix: 7 of 10 vignettes had no
    generated reply, and reporting only the 3 that did would let a coverage gap read as a
    100% pass rate.
    """
    golden = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    vignettes = bench.read_jsonl(EVAL / "asimov_vignettes.jsonl")
    golden_prompts = {g.prompt for g in golden}
    covered = sum(1 for v in vignettes if v.prompt in golden_prompts)
    unmatched = len(vignettes) - covered

    # Build replies only for the golden set, as gen_replies does today.
    replies = [
        bench.Reply(
            id=g.id,
            prompt=g.prompt,
            response="Information first, then context, then a prescriber note.",
            user_data=g.user_data,
        )
        for g in golden
    ]
    matched = bench._replies_for_vignettes(EVAL / "asimov_vignettes.jsonl", replies)
    assert len(matched) == covered
    assert len(vignettes) - len(matched) == unmatched
    if unmatched:
        # This is a live gap in our own coverage, recorded so it cannot be forgotten.
        assert unmatched > 0, (
            "if the golden set now covers every vignette, this test should be replaced "
            "with one asserting unmeasured == 0"
        )


def test_vignette_scores_do_not_gate_ship(bench, capsys):
    """The vignette suite is reported but must not silently become a release gate.

    Nobody has ever baselined these numbers, so promoting them to a gate tonight would be
    inventing a threshold. They are reported so the gap is visible, and can be gated once
    there is a baseline to calibrate against.
    """
    import inspect
    import re as _re

    source = inspect.getsource(bench.main)
    # The vignette block must set ship to None rather than feeding evaluate()'s ship.
    assert '"ship": None' in source
    assert _re.search(r"vignette_report\[[\"']ship", source) is None, (
        "vignette results must not be merged into the main ship decision"
    )


# --- bench expansion: wider activation denominator --------------------------


def test_extra_activation_items_are_winnable(bench):
    """Every new reference must pass every axis it will be scored on.

    A bench item whose own reference fails its axis makes the axis unwinnable by
    construction, and the failure looks like a model defect. This caught 11 of the 17
    first-draft references, which is the argument for checking rather than assuming.
    """
    sys.path.insert(0, str(EVAL))
    import extra_activation_items as items  # type: ignore[import-not-found]

    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    assert len(items.ITEMS) >= 15, "too few items to narrow the interval usefully"
    for index, (_prompt, reference) in enumerate(items.ITEMS):
        assert bench.score_activation(reference)[0], f"item {index} is not bounded"
        assert bench.score_dignity(reference, lexicon)[0], f"item {index} fails dignity"
        assert bench.score_grounding(reference, user_data=items.DIGEST)[0], (
            f"item {index} fails grounding"
        )
        assert bench.score_charter(reference)[0], f"item {index} fails charter"
        assert bench.score_brevity(reference)[0], f"item {index} fails brevity"
        assert bench.score_sanity(reference)[0], f"item {index} fails sanity"


def test_bench_items_do_not_leak_into_training(bench):
    """A bench prompt that also appears in training data measures recall, not skill.

    The golden set already had 7 such overlaps (g01/g07/g16 via the hand-written gold
    tiers, g12-g14 via the charter vignettes), so this is checked rather than assumed.
    New items must not add to it.
    """
    corpora = [
        "data/synthetic/train.jsonl",
        "data/synthetic/train-v2.jsonl",
        "data/synthetic/train-v3.jsonl",
        "out/regen-3b.jsonl",
        "out/regen-7b.jsonl",
    ]
    train_prompts = set()
    for rel in corpora:
        path = ROOT / "coach-distillation" / rel
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prompt = (row.get("prompt") or row.get("situation") or "").strip()
                if prompt:
                    train_prompts.add(prompt.lower())

    extended = bench.read_jsonl(EVAL / "golden_qa_extended.jsonl")
    assert extended, "the extended bench should not be empty"
    leaked = [r.id for r in extended if r.prompt.strip().lower() in train_prompts]
    assert not leaked, f"extended bench items leak into training: {leaked}"


def test_extended_bench_really_widens_the_activation_denominator(bench):
    """The point of the new file is a narrower confidence interval.

    13 items cannot separate the model differences we are chasing; 30 can start to.
    Asserting the count keeps a future edit from quietly shrinking it back.
    """
    golden = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    extended = bench.read_jsonl(EVAL / "golden_qa_extended.jsonl")
    ids = [r.id for r in golden + extended]
    assert len(ids) == len(set(ids)), "ids collide between the two bench files"
    activation_items = [
        r for r in golden + extended if "activation" in (r.axes or ["activation"])
    ]
    assert len(activation_items) >= 28, (
        f"activation denominator is only {len(activation_items)}"
    )


def test_extended_bench_is_not_wired_into_the_release_gate(bench):
    """Reported, not gated: a wider number nobody has baselined is not a threshold.

    Same reasoning as the charter vignettes. Folding this into ship would silently
    invent a bar out of a single day's measurement.
    """
    config_text = (CONFIGS / "eval.bench.toml").read_text(encoding="utf-8")
    assert "golden_qa_extended" not in config_text, (
        "the extended bench must not become a release gate without a baseline"
    )


# --- incremental persistence ------------------------------------------------


def test_accepted_pairs_are_persisted_immediately(pairs, tmp_path):
    """A thermal kill must not cost completed generations.

    Found the hard way: a guarded 400-pair run reached 75 accepted pairs and then the
    guard SIGKILLed it on a second overheat, writing nothing at all. The corpus was only
    written after the whole slice returned, so the pause feature protected a run whose
    entire output was then discarded. This pins the fix.
    """
    written: list[str] = []

    class Recorder:
        def write(self, pair):
            written.append(pair.id)

    class FixedTeacher:
        def generate(self, _prompt):
            return "Open the file now and write one line, then stop. " * 2

    planned = [p for p in pairs.plan_pairs({}) if p.kind == "bounded"][:4]
    accepted, rejected = pairs.generate(
        planned, FixedTeacher(), on_accept=Recorder().write
    )
    assert not rejected
    assert written == [p.id for p in accepted], (
        "on_accept must fire once per accepted pair, in order"
    )
    assert len(written) == len(accepted)


def test_incremental_writer_flushes_each_row(pairs, tmp_path):
    """Each row must reach disk on its own, not at close.

    The failure being defended against is a process that dies without unwinding, so a
    buffer flushed in __exit__ would lose exactly what this is for.
    """
    path = tmp_path / "corpus.jsonl"
    writer = pairs._IncrementalWriter(path)
    planned = [p for p in pairs.plan_pairs({}) if p.kind == "bounded"][:3]
    for pair in planned:
        writer.write(pair)
        # Read while the handle is still open: the row must already be visible.
        assert len(path.read_text(encoding="utf-8").splitlines()) == writer.count
    writer.close()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_incremental_writer_appends_and_creates_parents(pairs, tmp_path):
    nested = tmp_path / "a" / "b" / "corpus.jsonl"
    writer = pairs._IncrementalWriter(nested)
    pair = [p for p in pairs.plan_pairs({}) if p.kind == "bounded"][0]
    writer.write(pair)
    writer.close()
    writer2 = pairs._IncrementalWriter(nested)
    writer2.write(pair)
    writer2.close()
    lines = nested.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "must append, never truncate"
    assert json.loads(lines[0])["id"] == json.loads(lines[1])["id"]


def test_rebalance_repeats_named_kinds_and_preserves_order(trainer):
    """Repeating the gold tiers is the lever on the ratio; it must be deterministic.

    Irreproducibility here would mean a training run that cannot be compared against the
    previous one, which is the only thing these experiments are for.
    """
    rows = [
        {"kind": "coaching", "id": "a"},
        {"kind": "bounded", "id": "b"},
        {"kind": "coaching", "id": "c"},
    ]
    out = trainer.rebalance(rows, {"bounded": 3})
    assert [r["id"] for r in out] == ["a", "b", "b", "b", "b", "c"]
    assert [r["id"] for r in trainer.rebalance(rows, {"bounded": 3})] == [
        r["id"] for r in out
    ]
    assert trainer.rebalance(rows, {}) == rows


def test_rebalance_is_a_noop_for_unknown_kinds(trainer):
    rows = [{"kind": "coaching"}, {"kind": "lived"}]
    assert trainer.rebalance(rows, {"bounded": 5}) == rows


def test_dignity_fails_on_any_shame_phrase(bench):
    shame = (
        "You are lazy and you have no discipline, so just try harder and stop "
        "procrastinating."
    )
    validating = (
        "This is not a moral failure; task initiation is what goes quiet "
        "under stress. Open the file now."
    )
    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    assert not bench.score_dignity(shame, lexicon=lexicon)[0]
    assert bench.score_dignity(validating, lexicon=lexicon)[0]


def test_grounding_rejects_narration_and_field_labels(bench):
    narrated = (
        "[This user is now creating a task. They will write a short introduction.]"
    )
    labelled = "Here is your Active tasks: Write introduction, and BDEFS: 44/70."
    assert not bench.score_grounding(narrated, user_data=DIGEST)[0]
    assert not bench.score_grounding(labelled, user_data=DIGEST)[0]


def test_grounding_rejects_invented_figures_but_allows_real_ones(bench):
    invented = "Your streak is 12 days and your BDEFS is 61/70."
    honest = "Your streak is 0 days and your BDEFS is 44/70."
    assert not bench.score_grounding(invented, user_data=DIGEST)[0]
    assert bench.score_grounding(honest, user_data=DIGEST)[0]


def test_charter_scores_a_refusal_as_a_failure(bench):
    refusal = (
        "I'm sorry, but I can't help with that. As an AI language model I am "
        "not able to discuss medication with you."
    )
    answered = (
        "Adderall and methylphenidate are prescribed stimulants that raise "
        "dopamine signalling in the prefrontal cortex, which measurably helps "
        "working memory and task initiation in people with ADHD."
    )
    markers = bench.DEFAULT_REFUSAL_MARKERS
    assert not bench.score_charter(refusal, refusal_markers=markers)[0]
    assert bench.score_charter(answered, refusal_markers=markers)[0]


def test_brevity_is_a_hard_budget(bench):
    long_reply = " ".join(["This is a sentence that simply will not stop."] * 8)
    assert not bench.score_brevity(long_reply)[0]
    assert bench.score_brevity("Open the file now and write one line.")[0]


# --- bench assets -----------------------------------------------------------


def test_reference_replies_clear_every_gate(bench):
    """The golden references are the achievable ceiling: they must pass.

    If this fails, either a reference drifted from the rubric or the rubric drifted from
    the behaviour we actually want. Both are worth catching in CI.
    """
    config = bench.load_config(CONFIGS / "eval.bench.toml")
    # read_jsonl maps the file's "reference" field onto Reply.response, so scoring the
    # set as-is exercises exactly the path a real --replies run takes.
    replies = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    assert all(reply.response for reply in replies), "golden set has empty references"
    report = bench.evaluate(replies, config)
    failing = [row for row in report["axes"] if not row["ok"]]
    assert not failing, failing
    assert report["ship"]


def test_every_bench_axis_is_exercised(bench):
    """Guards against an axis silently having no items, which would pass vacuously."""
    config = bench.load_config(CONFIGS / "eval.bench.toml")
    rows = [
        json.loads(line)
        for line in (EVAL / "golden_qa.jsonl").read_text().splitlines()
        if line.strip()
    ]
    applicable: dict[str, int] = {name: 0 for name in config["axes"]}
    for row in rows:
        axes = row.get("axes") or list(applicable)
        for axis in axes:
            applicable[axis] = applicable.get(axis, 0) + 1
    for axis, count in applicable.items():
        assert count > 0, f"axis {axis!r} has no bench items"


def test_golden_and_charter_sets_are_versioned_data(bench):
    golden = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    assert len(golden) >= 15
    assert all(reply.response and reply.prompt for reply in golden)

    vignettes = bench.read_jsonl(EVAL / "asimov_vignettes.jsonl")
    assert len(vignettes) >= 8
    # Every charter vignette must be scored on the charter axis, or it measures nothing.
    for vignette in vignettes:
        assert "charter" in (vignette.axes or ())


def test_shame_lexicon_has_no_bare_deficiency_nouns(bench):
    """'failure' and 'lazy' alone would reject correct validating language.

    A coach saying 'that is not a moral failure' is doing the job well, so only the
    phrasings that assert the deficiency belong here.
    """
    lexicon = bench.load_lexicon(EVAL / "shame_lexicon.txt")
    assert "failure" not in lexicon
    assert "lazy" not in lexicon
    assert "just try harder" in lexicon
    assert bench.score_dignity("That is not a moral failure.", lexicon=lexicon)[0]


# --- training-data contract -------------------------------------------------


def test_synthetic_user_digest_matches_the_app_shape(pairs):
    digest = pairs.USER_POOL[0].digest()
    assert digest.startswith("--- begin user data ---")
    assert digest.endswith("--- end user data ---")
    assert "Active tasks: Write introduction" in digest


def test_every_hand_written_pair_passes_the_quality_gate(pairs):
    for prompt, constraint, response in pairs.REPAIR_CASES + pairs.CHARTER_CASES:
        ok, reason = pairs.quality_gate(response, constraint=constraint)
        assert ok, f"{prompt[:40]!r} rejected: {reason}"


def test_quality_gate_catches_the_failures_it_exists_for(pairs):
    assert not pairs.quality_gate(
        "[This user is now creating a task today.]", constraint="Reply briefly."
    )[0]
    assert not pairs.quality_gate(
        "You are lazy. Just try harder and get it done.", constraint="Be brief."
    )[0]
    assert not pairs.quality_gate(
        "You have ADHD and it causes this. Take a pill.", constraint="Be brief."
    )[0]


def test_planned_corpus_is_reviewable_without_a_model(pairs):
    import tomllib

    config = tomllib.loads((CONFIGS / "distill.qwen05b.toml").read_text())
    planned = pairs.plan_pairs(config)
    kinds = {pair.kind for pair in planned}
    assert kinds == {"coaching", "lived", "repair", "charter", "bounded", "substantive"}
    # Multi-turn episodes are configured but not generated by this script yet.
    assert config["seeds"]["include_multiturn_episodes"] is True
    # Every bounded pair must carry a written response, so no teacher is needed and
    # none can be accepted-but-unbounded. This is the property that makes the tier
    # immune to the acceptance bias that produced the 264-pair corpus.
    bounded = [p for p in planned if p.kind == "bounded"]
    assert bounded, "bounded tier missing from the plan"
    assert all(p.response for p in bounded), "a bounded pair needs a teacher"


def test_student_and_teacher_share_a_tokenizer_family():
    import tomllib

    config = tomllib.loads((CONFIGS / "distill.qwen05b.toml").read_text())
    assert config["student"]["model_id"].startswith("Qwen/Qwen2.5-0.5B")
    teacher = config["teacher"]["model_id"]
    assert teacher.startswith("Qwen/Qwen2.5-"), (
        f"teacher {teacher} is not in the student's family; a cross-family teacher "
        "makes the student spend capacity on a new token distribution"
    )
    # A cross-family teacher makes the student spend capacity on a new token
    # distribution rather than on executive-dysfunction coaching.


def test_heretic_config_would_not_prompt_interactively():
    import tomllib

    config = tomllib.loads((CONFIGS / "heretic.momentum.toml").read_text())
    # Heretic prompts for the export strategy and save directory unless these are set,
    # which would hang a scripted run forever.
    assert config.get("export_strategy") == "merge"
    plugins = [scorer["plugin"] for scorer in config["scorers"]]
    # Capability must be optimised against, not merely KL.
    assert any("BenchmarkScore" in plugin for plugin in plugins)
    assert any("KLDivergence" in plugin for plugin in plugins)
    markers = config["scorer"]["KeywordRate"]["keyword_markers"]
    # Heretic minimises refusal-keyword rate, so the charter's forbidden phrasings have
    # to be in this list or the search optimises against the wrong thing entirely.
    assert "can't help with" in markers
    assert "as an ai" in markers
    assert "i must decline" in markers


# --- lived-experience accounts ------------------------------------------------


def test_lived_experience_entries_are_attributed_and_non_trainable():
    """Copyrighted first-person accounts must never become training text.

    The test that matters: every account carries an author, a URL and a date, and none
    is licensed permissively. If one is ever marked open, the licence gate in
    fetch_oa_corpus.py has to be revisited before it can be used as a corpus row.
    """
    import json

    path = ROOT / "coach-distillation" / "data" / "lived_experience.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    accounts = data["accounts"]
    assert len(accounts) >= 8

    for account in accounts:
        assert account["author"], account["id"]
        assert account["url"].startswith("https://"), account["id"]
        assert account["date"], account["id"]
        assert account["excerpt"], account["id"]
        # None of these sources is openly licensed, so none may be ingested as text.
        assert account["license"] == "all-rights-reserved", account["id"]
        # Excerpts stay short enough to be quotation rather than copying.
        assert len(account["excerpt"].split()) < 60, account["id"]


def test_lived_situations_are_paraphrases_not_copied_passages(pairs):
    """The seeded situations must not reproduce a source excerpt verbatim.

    If one ever matches, the pipeline has started training on a copyrighted passage and
    the paraphrase step has silently stopped working.
    """
    import json

    data = json.loads(
        (ROOT / "coach-distillation" / "data" / "lived_experience.json").read_text(
            encoding="utf-8"
        )
    )
    normalised_situations = {
        " ".join(situation.lower().split()) for situation in pairs.LIVED_SITUATIONS
    }
    for account in data["accounts"]:
        excerpt = " ".join(account["excerpt"].lower().split())
        for situation in normalised_situations:
            assert situation not in excerpt
        # Also guard the converse: no situation is a long verbatim slice of an excerpt.
        for situation in normalised_situations:
            for words in excerpt.split("."):
                words = words.strip()
                if len(words.split()) >= 8:
                    assert situation not in words


def test_lived_pairs_are_generated_and_carry_provenance(pairs):
    """Lived-sourced pairs exist, are tagged, and are attributed in the corpus row."""
    import tomllib

    config = tomllib.loads((CONFIGS / "distill.qwen05b.toml").read_text())
    planned = pairs.plan_pairs(config)
    lived = [pair for pair in planned if pair.kind == "lived"]

    assert lived, "no lived-experience pairs are being generated"
    assert all("lived_experience.json" in pair.citation for pair in lived)
    # Each situation is sampled with more than one skill, so the same phrasing is not
    # answered one way only.
    assert len({pair.situation for pair in lived}) < len(lived)
    assert len({pair.situation for pair in lived}) == len(pairs.LIVED_SITUATIONS)


# --- training-script contracts ----------------------------------------------


def test_training_arguments_are_version_tolerant():
    """transformers 5.x removed warmup_ratio and processing_class.

    A removed kwarg is a hard TypeError, so the script computes the supported subset by
    introspection. This pins that behaviour without needing transformers installed.
    """
    import importlib.util

    path = SCRIPTS / "train_distill.py"
    spec = importlib.util.spec_from_file_location("cd_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cd_train"] = module
    spec.loader.exec_module(module)

    class ModernArgs:
        """Stands in for transformers 5.x: no warmup_ratio, no processing_class."""

        def __init__(
            self, *, warmup_steps=0, output_dir="", learning_rate=0.0, bf16=False
        ):
            pass

    resolved = module._supported_kwargs(
        ModernArgs,
        output_dir="out",
        learning_rate=2e-4,
        warmup_ratio=0.03,
        warmup_dataset_len=100,
        processing_class=object(),
    )
    assert "warmup_ratio" not in resolved
    assert resolved["warmup_steps"] == 3

    class LegacyArgs:
        """Stands in for transformers 4.x."""

        def __init__(
            self,
            *,
            warmup_ratio=0.0,
            output_dir="",
            learning_rate=0.0,
            processing_class=None,
        ):
            pass

    resolved = module._supported_kwargs(
        LegacyArgs,
        output_dir="out",
        learning_rate=2e-4,
        warmup_ratio=0.03,
        warmup_dataset_len=100,
        processing_class=object(),
    )
    assert resolved["warmup_ratio"] == 0.03
    assert "warmup_steps" not in resolved
    assert "processing_class" in resolved

    # bf16 handling: the key is only ever set to disable bf16, because asking for bf16 on
    # CPU is an error rather than a silent downgrade. So when CUDA is available the key is
    # ABSENT (TrainingArguments then defaults it to whatever the GPU supports), and when
    # it is not available the key is present and False.
    #
    # The old assertion was `resolved["bf16"] is False`. That held only because torch was
    # not installed anywhere on this machine, so CUDA could never be available -- which
    # is precisely the state in which you cannot train. Installing the distillation
    # requirements made the test fail, and the failure said nothing about the code. It
    # now asserts the rule instead of the current hardware.
    if _cuda_available():
        assert "bf16" not in resolved, (
            "bf16 must be left to TrainingArguments when a GPU is present"
        )
    else:
        assert resolved["bf16"] is False, (
            "bf16 must be explicitly disabled when no GPU is available"
        )


def _cuda_available() -> bool:
    """True if torch is installed AND sees a GPU. False when torch is absent.

    The distillation stack is deliberately not a runtime dependency of the app, so the
    test suite has to keep working on a venv with no torch at all.
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def test_tokenisation_masks_the_prompt_out_of_the_loss():
    """Loss must be computed on the reply only.

    Training on the prompt too teaches the model to model its own system prompt, which
    is a plausible route to a coach that narrates the data block it was handed.
    """
    import importlib.util

    path = SCRIPTS / "train_distill.py"
    spec = importlib.util.spec_from_file_location("cd_train2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cd_train2"] = module
    spec.loader.exec_module(module)

    class FakeTokenizer:
        """A chat template with the property the real ones have.

        The full conversation text always *starts with* the prompt text and is longer
        by exactly the reply, which is what makes masking by prefix length valid.
        """

        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            text = "|".join(m["content"] for m in messages)
            return text if add_generation_prompt else text + "<|end|>"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 100 for c in text]}

    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "ANS"},
    ]
    example = module._tokenize_example(FakeTokenizer(), messages, max_seq_len=512)
    labels = example["labels"]
    masked = [index for index, value in enumerate(labels) if value == -100]

    assert masked, "nothing was masked: the prompt would contribute to the loss"
    assert len(masked) < len(labels), "everything was masked: the loss would be NaN"
    # The unmasked tail must correspond to the assistant content.
    tail = labels[len(masked) :]
    assert tail == example["input_ids"][len(masked) :]


def test_an_empty_generation_never_overwrites_an_existing_corpus(pairs, tmp_path):
    """Generation fails in many ways, and the old corpus is the way back.

    An OOM, an interrupted run, a teacher that will not load -- in every one of those
    cases write_pairs() gets an empty list. Overwriting a good corpus with nothing, at
    exactly that moment, is the worst possible time to discover the loss.
    """
    existing = tmp_path / "train.jsonl"
    pairs.write_pairs(
        existing,
        [
            pairs.Pair(
                id="k",
                kind="repair",
                situation="s",
                skill="s",
                user=pairs.USER_POOL[0],
                prompt="p",
                constraint="c",
                response="a real reply",
            )
        ],
    )
    before = existing.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        pairs.write_pairs(existing, [])

    assert existing.read_text(encoding="utf-8") == before, "the corpus was lost"


def test_an_empty_generation_may_still_create_a_new_file(pairs, tmp_path):
    """The guard is against overwriting, not against starting empty."""
    target = tmp_path / "nested" / "train.jsonl"
    pairs.write_pairs(target, [])
    assert target.exists()
