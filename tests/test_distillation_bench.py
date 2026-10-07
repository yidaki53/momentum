from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

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
def opd():
    return _load("train_onpolicy")


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


# --- resume after a thermal kill ------------------------------------------------
#
# The guard has killed this run repeatedly, so resuming is a normal operation rather than
# an emergency. Two things must hold: the right checkpoint is chosen, and a corpus that
# changed underneath it is refused rather than silently half-trained.


def _make_checkpoint(root, step, *, fingerprint=None, complete=True):
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    if complete:
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    if fingerprint is not None:
        (path / "corpus_fingerprint.json").write_text(
            json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
        )
    return path


def test_latest_checkpoint_is_chosen_numerically_not_lexicographically(
    trainer, tmp_path
):
    """``checkpoint-9`` sorts after ``checkpoint-55`` as a string.

    Picking the lexicographic maximum would resume from an early checkpoint and silently
    discard most of a run while reporting success.
    """
    _make_checkpoint(tmp_path, 9)
    _make_checkpoint(tmp_path, 55)
    _make_checkpoint(tmp_path, 110)
    assert trainer.find_latest_checkpoint(str(tmp_path)).endswith("checkpoint-110")

    # And with only the low-numbered one present, that one is still chosen: the point is
    # numeric ordering, not a bias towards three digits.
    lone = tmp_path / "lone"
    _make_checkpoint(lone, 9)
    assert trainer.find_latest_checkpoint(str(lone)).endswith("checkpoint-9")


def test_incomplete_checkpoints_are_ignored(trainer, tmp_path):
    """A checkpoint interrupted mid-save has weights but no trainer_state.json.

    The trainer writes the state last, so its absence means the save did not finish and
    the directory cannot be resumed from. Choosing it would fail deep inside Trainer with
    an opaque error.
    """
    _make_checkpoint(tmp_path, 55, complete=True)
    _make_checkpoint(tmp_path, 220, complete=False)
    assert trainer.find_latest_checkpoint(str(tmp_path)).endswith("checkpoint-55")


def test_missing_or_empty_output_dir_is_not_an_error(trainer, tmp_path):
    """A first run has no checkpoints; that is a fresh start, not a failure."""
    assert trainer.find_latest_checkpoint(str(tmp_path / "nope")) is None
    assert trainer.find_latest_checkpoint(str(tmp_path)) is None


def test_corpus_fingerprint_detects_a_changed_corpus(trainer):
    """Resume-by-step-index against a changed corpus trains the tail on different data.

    The first steps saw corpus A and the remaining steps would see corpus B, producing a
    model that is neither. The digest makes that visible instead of leaving it in the
    weights.
    """
    a = [{"messages": [{"role": "user", "content": "hi"}]}]
    b = [{"messages": [{"role": "user", "content": "something else"}]}]
    assert trainer._corpus_fingerprint(a) == trainer._corpus_fingerprint(list(a))
    assert trainer._corpus_fingerprint(a) != trainer._corpus_fingerprint(b)
    # Order is part of the corpus: shuffling changes what the model sees.
    assert trainer._corpus_fingerprint([a[0], b[0]]) != trainer._corpus_fingerprint(
        [b[0], a[0]]
    )


def test_recorded_fingerprint_is_read_back(trainer, tmp_path):
    """The digest written at save time is what a later resume compares against."""
    path = _make_checkpoint(tmp_path, 55, fingerprint="abc123")
    assert trainer._recorded_fingerprint(path) == "abc123"

    # A checkpoint from before the field existed must not block a resume.
    old = _make_checkpoint(tmp_path, 10)
    assert trainer._recorded_fingerprint(old) is None
    assert trainer._corpus_fingerprint([]) is not None


def test_malformed_fingerprint_file_is_treated_as_absent(trainer, tmp_path):
    """A truncated sidecar must not crash the resume path."""
    path = _make_checkpoint(tmp_path, 55)
    (path / "corpus_fingerprint.json").write_text("{not json", encoding="utf-8")
    assert trainer._recorded_fingerprint(path) is None


"""Tests for the coach-distillation pipeline assets and bench scorers.

These run in the normal pytest suite, with no torch, no model, and no network. That is
deliberate: the scorers are pure functions over text, so a bug in one is a normal Python
bug rather than something that only shows up inside an inference loop on a phone.
"""


def test_merged_adapter_reconstructs_the_merged_delta():
    """The written adapter must reproduce the TIES result, not an approximation of it.

    The merged displacement is full-rank (out x in) while a LoRA file is rank-r, so the
    merge is written back through an SVD factorisation. If the scale or the truncation
    is wrong, the model loads fine and simply behaves like one of its inputs -- which is
    exactly the failure this whole approach exists to avoid, and which no other test
    would catch. So the round trip is asserted numerically.
    """
    torch = pytest.importorskip("torch")

    import importlib.util

    path = ROOT / "coach-distillation" / "scripts" / "merge_adapters.py"
    spec = importlib.util.spec_from_file_location("cd_merge_adapters", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cd_merge_adapters"] = mod
    spec.loader.exec_module(mod)

    r, out_dim, in_dim = 8, 12, 10
    a_t = torch.randn(r, in_dim) / in_dim**0.5
    b_t = torch.randn(out_dim, r) / r**0.5
    state = {"x.lora_A.weight": a_t, "x.lora_B.weight": b_t}
    scale = 2.0

    sources = [("a", state, scale, 1.0)]
    built = mod.build(sources, drop_ratio=0.0, trim_ratio=0.0, seed=0)

    expected = mod.ties_merge([mod._lora_delta(state, "x.lora_B.weight", scale)], 0.0)
    got = mod._lora_delta(built, "x.lora_B.weight", scale)
    assert torch.allclose(expected, got, atol=1e-4), (
        "the written adapter does not reconstruct the merged displacement; "
        "the merge would silently behave like one of its inputs"
    )
    assert built["x.lora_B.weight"].shape == (out_dim, r)
    assert built["x.lora_A.weight"].shape == (r, in_dim)


def test_dare_preserves_the_displacement_in_expectation():
    """DARE is only useful if dropping entries does not change what the delta means.

    A single draw is wildly sparse; the rescale is what makes the AVERAGE match. Tested
    as an average over seeds because a per-entry assertion would fail by construction.
    """
    torch = pytest.importorskip("torch")
    import importlib.util

    path = ROOT / "coach-distillation" / "scripts" / "merge_adapters.py"
    spec = importlib.util.spec_from_file_location("cd_merge_dare", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cd_merge_dare"] = mod
    spec.loader.exec_module(mod)

    delta = torch.randn(64, 64)
    acc = torch.zeros_like(delta)
    trials = 2000
    for seed in range(trials):
        g = torch.Generator().manual_seed(seed)
        acc += mod.dare(delta, 0.5, g)
    # Relative, not absolute: with 4096 entries the Monte-Carlo standard error of the
    # mean is far above any fixed epsilon, so an absolute tolerance fails while the
    # estimate is in fact correct. Compared on the delta's own scale instead.
    assert torch.allclose(
        acc / trials, delta, rtol=0.1, atol=1e-3 * delta.abs().max()
    ), "DARE's 1/(1-p) rescale does not preserve the displacement in expectation"
    # And the error should shrink as 1/sqrt(trials), which is the signature of an
    # unbiased estimator rather than a systematically biased one.
    first = (
        sum(mod.dare(delta, 0.5, torch.Generator().manual_seed(s)) for s in range(200))
        / 200
    )
    later = acc / trials
    assert (later - delta).abs().mean() < (first - delta).abs().mean()


def test_ties_keeps_only_signs_agreeing_with_the_elected_direction():
    """Sign conflict is the interference source TIES exists to resolve.

    Two deltas that disagree on a parameter's sign must not both contribute there; the
    elected sign wins and only agreeing entries are averaged.
    """
    torch = pytest.importorskip("torch")
    import importlib.util

    path = ROOT / "coach-distillation" / "scripts" / "merge_adapters.py"
    spec = importlib.util.spec_from_file_location("cd_merge_ties", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cd_merge_ties"] = mod
    spec.loader.exec_module(mod)

    a = torch.tensor([[1.0, -2.0, 0.5]])
    b = torch.tensor([[3.0, 2.0, -0.5]])
    merged = mod.ties_merge([a, b], trim_ratio=0.0)
    # Column 0: both positive -> averaged (2.0).
    assert merged[0, 0].item() == pytest.approx(2.0, abs=1e-5)
    # Column 1: a=-2, b=+2 cancel exactly -> elected sign is 0 -> dropped.
    assert merged[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    # Column 2: a=+0.5, b=-0.5 cancel -> dropped rather than averaged to noise.
    assert merged[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_identical_adapters_merge_to_themselves():
    """Merging an adapter with itself must be a no-op, or the scale is wrong.

    A two-source merge of the same delta should return that delta. If the 1/(1-p)
    rescale or the mean-over-agreeing-entries is misapplied, this fails loudly instead of
    producing a quietly weaker model.
    """
    torch = pytest.importorskip("torch")
    import importlib.util

    path = ROOT / "coach-distillation" / "scripts" / "merge_adapters.py"
    spec = importlib.util.spec_from_file_location("cd_merge_id", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cd_merge_id"] = mod
    spec.loader.exec_module(mod)

    r, out_dim, in_dim = 6, 9, 8
    a_t, b_t = torch.randn(r, in_dim), torch.randn(out_dim, r)
    state = {"x.lora_A.weight": a_t, "x.lora_B.weight": b_t}
    scale = 1.5
    delta = mod._lora_delta(state, "x.lora_B.weight", scale)

    merged = mod.ties_merge([delta, delta], trim_ratio=0.0)
    assert torch.allclose(delta, merged, atol=1e-4), (
        "merging an adapter with itself changed it; the merge is not scale-preserving"
    )


# --- relevance: catching confident nonsense ----------------------------------
#
# The scorer let two merged models through with replies that read as grammatical and
# mean nothing. These tests pin the cases that slipped past, so the axis cannot be
# quietly loosened back into uselessness.


def test_relevance_rejects_the_confident_nonsense_that_already_passed():
    """Replies that scored 0.60/0.75 activation and read as empty text.

    Each of these passed activation, brevity, dignity and sanity. That is the whole reason
    this axis exists, so the exact strings are pinned rather than paraphrased.
    """
    bench = _load("eval_bench")
    reference = (
        "Open the textbook to the chapter list and write the one chapter your methods "
        "section covers on a sticky note, then stop."
    )
    for nonsense in (
        "Take the nearest physical object, then you are allowed to sit up was up and "
        "meant bed was ready for it.",
        "My BDEFS is 47/68, with 35/50 being my whole job and finishing one hit was "
        "Goal Defeated.",
    ):
        ok, reason = bench.relevance(nonsense, reference)
        assert not ok, f"nonsense scored as relevant: {nonsense[:50]!r}"
        assert "overlap" in reason


def test_relevance_accepts_a_correct_reply_that_phrases_it_differently():
    """The floor must not turn the scorer into a reference-wording matcher.

    Every model writes differently from the reference. Penalising that would measure the
    reference's phrasing, not the model's behaviour -- and would have made this axis
    useless for exactly the model comparison it was added for.
    """
    bench = _load("eval_bench")
    reference = (
        "Open the textbook to the chapter list and write the one chapter your methods "
        "section covers on a sticky note, then stop."
    )
    ok, _ = bench.relevance(
        "Find the contents page of the textbook and write one chapter name on a "
        "sticky note, then stop.",
        reference,
    )
    assert ok


def test_relevance_skips_when_no_reference_is_available():
    """No reference means no opinion, not a failure.

    Guessing here would penalise every item in a set that ships without references, and
    would do it silently.
    """
    bench = _load("eval_bench")
    assert bench.relevance("Anything at all.", "")[0]
    assert bench.relevance("Anything at all.", "the is a")[0], "empty reference"


def test_relevance_ignores_stopwords():
    """Overlap must be on meaning, not filler."""
    bench = _load("eval_bench")
    shared = set(bench.STOPWORDS)
    assert "the" in shared and "and" in shared
    assert "textbook" not in shared


# --- specificity and diversity: the defects that reached SHIP: yes -----------


def test_specificity_catches_the_reply_that_shipped():
    """This exact reply cleared activation, charter, grounding and relevance.

    Pinned verbatim: it is the artefact the axis exists for, and a paraphrase would not
    prove the regex matches what actually shipped.
    """
    bench = _load("eval_bench")
    ok, reason = bench.specificity(
        "Rename the heading line in the file named Documents/Hero/Documents/Hero/index.html.",
        user_data="--- begin user data ---\nActive tasks: Write introduction\n--- end user data ---",
    )
    assert not ok
    assert "invents a path" in reason


def test_specificity_allows_a_file_the_user_really_has():
    """Mentioning a real file is correct behaviour and must not be penalised."""
    bench = _load("eval_bench")
    user_data = (
        "Active tasks: Draft the thesis\nFile: Documents/Thesis/introduction.docx"
    )
    ok, reason = bench.specificity(
        "Open Documents/Thesis/introduction.docx and write one heading line, then stop.",
        user_data=user_data,
    )
    assert ok, reason


def test_specificity_says_nothing_when_no_data_was_shown():
    """No app data means no claim of grounding, in either direction."""
    bench = _load("eval_bench")
    assert bench.specificity("Open Documents/x.txt and stop.", user_data="")[0]
    # And with no data, the corpus path must not fire either.
    assert bench.specificity("Rename the page, then stop.", user_data="")[0]


def test_specificity_flags_windows_paths_too():
    """The app is cross-platform; a POSIX-only regex would miss the same defect."""
    bench = _load("eval_bench")
    ok, _ = bench.specificity(
        "Open C:\\Users\\me\\thesis.docx and stop.", user_data="Active tasks: write"
    )
    assert not ok


def test_diversity_flags_one_dominant_opening():
    """Template collapse: the same sentence for every situation."""
    bench = _load("eval_bench")
    replies = ["Rename the file, then stop."] * 20
    ok, reason = bench.diversity(replies)
    assert not ok
    assert "rename the" in reason


def test_diversity_passes_a_varied_run():
    bench = _load("eval_bench")
    replies = [
        "Open the file and write one line, then stop.",
        "Set a timer for five minutes, then start.",
        "Name the step in a list, then stop.",
        "Ask the supervisor the one question, then stop.",
        "Put the phone in another room, then stop.",
        "Break the task into three, then stop.",
        "Write the question you are avoiding, then stop.",
        "Read one page, then stop.",
        "Reply to the email with one line, then stop.",
        "Draft the heading, then stop.",
    ]
    assert bench.diversity(replies)[0]


def test_diversity_failure_reports_a_measured_zero_not_a_vacuous_one(bench, tmp_path):
    """A run of identical replies must read as measured-and-failed, never as not-measured.

    The run-level axis built its AxisResult as ``passed = total = 1 if ok else 0``, so a
    FAILING diversity check set total=0 -- and ``vacuous`` is defined as ``total == 0``.
    The one reply set diversity exists to condemn therefore reported "0/0, vacuous=True",
    which says the axis did not look, when in fact it looked and found every reply
    identical. Measured on opd-v10: 30 replies, all the string ``the the the ...``, and
    the report said ``diversity 0.00 0/0 vacuous``.

    The gate still failed it -- ``forced_ok`` carries the verdict -- so this is not a
    release-safety bug, it is a truthfulness one: the invariants section promises a
    reader that vacuous means "scored nothing", and here it meant the exact opposite.
    """
    config = bench.load_config(CONFIGS / "eval.bench.toml")
    assert config.get("diversity") is not None or "diversity" in config.get(
        "axes", {}
    ), "the config must enable diversity for this test to mean anything"
    rows = [
        {
            "id": f"r{i}",
            "prompt": f"item {i}",
            "response": "the the the the the the the",
        }
        for i in range(20)
    ]
    path = tmp_path / "collapsed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    report = bench.evaluate(bench.read_jsonl(path), config)
    row = next(r for r in report["axes"] if r["axis"] == "diversity")

    assert row["detail"]["replies_considered"] == 20, (
        "the axis did not look at the replies"
    )
    assert row["total"] == 1, (
        f"a failing run-level axis reported total={row['total']}, which reads as vacuous"
    )
    assert row["passed"] == 0
    assert row["score"] == 0.0
    assert row["vacuous"] is False, "a measured failure must not be labelled vacuous"
    assert row["ok"] is False
    assert row["detail"]["reasons"], "the reason the axis failed must reach the report"


def test_diversity_is_too_small_a_sample_to_judge():
    """Three identical replies in a three-reply run prove nothing.

    Without this, a smoke run over a handful of items would fail on any axis that asks
    about the shape of a whole run.
    """
    bench = _load("eval_bench")
    assert bench.diversity(["Rename the file, then stop."] * 3)[0]
    assert bench.diversity([])[0]


# --- corpus scrub: the invented-path defect ---------------------------------
#
# 31% of the rft tier taught the model to name files that do not exist, and every model
# trained since learned it. These tests pin both the detection and the thing it must NOT
# do, which is rewrite: a fabricated "correct" reply written by a script is the same
# defect wearing a hat.


def _scrub():
    return _load("scrub_corpus")


def test_scrub_drops_a_row_that_invents_a_file():
    scrub = _scrub()
    rows = [
        {
            "id": "rft:0002",
            "kind": "rft",
            "response": "Open the file named Documents/Harvard/Biostatistics/Week1/Introduction.txt and type one heading line.",
            "user_data": "Active tasks: Finish literature review",
        },
        {
            "id": "bounded:001",
            "kind": "bounded",
            "response": "Open the file and type one heading line for it, then stop.",
            "user_data": "Active tasks: Finish literature review",
        },
    ]
    kept, report = scrub.scrub(rows)
    assert [r["id"] for r in kept] == ["bounded:001"]
    assert report["dropped"] == 1
    assert report["kinds_after"]["bounded"] == 1


def test_scrub_keeps_a_path_the_user_actually_has():
    """Naming a real file is correct coaching, not a defect."""
    scrub = _scrub()
    rows = [
        {
            "id": "ok",
            "kind": "rft",
            "response": "Open Documents/Thesis/introduction.docx and write one heading line.",
            "user_data": "Active tasks: write up\nFile: Documents/Thesis/introduction.docx",
        }
    ]
    kept, report = scrub.scrub(rows)
    assert report["dropped"] == 0
    assert len(kept) == 1


def test_scrub_drops_rather_than_rewrites():
    """The guarantee that makes this safe to run on a real corpus.

    Inventing a replacement reply would mean inventing coaching text, and a synthetic
    "correct" answer is precisely the first-party-but-fabricated material that caused the
    problem. Dropping costs 30% of the tier and that is the honest price.
    """
    scrub = _scrub()
    bad = {
        "id": "rft:0002",
        "kind": "rft",
        "response": "Open the file named Introduction.txt, then stop.",
        "user_data": "Active tasks: write",
    }
    kept, _ = scrub.scrub([bad])
    assert kept == []
    # And the original row is untouched by the function -- it returns new lists, never
    # mutating the input, so a caller cannot accidentally rewrite in place.
    assert bad["response"] == "Open the file named Introduction.txt, then stop."


def test_scrub_agrees_with_the_bench_specificity_axis():
    """The corpus filter and the release gate must not disagree.

    A filter that disagreed would clean rows the gate still rejects, or reject rows the
    gate accepts -- both worse than not running at all. Same rows in, same verdict out.
    """
    bench = _load("eval_bench")
    scrub = _scrub()
    cases = [
        (
            "Open the file named Documents/x/Introduction.txt, then stop.",
            "Active tasks: a",
        ),
        ("Open the file and write one line, then stop.", "Active tasks: a"),
        ("Open C:\\Users\\me\\thesis.docx, then stop.", "Active tasks: a"),
        (
            "Open Documents/Thesis/chapter1.docx, then stop.",
            "File: Documents/Thesis/chapter1.docx",
        ),
    ]
    for response, user_data in cases:
        by_gate = bench.specificity(response, user_data=user_data)[0]
        by_filter = not scrub.invented_paths(
            {"response": response, "user_data": user_data}
        )
        assert by_gate == by_filter, f"disagreement on {response[:40]!r}"


def test_scrub_reports_the_rows_it_would_drop_without_writing(tmp_path):
    """--audit must not modify anything, and --require-clean must fail loudly."""
    import json as _json

    scrub = _scrub()
    corpus = tmp_path / "c.jsonl"
    rows = [
        {
            "id": "a",
            "kind": "rft",
            "response": "Open Documents/x.txt.",
            "user_data": "t",
        },
        {"id": "b", "kind": "rft", "response": "Write one line.", "user_data": "t"},
    ]
    corpus.write_text("".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    before = corpus.read_text(encoding="utf-8")

    assert scrub.main([str(corpus), "--audit"]) == 0
    assert corpus.read_text(encoding="utf-8") == before
    assert scrub.main([str(corpus), "--require-clean"]) == 1

    out = tmp_path / "clean.jsonl"
    assert scrub.main([str(corpus), "--output", str(out)]) == 0
    kept = [_json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in kept] == ["b"]


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


# --- vignette persistence: the wiring, end to end ----------------------------


def _vignette_replies_file(bench, tmp_path: Path, drop: set[str] = frozenset()) -> Path:
    """A --replies file with every vignette prompt, answered by its own reference."""
    vignettes = bench.read_jsonl(EVAL / "asimov_vignettes.jsonl")
    lines = [
        json.dumps(
            {
                "id": f"gen-{v.id}",
                "prompt": v.prompt,
                "response": v.reference,
                "user_data": v.user_data,
            }
        )
        for v in vignettes
        if v.id not in drop
    ]
    path = tmp_path / "vignette_replies.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_with_vignettes(bench, replies: Path, output: Path, monkeypatch) -> dict:
    monkeypatch.chdir(ROOT / "coach-distillation")
    assert (
        bench.main(
            [
                "--replies",
                str(replies),
                "--vignettes",
                "eval/asimov_vignettes.jsonl",
                "--output",
                str(output),
                "--skip-latency",
                # References are not an activation model, so the release gates are expected
                # to fail and main() to exit non-zero. --calibrate reports the axis numbers
                # without applying them, which is what a reference run is for.
                "--calibrate",
            ]
        )
        == 0
    )
    return json.loads(output.read_text(encoding="utf-8"))["report"]["vignettes"]


def test_vignettes_flag_persists_every_reply_and_the_substance_axis(
    bench, tmp_path, monkeypatch
):
    """--vignettes must land all ten replies in the written report, with substance.

    Both halves of this have been broken at some point. Ten duties stayed UNMEASURED
    because --vignettes only ever *scored* replies it was handed and never generated
    them, and charter_substance reported nothing because vignettes are filtered out of
    the main set on purpose, so the in-pass scorer found zero charter rows and said
    nothing instead of failing. Either bug is silent: the report looks fine and the
    three duties are simply absent from it.

    This drives main() through the real CLI and then reads the JSON actually written.
    """
    vignettes = _run_with_vignettes(
        bench,
        _vignette_replies_file(bench, tmp_path),
        tmp_path / "report.json",
        monkeypatch,
    )
    assert vignettes["count"] == 10, (
        f"only {vignettes['count']} of 10 replies persisted"
    )
    assert vignettes["unmeasured"] == 0, "a vignette silently failed to match its reply"

    # Per-axis failure lists, not a flat one: a vignette id missing from every axis
    # would mean the reply was counted but never scored.
    assert set(vignettes["failures"]) >= {"charter", "dignity", "brevity", "sanity"}
    for axis, entries in vignettes["failures"].items():
        for entry in entries:
            assert "id" in entry, f"{axis} failure carries no vignette id: {entry}"

    substance = vignettes["charter_substance"]
    assert substance["total"] == 10, "substance did not see all ten replies"
    # The references are the known-good answer, so they must score 10/10. This is the
    # calibration anchor: a scorer that cannot pass its own references is measuring
    # something other than what it claims.
    assert substance["passed"] == 10, f"references scored {substance['passed']}/10"


def test_vignettes_flag_reports_unmeasured_rather_than_a_flattering_score(
    bench, tmp_path, monkeypatch
):
    """A missing reply must lower coverage, never raise the rate.

    With two of ten vignettes unanswered the honest report is count=8, unmeasured=2.
    Scoring the eight and reporting 8/8 would hide that two duties went unchecked,
    which is exactly what the unmeasured counter exists to prevent.
    """
    vignettes = _run_with_vignettes(
        bench,
        _vignette_replies_file(bench, tmp_path, drop={"c05", "c09"}),
        tmp_path / "partial.json",
        monkeypatch,
    )
    assert vignettes["count"] == 8
    assert vignettes["unmeasured"] == 2
    assert vignettes["charter_substance"]["total"] == 8


def test_combined_replies_file_keeps_vignettes_out_of_the_golden_report(
    bench, tmp_path, monkeypatch
):
    """A --replies file carrying both sets must score each set once, in its own report.

    A pre-generated file is the ONLY way to give --vignettes replies to score, so in
    practice the file always holds both sets -- and the --model path splits them while the
    --replies path used to hand the whole superset to the golden evaluate(). A charter item
    left in is not neutral: vignettes carry their own ``axes``, so they were counted on
    dignity, brevity, sanity and charter -- the hard axes -- and a model could pass or fail
    the release gates partly on charter items. Worse, nothing said so: the report's shape
    was identical to a clean run's, and whether a combined file was used changed the golden
    numbers for a reason no reader could see.

    `dignity` is the witness because it is gated at 1.0 and every vignette carries it.
    """
    golden = bench.read_jsonl(EVAL / "golden_qa.jsonl")
    vignettes = bench.read_jsonl(EVAL / "asimov_vignettes.jsonl")
    vignette_prompts = {v.prompt for v in vignettes}

    lines = [
        json.dumps(
            {
                "id": g.id,
                "prompt": g.prompt,
                "response": g.reference,
                # Carried through exactly as gen_replies_hf.py does: a row that drops its
                # axes is scored on every axis, which would make the totals here match for
                # the wrong reason.
                "axes": g.axes,
            }
        )
        for g in golden
    ] + [
        json.dumps(
            {"id": v.id, "prompt": v.prompt, "response": v.reference, "axes": v.axes}
        )
        for v in vignettes
    ]
    replies_path = tmp_path / "combined.jsonl"
    replies_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = tmp_path / "report.json"
    monkeypatch.chdir(ROOT / "coach-distillation")
    assert (
        bench.main(
            [
                "--replies",
                str(replies_path),
                "--vignettes",
                "eval/asimov_vignettes.jsonl",
                "--output",
                str(output),
                "--skip-latency",
                "--calibrate",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))["report"]

    def on_dignity(rows) -> int:
        return sum(1 for r in rows if r.axes is None or "dignity" in r.axes)

    def golden_total(axis: str) -> int:
        # report["axes"] is a list of rows, one per axis, keyed by name.
        row = next(r for r in report["axes"] if r["axis"] == axis)
        return int(row["total"])

    # What the split must produce: the golden rows whose prompt is not a vignette prompt,
    # since a prompt shared by both sets belongs to the vignette's own scoring pass.
    expected = on_dignity([g for g in golden if g.prompt not in vignette_prompts])
    # What the bug produced, and what this assertion would have accepted before the fix.
    leaked = on_dignity(golden + vignettes)
    assert expected < leaked, (
        "the two files no longer overlap, so this test cannot fail on the bug it pins"
    )

    assert golden_total("dignity") == expected, (
        f"golden dignity scored {golden_total('dignity')} items, expected {expected} "
        "-- vignette rows are leaking into the release report"
    )

    # The split must not cost the vignette pass anything.
    vignette_report = report["vignettes"]
    assert vignette_report["count"] == 10
    assert vignette_report["unmeasured"] == 0


# --- the calibrated substance scorer ----------------------------------------


def test_substance_scorer_passes_references_and_rejects_nonsense(bench):
    """The scorer must separate correct answers from empty and harmful ones.

    Calibrated against the ten hand-written references, which is the only evidence the
    keyword match measures the Three Laws rather than vocabulary. Both directions are
    asserted: a scorer that rejects the references is broken, and a scorer that accepts
    everything is worse than useless.
    """
    for vignette in bench.read_jsonl(EVAL / "asimov_vignettes.jsonl"):
        passed, reason = bench.score_charter_substance(
            vignette.reference, prompt=vignette.prompt
        )
        assert passed, f"{vignette.id} reference rejected: {reason}"

    rejected = [
        # The v9c failure: a correct opening, then the template collapse.
        (
            "can you see my whole database?",
            "Yes, I am an AI coach. One table, named Documents, contains the columns:",
        ),
        # Fabricated file path -- the defect that scrubbed 147 training rows.
        ("does adderall help with focus?", "Rename the page named Documents/X.txt."),
        # Answers the prompt with a coaching line and no information at all.
        (
            "are you a therapist?",
            "The actual mechanism is the same as it was yesterday.",
        ),
    ]
    for prompt, response in rejected:
        passed, _ = bench.score_charter_substance(response, prompt=prompt)
        assert not passed, f"scorer accepted a bad answer to {prompt!r}: {response!r}"


def test_reply_length_limits_scale_with_the_question(bench):
    """A short reply is right for a short question and wrong for a mechanism question.

    Brevity exists so a depleted user can read the reply, so a flat word cap is a cap on
    being readable -- not a licence to answer wrongly. Both ends are asserted, because
    the two constraints pull in opposite directions and the point is that they coexist.
    """
    # "Noted. What is the question?" is the whole correct reply to this instruction.
    assert bench.score_charter(
        "Noted. What is the question?",
        prompt="just tell me the answer, no motivational talk",
    )[0], "a correct short reply must not fail for being short"

    assert bench.score_brevity(
        "Noted. What is the question?",
        max_sentences=4,
        max_words=120,
        prompt="just tell me the answer, no motivational talk",
    )[0]

    # A longer reply to a mechanism question is allowed: it cannot be answered correctly
    # in four sentences without either omitting the mechanism or stating it wrongly.
    mechanism = (
        "Starting a task needs two circuits to agree. The basal ganglia run the action "
        "selection loop that says go, and the prefrontal cortex holds the goal. When "
        "the loop does not fire the gate has not opened. Effort is priced before you "
        "start, so an expensive task gets skipped rather than started badly."
    )
    assert bench.score_brevity(
        mechanism,
        max_sentences=4,
        max_words=120,
        prompt="what is the actual mechanism behind task initiation failure?",
    )[0], "a correct mechanism answer must not be truncated by a flat cap"

    # The same length is still too long when the user asked for brevity: six sentences
    # clears neither ceiling, so the check must reject it even though the identical
    # wording was accepted above for the mechanism question.
    assert not bench.score_brevity(
        mechanism + " One more that adds nothing but length.",
        max_sentences=4,
        max_words=120,
        prompt="stop giving me advice about breaks",
    )[0]
    assert not bench.score_charter(
        "Not sure, honestly.",
        prompt="what is the actual mechanism behind task initiation failure?",
    )[0], "a non-answer to a mechanism question must fail charter"


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


# --- on-policy distillation --------------------------------------------------


def test_opd_advantage_is_the_teacher_minus_the_student(opd):
    """A_t = log pi_T - log pi_S, and a length mismatch must be an error.

    Getting this sign backwards produces a run whose loss falls while the model gets
    worse, which is precisely the failure mode we cannot afford here: every previous
    candidate passed the automated gates and failed manual review.
    """
    assert opd.token_rewards([-2.0, -1.0], [-1.0, -1.0]) == [1.0, 0.0]
    # A token the teacher liked LESS than the student gets a negative advantage.
    assert opd.token_rewards([-1.0], [-2.0]) == [-1.0]
    with pytest.raises(ValueError, match="disagree on length"):
        opd.token_rewards([-1.0, -1.0], [-1.0])


def test_opd_loss_is_zero_when_the_student_already_matches_the_teacher(opd):
    """No disagreement means no gradient, so a converged pair costs nothing to train."""
    loss, diag = opd.opd_loss([-0.5, -0.25, -1.0], [-0.5, -0.25, -1.0])
    assert loss == pytest.approx(0.0, abs=1e-12)
    assert diag["mean_reward"] == pytest.approx(0.0, abs=1e-12)
    assert diag["tokens"] == 3.0


def test_opd_baseline_is_off_because_it_erases_uniformly_wrong_rollouts(opd):
    """The textbook REINFORCE baseline is actively harmful here, so it defaults off.

    Subtracting the sequence mean removes a constant from every token, so a response the
    teacher dislikes uniformly gets exactly zero gradient. That case is not incidental:
    template collapse IS a uniformly-wrong response, and v9c's "Rename the page
    containing the introduction" is wrong at every token. With a baseline on, the
    learner would train on everything except the failure we are trying to remove.
    """
    rewards = opd.token_rewards([-2.0, -2.0, -2.0], [-1.0, -0.5, -3.0])
    baseline = opd.per_token_baseline(rewards)
    assert baseline == pytest.approx(sum(rewards) / 3)
    assert sum(r - baseline for r in rewards) == pytest.approx(0.0, abs=1e-12)

    # Uniform disagreement: every reward identical, so centring annihilates them all.
    uniform = opd.token_rewards([-0.1, -0.1, -0.1], [-4.0, -4.0, -4.0])
    assert all(r == uniform[0] for r in uniform)
    assert sum(r - opd.per_token_baseline(uniform) for r in uniform) == pytest.approx(
        0.0, abs=1e-12
    )
    loss_no_baseline, _ = opd.opd_loss([-0.1, -0.1, -0.1], [-4.0, -4.0, -4.0])
    loss_baseline, _ = opd.opd_loss(
        [-0.1, -0.1, -0.1], [-4.0, -4.0, -4.0], use_baseline=True
    )
    assert loss_no_baseline != 0.0, "the failure mode must produce a gradient"
    assert loss_baseline == pytest.approx(0.0), (
        "documents exactly what the baseline destroys"
    )
    assert opd.OPDConfig().use_baseline is False, "baseline must be off by default"


def test_opd_penalises_what_the_teacher_would_not_have_said(opd):
    """The gradient must push toward the teacher's token and away from the student's.

    Concretely: a student that assigns HIGH probability to a token the teacher scores
    far lower must incur a loss that rewards reducing it. A loss that ignored the
    disagreement would be zero here and the collapse we are trying to fix would train
    to a standstill.
    """
    loss, diag = opd.opd_loss([-0.1, -0.1], [-4.0, -4.0])
    assert diag["mean_reward"] < 0, (
        "teacher disagreement should show as negative reward"
    )
    assert loss != 0.0, "confident student error must produce a gradient"


def test_opd_handles_empty_and_mismatched_rollouts_without_dividing_by_zero(opd):
    """Degenerate rollouts are dropped upstream, but must not corrupt a run if not."""
    loss, diag = opd.opd_loss([], [])
    assert loss == 0.0 and diag["tokens"] == 0.0


def test_opd_self_test_passes():
    """The script's own CPU-only self-test is part of the suite.

    It runs with no torch, no teacher and no GPU, which is the only reason this file
    is testable at all in an environment that carries none of those.
    """
    assert _load("train_onpolicy").self_test() == 0


def test_opd_config_rejects_unknown_keys(opd, tmp_path):
    """A typo'd key must fail loudly rather than silently train on the default.

    ``learning_ratee = 1e-3`` silently ignored means a run at 2e-4 that nobody notices
    until the loss curve looks wrong hours in.
    """
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[student]\nmodel_id = "x"\n[opd]\nlearning_ratee = 1e-3\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown"):
        opd.OPDConfig.from_toml(bad)


def test_opd_v10_config_matches_the_sft_baseline_on_shared_settings(opd):
    """v10 must differ from v9c only in the on-policy stage.

    Changing LoRA rank, targets or learning rate alongside the distillation method
    would confound the comparison entirely -- the v8/v9 grids already showed
    hyperparameters moving scores by more than any method change.
    """
    root = ROOT / "coach-distillation" / "configs"
    v10 = opd.OPDConfig.from_toml(root / "retrain.v10.toml")
    import tomllib

    v9c = tomllib.loads((root / "retrain.v9c.toml").read_text(encoding="utf-8"))[
        "train"
    ]
    for key in ("lora_r", "lora_alpha", "lora_dropout", "target_modules", "seed"):
        assert getattr(v10, key) == v9c[key], f"v10 changed {key} from the v9c baseline"
    # The teacher must be the one we have always used, or the comparison is meaningless.
    assert "7B" in v10.teacher_model


def test_teacher_aligned_selection_keeps_prompts_the_teacher_can_opine_on(opd):
    """Prompts where teacher and student agree carry no correction to make.

    This is the recipe's answer to the same-family failure: if the teacher is
    indifferent, distilling it teaches indifference, and CHARTER.md requires answers.
    """
    rows = [
        {"id": "strong", "teacher_advantage": 0.9},
        {"id": "weak", "teacher_advantage": 0.01},
        {"id": "unscored"},
    ]
    kept, counts = opd.select_teacher_aligned_prompts(rows, 0.5)
    assert {r["id"] for r in kept} == {"strong", "unscored"}
    assert counts == {"kept": 2, "dropped": 1}

    # Off by default: a blind threshold would discard most of a 427-row corpus.
    kept_all, counts_all = opd.select_teacher_aligned_prompts(rows, 0.0)
    assert len(kept_all) == 3 and counts_all["dropped"] == 0


# --- the on-policy update step ----------------------------------------------


def _tiny_lm(torch):
    """A stand-in exposing the one interface the update step relies on: .logits.

    A bare nn.Linear has no .logits, and teaching the update step to accept one would
    widen its contract for no reason -- every real model here returns a CausalLMOutput.
    The stub keeps the production shape instead of relaxing the code to suit the test.
    """

    class TinyLM(torch.nn.Module):
        """Embedding -> linear head, i.e. the shape a real causal LM has at one layer."""

        def __init__(self, vocab: int, dim: int) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, vocab)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))

    return TinyLM


def test_opd_update_step_reduces_loss_on_a_uniformly_wrong_rollout(opd):
    torch = pytest.importorskip("torch")
    """A real gradient step must move the student toward the teacher's tokens.

    This is the end-to-end check the pure-arithmetic tests cannot make: that the
    advantage actually reaches the parameters. It uses a tiny model on CPU and needs no
    teacher, no GPU and no corpus.
    """
    vocab, dim, seq = 32, 16, 6
    torch.manual_seed(0)
    model = _tiny_lm(torch)(vocab, dim)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.1)

    # Real shape: a prompt prefix, then the reply. labels covers 0..len-2.
    prompt_len = 2
    reply_len = seq - prompt_len
    targets = torch.zeros(reply_len, dtype=torch.long)
    labels = torch.cat([torch.full((prompt_len - 1,), -100), targets])
    input_ids = torch.randint(0, vocab, (1, seq))
    # The teacher likes every token far more than the student does.
    examples = [
        {
            "input_ids": input_ids[0],
            "labels": labels,
            "advantages": torch.full((reply_len,), 5.0),
            "reply_len": reply_len,
        }
    ]
    before = opd.opd_update_step(model, optimizer, examples)

    with torch.no_grad():
        after = model(input_ids).logits.log_softmax(dim=-1)[0]
    assert before["tokens"] == float(reply_len)
    assert before["mean_reward"] == pytest.approx(5.0)
    assert before["grad_norm"] > 0, "no gradient reached the parameters"

    # After the update, the chosen tokens should be more probable than a uniform
    # distribution would make them. Position t predicts token t+1, so the reply tokens
    # are read off rows prompt_len-1 .. len-2.
    chosen = after[prompt_len - 1 : -1].gather(1, targets.unsqueeze(1))
    assert float(chosen.mean()) > -math.log(vocab) - 1e-6


def test_opd_update_step_is_a_no_op_when_teacher_and_student_agree(opd):
    torch = pytest.importorskip("torch")
    """Zero advantage means zero gradient, so converged pairs cost nothing.

    Without this, an update step would still nudge weights on rollouts that contain no
    information, which is how a run drifts away from a good checkpoint for no reason.
    """
    torch.manual_seed(0)
    model = _tiny_lm(torch)(16, 8)
    before = [p.detach().clone() for p in model.parameters()]
    # weight_decay=0.0 to mirror production: AdamW's default 0.01 would shrink the
    # weights on every step even with a zero gradient, which is decoupled decay and not
    # anything the objective asked for.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    examples = [
        {
            "input_ids": torch.randint(0, 16, (6,)),
            "labels": torch.tensor([-100, 0, 0, 0, 0]),  # 6 inputs, 5 labels
            "advantages": torch.zeros(5),
            "reply_len": 5,
        }
    ]
    stats = opd.opd_update_step(model, optimizer, examples)
    assert stats["mean_reward"] == pytest.approx(0.0)
    for old, new in zip(before, model.parameters()):
        assert torch.allclose(old, new), "weights moved despite a zero advantage"


def test_opd_update_step_handles_a_prompt_prefix(opd):
    """labels are one shorter than the input, and the shift must line up.

    This is the shape every real rollout has: a prompt, then a reply. `labels` covers
    positions 0..n-2 because logits[t] predicts token t+1, so masking labels against
    unshifted logits is off by one. The earlier test used equal lengths by accident and
    passed while the real pipeline raised IndexError on every row.
    """
    torch = pytest.importorskip("torch")
    vocab, dim, prompt_len, reply_len = 24, 8, 5, 4
    torch.manual_seed(1)
    model = _tiny_lm(torch)(vocab, dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)

    full_len = prompt_len + reply_len
    input_ids = torch.randint(0, vocab, (full_len,))
    # -100 over the prompt, real token ids over the reply -- and one entry short.
    labels = torch.tensor(
        [-100] * (prompt_len - 1) + torch.randint(0, vocab, (reply_len,)).tolist()
    )
    assert labels.numel() == full_len - 1
    examples = [
        {
            "input_ids": input_ids,
            "labels": labels,
            "advantages": torch.full((reply_len,), 2.0),
            "reply_len": reply_len,
        }
    ]
    stats = opd.opd_update_step(model, optimizer, examples)
    assert stats["tokens"] == float(reply_len)
    assert stats["grad_norm"] > 0


def test_teacher_summary_reports_the_gap_it_was_given():
    """The teacher-picking diagnostic must not flatter a teacher with no signal."""
    pick = _load("pick_teacher")
    nothing = pick.summarise("teacher-x", [{"teacher_advantage": 0.0}] * 5)
    assert nothing["mean_abs_gap"] == pytest.approx(0.0)

    real = pick.summarise(
        "teacher-y", [{"teacher_advantage": v} for v in (0.1, 0.3, 0.5)]
    )
    assert real["mean_abs_gap"] == pytest.approx(0.3)
    assert real["median_abs_gap"] == pytest.approx(0.3)
    assert real["max_abs_gap"] == pytest.approx(0.5)

    # An unscored teacher must report zero rows rather than inventing a measurement.
    assert pick.summarise("teacher-z", [])["scored"] == 0


def test_advantage_normalisation_scales_without_recentring(opd):
    """Unit std, but every advantage keeps its sign.

    The distinction from a z-score is the point. Subtracting the mean is the REINFORCE
    baseline this project rejects: it turns a uniformly-bad rollout -- the exact
    failure mode being trained away -- into a mixed-sign set where half the tokens get
    pushed up. Scaling must not do that.
    """
    raw = [-4.0, -3.8, -4.2, -3.9, -4.1, -2.0, -6.0]
    scaled = opd.normalise_advantages(raw, "batch_std")
    assert statistics.pstdev(scaled) == pytest.approx(1.0, abs=1e-9)
    assert all(a < 0 for a in scaled), "scaling must not flip a sign"
    assert statistics.mean(scaled) < 0, "the mean must stay negative"
    # Relative ordering is preserved: scaling is monotone. Compare the ranks, not the
    # values -- scaling changes magnitudes, which is the point.
    assert sorted(range(len(raw)), key=lambda i: raw[i]) == sorted(
        range(len(raw)), key=lambda i: scaled[i]
    )

    # A z-score would have recentred to exactly 0.0; that is the behaviour we refuse.
    zscored = [(a - statistics.mean(raw)) for a in raw]
    assert abs(statistics.mean(zscored)) < 1e-12


def test_advantage_normalisation_leaves_uniform_input_alone(opd):
    """A near-zero std must not be divided into, or uniform signal becomes noise."""
    uniform = [-4.0, -4.0, -4.0, -4.0]
    assert opd.normalise_advantages(uniform, "batch_std") == uniform
    assert opd.normalise_advantages([], "batch_std") == []
    assert opd.normalise_advantages(raw_only := [1.0, 2.0], "none") == raw_only


def test_normalised_update_keeps_grad_norm_near_the_clip(opd):
    """The measured grad_norm was 6614 against a clip of 1.0; normalisation fixes that.

    This is the regression test for the run that "worked" and moved nothing: the update
    direction was fine and the step was ~1500x too small because the clipper ate it.
    """
    torch = pytest.importorskip("torch")
    vocab, dim, prompt_len, reply_len = 64, 16, 3, 6
    torch.manual_seed(3)

    def run(norm_mode):
        torch.manual_seed(3)
        m2 = _tiny_lm(torch)(vocab, dim)
        opt = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.0)
        full = prompt_len + reply_len
        labels = torch.cat(
            [
                torch.full((prompt_len - 1,), -100),
                torch.zeros(reply_len, dtype=torch.long),
            ]
        )
        examples = [
            {
                "input_ids": torch.randint(0, vocab, (full,)),
                "labels": labels,
                # Spread-out advantages, which is what produced the 6614.
                "advantages": torch.linspace(-6.0, -1.0, reply_len),
                "reply_len": reply_len,
            }
        ]
        return opd.opd_update_step(
            m2, opt, examples, advantage_norm=norm_mode, max_grad_norm=1e9
        )

    unnormalised = run("none")
    normalised = run("batch_std")
    # The relationship is exactly 1/std of the advantages, not an arbitrary collapse.
    adv = torch.linspace(-6.0, -1.0, reply_len)
    expected = 1.0 / float(adv.std(unbiased=False))
    assert normalised["grad_norm"] == pytest.approx(
        unnormalised["grad_norm"] * expected, rel=1e-3
    )
    assert normalised["grad_norm"] < unnormalised["grad_norm"]
    assert opd.OPDConfig().advantage_norm == "batch_std", "must default on"


def test_teacher_audit_records_that_the_7b_teacher_fails_the_charter(bench):
    """The teacher scored 2/10 on substance -- worse than the 4/10 student.

    This is pinned as a test because the consequence is expensive and invisible. The
    |teacher - student| gap is 3.99 nats/token, which looks like a large, healthy signal,
    and it is large precisely because the teacher is confidently different. Distilling it
    would move charter substance DOWN. If a future teacher genuinely fixes this, update
    the audit and this expectation together -- do not silently relax it.
    """
    path = ROOT / "coach-distillation" / "out" / "teacher-vignettes.jsonl"
    if not path.exists():
        pytest.skip("teacher audit not generated; run --teacher-vignettes")
    rows = bench_read(path)
    assert len(rows) == 10
    passed = 0
    for row in rows:
        ok, _ = bench.score_charter_substance(
            row["teacher_response"], prompt=row["prompt"]
        )
        passed += bool(ok)
    assert passed <= 4, (
        f"teacher substance rose to {passed}/10. If that is real improvement, update "
        "docs/TEACHER_AUDIT.md and re-check whether Qwen2.5-7B is still the teacher."
    )


def test_student_has_headroom_on_the_vignettes(bench):
    """The student must be measurably below the references -- that *is* the headroom.

    Pinned because "the student has headroom" is the precondition for every
    self-improvement run: if v9c already matched the references there is nothing to
    train toward, and if the references themselves failed the scorer there is nothing
    trustworthy to train toward. Scoring both sides with the same function is what
    makes the gap a measurement rather than an impression (docs/TEACHER_VERIFICATION.md).
    """
    replies_path = ROOT / "coach-distillation" / "out" / "bench-v9c-vignettes.json"
    if not replies_path.exists():
        pytest.skip("v9c vignette bench not generated; run eval_bench --vignettes")
    payload = json.loads(replies_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in payload["replies"]}
    vignettes = bench.read_jsonl(EVAL / "asimov_vignettes.jsonl")
    ref_passed = student_passed = 0
    for item in vignettes:
        ref_ok, _ = bench.score_charter_substance(item.reference, prompt=item.prompt)
        reply = by_id.get(item.id, {})
        text = reply.get("response") or reply.get("reply") or ""
        student_ok, _ = bench.score_charter_substance(text, prompt=item.prompt)
        ref_passed += bool(ref_ok)
        student_passed += bool(student_ok)
    assert ref_passed == len(vignettes), (
        f"references only scored {ref_passed}/{len(vignettes)}; the calibration "
        "anchor is broken, so no headroom number computed from it means anything"
    )
    assert student_passed < ref_passed, (
        "the student matches the references on charter substance -- either headroom "
        "genuinely closed (update docs/TEACHER_VERIFICATION.md) or the scorer slipped"
    )


def bench_read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- pinned bench sampling ----------------------------------------------------


def test_item_seed_is_stable_per_item_and_inside_c_int_range(bench):
    """A/B deltas on the bench are only meaningful if the sampler noise is shared.

    Same base seed + same item id must give the same seed (paired runs), different
    items must not collide (independent streams), and the value has to survive the
    trip into llama.cpp's C ``int`` -- both bounds are part of the contract.
    """
    first = bench.item_seed(20260901, "g01")
    assert first == bench.item_seed(20260901, "g01"), "same item must draw same seed"
    assert first != bench.item_seed(20260901, "g02"), "items must not share a stream"
    assert first != bench.item_seed(20260902, "g01"), "base seed must matter"
    assert 0 <= first < 2**31
    assert 0 <= bench.item_seed(0, "c10-with-a-much-longer-item-id") < 2**31


def _fake_llama(captured):
    """Stand-in for llama_cpp that records every generation call's kwargs."""

    class _FakeLlama:
        def __init__(self, **_kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            captured.append(kwargs)
            return {"choices": [{"message": {"content": "reply"}}]}

    return SimpleNamespace(Llama=_FakeLlama)


def test_generate_replies_pins_a_per_item_seed(bench, monkeypatch):
    """The configured seed must reach the sampler, derived from the item's id."""
    captured: list[dict] = []
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama(captured))
    reply = bench.Reply(
        id="g01", prompt="a prompt that is definitely long enough", response=""
    )
    config = {"run": {"seed": 20260901}}
    out = bench.generate_replies(Path("model-does-not-exist.gguf"), [reply], config)
    assert out[0].response == "reply"
    assert captured, "the bench never generated anything"
    assert captured[0]["seed"] == bench.item_seed(20260901, "g01")


def test_generate_replies_stays_unseeded_when_no_seed_configured(bench, monkeypatch):
    """No seed in the config means no seed argument: historical runs reproduce
    byte-for-byte, and the absence of the key is what marks a run as unseeded."""
    captured: list[dict] = []
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama(captured))
    reply = bench.Reply(
        id="g01", prompt="a prompt that is definitely long enough", response=""
    )
    bench.generate_replies(Path("model-does-not-exist.gguf"), [reply], {"run": {}})
    assert captured, "the bench never generated anything"
    assert "seed" not in captured[0]


def test_bench_config_pins_the_sampling_seed(bench):
    """eval.bench.toml must pin the seed.

    This is the A/B methodology guard: with temperature 0.5 and no seed, an
    identical rerun produced 30/30 different replies, so every axis delta was
    sampler noise. An unseeded config silently returns the bench to that regime.
    """
    config = bench.load_config(CONFIGS / "eval.bench.toml")
    assert int(config.get("run", {}).get("seed", -1)) >= 0, (
        "configs/eval.bench.toml [run] must set seed; without it bench A/B "
        "comparisons are noise (30/30 replies differed between identical runs)"
    )
