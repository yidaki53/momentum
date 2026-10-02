"""Tests for the coach-distillation pipeline assets and bench scorers.

These run in the normal pytest suite, with no torch, no model, and no network. That is
deliberate: the scorers are pure functions over text, so a bug in one is a normal Python
bug rather than something that only shows up inside an inference loop on a phone.
"""

from __future__ import annotations

import importlib.util
import json
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
    assert kinds == {"coaching", "repair", "charter"}
    # Multi-turn episodes are configured but not generated by this script yet.
    assert config["seeds"]["include_multiturn_episodes"] is True


def test_student_and_teacher_share_a_tokenizer_family():
    import tomllib

    config = tomllib.loads((CONFIGS / "distill.qwen05b.toml").read_text())
    assert config["student"]["model_id"].startswith("Qwen/Qwen2.5-0.5B")
    assert config["teacher"]["model_id"].startswith("Qwen/Qwen2.5-7B")
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
