"""Regression tests for the OPD student loader: cold start versus fresh LoRA.

The bug these exist for: ``_load_models`` used to load the cold-start adapter
and then *always* call ``_attach_lora``, stacking a freshly initialised LoRA on
top of a cold start that ``PeftModel.from_pretrained`` had frozen
(``is_trainable`` defaults to False). The optimizer therefore trained only the
throwaway outer adapter, and ``save_pretrained`` wrote only that adapter: keys
with a doubled ``base_model.model.`` prefix, ``base_model_name_or_path: null``,
and none of the cold start's weights. Both pre-fix v10 OPD runs were
invalidated that way, so the tests below pin four properties:

  1. the cold start is resumed, its weights are the ones on disk, and every
     LoRA parameter is trainable;
  2. the real ``_load_models`` entry point never double-wraps;
  3. a missing cold start still gets a fresh adapter (the documented fallback);
  4. an adapter saved after a resume round-trips the cold start and names its
     base model -- the three artifact symptoms found by the forensics.

Plus one for the logged diagnostic: ``grad_norm`` is the *pre-clip* norm.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "coach-distillation" / "scripts"

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
get_peft_model_state_dict = pytest.importorskip("peft").get_peft_model_state_dict
load_file = pytest.importorskip("safetensors.torch").load_file

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:  # pragma: no cover - depends on the transformers build
    pytest.skip("Qwen2 architecture not available", allow_module_level=True)

# Registered under its own name so loading this module cannot clobber the
# cd_train_onpolicy entry that test_distillation_bench's fixture relies on.
_MODULE_NAME = "cd_train_onpolicy_coldstart"


def _load_opd():
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME, SCRIPTS / "train_onpolicy.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def opd():
    return _load_opd()


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """Two-layer Qwen2 -- the production student's architecture, at 1/1000 scale."""
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    path = tmp_path_factory.mktemp("qwen2-tiny")
    model.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def cold_start(tiny_base, tmp_path_factory):
    """A saved adapter with *identifiable* weights.

    PEFT zero-initialises ``lora_B``, so an untouched fresh LoRA and an
    untouched cold start are both zeros -- without this marker the test could
    not tell "resumed the cold start" from "re-initialised it".
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(1)
    base = AutoModelForCausalLM.from_pretrained(tiny_base)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.05)
    path = tmp_path_factory.mktemp("cold-start-adapter")
    model.save_pretrained(path)
    # v9c's saved config says inference_mode: true (it was written after
    # training); the loader must still resume it as trainable.
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inference_mode"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _cfg(opd, base, cold_start=None):
    return opd.OPDConfig(
        student_model=str(base),
        cold_start_dir=str(cold_start) if cold_start else None,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
    )


def test_cold_start_is_resumed_trainable_and_not_double_wrapped(
    opd, tiny_base, cold_start
):
    """The v10 bug in one test: frozen cold start + fresh outer LoRA."""
    student = opd._load_student(_cfg(opd, tiny_base, cold_start), "cpu")
    state = student.state_dict()

    # One wrap prefixes keys once; the old code produced two.
    assert not any(
        key.startswith("base_model.model.base_model.model.") for key in state
    )

    # state_dict() carries the adapter name (lora_A.default.weight) while the
    # saved file does not; get_peft_model_state_dict is the save path's own view.
    adapter_state = get_peft_model_state_dict(student)
    saved = load_file(str(cold_start / "adapter_model.safetensors"))
    assert saved and set(adapter_state) == set(saved)
    for key, expected in saved.items():
        assert torch.allclose(adapter_state[key], expected, atol=1e-6), key

    trainable = {name for name, p in student.named_parameters() if p.requires_grad}
    assert trainable == {
        name for name, p in student.named_parameters() if "lora_" in name
    }, "every LoRA weight must be trainable, and only LoRA weights"


def test_load_models_entry_point_never_double_wraps(
    opd, tiny_base, cold_start, monkeypatch
):
    """If ``_attach_lora`` is ever re-added after the cold start, this fails."""
    from transformers import AutoTokenizer

    class _FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer()
    )
    # Keep the test on CPU: the loader's device choice is not what is under test.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    student, teacher, tokenizer = opd._load_models(
        _cfg(opd, tiny_base, cold_start), load_teacher=False
    )
    assert teacher is None
    assert isinstance(tokenizer, _FakeTokenizer)

    state = student.state_dict()
    assert not any(
        key.startswith("base_model.model.base_model.model.") for key in state
    )
    adapter_state = get_peft_model_state_dict(student)
    saved = load_file(str(cold_start / "adapter_model.safetensors"))
    assert set(adapter_state) == set(saved)
    for key, expected in saved.items():
        assert torch.allclose(adapter_state[key], expected, atol=1e-6), key


def test_without_a_cold_start_a_fresh_lora_is_attached(opd, tiny_base):
    """The documented fallback still attaches exactly one trainable adapter."""
    student = opd._load_student(_cfg(opd, tiny_base), "cpu")
    state = student.state_dict()
    assert not any(
        key.startswith("base_model.model.base_model.model.") for key in state
    )
    assert any(key.startswith("base_model.model.") and "lora_" in key for key in state)
    assert any(p.requires_grad for p in student.parameters())


def test_saved_adapter_round_trips_the_cold_start(opd, tiny_base, cold_start, tmp_path):
    """The artifact symptoms found by the forensics, pinned to the fix.

    The invalidated run's adapter had doubled key prefixes, a null
    ``base_model_name_or_path`` and none of the cold start's weights. All
    three must be absent from anything ``phase_train`` would save.
    """
    student = opd._load_student(_cfg(opd, tiny_base, cold_start), "cpu")
    out = tmp_path / "adapter"
    student.save_pretrained(out)

    saved = load_file(str(out / "adapter_model.safetensors"))
    original = load_file(str(cold_start / "adapter_model.safetensors"))
    assert set(saved) == set(original)
    assert not any(
        key.startswith("base_model.model.base_model.model.") for key in saved
    )
    for key, expected in original.items():
        assert torch.allclose(saved[key], expected, atol=1e-6), key

    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"], "the artifact must name its base model"


def test_update_step_logs_the_pre_clip_grad_norm(opd):
    """``grad_norm`` is ``clip_grad_norm_``'s return value: the norm before scaling.

    The run log showed grad_norm=1417 with max_grad_norm=1.0. A post-clip
    report could never exceed the clip, and the diagnostic that motivated
    advantage normalisation (6614 against a clip of 1.0) would be impossible.
    """

    class TinyLM(torch.nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 16) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, vocab)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))

    torch.manual_seed(5)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    prompt_len, reply_len = 3, 6
    labels = torch.cat(
        [
            torch.full((prompt_len - 1,), -100),
            torch.zeros(reply_len, dtype=torch.long),
        ]
    )
    examples = [
        {
            "input_ids": torch.randint(0, 32, (prompt_len + reply_len,)),
            "labels": labels,
            "advantages": torch.full((reply_len,), 5.0),
            "reply_len": reply_len,
        }
    ]
    stats = opd.opd_update_step(
        model, optimizer, examples, max_grad_norm=1e-6, advantage_norm="none"
    )
    # Post-clip the reported norm would be exactly 1e-6; pre-clip it is orders above.
    assert stats["grad_norm"] > 1e-3


def _tiny_batch(opd, advantages):
    """A single example whose reply positions line up with `advantages`."""
    prompt_len, reply_len = 3, len(advantages)
    labels = torch.cat(
        [
            torch.full((prompt_len - 1,), -100),
            torch.zeros(reply_len, dtype=torch.long),
        ]
    )
    return [
        {
            "input_ids": torch.arange(prompt_len + reply_len) % 32,
            "labels": labels,
            "advantages": torch.tensor(advantages),
            "reply_len": reply_len,
        }
    ]


def _tiny_model(seed: int = 5):
    class TinyLM(torch.nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 16) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, vocab)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))

    torch.manual_seed(seed)
    return TinyLM()


def test_update_step_reports_the_raw_mean_advantage_and_leaves_examples_intact(opd):
    """``mean_reward`` is the teacher-student gap in nats, and ``examples`` is read-only.

    ``opd_update_step`` used to write the batch_std copy straight back into
    ``example["advantages"]``. Because scaling without recentring is idempotent, epoch 2
    normalised already-normalised values, every later epoch repeated that, and the logged
    ``mean_reward`` settled at mean/std -- a constant -0.9145 across all ten epochs of the
    v10 run -- instead of the -4.97 nats it is documented to be. Nothing else looked
    wrong: the loss had the correctly scaled advantages to learn from. Only the number a
    reader judges the run by was wrong, which is the worst kind of wrong for a diagnostic
    whose whole purpose is to report a gap.

    Both halves are pinned: the reported value must be the raw mean, and the raw
    advantages must survive the call untouched so the next epoch normalises from them.
    """
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    advantages = [6.0, 2.0, -1.0, -3.0]
    examples = _tiny_batch(opd, advantages)
    raw = examples[0]["advantages"].clone()

    raw_mean = sum(advantages) / len(advantages)
    variance = sum((a - raw_mean) ** 2 for a in advantages) / len(advantages)
    mean_over_std = raw_mean / variance**0.5
    assert raw_mean == pytest.approx(1.0)
    assert mean_over_std != pytest.approx(raw_mean), (
        "this batch must make the two readings distinguishable"
    )

    stats = opd.opd_update_step(
        model, optimizer, examples, max_grad_norm=1.0, advantage_norm="batch_std"
    )

    assert stats["mean_reward"] == pytest.approx(raw_mean)
    assert stats["mean_reward"] != pytest.approx(mean_over_std), (
        "mean_reward reported the normalisation's mean/std, not the teacher-student gap"
    )
    assert torch.equal(examples[0]["advantages"], raw), (
        "opd_update_step wrote scaled advantages back into the examples"
    )


def test_suppression_warning_fires_only_when_the_batch_is_mostly_negative(opd, caplog):
    """A batch that only teaches the student to shut up must say so, loudly.

    v10 measured 87.95% negative advantages, took ten steps over a frozen rollout set,
    and the student's replies became ``the the the`` -- charter substance 10/10 to 1/10
    -- while the loss fell monotonically to -62. The loss cannot carry this warning: it
    goes the right way throughout a collapse. So the sign distribution is checked before
    the first update and reported.
    """
    cfg = opd.OPDConfig()
    mostly_negative = [{"advantages": torch.tensor([-1.0] * 9 + [1.0])}]

    with caplog.at_level(logging.WARNING):
        stats = opd._warn_if_update_is_net_suppressive(mostly_negative, cfg)

    assert stats["fraction_negative"] == pytest.approx(0.9)
    assert stats["mean_advantage"] == pytest.approx(-0.8)
    assert stats["tokens"] == 10.0
    assert "NEGATIVE advantage" in caplog.text
    assert "eval_bench.py --vignettes" in caplog.text, (
        "the warning must point at the only measurement that caught the collapse"
    )

    # A balanced batch is the normal case and must stay quiet, or the warning is noise.
    caplog.clear()
    balanced = [{"advantages": torch.tensor([-1.0, 1.0, -1.0, 1.0])}]
    with caplog.at_level(logging.WARNING):
        quiet = opd._warn_if_update_is_net_suppressive(balanced, cfg)
    assert quiet["fraction_negative"] == pytest.approx(0.5)
    assert "NEGATIVE advantage" not in caplog.text


# --- per-epoch rollout resampling (STATUS.md Phase 2 step 6) ---------------------
#
# v10 collapsed with a configuration that these tests must make impossible to
# reintroduce silently. It built its examples once, took ten full-batch steps over
# the same 11308 tokens, 87.95% of which carried a NEGATIVE advantage, and pushed
# every one of them DOWN ten times with nothing to raise instead. The loss fell
# monotonically to -62.27 while all 30 bench replies became `the the the ...`.
#
# The seam under test is the `examples` argument of `opd_update_step`: it is the only
# consumer of that list, so recording it records exactly what would have been trained
# on. A "resample" that logged the word but kept the frozen list would produce a
# plausible history and fail here.


def _scored_row(opd, batch: int, reply_len: int = 4):
    """A scored rollout tagged with which resample pass produced it.

    ``marker`` is what the stub builder turns into the advantage vector, so the
    recorded advantages name their batch. Real tokenisation is deliberately bypassed:
    ``build_rollout_example`` needs the app's chat template and a real tokenizer, and
    this test is about which batch the loop hands to the update step, not about
    alignment -- which ``build_rollout_example``'s own tests already cover.
    """
    return {"marker": [float(batch)] * reply_len, "reply_len": reply_len}


class _StubModel:
    """Trainable parameters, so the loop's optimizer guard passes."""

    def parameters(self):
        return iter([torch.zeros(1, requires_grad=True)])

    def train(self):
        return self


class _StubTokenizer:
    """Present but unused: the stub builder never reaches the tokenizer."""


@pytest.fixture
def spy_update(opd, monkeypatch):
    """Record each batch the update step receives; return the list of recordings."""
    seen: list[list[float]] = []

    def _spy(_model, _optimizer, examples, **_kwargs):
        flat = [
            float(a) for example in examples for a in example["advantages"].tolist()
        ]
        seen.append(flat)
        return {
            "loss": 0.0,
            "mean_reward": sum(flat) / len(flat) if flat else 0.0,
            "tokens": float(len(flat)),
            "grad_norm": 1.0,
        }

    monkeypatch.setattr(opd, "opd_update_step", _spy)
    return seen


@pytest.fixture
def stub_build(opd, monkeypatch):
    """Turn a tagged row straight into an example, tagged by the same marker.

    Called through the module attribute rather than reimplemented here, so
    ``run_opd_epochs`` still exercises its own ``build_examples`` call path --
    including the "no usable rollout" guard.
    """

    def _build(tokenizer, scored, cfg):
        examples = []
        for row in scored:
            marker = row["marker"]
            examples.append(
                {
                    "input_ids": torch.arange(len(marker)),
                    "labels": torch.zeros(len(marker), dtype=torch.long),
                    "advantages": torch.tensor(marker),
                    "reply_len": row["reply_len"],
                }
            )
        return examples

    monkeypatch.setattr(opd, "build_examples", _build)


def test_resampling_replaces_the_batch_each_epoch(
    opd, monkeypatch, spy_update, stub_build
):
    """With ``refresh_rollouts_every = 1`` every epoch must train on a NEW batch.

    This is the whole of the v10 fix. The old loop built examples once, so a
    "resample" that failed to reassign ``examples`` would leave the history looking
    entirely plausible while training on the same frozen tokens that collapsed the
    student -- which is why the assertion is on the batches seen, not on the history.
    """
    cfg = opd.OPDConfig(opd_epochs=3, refresh_rollouts_every=1, advantage_norm="none")
    calls = {"n": 0}

    def _resample(_model):
        calls["n"] += 1
        return [_scored_row(opd, calls["n"])]

    opd.run_opd_epochs(
        _StubModel(),
        _StubTokenizer(),
        [_scored_row(opd, 0)],
        cfg,
        resample=_resample,
    )

    # Epoch 1 must train on the rollouts it was handed, not resample first: doing it
    # the other way discards the only measured gap a fresh cold start produces.
    assert calls["n"] == 2, "epoch 1 resampled instead of using the given rollouts"
    assert spy_update == [[0.0] * 4, [1.0] * 4, [2.0] * 4], (
        "the update step saw a frozen batch -- this is the v10 collapse path"
    )


def test_no_resampling_keeps_the_v10_behaviour_byte_for_byte(
    opd, monkeypatch, spy_update, stub_build
):
    """``refresh_rollouts_every = 0`` must never call the resampler.

    The frozen-set path is not merely a legacy default: it is how the v10 run that
    produced the collapse measurement is reproduced. If refreshing leaked into it,
    that run would stop being reproducible and its numbers would become unexplainable
    -- so the resampler explodes rather than being called.
    """
    cfg = opd.OPDConfig(opd_epochs=4, refresh_rollouts_every=0)

    def _explode(_model):
        raise AssertionError("resampler called with refresh_rollouts_every = 0")

    opd.run_opd_epochs(
        _StubModel(),
        _StubTokenizer(),
        [_scored_row(opd, 7)],
        cfg,
        resample=_explode,
    )

    assert spy_update == [[7.0] * 4] * 4, (
        "every epoch must reuse the one given batch (this is the frozen-set path)"
    )


def test_refresh_interval_is_an_interval_not_every_epoch(
    opd, monkeypatch, spy_update, stub_build
):
    """``refresh_rollouts_every = 2`` resamples every SECOND epoch.

    The value is an interval, and an off-by-one here either burns a full teacher pass
    per epoch -- hours on a card that cannot hold teacher and student together -- or
    silently reintroduces the frozen-set collapse while the config claims otherwise.
    """
    cfg = opd.OPDConfig(opd_epochs=5, refresh_rollouts_every=2)
    calls = {"n": 0}

    def _resample(_model):
        calls["n"] += 1
        return [_scored_row(opd, calls["n"])]

    opd.run_opd_epochs(
        _StubModel(),
        _StubTokenizer(),
        [_scored_row(opd, 0)],
        cfg,
        resample=_resample,
    )

    # 0-based epochs 0..4; refresh fires on epoch 2 and 4 (i.e. after every 2 trained
    # epochs, 1-based epochs 3 and 5). Epoch 1 is excluded by design: it must train on
    # the rollouts it was handed rather than resampling first.
    assert calls["n"] == 2, f"expected 2 refreshes across 5 epochs, got {calls['n']}"
    assert spy_update == [[0.0] * 4, [0.0] * 4, [1.0] * 4, [1.0] * 4, [2.0] * 4]


def test_refresh_of_every_third_epoch_skips_the_boundaries(
    opd, monkeypatch, spy_update, stub_build
):
    """A non-divisor interval must not schedule a refresh past the last epoch.

    With 5 epochs and an interval of 3, exactly one refresh is due (before 1-based
    epoch 4). A sixth epoch's worth of scheduling would mean a wasted teacher pass --
    hours on a card that cannot hold teacher and student together -- for no update.
    """
    cfg = opd.OPDConfig(opd_epochs=5, refresh_rollouts_every=3)
    calls = {"n": 0}

    def _resample(_model):
        calls["n"] += 1
        return [_scored_row(opd, calls["n"])]

    opd.run_opd_epochs(
        _StubModel(),
        _StubTokenizer(),
        [_scored_row(opd, 0)],
        cfg,
        resample=_resample,
    )
    # Only 0-based epoch 3 is divisible by 3 and > 0.
    assert calls["n"] == 1, f"epochs 1..5 at interval 3 -> 1 refresh, got {calls['n']}"
    assert spy_update == [[0.0] * 4, [0.0] * 4, [0.0] * 4, [1.0] * 4, [1.0] * 4]


def test_resampler_is_never_called_without_a_refresh_interval(
    opd, monkeypatch, spy_update, stub_build
):
    """A callable alone is not enough: the config must also ask for refreshing.

    ``run_opd_epochs`` is called from ``phase_train`` with the resampler only when
    ``refresh_rollouts_every > 0``. If the loop honoured the callable on its own, a
    config that says ``0`` would silently become the v11 configuration.
    """
    cfg = opd.OPDConfig(opd_epochs=3, refresh_rollouts_every=0)
    calls = {"n": 0}

    def _resample(_model):
        calls["n"] += 1
        return [_scored_row(opd, 99)]

    opd.run_opd_epochs(
        _StubModel(),
        _StubTokenizer(),
        [_scored_row(opd, 5)],
        cfg,
        resample=_resample,
    )
    assert calls["n"] == 0
    assert spy_update == [[5.0] * 4] * 3


def test_empty_refresh_warns_and_continues_on_the_previous_set(
    opd, monkeypatch, spy_update, stub_build, caplog
):
    """A refresh that yields nothing must be visible, not silent.

    Continuing is the right call -- one bad teacher pass should not waste an
    overnight run -- but "trained on stale data" is the v10 story retold, so it has
    to be in the log in words, because the loss curve will look fine either way.
    """
    cfg = opd.OPDConfig(opd_epochs=2, refresh_rollouts_every=1)
    calls = {"n": 0}

    def _resample(_model):
        calls["n"] += 1
        return []

    with caplog.at_level(logging.WARNING):
        opd.run_opd_epochs(
            _StubModel(),
            _StubTokenizer(),
            [_scored_row(opd, 3)],
            cfg,
            resample=_resample,
        )

    assert calls["n"] == 1
    assert "no usable rollouts" in caplog.text
    assert "continuing on the previous set" in caplog.text
    assert spy_update == [[3.0] * 4] * 2


def test_suppression_warning_reruns_on_each_refreshed_batch(
    opd, monkeypatch, spy_update, stub_build, caplog
):
    """The sign check must judge the batch that is about to be trained on.

    v10 measured 87.95% negative advantages. If the warning only ever ran on the
    first batch, a refresh that came back *more* suppressive would train silently.
    """
    cfg = opd.OPDConfig(opd_epochs=2, refresh_rollouts_every=1)

    def _resample(_model):
        # 9 of 10 negative: the v10 distribution, and it must be caught.
        return [
            {
                "marker": [-1.0] * 9 + [1.0],
                "reply_len": 10,
            }
        ]

    with caplog.at_level(logging.WARNING):
        opd.run_opd_epochs(
            _StubModel(),
            _StubTokenizer(),
            # All-positive: this batch is NOT suppressive and must stay quiet.
            [_scored_row(opd, 1)],
            cfg,
            resample=_resample,
        )

    # Exactly one warning, from the REFRESHED batch. If the check ran only on the
    # first batch the count would be 0 and a collapse-in-progress would train
    # silently -- which is the assertion that bites. Counting 2 instead would be
    # wrong here: the initial batch is all-positive and correctly stays quiet.
    assert caplog.text.count("NEGATIVE advantage") == 1, (
        "the refreshed batch was not sign-checked"
    )
    # The logged mean pins the warning to the refreshed batch (mean -0.8), not to the
    # initial one (mean +1.0). Without this, a warning computed from stale numbers
    # would still satisfy the count.
    assert "mean -0.800 nats" in caplog.text, (
        "the warning reported the wrong batch's statistics"
    )
    # And the refresh genuinely took effect on what was trained on.
    assert spy_update == [[1.0] * 4, [-1.0] * 9 + [1.0]]


def _refresh_fixture(tmp_path):
    """The minimum on-disk state ``_make_resampler`` reads and writes."""
    rows = [{"id": "p1", "prompt": "I want to stop taking my antidepressants."}]
    config_path = tmp_path / "retrain.v11.toml"
    config_path.write_text("[opd]\n", encoding="utf-8")
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    return rows, config_path, corpus_path


class _SavingModel:
    """Records where the live adapter is written; creates the directory."""

    def __init__(self):
        self.saved: list[Path] = []

    def save_pretrained(self, path):
        self.saved.append(Path(path))
        Path(path).mkdir(parents=True, exist_ok=True)


def test_resampler_points_the_child_at_the_live_adapter(opd, tmp_path, monkeypatch):
    """The child must roll out from the CURRENT policy, not the cold start.

    If ``--cold-start-dir`` were omitted the refresh cycle would re-score the cold
    start's rollouts -- the same ones epoch 1 already trained on. That is the v10
    collapse with extra GPU hours spent looking like a fix.
    """
    rows, config_path, corpus_path = _refresh_fixture(tmp_path)
    # The child writes scored.jsonl into output_dir; give it something to read back.
    (tmp_path / "scored.jsonl").write_text(
        json.dumps(
            {
                "prompt": rows[0]["prompt"],
                "rollout": "fresh",
                "student_logprobs": [-1.0],
                "teacher_logprobs": [-0.5],
                "teacher_advantage": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    argv_seen: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(argv, **_kwargs):
        argv_seen.append(list(argv))
        return _Completed()

    import subprocess

    monkeypatch.setattr(subprocess, "run", _fake_run)

    model = _SavingModel()
    resampler = opd._make_resampler(
        opd.OPDConfig(output_dir=str(tmp_path)),
        rows,
        tmp_path,
        config_path=config_path,
        corpus_path=corpus_path,
        max_prompts=7,
    )
    kept = resampler(model)

    assert model.saved, "the live adapter must be persisted before the child runs"
    assert model.saved[0] == tmp_path / "refresh-adapter"
    assert argv_seen, "the refresh cycle must shell out"
    argv = argv_seen[0]
    assert "--score-only" in argv, "the refresh is a rollout+score cycle"
    assert "--cold-start-dir" in argv
    assert argv[argv.index("--cold-start-dir") + 1] == str(model.saved[0]), (
        "the child would roll out from the cold start and resample nothing new"
    )
    assert str(config_path) in argv
    assert str(corpus_path) in argv
    assert argv[argv.index("--max-prompts") + 1] == "7"
    assert kept, "the refreshed rows must reach the epoch loop"


def test_failed_child_process_raises_instead_of_training_on_stale_data(
    opd, tmp_path, monkeypatch
):
    """A crashed refresh must abort, not quietly reuse last epoch's batch.

    Training on the previous frozen set while logging "resampled" is precisely how v10
    destroyed the student and still reported a healthy loss curve.
    """
    rows, config_path, corpus_path = _refresh_fixture(tmp_path)

    class _Failed:
        returncode = 1
        stdout = "Traceback (most recent call last):"
        stderr = "torch.cuda.OutOfMemoryError"

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Failed())

    resampler = opd._make_resampler(
        opd.OPDConfig(output_dir=str(tmp_path)),
        rows,
        tmp_path,
        config_path=config_path,
        corpus_path=corpus_path,
    )
    with pytest.raises(RuntimeError, match="refusing to reuse stale rollouts"):
        resampler(_SavingModel())


def test_resampler_applies_the_same_prompt_selection_as_the_train_phase(
    opd, tmp_path, monkeypatch
):
    """A refresh that bypassed ``min_teacher_confidence`` would widen the corpus.

    Li et al.'s prompt selection is what keeps uninformative prompts out of the
    update; a refreshed batch that skipped it would train on rows the first batch
    deliberately dropped, and the two epochs would not be comparable.
    """
    rows, config_path, corpus_path = _refresh_fixture(tmp_path)
    # One strong-preference prompt and one the teacher is indifferent about.
    (tmp_path / "scored.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "id": "strong",
                    "prompt": "p",
                    "rollout": "r",
                    "student_logprobs": [-1.0],
                    "teacher_logprobs": [-0.2],
                    "teacher_advantage": 0.8,
                },
                {
                    "id": "weak",
                    "prompt": "p2",
                    "rollout": "r2",
                    "student_logprobs": [-1.0],
                    "teacher_logprobs": [-0.99],
                    "teacher_advantage": 0.01,
                },
            )
        ),
        encoding="utf-8",
    )

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Completed())

    resampler = opd._make_resampler(
        opd.OPDConfig(output_dir=str(tmp_path), min_teacher_confidence=0.5),
        rows,
        tmp_path,
        config_path=config_path,
        corpus_path=corpus_path,
    )
    kept = resampler(_SavingModel())
    assert [row["id"] for row in kept] == ["strong"], (
        "the refresh cycle dropped min_teacher_confidence"
    )


def test_train_phase_refuses_to_silently_drop_the_refresh(opd, tmp_path):
    """``refresh_rollouts_every > 0`` without the corpus must fail loudly.

    A bare ``--phase train`` invocation cannot resample from scored.jsonl alone.
    Falling back to the frozen set here would run the exact configuration that
    collapsed v10 while the config file on disk says otherwise.
    """
    scored_path = tmp_path / "scored.jsonl"
    scored_path.write_text(
        json.dumps(
            {
                "prompt": "x",
                "rollout": "y",
                "student_logprobs": [-1.0],
                "teacher_logprobs": [-0.5],
                "teacher_advantage": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = opd.OPDConfig(
        refresh_rollouts_every=1, output_dir=str(tmp_path), max_new_tokens=4
    )
    with pytest.raises(RuntimeError, match="needs the corpus rows"):
        opd.phase_train(cfg, scored_path, tmp_path)
