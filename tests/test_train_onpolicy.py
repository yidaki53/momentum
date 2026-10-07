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
