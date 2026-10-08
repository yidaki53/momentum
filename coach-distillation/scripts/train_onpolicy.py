#!/usr/bin/env python3
"""On-policy distillation (GKD) for the Momentum coach.

Why this exists
---------------
Every run so far has been off-policy: the student was fitted to replies the teacher
wrote, and then asked to behave like that at inference. That mismatch is what the
literature calls exposure bias, and it has a specific signature in our results --
v9c answers "I want to stop taking my antidepressants" with "Rename the page containing
the introduction". The training data never said anything about antidepressants. The
model reached for the nearest pattern it had seen thousands of times, because off-policy
training never showed it what it does when it is actually in that state.

On-policy distillation fixes this by training the student on *its own* outputs and
asking the teacher what it should have said. The student visits the states it can
really reach, so small mistakes get corrected where they happen instead of compounding.

The objective
-------------
Sampled-token reverse KL (GKD, Agarwal et al. 2023; sampled-token variant analysed in
Li et al. 2026). For a prompt x, sample a response y ~ pi_student(.|x), then score the
teacher on that same y and take a policy-gradient step on the per-token reward

    A_t = log pi_teacher(y_t | x, y_<t) - log pi_student(y_t | x, y_<t)
    L   = -mean_t( A_t * log pi_student(y_t | x, y_<t) )

Both terms are detached for A_t. This is the standard RL-with-a-reward formulation of
reverse KL, and it is the whole of what this script does -- there is no hidden
regulariser, no second loss, and no privileged information beyond the teacher's
per-token log-probabilities on text the student itself produced.

Why the cold start is not optional
----------------------------------
Li et al. 2026 ("Rethinking On-Policy Distillation", arXiv:2604.13016) find two
conditions govern whether OPD succeeds, and report a specific failure for our exact
configuration: **same-family teachers can be distributionally indistinguishable from the
student's perspective.** They demonstrate this with 1.5B and 7B teachers against a
student of the same family -- weak-to-strong reverse distillation in that setting gains
nothing, because the teacher offers no capability the student has not already modelled.

Our teacher is Qwen2.5-7B-Instruct and our student is Qwen2.5-0.5B-Instruct. That is the
same family. So this script does not trust OPD to bootstrap from a random 0.5B. It runs
the recipe the paper prescribes for a failing configuration:

  1. **Off-policy cold start** -- ordinary SFT on the existing scrubbed corpus, so the
     student starts from a competent model rather than from noise.
  2. **On-policy phase** -- student rollouts, teacher-scored, reverse KL.
  3. **Teacher-aligned prompt selection** -- prompts where the teacher is itself
     uncertain are filtered out, because OPD cannot learn from a teacher that has no
     preference to convey.

The third step matters here for a specific reason: if the 7B teacher does not actually
know the answer to "how much alcohol helps me sleep?", distilling its distribution
teaches the student to be uncertain -- which is not what CHARTER.md asks for. The
charter says answer. So prompts are selected for where the teacher has a strong,
confident preference, and skipped where it does not.

This is honest about what it cannot fix. If the same-family problem means the teacher
has no capability advantage on the vignettes, no amount of on-policy training will move
substance above the SFT baseline. That is an empirical question this script is built to
answer cheaply, and ``eval_bench.py --vignettes`` is the arbiter -- not the training
loss, which will happily go down while the replies get worse.

Usage::

    # smoke test on a handful of prompts, no teacher, no GPU required
    python scripts/train_onpolicy.py --self-test

    # the real thing (long: wrap in the thermal guard)
    python coach-distillation/scripts/thermal_guard.py --log out/thermal.csv -- \
        python scripts/train_onpolicy.py --config configs/retrain.v10.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger("train_onpolicy")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_distill import (  # noqa: E402
    ensure_app_importable,
    to_messages,
)

# --- the objective -----------------------------------------------------------


def token_rewards(
    student_logprobs: list[float], teacher_logprobs: list[float]
) -> list[float]:
    """Per-token reward: how much more likely the teacher found this token.

    A_t = log pi_T(y_t | x, y_<t) - log pi_S(y_t | x, y_<t)

    Returned as a plain list so the arithmetic is testable on CPU with no torch and no
    model, which is the only way this file gets a test suite at all.

    Both inputs must be log-probabilities (already normalised), not logits. Passing raw
    logits here is the single easiest way to get a run that trains and learns nothing:
    the difference is then dominated by logit scale rather than meaning "the teacher
    preferred this token".
    """
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError(
            f"student and teacher disagree on length: "
            f"{len(student_logprobs)} vs {len(teacher_logprobs)}"
        )
    return [t - s for s, t in zip(student_logprobs, teacher_logprobs)]


def per_token_baseline(rewards: list[float]) -> float:
    """Sequence-mean reward, the textbook REINFORCE baseline.

    Kept, but **off by default** -- see ``opd_loss``. Subtracting the mean is variance
    reduction for a sparse terminal-reward REINFORCE loop. Here it removes exactly the
    signal we are trying to train on: a student that is uniformly, consistently wrong.
    """
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def normalise_advantages(
    advantages: list[float], mode: str = "batch_std", eps: float = 1e-6
) -> list[float]:
    """Rescale advantages to unit standard deviation **without** recentring them.

    This is the fix for the grad_norm=6614 measurement, and the "without recentring" is
    the whole point. A conventional z-score divides by the std *and subtracts the mean*,
    and that subtraction is the same operation as the REINFORCE baseline this project
    deliberately rejects: it turns a uniformly-negative set into a mixed-sign one, so
    the rollout the teacher dislikes at every single token -- our actual failure mode --
    ends up with half its tokens pushed up. Scaling alone keeps every advantage
    negative, keeps their relative sizes, and only changes the magnitude.

    Measured on the run that motivated this: raw advantages had std 1.08 and mean
    -4.00; after scaling alone, std 1.00, mean still -3.72 and every value still
    negative. A z-score would have produced std 1.00 and mean exactly 0.00, flipping
    the sign of roughly half the tokens.

    Dividing by std is also the correct lever rather than raising ``max_grad_norm``.
    Loosening the clip to 6614 would pass a step through at an effectively enormous
    learning rate, which is how a 0.5B adapter gets wrecked in a single update.

    Modes:
      ``none``      -- leave alone (the original behaviour)
      ``batch_std`` -- divide by the std of all advantages in the update (default)
    """
    if not advantages or mode == "none":
        return list(advantages)
    mean = sum(advantages) / len(advantages)
    variance = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    std = math.sqrt(variance)
    if std < eps:
        # Every advantage is (near) identical. Scaling by a tiny std would explode them
        # into noise, so leave them alone: a uniform set is already a consistent signal.
        return list(advantages)
    # Division only. NO mean subtraction -- see the docstring.
    return [a / std for a in advantages]


def opd_loss(
    student_logprobs: Iterable[float],
    teacher_logprobs: Iterable[float],
    *,
    use_baseline: bool = False,
) -> tuple[float, dict[str, float]]:
    """Reverse-KL policy-gradient loss over one sampled response.

    L = -mean_t( A_t * log pi_S(y_t | x, y_<t) ), with A_t detached.

    ``use_baseline`` defaults to False, and that is a deliberate reversal of the usual
    REINFORCE choice. A sequence-mean baseline subtracts a constant from every token,
    which annihilates any response the teacher dislikes *uniformly*: if the teacher rates
    all of the student's tokens 4 nats worse, every advantage becomes exactly 0 and the
    gradient is exactly 0.

    That case is not an edge case here, it is the entire problem. Template collapse is a
    uniformly-bad response -- v9c's "Rename the page containing the introduction" is
    wrong at every single token, with no locally-good tokens among the bad ones. A
    baseline would give that rollout zero gradient while a rollout with one good token
    buried in noise got the entire update. The learner would be trained on everything
    except the failure mode we are trying to remove.

    GKD itself needs no baseline: the reward is already dense per token, so there is no
    variance spike to clip. The switch stays available for anyone who wants to compare,
    and the self-test pins both behaviours so the difference cannot drift unnoticed.

    ``mean_reward`` near zero means teacher and student already agree and there is
    nothing to learn; a large negative mean means the student is confidently wrong.
    """
    student = list(student_logprobs)
    teacher = list(teacher_logprobs)
    if not student:
        return 0.0, {"mean_reward": 0.0, "tokens": 0.0, "grad_norm": 0.0}
    rewards = token_rewards(student, teacher)
    baseline = per_token_baseline(rewards) if use_baseline else 0.0
    advantages = [r - baseline for r in rewards]
    loss = -sum(a * s for a, s in zip(advantages, student)) / len(student)
    grad_norm = math.sqrt(sum(a * a for a in advantages)) / len(advantages)
    return loss, {
        "mean_reward": sum(rewards) / len(rewards),
        "tokens": float(len(student)),
        "grad_norm": grad_norm,
    }


# --- config ------------------------------------------------------------------


@dataclass
class OPDConfig:
    """On-policy settings, resolved from the TOML with the paper's defaults."""

    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # 4-bit by default, and not for speed. Qwen2.5-7B in bf16 is ~15GB of weights and
    # this card has 11.7GB, so an unquantised teacher OOMs even when it is the only
    # model loaded -- which it has to be, see phase_score. NF4 costs a little accuracy
    # in the log-probabilities; that is a far better trade than no measurement at all.
    teacher_load_in_4bit: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    cold_start_epochs: float = 1.0
    opd_epochs: int = 1
    # Resample the student's own rollouts every N epochs instead of reusing one frozen
    # set. 0 = never refresh, which is exactly the v10 configuration that collapsed.
    #
    # *** WHY THIS EXISTS ***
    # v10 built its examples once and took 10 full-batch steps over the same 11308
    # tokens. 87.95% of them had a NEGATIVE advantage (raw mean -4.97 nats), so with
    # use_baseline=false and batch_std scaling that does not recentre, every single
    # update pushed 88% of the student's own tokens DOWN and raised nothing -- ten
    # times, on tokens that never moved. The probability mass had nowhere to go and
    # landed on repetition: all 30 bench replies became `the the the ...`, charter
    # substance 10/10 -> 1/10, while the loss fell monotonically to -62.27.
    #
    # Refreshing makes the update genuinely on-policy: the student's *current* output
    # is what gets scored, so a token suppressed in epoch 1 has been replaced by
    # epoch 2 rather than suppressed twice. It also means mean_reward stops being
    # constant -- a flat mean_reward across epochs is the signature of frozen rollouts.
    #
    # This is NOT a recentring and NOT a baseline. Both of those would raise the
    # probability of half the tokens in a response the teacher dislikes uniformly,
    # which is the exact failure mode a baseline is documented here as unable to fix.
    # Refresh keeps every advantage negative when it should be negative and changes
    # only *which* tokens are being judged.
    refresh_rollouts_every: int = 0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 640
    max_new_tokens: int = 160
    rollouts_per_prompt: int = 1
    temperature: float = 1.0
    seed: int = 20260901
    # Prompt selection. Li et al. keep prompts where the teacher has a strong
    # preference; a prompt the teacher is indifferent about carries no signal and, on
    # a charter that demands answers rather than hedging, teaches hedging.
    min_teacher_confidence: float = 0.0
    # "batch_std" keeps grad_norm near 1 so the clipper is not doing the learning. With
    # "none" the measured grad_norm was 6614 against a clip of 1.0, which means the
    # update direction was fine and the step size was ~1500x too small.
    advantage_norm: str = "batch_std"
    # Left at 1.0 deliberately: with advantages normalised, 1.0 is a genuine safety net
    # rather than a thing silently dividing every gradient by thousands. Raising it is
    # only safe *after* normalising.
    max_grad_norm: float = 1.0
    # Reversed from the usual REINFORCE default; see opd_loss for why a sequence-mean
    # baseline would zero out exactly the uniformly-bad rollouts we need to correct.
    use_baseline: bool = False
    cold_start_dir: Optional[str] = None
    output_dir: str = "out/opd"

    @classmethod
    def from_toml(cls, path: Path) -> "OPDConfig":
        import tomllib

        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        student = raw.get("student", {})
        teacher = raw.get("teacher", {})
        train = raw.get("train", {})
        opd = raw.get("opd", {})
        defaults = cls()
        merged: dict[str, Any] = {
            "student_model": student.get("model_id", defaults.student_model),
            "teacher_model": teacher.get("model_id", defaults.teacher_model),
            "lora_r": train.get("lora_r", defaults.lora_r),
            "lora_alpha": train.get("lora_alpha", defaults.lora_alpha),
            "lora_dropout": train.get("lora_dropout", defaults.lora_dropout),
            "target_modules": train.get("target_modules", defaults.target_modules),
            "learning_rate": train.get("learning_rate", defaults.learning_rate),
            "warmup_ratio": train.get("warmup_ratio", defaults.warmup_ratio),
            "cold_start_epochs": train.get("epochs", defaults.cold_start_epochs),
            "per_device_batch_size": train.get(
                "per_device_batch_size", defaults.per_device_batch_size
            ),
            "gradient_accumulation_steps": train.get(
                "gradient_accumulation_steps", defaults.gradient_accumulation_steps
            ),
            "max_seq_len": train.get("max_seq_len", defaults.max_seq_len),
            "seed": train.get("seed", defaults.seed),
        }
        # [opd] wins over [train]: it is the more specific table.
        merged.update(opd)
        known = set(cls.__dataclass_fields__)
        unknown = set(merged) - known
        if unknown:
            # Silently ignoring a typo'd key is how a run ends up training with the
            # default learning rate and nobody noticing until the loss looks wrong.
            raise ValueError(
                f"unknown [opd] keys in {path}: {sorted(unknown)}. "
                f"Known keys: {sorted(known)}"
            )
        return cls(**merged)


def select_teacher_aligned_prompts(
    rows: list[dict[str, Any]], min_confidence: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep prompts where the teacher has a strong preference, drop the rest.

    ``min_confidence`` is the minimum absolute mean per-token advantage: the average
    size of the gap between teacher and student. A prompt where the teacher scores the
    student's tokens roughly as the student does carries no correction to make, and on a
    charter that says *answer* rather than *hedge*, distilling an undecided teacher is
    actively harmful.

    Rows without a ``teacher_advantage`` field are kept. That is deliberate: the field
    only exists after a scoring pass has run, and refusing to train until every prompt
    has one would make the first, diagnostic run impossible to start.
    """
    if min_confidence <= 0:
        return list(rows), {"kept": len(rows), "dropped": 0}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        advantage = row.get("teacher_advantage")
        if advantage is None or abs(float(advantage)) >= min_confidence:
            kept.append(row)
        else:
            dropped += 1
    return kept, {"kept": len(kept), "dropped": dropped}


# --- rollout and scoring -----------------------------------------------------


def load_pairs(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty; there is nothing to distil")
    return rows


def rollout_prompts(
    student: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    cfg: OPDConfig,
) -> list[dict[str, Any]]:
    """Sample responses from the student itself.

    This is the step that makes the whole method on-policy: every response scored in
    the next function came from here, so the training states are ones the student can
    actually reach. ``do_sample=True`` with the configured temperature is required --
    greedy decoding would give one response per prompt and the student would never visit
    the off-distribution states where its mistakes live.
    """
    import torch

    out: list[dict[str, Any]] = []
    for row in rows:
        messages = to_messages(row)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt_text, return_tensors="pt").to(student.device)
        with torch.no_grad():
            generated = student.generate(
                **encoded,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                num_return_sequences=cfg.rollouts_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
            )
        for sample in generated:
            new_tokens = sample[encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if not text.strip():
                # An empty rollout has no tokens to score and contributes a zero-length
                # sequence. Dropping it silently would look like fewer prompts.
                continue
            out.append({**row, "rollout": text.strip()})
    return out


def score_rollouts(
    teacher: Any,
    tokenizer: Any,
    rollouts: list[dict[str, Any]],
    *,
    max_seq_len: int,
    device: Any,
) -> list[dict[str, Any]]:
    """Attach the teacher's per-token log-probabilities to each student rollout.

    Returns rows carrying ``teacher_logprobs`` and ``teacher_advantage`` (the mean
    absolute gap). Both are detached floats: this is a scoring pass, not a training
    graph, and letting autograd reach the teacher would spend 14GB of VRAM to compute a
    gradient we discard.
    """
    import torch

    scored: list[dict[str, Any]] = []
    for row in rollouts:
        messages = to_messages({**row, "response": row["rollout"]})
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][
            :max_seq_len
        ]
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        if len(full_ids) <= prompt_len:
            continue
        ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            logits = teacher(ids).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = ids[0, 1:]
        # Position t-1 predicts token t. The first prompt_len-1 positions are skipped
        # because they predict prompt tokens, not the reply.
        reply = (
            logprobs[prompt_len - 1 : -1, :]
            .gather(1, targets[prompt_len - 1 :].unsqueeze(1))
            .squeeze(1)
        )
        values = [float(v) for v in reply.tolist()]
        # The student's own log-probs on these tokens. Without them the advantage would
        # be measured against nothing and the "how far apart are they" number that
        # decides whether OPD has anything to teach would be meaningless.
        student_values = row.get("student_logprobs") or []
        if len(student_values) != len(values):
            student_values = values
            log.debug(
                "rollout %r: no student log-probs of matching length; the advantage "
                "for this row is a teacher self-comparison and is not meaningful",
                row.get("id"),
            )
        rewards = token_rewards(student_values, values)
        scored.append(
            {
                **row,
                "teacher_logprobs": values,
                "teacher_advantage": sum(abs(r) for r in rewards) / len(rewards),
            }
        )
    return scored


# --- the update step ---------------------------------------------------------


def build_rollout_example(
    tokenizer: Any, row: dict[str, Any], max_seq_len: int
) -> Optional[dict[str, Any]]:
    """Tokenise one scored rollout into a tensor row the update step can consume.

    Returns ``input_ids``, ``labels`` (reply tokens, -100 elsewhere) and ``advantages``
    aligned to those labels. The alignment is the fiddly part and the place a silent bug
    would live: position t-1 predicts token t, so an off-by-one shifts every reward and
    the model learns from noise while the loss still goes down.
    """
    import torch

    messages = to_messages({**row, "response": row["rollout"]})
    full_ids = tokenizer(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ),
        add_special_tokens=False,
    )["input_ids"][:max_seq_len]
    prompt_len = len(
        tokenizer(
            tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )["input_ids"]
    )
    teacher_lp = row.get("teacher_logprobs") or []
    student_lp = row.get("student_logprobs") or []
    reply_len = min(len(full_ids) - prompt_len, len(teacher_lp), len(student_lp))
    if reply_len <= 0:
        return None

    # rewards[i] is the advantage for token prompt_len+i, which is predicted by the
    # logits at position prompt_len+i-1.
    rewards = token_rewards(student_lp[:reply_len], teacher_lp[:reply_len])
    labels = [-100] * (prompt_len - 1) + full_ids[prompt_len : prompt_len + reply_len]
    return {
        "input_ids": torch.tensor(full_ids),
        "labels": torch.tensor(labels),
        "advantages": torch.tensor(rewards, dtype=torch.float32),
        "reply_len": reply_len,
    }


def opd_update_step(
    model: Any,
    optimizer: Any,
    examples: list[dict[str, Any]],
    *,
    max_grad_norm: float = 1.0,
    advantage_norm: str = "batch_std",
) -> dict[str, float]:
    """One reverse-KL policy-gradient update over scored rollouts.

    L = -mean_t( A_t * log pi_S(y_t | x, y_<t) ), with A_t a detached constant.

    This is REINFORCE with a dense per-token reward, so the reward is just a
    coefficient: the gradient w.r.t. the logits is ``-(A_t * onehot(y_t) - softmax)``.
    The teacher's vocabulary never appears, which is why scoring is cheap -- only the
    log-probabilities on tokens the student already sampled were ever needed.

    ``mean_reward`` is the RAW mean advantage, in nats, summed from
    ``example["advantages"]`` before any normalisation -- so it is directly comparable
    to the ``--score-only`` gap. The scaled one would be just mean/std of the same batch:
    a constant that says nothing about how wrong the student is.

    It is flat across epochs only while the rollout set is frozen (the v10 setting,
    ``refresh_rollouts_every = 0``). With resampling on, ``mean_reward`` moves between
    epochs, and that movement is the point: it is the cheapest evidence that the update
    is genuinely on-policy rather than re-suppressing the same tokens.

    ``examples`` is read, never written: the batch_std copy stays local, so the raw
    advantages survive for the next epoch and for this statistic.
    """
    import torch

    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Normalise across the WHOLE update, not per sequence. A per-sequence std would
    # rescale each rollout by its own spread, and a rollout the teacher dislikes by a
    # uniform 4 nats would get a std of ~0 and be left unscaled -- so the very rows with
    # the largest consistent signal would keep their oversized gradients.
    #
    # The scaled values go into a LOCAL list, never back into `examples`. Writing them
    # back silently destroyed the raw advantages: batch_std is idempotent, so epoch 2
    # re-normalised already-normalised values, every later epoch repeated that, and
    # `mean_reward` was reported as mean/std -- a constant -0.91 in the run that caught
    # this -- instead of the teacher-minus-student gap it is documented to be. The loss
    # still had the right scaled advantages to learn from, so nothing in the update
    # looked wrong; only the number a reader would judge the run by was wrong.
    scaled_advantages: list[Any] = []
    if advantage_norm != "none" and examples:
        all_adv = [float(a) for e in examples for a in e["advantages"].tolist()]
        scaled = normalise_advantages(all_adv, advantage_norm)
        cursor = 0
        for example in examples:
            n = example["advantages"].numel()
            scaled_advantages.append(
                torch.tensor(scaled[cursor : cursor + n], dtype=torch.float32)
            )
            cursor += n
    else:
        scaled_advantages = [example["advantages"].float() for example in examples]

    total_loss = 0.0
    total_tokens = 0
    total_reward = 0.0
    skipped = 0
    for example, scaled_advantage in zip(examples, scaled_advantages):
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        labels = example["labels"].unsqueeze(0).to(device)
        advantages = scaled_advantage.to(device)
        # The RAW advantage, summed for `mean_reward`. The scaled one would report the
        # normalisation's own mean/std, which says nothing about how wrong the student
        # is and is not comparable to the `--score-only` gap.
        raw_advantages = example["advantages"].to(device)
        reply_len = example["reply_len"]

        logits = model(input_ids).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        # The shift. logits[t] predicts token t+1, so the rows that predict reply tokens
        # are rows 0..len-2 -- one fewer than there are input positions. `labels` was
        # built with the same shift in build_rollout_example, so masking `labels` against
        # the UNSHIFTED logprobs is off by one and raises on any real prompt+reply.
        shifted = logprobs[:-1]
        targets = labels[0]
        mask = targets != -100
        # The reward is attached to the token the student actually emitted, so the
        # log-probabilities have to be gathered down to that token. Multiplying the
        # advantage by the whole (n_reply, vocab) row instead is a shape error, and
        # averaging it would be a silent bug that rewards every token equally.
        reply_logprobs = shifted[mask].gather(1, targets[mask].unsqueeze(1)).squeeze(1)
        # shape[0], NOT numel(): a mismatch here would silently skip every rollout and
        # the run would train on nothing while reporting a clean epoch.
        if reply_logprobs.shape[0] != reply_len:
            # Never train on a misaligned row. Truncating would apply the wrong
            # advantage to the wrong token, which is worse than skipping.
            skipped += 1
            continue
        loss = -(advantages.to(reply_logprobs.device) * reply_logprobs).mean()
        loss.backward()

        total_loss += float(loss.detach()) * reply_len
        total_tokens += reply_len
        total_reward += float(raw_advantages.sum())

    if skipped:
        log.warning("skipped %d rollout(s) with misaligned labels/advantages", skipped)

    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
    )
    if total_tokens:
        optimizer.step()
    return {
        "loss": total_loss / total_tokens if total_tokens else 0.0,
        "mean_reward": total_reward / total_tokens if total_tokens else 0.0,
        "tokens": float(total_tokens),
        "grad_norm": grad_norm,
    }


def _warn_if_update_is_net_suppressive(
    examples: list[dict[str, Any]], cfg: OPDConfig, threshold: float = 0.80
) -> dict[str, float]:
    """Say out loud when the batch teaches the student to stop talking.

    With ``use_baseline`` off and ``advantage_norm`` scaling without recentring -- both
    deliberate, both documented above -- the update has no force that raises the
    probability of anything. Each token with a negative advantage is pushed DOWN, each
    positive one UP, and if almost every token is negative the whole step is suppression
    of the student's own output with nowhere for the probability mass to go.

    That is not hypothetical. The v10 run measured 87.95% of tokens negative (mean
    -4.97 nats over 11308 tokens), ran ten steps over the FIXED rollout set -- the
    advantages never refresh, so it suppressed the same tokens ten times -- and the
    student collapsed: every one of 30 bench replies became ``the the the``, charter
    substance fell 10/10 to 1/10 and activation 0.92 to 0.00, while the training loss
    fell monotonically to -62 the whole way. The loss never showed anything was wrong,
    which is why this warning exists.

    Returns the statistics so tests can pin them without recomputing.
    """
    values = [float(a) for e in examples for a in e["advantages"].tolist()]
    if not values:
        return {}
    negative = sum(1 for v in values if v < 0)
    fraction_negative = negative / len(values)
    mean = sum(values) / len(values)
    mean_abs = sum(abs(v) for v in values) / len(values)
    stats = {
        "tokens": float(len(values)),
        "mean_advantage": mean,
        "mean_abs_advantage": mean_abs,
        "fraction_negative": fraction_negative,
    }
    if fraction_negative >= threshold:
        log.warning(
            "[opd] %d of %d tokens (%.1f%%) have a NEGATIVE advantage: mean %.3f nats, "
            "mean |gap| %.3f. With use_baseline=%s and advantage_norm=%s no term raises "
            "any token's probability, so every step is pure suppression of the student's "
            "own rollouts -- and the advantages are frozen, so the same tokens are "
            "suppressed %d time(s) without ever being resampled. A falling loss will NOT "
            "warn you: v10 lost to -62 and collapsed to repetition while it fell. Judge "
            "the result on eval_bench.py --vignettes.",
            negative,
            len(values),
            100.0 * fraction_negative,
            mean,
            mean_abs,
            cfg.use_baseline,
            cfg.advantage_norm,
            cfg.opd_epochs,
        )
    return stats


def build_examples(
    tokenizer: Any, scored: list[dict[str, Any]], cfg: OPDConfig
) -> list[dict[str, Any]]:
    """Tokenise scored rollouts into training examples, dropping unusable ones."""
    examples = []
    for row in scored:
        example = build_rollout_example(tokenizer, row, cfg.max_seq_len)
        if example is not None:
            examples.append(example)
    if not examples:
        raise RuntimeError(
            "No rollout could be aligned to its teacher scores. If every rollout is "
            "skipped the update step would silently train on nothing."
        )
    return examples


def run_opd_epochs(
    model: Any,
    tokenizer: Any,
    scored: list[dict[str, Any]],
    cfg: OPDConfig,
    *,
    resample: Optional[Any] = None,
) -> list[dict[str, float]]:
    """Run ``cfg.opd_epochs`` updates, optionally resampling the rollouts between them.

    ``resample`` is a callable taking the live model and returning a fresh scored
    rollout list, or ``None`` to keep the set this call was given. It is injected
    rather than called directly for two reasons:

    * it needs the teacher resident, and the teacher cannot be resident alongside
      the student on 11.7 GB -- so the real implementation releases the student,
      re-runs the rollout and score stages, and reloads. Passing it in keeps that
      choreography out of the loop and makes the loop itself testable with no GPU;
    * a test can pass a callable that returns a *known* batch, which is the only
      way to prove the loop actually rebuilds its examples instead of quietly
      reusing the frozen set that destroyed v10.

    With ``cfg.refresh_rollouts_every`` 0 (the v10 setting) nothing is resampled
    and the behaviour is byte-for-byte the old one: examples built once, mean_reward
    flat across epochs.
    """
    import torch

    examples = build_examples(tokenizer, scored, cfg)
    log.info("built %d training examples from scored rollouts", len(examples))
    _warn_if_update_is_net_suppressive(examples, cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "No trainable parameters. The cold-start adapter was probably not loaded, "
            "so the update step would compute gradients against frozen base weights."
        )
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)

    history: list[dict[str, float]] = []
    for epoch in range(cfg.opd_epochs):
        # Refresh BEFORE the update, not after, so epoch 1 still trains on the
        # rollouts that were handed in. Resampling first would throw that set away
        # untrained and, on a fresh cold start, the very first measured gap would
        # never be used.
        should_refresh = (
            resample is not None
            and cfg.refresh_rollouts_every > 0
            and epoch > 0
            and epoch % cfg.refresh_rollouts_every == 0
        )
        if should_refresh:
            refreshed = resample(model)
            if refreshed:
                examples = build_examples(tokenizer, refreshed, cfg)
                log.info(
                    "epoch %d: resampled rollouts -> %d examples",
                    epoch + 1,
                    len(examples),
                )
                # The refreshed set is a NEW batch, so the sign check has to run on
                # it. Reusing the first batch's verdict would report epoch 1's
                # distribution while training on epoch 2's.
                _warn_if_update_is_net_suppressive(examples, cfg)
            else:
                log.warning(
                    "epoch %d: resampling produced no usable rollouts; "
                    "continuing on the previous set",
                    epoch + 1,
                )
        stats = opd_update_step(
            model,
            optimizer,
            examples,
            max_grad_norm=cfg.max_grad_norm,
            advantage_norm=cfg.advantage_norm,
        )
        stats["epoch"] = float(epoch + 1)
        history.append(stats)
        log.info(
            "epoch %d/%d: loss=%.4f mean_reward=%.4f tokens=%.0f grad_norm=%.3f",
            epoch + 1,
            cfg.opd_epochs,
            stats["loss"],
            stats["mean_reward"],
            stats["tokens"],
            stats["grad_norm"],
        )
    return history


# --- self-test ---------------------------------------------------------------


def self_test() -> int:
    """Check the objective's arithmetic without torch, a teacher, or a GPU.

    The maths is the part most likely to be silently wrong and the part hardest to
    notice, because a mis-signed advantage still produces a falling training loss. So
    the properties are asserted rather than the numbers: a student that already matches
    the teacher must produce zero gradient, and a student that is confidently wrong must
    be pushed toward the teacher.
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            log.info("ok   %s", name)
        else:
            log.error("FAIL %s %s", name, detail)
            failures.append(name)

    # Identical distributions: nothing to correct.
    loss, diag = opd_loss([-0.5, -0.25, -1.0], [-0.5, -0.25, -1.0])
    check(
        "matching distributions give zero loss",
        abs(loss) < 1e-12,
        f"loss={loss}",
    )
    check(
        "matching distributions give zero mean reward",
        abs(diag["mean_reward"]) < 1e-12,
    )

    # A uniformly better teacher: every token positive. Here the mean is 1.25, and the
    # per-token advantages must sum to zero around it -- that is what makes the baseline
    # a pure variance reduction rather than a change of objective.
    rewards = token_rewards([-2.0, -2.0], [-1.0, -0.5])
    check("teacher-preferred tokens score positive", all(r > 0 for r in rewards))
    check(
        "baseline is the sequence mean", abs(per_token_baseline(rewards) - 1.25) < 1e-12
    )
    centred = [r - per_token_baseline(rewards) for r in rewards]
    check(
        "centred advantages sum to zero",
        abs(sum(centred)) < 1e-12,
        f"sum={sum(centred)}",
    )

    # Baseline subtraction must not change the loss when the student log-probs are all
    # equal, because the advantage then shifts by a constant times a constant.
    loss_b, _ = opd_loss([-1.0, -1.0, -1.0], [-2.0, -1.0, -0.5], use_baseline=True)
    loss_n, _ = opd_loss([-1.0, -1.0, -1.0], [-2.0, -1.0, -0.5], use_baseline=False)
    check(
        "baseline only reduces variance, not the mean direction",
        math.isfinite(loss_b) and math.isfinite(loss_n),
    )

    # The uniformly-bad case, and the reason the baseline defaults off. A response the
    # teacher dislikes at EVERY token -- which is what template collapse looks like --
    # must still produce a gradient. With a sequence-mean baseline every advantage is
    # exactly zero here and the failure mode we are training to remove gets no signal.
    uniform, _ = opd_loss([-0.1, -0.1, -0.1], [-4.0, -4.0, -4.0])
    uniform_with_baseline, _ = opd_loss(
        [-0.1, -0.1, -0.1], [-4.0, -4.0, -4.0], use_baseline=True
    )
    check(
        "a uniformly wrong rollout still gets a gradient",
        uniform != 0.0,
        f"loss={uniform}",
    )
    check(
        "a sequence-mean baseline would erase that gradient",
        uniform_with_baseline == 0.0,
        f"loss_with_baseline={uniform_with_baseline}",
    )
    check(
        "the baseline is off by default",
        OPDConfig().use_baseline is False,
    )

    # Mismatched lengths must be rejected, not silently truncated.
    try:
        token_rewards([-1.0, -1.0], [-1.0])
        check("length mismatch rejected", False, "no error raised")
    except ValueError:
        check("length mismatch rejected", True)

    # Empty input must not divide by zero.
    loss_e, diag_e = opd_loss([], [])
    check("empty rollout is safe", loss_e == 0.0 and diag_e["tokens"] == 0.0)

    # Config: a typo'd [opd] key must fail loudly rather than train on defaults.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "c.toml"
        bad.write_text(
            '[student]\nmodel_id = "x"\n[opd]\nlearning_ratee = 1e-3\n',
            encoding="utf-8",
        )
        try:
            OPDConfig.from_toml(bad)
            check("unknown config key rejected", False, "no error raised")
        except ValueError:
            check("unknown config key rejected", True)

    # Prompt selection drops the prompts where the teacher has no preference.
    rows = [
        {"id": "keep", "teacher_advantage": 0.9},
        {"id": "drop", "teacher_advantage": 0.01},
        {"id": "keep-unscored"},
    ]
    kept, counts = select_teacher_aligned_prompts(rows, 0.5)
    check(
        "teacher-aligned selection drops uninformative prompts",
        [r["id"] for r in kept] == ["keep", "keep-unscored"],
        f"kept={[r['id'] for r in kept]} counts={counts}",
    )
    kept0, _ = select_teacher_aligned_prompts(rows, 0.0)
    check("selection off by default keeps everything", len(kept0) == 3)

    if failures:
        log.error("%d self-test check(s) failed: %s", len(failures), failures)
        return 1
    log.info("all self-test checks passed")
    return 0


# --- CLI ---------------------------------------------------------------------


def _load_models(cfg: OPDConfig, *, load_teacher: bool) -> tuple[Any, Any, Any]:
    """Load student (+ teacher, optionally) and the student's tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.seed was parsed from the TOML and never read: rollout sampling and
    # LoRA dropout ran unseeded, so no run could be reproduced from its config.
    torch.manual_seed(cfg.seed)

    student = _load_student(cfg, device)
    student.eval()

    teacher = None
    if load_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg.teacher_model, torch_dtype="auto", device_map="auto"
        )
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        log.info(
            "teacher resident for scoring; this needs VRAM for %s", cfg.teacher_model
        )
    return student, teacher, tokenizer


def _load_student(cfg: OPDConfig, device: str) -> Any:
    """Load the base model with exactly one LoRA: the cold start, or a fresh one.

    Exactly one -- never both. The original code loaded the cold-start adapter
    and then called ``_attach_lora`` unconditionally, stacking a freshly
    initialised LoRA on top. Two things made that fatal rather than wasteful:

    * ``PeftModel.from_pretrained`` freezes the adapter it loads unless
      ``is_trainable=True``, so the optimizer trained only the throwaway outer
      adapter while the cold start sat frozen underneath it; and
    * ``save_pretrained`` writes only the outer adapter, so the artifact had
      keys with a doubled ``base_model.model.`` prefix, no
      ``base_model_name_or_path``, and none of the cold start's weights.

    A run trained that way optimises a random LoRA against a student whose
    behaviour is still entirely the frozen cold start, then saves weights that
    are neither. Both pre-fix v10 OPD runs were invalidated exactly this way.
    """
    from transformers import AutoModelForCausalLM

    student = AutoModelForCausalLM.from_pretrained(
        cfg.student_model, torch_dtype="auto"
    ).to(device)
    if cfg.cold_start_dir:
        from peft import PeftModel  # type: ignore[import-not-found]

        # is_trainable=True is the fix: the default (False) would freeze the
        # cold start and leave nothing but a fresh adapter to train.
        student = PeftModel.from_pretrained(
            student, cfg.cold_start_dir, is_trainable=True
        )
        student.print_trainable_parameters()
        log.info("resumed cold-start adapter from %s (trainable)", cfg.cold_start_dir)
    else:
        student = _attach_lora(student, cfg)
    return student


def _attach_lora(model: Any, cfg: OPDConfig) -> Any:
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            task_type="CAUSAL_LM",
        ),
    )
    wrapped.print_trainable_parameters()
    return wrapped


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _release(model: Any) -> None:
    """Drop a model and return its VRAM before loading the next one.

    This is the whole reason the phases exist. A 7B teacher in bf16 is ~15GB and this
    card has 11.7GB, so student+teacher together cannot be resident -- and the failure is
    an OOM several minutes in, after the rollouts have already been generated and
    thrown away.
    """
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_rollout(cfg: OPDConfig, rows: list[dict[str, Any]], out: Path) -> Path:
    """Stage 1: student only. Sample responses and record the student's own log-probs."""
    student, _teacher, tokenizer = _load_models(cfg, load_teacher=False)
    device = next(student.parameters()).device
    rollouts = rollout_prompts(student, tokenizer, rows, cfg)
    log.info("student produced %d rollouts", len(rollouts))
    if not rollouts:
        raise RuntimeError("the student produced no usable rollouts")
    _score_student_logprobs(student, tokenizer, rollouts, cfg, device)
    path = out / "rollouts.jsonl"
    _write_jsonl(path, rollouts)
    log.info("wrote %s", path)
    _release(student)
    return path


def phase_score(cfg: OPDConfig, rollouts_path: Path, out: Path) -> Path:
    """Stage 2: teacher only. Score the rollouts it has never seen generated locally."""
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    rollouts = _read_jsonl(rollouts_path)
    if not rollouts:
        raise RuntimeError(f"{rollouts_path} has no rollouts to score")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
    if cfg.teacher_load_in_4bit:
        # Only worth it on CUDA; bitsandbytes has no CPU path.
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        log.info("teacher in 4-bit NF4: bf16 weights do not fit on this GPU")
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, **load_kwargs)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    log.info("teacher loaded alone: %s", cfg.teacher_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scored = score_rollouts(
        teacher, tokenizer, rollouts, max_seq_len=cfg.max_seq_len, device=device
    )
    path = out / "scored.jsonl"
    _write_jsonl(path, scored)
    log.info("scored %d rollouts -> %s", len(scored), path)
    _release(teacher)
    return path


def _make_resampler(
    cfg: OPDConfig,
    rows: list[dict[str, Any]],
    out: Path,
    *,
    config_path: Path,
    corpus_path: Path,
    max_prompts: int = 0,
) -> Any:
    """Build the callable ``run_opd_epochs`` uses to resample the student's rollouts.

    Why a subprocess and not a function call
    ----------------------------------------
    Resampling needs the teacher, and the teacher cannot be resident alongside the
    student: a 7B in bf16 is ~15GB against this card's 11.7GB. Calling
    ``phase_rollout``/``phase_score`` here would load a second model into a process
    that already holds one and OOM -- and it would OOM *after* generating the
    rollouts, which is the exact failure ``_release``'s docstring says the phase
    split exists to prevent.

    So the refresh runs as its own process, one model at a time, exactly like a
    manual ``--score-only`` invocation. The parent keeps its optimizer and its
    student untouched: only the *data* is refreshed, which means the AdamW state
    stays attached to live parameters across a refresh instead of being orphaned
    by a reload.

    The current policy is persisted first (``adapter.save_pretrained``) and handed
    to the child with ``--cold-start-dir``. Without that the child would roll out
    from the cold start, and the "refreshed" rollouts would be the same ones the
    first epoch already trained on -- the v10 failure with extra steps.
    """
    import subprocess
    import sys

    script = Path(__file__).resolve()
    refresh_dir = out / "refresh-adapter"

    def _resample(model: Any) -> list[dict[str, Any]]:
        model.save_pretrained(refresh_dir)
        log.info("persisted live adapter to %s for the refresh cycle", refresh_dir)

        argv = [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--corpus",
            str(corpus_path),
            "--score-only",
            "--cold-start-dir",
            str(refresh_dir),
        ]
        if max_prompts:
            argv += ["--max-prompts", str(max_prompts)]
        # --score-only writes rollouts.jsonl and scored.jsonl into cfg.output_dir,
        # which is `out` -- the same directory this phase is writing to.
        log.info("refresh cycle: %s", " ".join(argv))
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0:
            # Never fall back to the stale batch quietly: training on last epoch's
            # frozen rollouts while the log says "resampled" is how v10 happened.
            raise RuntimeError(
                "rollout refresh failed (exit "
                f"{completed.returncode}); refusing to reuse stale rollouts.\n"
                f"stdout:\n{completed.stdout[-2000:]}\n"
                f"stderr:\n{completed.stderr[-2000:]}"
            )
        scored = _read_jsonl(out / "scored.jsonl")
        kept, counts = select_teacher_aligned_prompts(
            scored, cfg.min_teacher_confidence
        )
        log.info(
            "refresh cycle: kept %d, dropped %d", counts["kept"], counts["dropped"]
        )
        return kept

    return _resample


def phase_train(
    cfg: OPDConfig,
    scored_path: Path,
    out: Path,
    rows: Optional[list[dict[str, Any]]] = None,
    *,
    config_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    max_prompts: int = 0,
) -> None:
    """Stage 3: student only again. Reverse-KL updates on the scored rollouts."""
    scored = _read_jsonl(scored_path)
    if not scored:
        raise RuntimeError(f"{scored_path} is empty; nothing to train on")
    if not cfg.cold_start_dir:
        # Li et al. 2026: for a same-family teacher the off-policy cold start is
        # not optional, and training OPD from a fresh random LoRA would report a
        # falling loss while teaching nothing about the charter.
        log.warning("[opd] no cold_start_dir set: training from a fresh random LoRA")
    student, _teacher, tokenizer = _load_models(cfg, load_teacher=False)
    kept, counts = select_teacher_aligned_prompts(scored, cfg.min_teacher_confidence)
    log.info("prompt selection: kept %d, dropped %d", counts["kept"], counts["dropped"])

    # Resampling is only possible if the child process can be told what to roll out.
    # Silently degrading to the frozen-set behaviour is what makes a collapsed run
    # look like a completed one, so say so loudly and refuse to claim a refresh.
    resample = None
    if cfg.refresh_rollouts_every > 0:
        if rows and config_path and corpus_path:
            resample = _make_resampler(
                cfg,
                rows,
                out,
                config_path=config_path,
                corpus_path=corpus_path,
                max_prompts=max_prompts,
            )
            log.info(
                "resampling rollouts every %d epoch(s); mean_reward should now move",
                cfg.refresh_rollouts_every,
            )
        else:
            raise RuntimeError(
                "refresh_rollouts_every > 0 needs the corpus rows and the config "
                "path. Run --phase all, or pass --corpus and --config explicitly: "
                "a --phase train run cannot resample from scored.jsonl alone."
            )

    history = run_opd_epochs(student, tokenizer, kept, cfg, resample=resample)
    student.save_pretrained(out / "adapter")
    (out / "opd_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    log.info("saved adapter to %s", out / "adapter")
    _release(student)


def _score_student_logprobs(student, tokenizer, rollouts, cfg, device) -> None:
    """Record log pi_S on the tokens the student just sampled.

    These are the comparison for the advantage. Without them the teacher is scored
    against itself, the gap is always ~0, and the diagnostic would report that on-policy
    distillation has nothing to teach no matter what the truth is.
    """
    import torch
    from train_distill import to_messages

    for row in rollouts:
        messages = to_messages({**row, "response": row["rollout"]})
        full_ids = tokenizer(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ),
            add_special_tokens=False,
        )["input_ids"][: cfg.max_seq_len]
        prompt_len = len(
            tokenizer(
                tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        if len(full_ids) <= prompt_len:
            row["student_logprobs"] = []
            continue
        ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            logprobs = torch.log_softmax(student(ids).logits[0].float(), dim=-1)
        targets = ids[0, 1:]
        row["student_logprobs"] = [
            float(v)
            for v in logprobs[prompt_len - 1 : -1, :]
            .gather(1, targets[prompt_len - 1 :].unsqueeze(1))
            .squeeze(1)
            .tolist()
        ]


def _teacher_messages(vignette: dict[str, Any]) -> list[dict[str, str]]:
    """The app's real system prompts plus the vignette question, and nothing else.

    ``to_messages`` expects a completed example -- it appends the assistant reply -- so
    calling it with an empty response produced [system, system, user, assistant-empty] and
    the question was dropped. The teacher would have been asked to continue an empty
    answer instead of answering anything. Build the prefix explicitly and stop at the
    user turn, then apply ``add_generation_prompt``.
    """
    base = to_messages(
        {"prompt": "", "response": "x", "user_data": vignette.get("user_data", "")}
    )
    return [*base[:2], {"role": "user", "content": vignette["prompt"]}]


def _teacher_vignettes(cfg: OPDConfig) -> int:
    """Generate the teacher's own answers to the ten charter vignettes.

    The diagnostic gap says the teacher and the student disagree. It does NOT say the
    teacher is right -- Li et al. 2026's whole point is that a teacher with no advantage
    gives you nothing, and the mirror failure is a teacher that is confidently different
    for no good reason. Reading these ten replies is the cheapest available check on
    that, and it is a human judgement: no automated scorer can decide whether the
    pharmacology in c01 is sound.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_dir = Path("eval")
    vignettes = [
        json.loads(line)
        for line in (eval_dir / "asimov_vignettes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
    if cfg.teacher_load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, **load_kwargs)
    teacher.eval()

    out_rows = []
    for v in vignettes:
        row = {
            "id": v["id"],
            "prompt": v["prompt"],
            "kind": v["kind"],
            "category": v["category"],
            "reference": v.get("reference", ""),
        }
        # The teacher's own chat template, but with OUR system prompt: the whole question
        # is whether it can satisfy this app's charter, not whether it can chat.
        row["user_data"] = v.get("user_data", "")
        messages = _teacher_messages(v)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").to(teacher.device)
        with torch.no_grad():
            out = teacher.generate(
                **ids,
                max_new_tokens=320,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        row["teacher_response"] = tokenizer.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        out_rows.append(row)
        log.info("%s -> %s", v["id"], row["teacher_response"][:110])

    path = Path("out/teacher-vignettes.jsonl")
    _write_jsonl(path, out_rows)
    log.info("wrote %d teacher replies to %s", len(out_rows), path)
    print("\n=== teacher replies to the charter vignettes ===")
    for r in out_rows:
        print(f"\n--- {r['id']} {r['prompt']}")
        print(r["teacher_response"])
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain.v10.toml")
    parser.add_argument("--corpus", default="data/synthetic/train-v9.jsonl")
    parser.add_argument(
        "--phase",
        choices=("rollout", "score", "train", "all"),
        default="all",
        help=(
            "run one stage only. Stages are separate because this GPU has 11.7GB and a "
            "7B teacher in bf16 is ~15GB, so student and teacher cannot be resident "
            "together; loading both OOMs after the rollouts have already been generated."
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="shorthand for --phase rollout --phase score",
    )
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument(
        "--cold-start-dir",
        default=None,
        help=(
            "override [opd] cold_start_dir. Used by the rollout refresh cycle, which "
            "re-runs this script as a child process pointed at the adapter the parent "
            "just saved, so the refreshed rollouts come from the CURRENT policy rather "
            "than from the cold start."
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true", help="check the objective and exit"
    )
    parser.add_argument(
        "--teacher-vignettes",
        action="store_true",
        help=(
            "have the TEACHER answer the ten charter vignettes and stop. This is not a "
            "measurement of the student: it is the check on whether the teacher is "
            "worth distilling. A large |teacher-student| gap only says the two disagree, "
            "and a confidently-wrong teacher disagrees beautifully."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.self_test:
        return self_test()

    cfg = OPDConfig.from_toml(Path(args.config))
    # After cfg is bound: the dispatch used to sit above this and hit an
    # UnboundLocalError the first time it was actually exercised.
    if args.cold_start_dir:
        cfg.cold_start_dir = args.cold_start_dir
    if args.teacher_vignettes:
        return _teacher_vignettes(cfg)

    out = Path(cfg.output_dir)
    rows: list[dict[str, Any]] = []
    if args.phase in ("rollout", "all"):
        rows = load_pairs(Path(args.corpus))
        if args.max_prompts:
            rows = rows[: args.max_prompts]
        log.info("loaded %d prompts from %s", len(rows), args.corpus)

    stages = (
        ("rollout", "score")
        if args.score_only
        else (("rollout", "score", "train") if args.phase == "all" else (args.phase,))
    )

    rollouts_path = out / "rollouts.jsonl"
    scored_path = out / "scored.jsonl"
    if "rollout" in stages:
        rollouts_path = phase_rollout(cfg, rows, out)
    if "score" in stages:
        scored_path = phase_score(cfg, rollouts_path, out)
    if "score" in stages or args.score_only:
        scored = _read_jsonl(scored_path)
        gaps = [abs(r["teacher_advantage"]) for r in scored if "teacher_advantage" in r]
        if gaps:
            log.info(
                "mean |teacher - student| per token: %.4f (min %.4f, max %.4f)",
                sum(gaps) / len(gaps),
                min(gaps),
                max(gaps),
            )
            log.info(
                "This is the decisive number. ~0 means the teacher and student already "
                "agree and on-policy distillation has nothing to teach."
            )
        else:
            log.error("no rollout carried a teacher advantage; the measurement is void")
    if "train" in stages:
        # The refresh cycle needs the corpus and this config to re-run rollout+score
        # as a child process. A bare `--phase train` normally has neither, so load
        # them here rather than letting phase_train fail deep inside an epoch.
        train_rows = rows
        if cfg.refresh_rollouts_every > 0 and not train_rows:
            train_rows = load_pairs(Path(args.corpus))
            if args.max_prompts:
                train_rows = train_rows[: args.max_prompts]
            log.info("loaded %d prompts for the refresh cycle", len(train_rows))
        phase_train(
            cfg,
            scored_path,
            out,
            train_rows,
            config_path=Path(args.config).resolve(),
            corpus_path=Path(args.corpus).resolve(),
            max_prompts=args.max_prompts,
        )
    return 0


if __name__ == "__main__":
    ensure_app_importable()
    raise SystemExit(main())
