#!/usr/bin/env python3
"""Choose a teacher empirically, by measuring how far it is from the student.

Why this exists
---------------
Li et al. 2026 (arXiv:2604.13016) report that same-family teachers can be
distributionally indistinguishable from the student, in which case on-policy
distillation has nothing to teach. Our teacher (Qwen2.5-7B-Instruct) and our student
(Qwen2.5-0.5B-Instruct) are the same family, so "pick a better teacher" is not a matter
of taste -- it is a measurement, and guessing wastes a multi-hour run on a laptop that
has crashed twice for heat.

The measurement is deliberately crude and deliberately cheap: sample rollouts from the
student once, then ask each candidate teacher to score those same tokens. The mean
|log pi_T - log pi_S| per token says how much disagreement there is to train on.

Read it carefully, because a large number is necessary but not sufficient:
  - ~0 everywhere: the teacher agrees with the student already. Nothing to learn.
  - large: there is signal. It is only useful signal if the teacher's preference is
    CORRECT, which this script cannot tell you. Score the teacher's own replies on the
    vignettes before trusting a high number.

The correctness check is the other half of the job and is deliberately not automated
here: `eval_bench.py --vignettes` scores the student, and the teacher's replies have to
be read by a person. A teacher that is confidently wrong will widen the gap and make
this number look excellent.

Usage::

    python scripts/pick_teacher.py --prompts 40 \\
        --teachers Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_onpolicy import (  # noqa: E402
    OPDConfig,
    load_pairs,
    rollout_prompts,
    score_rollouts,
)

log = logging.getLogger("pick_teacher")


def summarise(name: str, scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and spread of the per-token teacher/student gap for one teacher."""
    advantages = [r["teacher_advantage"] for r in scored if "teacher_advantage" in r]
    if not advantages:
        return {"teacher": name, "scored": 0, "mean_abs_gap": 0.0}
    ordered = sorted(advantages)
    return {
        "teacher": name,
        "scored": len(advantages),
        "mean_abs_gap": sum(advantages) / len(advantages),
        "median_abs_gap": ordered[len(ordered) // 2],
        "max_abs_gap": ordered[-1],
        # A teacher that disagrees enormously on a couple of prompts and not at all on
        # the rest is a different failure mode from one that is uniformly a bit further
        # away, and only the second is useful for steady training.
        "p10_abs_gap": ordered[max(0, len(ordered) // 10)],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain.v10.toml")
    parser.add_argument("--corpus", default="data/synthetic/train-v9.jsonl")
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="Hugging Face ids to score the student against, one at a time",
    )
    parser.add_argument("--prompts", type=int, default=40)
    parser.add_argument("--output", default="out/teacher-diagnostic.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = OPDConfig.from_toml(Path(args.config))
    rows = load_pairs(Path(args.corpus))[: args.prompts]
    log.info("scoring %d prompts against %d teachers", len(rows), len(args.teachers))

    results = []
    for teacher_id in args.teachers:
        log.info("=== teacher: %s ===", teacher_id)
        cfg.teacher_model = teacher_id
        try:
            # The student rollouts are regenerated per teacher only because loading the
            # model twice is cheap relative to a wrong conclusion; reuse them by passing
            # --prompts small if this becomes the bottleneck.
            from train_onpolicy import _load_models

            student, teacher, tokenizer = _load_models(cfg, load_teacher=True)
            import torch

            device = next(student.parameters()).device
            rollouts = rollout_prompts(student, tokenizer, rows, cfg)
            for row in rollouts:
                row["student_logprobs"] = []
            _score_student(student, tokenizer, rollouts, cfg, device)
            scored = score_rollouts(
                teacher,
                tokenizer,
                rollouts,
                max_seq_len=cfg.max_seq_len,
                device=device,
            )
            result = summarise(teacher_id, scored)
            results.append(result)
            log.info("%s -> mean |gap| %.4f", teacher_id, result["mean_abs_gap"])

            del student, teacher
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            # One unavailable teacher must not lose the measurements from the others.
            log.error("teacher %s failed: %s", teacher_id, exc)
            results.append({"teacher": teacher_id, "error": str(exc)})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    print("\n=== teacher gap (higher = more to learn, if the teacher is right) ===")
    for r in sorted(
        (r for r in results if "mean_abs_gap" in r),
        key=lambda r: -r["mean_abs_gap"],
    ):
        print(f"  {r['mean_abs_gap']:.4f}  {r['teacher']}  (scored {r['scored']})")
    print(
        "\nA high gap is only useful if the teacher's own replies are correct.\n"
        "Read them before choosing one -- see the module docstring."
    )
    return 0


def _score_student(student, tokenizer, rollouts, cfg, device) -> None:
    """Attach the student's own log-probs; identical to the step in train_onpolicy."""
    import torch
    from train_distill import to_messages
    from train_onpolicy import build_rollout_example  # noqa: F401

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
            lp = torch.log_softmax(student(ids).logits[0].float(), dim=-1)
        targets = ids[0, 1:]
        row["student_logprobs"] = [
            float(v)
            for v in lp[prompt_len - 1 : -1, :]
            .gather(1, targets[prompt_len - 1 :].unsqueeze(1))
            .squeeze(1)
            .tolist()
        ]


if __name__ == "__main__":
    raise SystemExit(main())
