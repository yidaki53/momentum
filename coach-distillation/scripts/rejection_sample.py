#!/usr/bin/env python3
"""Rejection sampling against the bench's own scorer -- Best-of-N with a programmatic verifier.

Why this exists instead of more 7B distillation
------------------------------------------------
Three results say the current approach is the wrong one:

- *Law of Capacity Gap* (arXiv:2311.07052): the optimal teacher is roughly linearly
  correlated with the student. A 7B teacher into a 0.5B student is a 14x gap, well
  past where a bigger teacher yields a better student.
- *Small Models Struggle to Learn from Strong Reasoners* (arXiv:2502.12143):
  Qwen2.5-0.5B specifically shows a NEGATIVE delta when fine-tuned on the larger
  teacher's style. We are not accidentally degrading; the method is known to fail at
  exactly this pair.
- *Learning to Choose or Choosing to Learn* (arXiv:2505.17288, AISTATS 2026): SFT
  wins when the target behaviour is REALIZABLE in the teacher's outputs. When
  realizability fails, Best-of-N converges faster. Ours fails -- 144 of the 7B's
  rejections were "action not bounded to a small step" -- which is precisely the
  non-realizable case the paper says to switch to BoN for.

So the teacher is the wrong source and the method is the wrong one. What we have
that most people in this situation do not is a programmatic verifier: the bench's own
scorers decide pass/fail exactly, with no model judge and no human loop. That makes
Best-of-N cheap and exact -- sample N, score with the same code that runs the release
gate, keep what passes.

Samples come from the STUDENT, not the teacher, which is what sidesteps the capacity
gap: this is on-policy, so there is no cross-model distribution to transfer.

Usage:
    rejection_sample.py --student out/student-v5 --limit 40 --n 8
"""

import argparse
import importlib.util
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
CD_ROOT = HERE.parent

# Any CJK or Cyrillic run. Qwen is bilingual and the 0.5B leaks the other language
# occasionally; a short fragment rather than a whole reply, which is why no length
# filter catches it.
_OFF_LANGUAGE_RE = re.compile(r"[\u0400-\u04ff\u4e00-\u9fff\u3040-\u30ff]")


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"_rs_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shape_of(text: str) -> str:
    """Normalise a reply so near-identical phrasings collapse to one key.

    The v4 corpus had two shapes at 14% between them while 256 distinct strings, so a
    distinct-string count would not have shown the concentration. This erases the task
    object and the numbers, which are the parts that legitimately vary.
    """
    text = re.sub(
        r"\b(for|to)\s+[^,.]*?(section|draft|introduction|review|task|it|them)\b",
        " for <T>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(draft|introduction|methods|literature review|thesis)\b",
        "<T>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\d+", "N", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z ]", " ", text).lower()).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="out/student-v5")
    parser.add_argument("--config", default="configs/retrain.v5.toml")
    parser.add_argument("--output", default="data/synthetic/train-v6-rft.jsonl")
    parser.add_argument("--limit", type=int, default=40, help="situations to expand")
    parser.add_argument("--n", type=int, default=8, help="samples per situation")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--min-shapes",
        type=int,
        default=2,
        help="max rows sharing one normalised shape",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    pairs_mod = _load("build_distill_pairs", HERE / "build_distill_pairs.py")
    bench = _load("eval_bench", HERE / "eval_bench.py")
    lexicon = bench.load_lexicon(CD_ROOT / "eval" / "shame_lexicon.txt")

    try:
        import torch
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError:  # pragma: no cover - environment dependent
        raise SystemExit(
            "transformers and torch are required; install requirements.txt"
        )

    config = pairs_mod.load_config(pathlib.Path(args.config))
    student_id = config.get("student", {}).get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")

    # Seeds: the bounded tier. These are the situations where a correct reply exists and
    # the student is closest to producing one.
    seeds = list(pairs_mod.BOUNDED_ACTIONS)[: args.limit]
    if not seeds:
        raise SystemExit("no bounded seeds found")

    tokenizer = AutoTokenizer.from_pretrained(student_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict] = []
    shape_counts: dict[str, int] = {}
    stats = {"samples": 0, "passed": 0, "rejected_shape": 0, "rejected_language": 0}

    for index, (situation, constraint, _reference) in enumerate(seeds):
        digest = pairs_mod.USER_POOL[index % len(pairs_mod.USER_POOL)].digest()
        prompt = pairs_mod.build_teacher_prompt(situation, constraint, digest)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = tokenizer([rendered] * args.n, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(
            outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for candidate in decoded:
            text = candidate.strip()
            stats["samples"] += 1
            if not text:
                continue
            # The same code the release gate runs. Not a training-specific scorer: a
            # separate scorer is how a corpus gets built that its own bench then fails.
            if not pairs_mod.quality_gate(text, constraint=constraint)[0]:
                continue
            if not bench.score_activation(text)[0]:
                continue
            if not bench.score_dignity(text, lexicon)[0]:
                continue
            if not bench.score_grounding(text, user_data=digest)[0]:
                continue
            # No axis checks that a reply is in English. The 0.5B occasionally emits a
            # trailing fragment in another language ("...then stop.一封简单的邮件解决了
            # 问题") which passes every scorer above, because a Chinese sentence has no
            # shame phrasing and no unbounded action in it. Verification is only as good
            # as the properties it checks, and this one was unchecked until someone read
            # a sample rather than a tally.
            if _OFF_LANGUAGE_RE.search(text):
                stats["rejected_language"] += 1
                continue
            key = shape_of(text)
            if shape_counts.get(key, 0) >= args.min_shapes:
                # Without this the sampler piles onto the mode and the corpus gains
                # nothing but copies: distinct strings, one shape.
                stats["rejected_shape"] += 1
                continue
            shape_counts[key] = shape_counts.get(key, 0) + 1
            stats["passed"] += 1
            rows.append(
                {
                    "id": f"rft:{index:03d}:{shape_counts[key]}",
                    "kind": "rft",
                    "situation": situation,
                    "skill": "bounded first action, self-generated and verified",
                    "prompt": situation,
                    "constraint": constraint,
                    "response": text,
                    "citation": "first-party: on-policy rejection sampling",
                    "user_data": digest,
                }
            )

        if (index + 1) % 10 == 0:
            print(
                f"[rft] {index + 1}/{len(seeds)} situations, "
                f"{stats['passed']} kept, {len(shape_counts)} distinct shapes",
                flush=True,
            )

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"\nsamples generated : {stats['samples']}")
    print(f"kept             : {stats['passed']}")
    print(f"dropped as shape : {stats['rejected_shape']}")
    print(f"dropped as lang  : {stats['rejected_language']}")
    print(f"distinct shapes  : {len(shape_counts)}")
    print(f"pass rate        : {100 * stats['passed'] / max(1, stats['samples']):.1f}%")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
