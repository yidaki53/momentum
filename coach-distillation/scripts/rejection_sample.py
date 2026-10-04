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


# Round-2 seed situations. v6 sampled only the bounded tier, so its 308 rows span 57
# situations however many samples each produced. The bottleneck for v7 is coverage, not
# volume: a seed pool of 57 cannot teach the model a situation it is never asked about,
# and the 120-item holdout is exactly the set of situations it IS asked about on unseen
# prompts.
#
# So these are new situations, topically adjacent to the bounded tier -- so the correct
# answer is in-register -- but lexically distinct from every holdout prompt. A test
# asserts zero overlap with eval/golden_qa_extended.jsonl, which is the only measurement
# that has separated v3 from v4 and v5 from v6.
#
# Kept as data rather than folded into the bounded tier because these have no hand-written
# reference: they are sampling prompts, not training pairs. A hand-written answer for each
# would just be a longer bounded tier wearing a different name.
RFT_SEEDS: tuple[str, ...] = (
    "the reply I owe my manager has been sitting in my drafts since Monday",
    "I have three tabs of research open and no notes in any of them",
    "the presentation is in an hour and the slides are still an outline",
    "I keep tidying the flat and avoiding the thing the tidying is for",
    "I have to give a talk on Thursday and I have not written a word",
    "the revision my supervisor sent back is sitting unread in my inbox",
    "I cannot begin the application even though the form is already open",
    "my headphones are on and the task still has not started",
    "I have to sign the form and I have been looking at it for days",
    "the washing needs doing and I have decided it is a tomorrow problem",
    "I keep writing emails in my head and never sending any of them",
    "the coursework brief is open and I have read the same paragraph twice",
    "I have to make the call to the clinic and I am putting it off",
    "I bought the equipment and have not set any of it up",
    "my notes from the seminar are just a photograph of a whiteboard",
    "I keep opening the tracker and closing it again",
    "the reply to my landlord is drafted and I have rewritten it four times",
    "I have to ring the bank about the fee and I cannot face the call",
    "the exercise plan says five days and I have done none of them",
    "I have to scan the documents and the scanner is a whole separate job",
    "I keep meaning to look up the address and never do",
    "I have to update my will and I have been putting it off for years",
    "the email to cancel the subscription is written but not sent",
    "I have to be at the appointment in an hour and have not left",
    "I keep rearranging my desk instead of starting the work",
    "the reading list is fourteen items long and I have read none of them",
    "I have to write the summary tonight and have not opened the file",
    "my password reset email is unopened and I cannot log in",
    "I have to return the library books and they are overdue",
    "I keep telling myself I will start properly after the weekend",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="out/student-v5")
    parser.add_argument("--config", default="configs/retrain.v5.toml")
    parser.add_argument("--output", default="data/synthetic/train-v6-rft.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "cap the situations sampled (0 = the whole pool). The default used to be 40,"
            " which silently truncated the pool to the first 40 bounded situations and"
            " made --seed-source wide a no-op -- the whole point of the flag was to add"
            " seeds and the seeds past position 40 were never sampled. A cap that"
            " quietly discards the thing you changed is worse than no cap."
        ),
    )
    parser.add_argument("--n", type=int, default=8, help="samples per situation")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--min-shapes",
        type=int,
        default=2,
        help="max rows sharing one normalised shape",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--seed-source",
        choices=("bounded", "wide"),
        default="wide",
        help="bounded = the hand-written tier only (v6); wide = plus RFT_SEEDS",
    )
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

    # Seeds: by default the bounded tier PLUS the wider RFT_SEEDS pool, because coverage
    # is what round 2 needs. --seed-source bounded reproduces v6 exactly, which is what
    # the one-sided comparison in docs/ACTIVATION.md rests on.
    if args.seed_source == "bounded":
        seeds = [(s, c, None) for s, c, _r in pairs_mod.BOUNDED_ACTIONS]
    else:
        seeds = [(s, c, None) for s, c, _r in pairs_mod.BOUNDED_ACTIONS]
        seeds += [(s, pairs_mod.BOUNDED_ACTION_CONSTRAINT, None) for s in RFT_SEEDS]
    if args.limit:
        seeds = seeds[: args.limit]
    if not seeds:
        raise SystemExit("no seeds found")

    tokenizer = AutoTokenizer.from_pretrained(student_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict] = []
    shape_counts: dict[str, int] = {}
    stats = {
        "samples": 0,
        "passed": 0,
        "rejected_shape": 0,
        "rejected_language": 0,
        "rejected_brief": 0,
    }

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
            # Brevity, and repetition with it. v7 failed brevity at 0.88 with replies
            # like "Rename the file named Documents/Resume.pdf." repeated four times --
            # the classic small-model decoding loop. Neither the corpus gate nor the
            # sampler looked at brevity, and the gate's own 140-word ceiling is looser
            # than the bench's 4-sentence rule, so a looping reply passed on length and
            # on everything else the gate checks.
            #
            # It is also self-reinforcing: training on looping replies teaches the
            # model to loop. This axis was 1.00 for v6 and the only regression in v7,
            # and the corpus that caused it was one I verified against four axes and
            # not this one. Verification is only as good as the properties it checks.
            if not bench.score_brevity(text)[0]:
                stats["rejected_brief"] += 1
                continue
            # Cheap belt-and-braces against near-duplicate sentences, which can pass
            # the sentence count while still reading as a stuck model.
            sentences = [
                s.strip().lower() for s in re.split(r"[.!?]", text) if s.strip()
            ]
            if len(sentences) != len(set(sentences)):
                stats["rejected_brief"] += 1
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
                    # A global counter, so this MUST be paired with a global unique
                    # suffix. Using shape_counts[key] here collided every seed's first
                    # accepted row onto the same id -- 495 rows collapsed to 104 on
                    # assembly. Worse, min_shapes is enforced against the same global
                    # count, so the cap behaved as "two per shape across the whole run"
                    # rather than "two per shape per situation", which quietly throttled
                    # coverage exactly when the pool was widened to get more of it.
                    "id": f"rft:{len(rows):04d}",
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
    print(f"dropped as brief : {stats['rejected_brief']}")
    print(f"distinct shapes  : {len(shape_counts)}")
    print(f"pass rate        : {100 * stats['passed'] / max(1, stats['samples']):.1f}%")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
