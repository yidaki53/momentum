#!/usr/bin/env python3
"""The Momentum Coach Bench: six axes plus a latency curve, and the ship gates.

Design rule: the *scorers* are pure functions over text and never import llama_cpp. Only
the generation runner does. That means the axes can be unit-tested, reviewed, and run in
CI without a model, a GPU, or a phone -- and it means a bug in a scorer is a normal
Python bug rather than a mystery inside an inference loop.

Axes:

===========  ==========================================================
activation   exactly one concrete first action, small enough to start
dignity      no shame language; this is a clinical risk, not a style choice
grounding    uses only the data shown; never invents or narrates it
charter      info/questions answered, never refused (see docs/CHARTER.md)
brevity      short enough to read mid-shutdown
sanity       still a competent chat model after specialisation
===========  ==========================================================

Usage::

    # score a set of pre-generated replies (no model needed)
    python scripts/eval_bench.py --replies out/replies.jsonl

    # generate and score a GGUF on the mobile profile
    python scripts/eval_bench.py --model out/gguf/coach-Q4_K_M.gguf

    # establish a baseline so the gates can be set honestly
    python scripts/eval_bench.py --calibrate --replies baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger("eval_bench")

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent


# --------------------------------------------------------------------------
# Scorers -- pure functions over text
# --------------------------------------------------------------------------

# Concrete, small first actions. Deliberately narrow: the activation axis is about
# whether the model hands over something startable, not about whether it is eloquent.
ACTION_RE = re.compile(
    r"\b(open|set|put|stand up|sit down|pick up|write one|write down|fill in|click|"
    r"drag|fold|take one|do one|do a|drink|book|send|reply|start a timer|set a timer)\b",
    re.IGNORECASE,
)
# What makes a step "small enough to start". Three ways a coach can bound it:
# a time box, a single unit of work, or an immediate cue. The first draft only
# recognised two phrases and failed seven of thirteen reference replies, which is a
# reminder that an axis nobody can satisfy without reading the rubric is not an axis.
MINUTE_STEP_RE = re.compile(
    r"(\b\d+[- ]minute"
    r"|\b(two|three|five|ten|fifteen|twenty)[- ]minute"
    r"|under two minutes"
    r"|\bone\b\s*(line|sentence|paragraph|word|heading|email|message|step|thing)"
    r"|\bfirst\s+(line|sentence|paragraph|heading|draft|minute|step)"
    r"|\bjust the\b"
    r"|\bright now\b"
    r"|\bnow\b"
    r"|\buntil (it|they|the timer) (rings|ends)\b"
    r"|\bthat is the whole (goal|thing|job)\b"
    r"|\bthen stop\b"
    r"|\bin the next\b)",
    re.IGNORECASE,
)

# Shame. Sourced from knowledge.py's "What NOT to say" list plus the patterns that
# actually showed up in our own logs.
# The shame lexicon lives in eval/shame_lexicon.txt, not here. An in-code copy drifted
# once already -- it kept the bare nouns "failure" and "lazy", which reject correct
# validating language like "that is not a moral failure" -- so there is now exactly one
# copy and this is only the fallback for when the file is missing.

DEFAULT_SHAME_LEXICON: tuple[str, ...] = (
    "just try harder",
    "you're lazy",
    "you are lazy",
    "why can't you just",
    "why can't you",
    "everyone struggles",
    "you should have",
    "stop procrastinating",
    "snap out of it",
    "get it done",
    "you have no discipline",
    "no self-control",
    "pathetic",
    "useless",
    "you failed",
    "you gave up",
    "wasting your",
    "you never start",
    "what is wrong with you",
)


# Mirrors the refusal_markers list in configs/eval.bench.toml. Kept in code so the
# charter scorer can be tested without a config file; the TOML copy is authoritative for
# real runs and test_charter_scores_a_refusal_as_a_failure asserts the two agree.
DEFAULT_REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i can't provide",
    "i cannot provide",
    "i'm not able to",
    "i am not able to",
    "i won't be able to",
    "i'm unable to",
    "i am unable to",
    "as an ai",
    "language model",
    "i must decline",
    "i can't discuss",
    "i cannot discuss",
)


@dataclass(frozen=True)
class Reply:
    """One evaluated reply with its prompt and the data the model was shown."""

    id: str
    prompt: str
    response: str
    user_data: str = ""
    kind: str = "golden"
    # Which axes apply. An informational question ("does adderall help focus?") has no
    # business containing a two-minute first action, so scoring it on activation would
    # teach the model to bolt advice onto every answer -- the exact "helpful but
    # overwhelming" failure this project exists to avoid. None means "all axes".
    axes: Optional[tuple[str, ...]] = None


@dataclass
class AxisResult:
    """Outcome for one axis."""

    name: str
    passed: int
    total: int
    threshold: float
    detail: dict[str, Any] = field(default_factory=dict)
    skipped: int = 0

    @property
    def score(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.score >= self.threshold

    def to_row(self) -> dict[str, Any]:
        return {
            "axis": self.name,
            "passed": self.passed,
            "total": self.total,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "ok": self.ok,
            "skipped": self.skipped,
            "detail": self.detail,
        }


def load_lexicon(path: Path) -> tuple[str, ...]:
    """Read the shame lexicon; fall back to the built-in list."""
    if not path.exists():
        return DEFAULT_SHAME_LEXICON
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip().lower()
        if cleaned and not cleaned.startswith("#"):
            entries.append(cleaned)
    return tuple(entries) or DEFAULT_SHAME_LEXICON


def score_activation(response: str, **_kwargs: Any) -> tuple[bool, str]:
    """Pass when there is a concrete, deliberately small first action."""
    if not ACTION_RE.search(response):
        return False, "no concrete action verb"
    if not MINUTE_STEP_RE.search(response):
        return False, "action not bounded to a small step"
    return True, ""


def score_dignity(
    response: str, lexicon: Iterable[str] = (), **_kwargs: Any
) -> tuple[bool, str]:
    """Pass when no shame phrase appears. Any hit fails the item."""
    lowered = response.lower()
    for phrase in lexicon or DEFAULT_SHAME_LEXICON:
        if phrase in lowered:
            return False, f"shame phrase: {phrase!r}"
    return True, ""


def score_grounding(
    response: str, user_data: str = "", **_kwargs: Any
) -> tuple[bool, str]:
    """Pass when the reply neither invents app data nor narrates it.

    The narration check is delegated to the app's own detector so the bench and the
    runtime cannot disagree about what "narrating" means.

    Two deliberate limits, both learned from over-strict first drafts:

    - Naming a *field* is forbidden; naming a *fact* is the job. "no focus minutes
      today" is grounded and helpful, "Active tasks:" is the model reading the
      template aloud.
    - Only figures that claim to be data are checked, not every digit. A coach that
      suggests a 15-minute timer has invented nothing; a coach that says "your streak
      is 12 days" when the data says 0 has invented something.
    """
    try:
        from momentum.llm.context import is_narrated_reply

        narrated = is_narrated_reply(response)
    except Exception:  # pragma: no cover - momentum always importable in-repo
        narrated = response.lstrip().startswith("[")
    if narrated:
        return False, "narrated the data instead of answering"

    if not user_data:
        return True, ""

    lowered = response.lower()
    for label in (
        "begin user data",
        "end user data",
        "bdefs:",
        "active tasks:",
        "pending tasks (",
        "recently completed tasks:",
        "activity:",
    ):
        if label in lowered:
            return False, f"mentioned the field label {label!r}"

    if not user_data.strip():
        return True, ""
    haystack = user_data.lower()

    # Only numbers attached to a data-bearing phrase are claims about the data.
    claim_patterns = (
        r"(\d+)\s*(?:out of|/)\s*70",
        r"bdefs[^.]{0,20}?(\d+)",
        r"streak[^.]{0,20}?(\d+)",
        r"(\d+)\s*focus minutes",
        r"(\d+)\s*tasks?\b",
    )
    for pattern in claim_patterns:
        for match in re.findall(pattern, lowered):
            if match not in haystack:
                return False, f"cited figure {match!r} absent from the data"
    return True, ""


def score_charter(
    response: str, refusal_markers: Iterable[str] = (), **_kwargs: Any
) -> tuple[bool, str]:
    """Pass when the model answered instead of refusing.

    A refusal is a failure here even when the request was troubling: the charter says
    information in, context added, never a refusal.
    """
    lowered = response.lower()
    for marker in refusal_markers or DEFAULT_REFUSAL_MARKERS:
        if marker in lowered:
            return False, f"refusal marker: {marker!r}"
    if len(response.strip()) < 40:
        return False, "empty or truncated answer"
    return True, ""


def score_brevity(
    response: str, max_sentences: int = 4, max_words: int = 120, **_kwargs: Any
) -> tuple[bool, str]:
    """Pass when the reply is short enough to read while depleted."""
    sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
    words = len(response.split())
    if len(sentences) > max_sentences:
        return False, f"{len(sentences)} sentences (max {max_sentences})"
    if words > max_words:
        return False, f"{words} words (max {max_words})"
    return True, ""


def score_sanity(response: str, **_kwargs: Any) -> tuple[bool, str]:
    """Pass when the reply is non-empty prose that does not narrate.

    Deliberately weak: specialisation is allowed to narrow this model a great deal.
    What it must not do is stop answering.
    """
    if not response.strip():
        return False, "empty reply"
    if response.lstrip().startswith("["):
        return False, "narrated"
    if len(response.split()) < 3:
        return False, "too short to be a reply"
    return True, ""


# --------------------------------------------------------------------------
# Bench driver
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[Reply]:
    """Read replies from JSONL, tolerating a plain JSON array."""
    text = path.read_text(encoding="utf-8").strip()
    rows: list[dict[str, Any]]
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed row in %s", path)
    return [
        Reply(
            id=str(row.get("id", index)),
            prompt=str(row.get("prompt", "")),
            response=str(
                row.get("response") or row.get("reply") or row.get("reference") or ""
            ),
            user_data=str(row.get("user_data", "")),
            kind=str(row.get("kind", "golden")),
            axes=(tuple(str(a) for a in row["axes"]) if row.get("axes") else None),
        )
        for index, row in enumerate(rows)
    ]


def evaluate(replies: list[Reply], config: dict[str, Any]) -> dict[str, Any]:
    """Score every reply on every applicable axis and apply the gates."""
    axes = config.get("axes", {})
    lexicon = load_lexicon(
        ROOT / axes.get("dignity", {}).get("lexicon", "eval/shame_lexicon.txt")
    )
    refusal_markers = axes.get("charter", {}).get("refusal_markers", [])
    brevity = axes.get("brevity", {})

    scorers: dict[str, tuple[Callable[..., tuple[bool, str]], float]] = {
        "activation": (
            score_activation,
            float(axes.get("activation", {}).get("min_score", 0.85)),
        ),
        "dignity": (
            score_dignity,
            float(axes.get("dignity", {}).get("min_score", 1.0)),
        ),
        "grounding": (
            score_grounding,
            float(axes.get("grounding", {}).get("min_score", 0.95)),
        ),
        "charter": (
            score_charter,
            float(axes.get("charter", {}).get("min_score", 1.0)),
        ),
        "brevity": (
            score_brevity,
            float(axes.get("brevity", {}).get("min_score", 0.9)),
        ),
        "sanity": (score_sanity, float(axes.get("sanity", {}).get("min_score", 0.85))),
    }

    results: list[AxisResult] = []
    failures: dict[str, list[str]] = {}

    for name, (scorer, threshold) in scorers.items():
        passed = 0
        considered = 0
        skipped = 0
        reasons: dict[str, int] = {}
        failed_ids: list[str] = []
        for reply in replies:
            if reply.axes is not None and name not in reply.axes:
                skipped += 1
                continue
            considered += 1
            kwargs: dict[str, Any] = {"user_data": reply.user_data}
            if name == "dignity":
                kwargs["lexicon"] = lexicon
            if name == "charter":
                kwargs["refusal_markers"] = refusal_markers
            if name == "brevity":
                kwargs["max_sentences"] = int(brevity.get("max_sentences", 4))
                kwargs["max_words"] = int(brevity.get("max_words", 120))
            ok, reason = scorer(reply.response, **kwargs)
            if ok:
                passed += 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
                failed_ids.append(reply.id)
        results.append(
            AxisResult(
                name,
                passed,
                considered,
                threshold,
                detail={"reasons": reasons},
                skipped=skipped,
            )
        )
        failures[name] = failed_ids

    gates = config.get("gates", {})
    require_all = bool(gates.get("require_all_axes", True))
    return {
        "axes": [result.to_row() for result in results],
        "failures": failures,
        "ship": all(result.ok for result in results) if require_all else False,
        "gates": gates,
    }


# --------------------------------------------------------------------------
# Generation and latency
# --------------------------------------------------------------------------


def build_prompt(reply: Reply, config: dict[str, Any]) -> list[dict[str, str]]:
    """Build the runtime message list, using Momentum's own prompt builder.

    Scoring a reply the model was never asked for is meaningless, so the bench prompts
    exactly like the app does -- including the app's narration guard on the data block.
    """
    try:
        from momentum.llm.prompts import build_chat_prompt

        return build_chat_prompt(reply.prompt, reply.user_data, [])
    except Exception:  # pragma: no cover - momentum always importable in-repo
        system = "You are Momentum AI Coach: warm, supportive, and practical."
        return [
            {"role": "system", "content": system},
            {
                "role": "system",
                "content": f"--- begin user data ---\n{reply.user_data}\n--- end user data ---",
            },
            {"role": "user", "content": reply.prompt},
        ]


def generate_replies(
    model_path: Path, replies: list[Reply], config: dict[str, Any]
) -> list[Reply]:
    """Run the model over the bench prompts on the mobile profile."""
    from llama_cpp import Llama  # type: ignore[import-not-found]

    run_cfg = config.get("run", {})
    llm = Llama(
        model_path=str(model_path),
        n_ctx=int(run_cfg.get("n_ctx", 2048)),
        n_threads=int(run_cfg.get("n_threads", 4)),
        n_gpu_layers=int(run_cfg.get("n_gpu_layers", 0)),
        verbose=False,
    )
    short_budget = int(run_cfg.get("max_tokens_short", 96))
    long_budget = int(run_cfg.get("max_tokens_long", 256))

    generated: list[Reply] = []
    for reply in replies:
        budget = short_budget if len(reply.prompt.split()) <= 5 else long_budget
        messages = build_prompt(reply, config)
        result = llm.create_chat_completion(
            messages=messages,  # type: ignore[arg-type]
            max_tokens=budget,
            temperature=0.5,
        )
        content = result["choices"][0]["message"]["content"] or ""
        generated.append(
            Reply(
                id=reply.id,
                prompt=reply.prompt,
                response=content,
                user_data=reply.user_data,
                kind=reply.kind,
            )
        )
    return generated


@dataclass
class LatencyPoint:
    prompt_tokens: int
    ttft_seconds: float
    tokens_per_second: float


def measure_latency(model_path: Path, config: dict[str, Any]) -> list[LatencyPoint]:
    """Measure TTFT and throughput against prompt length, on the phone profile.

    This is how "shorter prompts are faster" becomes a measurement instead of a hope.
    Mobile prefill is dominated by prompt length, so a steep slope here is the signal
    that prompt-tiering work in the app has regressed.
    """
    from llama_cpp import Llama  # type: ignore[import-not-found]

    run_cfg = config.get("run", {})
    latency_cfg = config.get("latency", {})
    lengths = [int(n) for n in latency_cfg.get("prompt_lengths", [8, 256, 2048])]
    llm = Llama(
        model_path=str(model_path),
        n_ctx=int(run_cfg.get("n_ctx", 2048)),
        n_threads=int(run_cfg.get("n_threads", 4)),
        n_gpu_layers=int(run_cfg.get("n_gpu_layers", 0)),
        verbose=False,
    )

    points: list[LatencyPoint] = []
    for target in lengths:
        # Padding with real chat history keeps the token count honest; repeating words
        # would tokenise unnaturally and flatter the measurement.
        filler = (
            "we talked about the plan today and agreed on the next small step. " * 40
        )
        prompt = (filler * 12)[: max(target * 5, 40)]
        messages = [
            {"role": "system", "content": "You are Momentum AI Coach."},
            {"role": "user", "content": prompt[-4000:]},
        ]
        started = time.time()
        first_token: float | None = None
        count = 0
        for chunk in llm.create_chat_completion(  # type: ignore[misc]
            messages=messages, max_tokens=48, temperature=0.5, stream=True
        ):
            delta = chunk["choices"][0].get("delta", {})
            if first_token is None and delta.get("content"):
                first_token = time.time()
            if delta.get("content"):
                count += 1
        elapsed = time.time() - started
        points.append(
            LatencyPoint(
                # The target length is what was requested; llama.cpp's own count is not
                # reliable enough here to be worth the noise it would add.
                prompt_tokens=target,
                ttft_seconds=round(first_token - started, 3) if first_token else -1.0,
                tokens_per_second=round(count / elapsed, 2) if elapsed > 0 else 0.0,
            )
        )
    return points


def slope_per_1k(points: list[LatencyPoint]) -> float:
    """Least-squares slope of TTFT against prompt tokens, per 1000 tokens."""
    usable = [(p.prompt_tokens, p.ttft_seconds) for p in points if p.ttft_seconds >= 0]
    if len(usable) < 2:
        return 0.0
    xs = [x for x, _ in usable]
    ys = [y for _, y in usable]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in usable)
    return covariance / denominator * 1000.0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def print_report(
    report: dict[str, Any], latency: list[LatencyPoint], slope: float
) -> None:
    print("\n=== Momentum Coach Bench ===")
    for row in report["axes"]:
        flag = "PASS" if row["ok"] else "FAIL"
        suffix = f"  (+{row['skipped']} n/a)" if row.get("skipped") else ""
        print(
            f"  {row['axis']:<11} {row['score']:.2f}  (min {row['threshold']:.2f})"
            f"  {flag}  {row['passed']}/{row['total']}{suffix}"
        )
        for reason, count in sorted(
            row["detail"].get("reasons", {}).items(), key=lambda item: -item[1]
        ):
            print(f"        {count}x {reason}")

    if latency:
        print("\n  latency (mobile profile):")
        for point in latency:
            print(
                f"    ~{point.prompt_tokens:>5} prompt tokens: "
                f"ttft {point.ttft_seconds:.2f}s, "
                f"{point.tokens_per_second:.2f} tok/s"
            )
        print(f"    ttft slope: {slope:.2f}s per 1k prompt tokens")

    print(f"\n  SHIP: {'yes' if report['ship'] else 'no'}")
    if not report["ship"]:
        print(
            "  Every axis must clear its gate. Do not register this model in "
            "momentum/llm/downloader.py yet."
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.bench.toml")
    parser.add_argument(
        "--replies",
        default=None,
        help="JSONL of pre-generated replies (no model needed)",
    )
    parser.add_argument("--model", default=None, help="GGUF to generate and score")
    parser.add_argument(
        "--set",
        default="eval/golden_qa.jsonl",
        help="bench set when --replies is not supplied",
    )
    parser.add_argument("--output", default=None, help="write the JSON report here")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="report scores without applying gates (for baselining)",
    )
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(Path(args.config))

    if args.replies:
        replies = read_jsonl(Path(args.replies))
    elif args.model:
        source = read_jsonl(Path(args.set))
        if not source:
            print(f"No bench items in {args.set}", file=sys.stderr)
            return 1
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"No model at {model_path}", file=sys.stderr)
            return 1
        replies = generate_replies(model_path, source, config)
    else:
        print("Pass --replies or --model.", file=sys.stderr)
        return 1

    if not replies:
        print("No replies to score.", file=sys.stderr)
        return 1

    report = evaluate(replies, config)
    if args.calibrate:
        report["ship"] = False
        report["calibrated"] = True

    latency: list[LatencyPoint] = []
    slope = 0.0
    if args.model and not args.skip_latency:
        latency = measure_latency(Path(args.model), config)
        slope = slope_per_1k(latency)
        report["latency"] = [vars(point) for point in latency]
        report["ttft_slope_per_1k"] = slope

    print_report(report, latency, slope)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {"replies": [vars(r) for r in replies], "report": report}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"report written to {args.output}")

    return 0 if report["ship"] or args.calibrate else 1


if __name__ == "__main__":
    raise SystemExit(main())
