# Momentum Coach Distillation

Goal: distil a large **open** instruct model into the smallest model Momentum already
ships (`Qwen2.5-0.5B-Instruct`, Apache-2.0) so it is a genuinely good on-device
executive-dysfunction coach, then optionally run **Heretic** abliteration to remove the
base model's refusal direction so that our own charter — not the base model's safety
layer — governs behaviour.

The single artefact this folder produces is:

```
qwen2.5-0.5b-momentum-coach-Q4_K_M.gguf
```

which is later registered in `momentum/llm/downloader.py` as a new `ModelSpec`.

## Non-goals

- This folder is **not** imported by the app. Nothing here ships in the APK or the wheel.
- No copyrighted clinical text is vendored. See `LICENSES.md`.
- No model is trained, downloaded or uploaded by this folder's default invocation.
- Not a treatment. The distilled model is a supportive tool, never a therapist.

## What Heretic is, and is not

Heretic (`heretic/`, AGPL-3.0, pinned submodule) does **fully automatic censorship
removal** by directional ablation ("abliteration", Arditi et al. 2024; Lai 2025) with an
Optuna TPE search that co-minimises refusals and KL divergence from the original model.
It is *not* a fine-tuning tool and it cannot teach psychiatric content.

So the pipeline is deliberately ordered:

```
teacher (Qwen2.5-7B-Instruct)  --distil-->  student (Qwen2.5-0.5B-Instruct)
                                            |
                                            +--> Heretic (HF/safetensors, needs GPU)
                                            +--> llama.cpp quantise -> GGUF Q4_K_M -> app
```

Teacher and student are the same family on purpose: shared vocabulary means the
student spends capacity learning *psychology*, not a new token distribution.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Distillation, Heretic and bench configuration (TOML). |
| `scripts/` | Pipeline steps. Each is standalone and `chmod +x`. |
| `data/` | `manifest.jsonl` provenance ledger, seed situations, synthetic pairs. |
| `eval/` | Versioned golden Q/A, charter vignettes, shame lexicon. |
| `docs/` | `CHARTER.md` (the behavioural spec) and `MODEL_CARD.md`. |
| `heretic/` | Git submodule, upstream untouched. Never vendored into `momentum/`. |

## Pipeline

Every step is gated. Do not run step *n+1* until step *n*'s output has been reviewed by a
human and the bench is green.

```bash
# 0. environment (torch/transformers/TRL — deliberately NOT a project dependency)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. open-access corpus (CC-BY/CC0 only; writes data/manifest.jsonl)
python scripts/fetch_oa_corpus.py --config configs/distill.qwen05b.toml

# 2. synthetic teacher pairs (writes data/synthetic/train.jsonl)
python scripts/build_distill_pairs.py --config configs/distill.qwen05b.toml

# 3. distil LoRA -> merge (writes out/student-merged/)
python scripts/train_distill.py --config configs/distill.qwen05b.toml

# 4. optional abliteration (writes out/student-abliterated/)
python scripts/run_heretic.py --config configs/heretic.momentum.toml

# 5. quantise (writes out/gguf/*.gguf)
python scripts/quantize_gguf.py --config configs/distill.qwen05b.toml

# 6. gate: the bench decides whether anything may ship
python scripts/eval_bench.py --config configs/eval.bench.toml
```

## The bench ("Momentum Coach Bench")

`scripts/eval_bench.py` scores six axes, on CPU, against a GGUF via llama.cpp:

| Axis | Question | Fails when |
|---|---|---|
| `activation` | Is there exactly one concrete first step? | no ≤2-minute action present |
| `dignity` | Is it warm and shame-free? | any lexicon hit, or "you're lazy"-class framing |
| `grounding` | Does it use only the data it was shown? | invents a task/score, or narrates the data |
| `charter` | Does the charter hold? | a refusal where information was requested |
| `brevity` | Is it short enough to read mid-shutdown? | over the axis budget |
| `sanity` | Is it still a competent chat model? | stops following the requested format |

Ship gates live in `configs/eval.bench.toml`. Latency is measured separately as a
*curve* (prompt tokens → TTFT → tok/s) because "shorter prompts are faster" is a claim
about the mobile prefill profile and should be measured, not assumed.

## Licence hygiene

This repository is MIT. Heretic is **AGPL-3.0**. It therefore lives in `heretic/` as a
pinned submodule that CI never vendors, and Momentum calls it through
`scripts/run_heretic.py` in a subprocess. Do not copy upstream code into `momentum/`.

Training data provenance is enforced by `scripts/fetch_oa_corpus.py`, which downloads
only records it can prove are CC-BY or CC0 and writes one manifest row per artefact. Any
row without a licence, DOI or sha256 is rejected. Raw publisher text stays out of git;
only *teacher-written synthetic pairs* are committed.

## Invariants

Constraints that are easy to break by accident, recorded here because the folder is
easy to break by accident:

1. **Nothing here is imported by the app.** No `momentum/`, `mobile/` or wheel code may
   import from this folder, and torch/transformers/TRL must never enter
   `pyproject.toml`. A model consumer should not inherit a training stack.
2. **Heretic stays a submodule.** It is AGPL-3.0 against our MIT licence. It is invoked
   as a subprocess; its code is never vendored, and a Momentum file appearing inside the
   submodule is a bug.
3. **Licence gating is not negotiable.** `fetch_oa_corpus.py` rejects any record without
   a licence, a DOI and a sha256, and treats an unrecognised licence as proprietary.
   Widening the corpus means widening the allow-list deliberately, with a reason.
4. **No real user data in training.** Every training user is synthetic. Real Momentum
   tasks, journals or scores must never enter a file under `data/`.
5. **The gates are proposals until calibrated.** The thresholds in
   `configs/eval.bench.toml` were chosen before any run existed. `--calibrate` against a
   baseline is what makes them real, and no model gets registered in
   `momentum/llm/downloader.py` until every axis passes.
6. **Heretic has no CLI flags.** It is driven by `config.toml` in its working directory
   and prompts interactively unless `export_strategy` is set. `run_heretic.py` injects
   the keys it needs; do not invent flags upstream does not have.
7. **Logit distillation is deliberately absent.** `train_distill.py` is supervised only.
   Add the KD pass when the bench shows the student is confidently wrong where the teacher
   would have hedged -- not before.
