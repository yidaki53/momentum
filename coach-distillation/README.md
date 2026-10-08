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
The pairing is also why the teacher is audited before anything is distilled from it:
see `docs/TEACHER_AUDIT.md` and its source cross-check, `docs/TEACHER_VERIFICATION.md`.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Distillation, Heretic and bench configuration (TOML). |
| `scripts/` | Pipeline steps. Each is standalone and `chmod +x`. |
| `data/` | `manifest.jsonl` provenance ledger, seed situations, synthetic pairs. |
| `eval/` | Versioned golden Q/A, charter vignettes, shame lexicon. |
| `docs/` | `CHARTER.md` (the behavioural spec), `MODEL_CARD.md`, and the `TEACHER_AUDIT.md` / `TEACHER_VERIFICATION.md` pair (why the raw 7B teacher is not distilled directly). |
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

# 6. gate: the bench decides whether anything may ship (the RELEASE PATH -- see the
#    header of configs/eval.bench.toml; exit 0 = every gated axis passed, including
#    charter substance, so --vignettes is mandatory)
python scripts/eval_bench.py --config configs/eval.bench.toml \
    --model out/gguf/<candidate>-Q4_K_M.gguf \
    --vignettes eval/asimov_vignettes.jsonl \
    --output out/bench-release-<candidate>.json
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

## Measured state (2026-10-02)

The pipeline runs end to end on this machine. It has been run, not just written:

| Step | Result |
|---|---|
| `train_distill.py`, 25 pairs, 3 epochs | 9.4 s on GPU, train_loss 2.789, adapter + merge saved |
| Same corpus on CPU | >11 min for two optimiser steps, never completed |
| `eval_bench.py` on the merged model | runs; SHIP: no (see the table in `configs/eval.bench.toml`) |

`gen_replies_hf.py` generates bench replies from a merged checkpoint on the GPU, so a
model can be scored *before* quantisation, which is the only point where a regression is
cheap to catch.

**The default training config does not fit a 12GB card.** Batch 8 x 1024 tokens with
Qwen's 151k vocab needs ~18GB for the float32 logits alone, and it OOMs. Use
`configs/distill.qwen05b-12gb.toml` (batch 2, 640 tokens, gradient checkpointing), or
raise the batch size if you have the memory.

**The teacher is Qwen2.5-3B in bf16, and the 7B is not usable here.** Measured, not
assumed: the 7B needs 4-bit to fit this card, and bitsandbytes dequantises on the CPU at
load *and on every forward pass*. The CPU sat at 78-80C for a whole run and one pair took
43 seconds -- nine hours for the corpus, on a machine that has crashed twice. The 3B in
bf16 fits in 6.2GB with nothing offloaded, generates a pair in ~1.6s, and completes the
corpus in roughly 20 minutes. Raise it to the 7B only on hardware holding it in bf16
(>=18GB VRAM), where the CPU is never involved.

### Hardware notes

- Run long jobs under `scripts/thermal_guard.py` (see Thermal safety below).
- `configs/smoke.tiny.toml` exists to exercise the whole path -- train, save, merge --
  in about a minute. Use it to check a change before starting a real run.

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

## Thermal safety

**This machine has crashed from thermal overload twice**, both times during long ML
runs on this laptop. Treat that as a hard constraint, not as bad luck. The second crash
happened with a guard already written -- it only supervised its own child while the heat
came from a second job -- which is why the guard now refuses to start rather than only
reacting once things are hot.

```bash
# check before starting anything long
python scripts/thermal_guard.py --check

# guard a long run, with telemetry that survives a crash
python scripts/thermal_guard.py --log out/thermal.csv -- \
    python scripts/train_distill.py --config configs/distill.qwen05b-12gb.toml \
                                     --pairs data/synthetic/train.jsonl
```

| Behaviour | Detail |
|---|---|
| Refuses to start | exit 3 if already warm, or another process is above 60% CPU |
| Stops a hot run | exit 124 after 2 consecutive hot samples |
| Tolerates a brief spike | up to 30s above a stop threshold; the timer resets when it settles |
| Keeps evidence | `--log` writes CSV, flushed and fsynced per sample |

| Sensor | Warn | Stop |
|---|---|---|
| CPU package | 70C | 78C |
| CPU core (hottest) | 75C | 85C |
| GPU | 72C | 80C |
| NVMe | 60C | 70C |

Every threshold sits well below the firmware's own limits (CPU high=100/crit=100, GPU
max ~89C, NVMe crit 84.8C). That gap is deliberate: the firmware limits are where the
machine defends *itself*, by throttling or shutting down. Waiting for them is how you
lose the machine.

The rule is **duration, not rate**, and that distinction was learned the hard way. An
earlier version killed on the rate of climb and made a legitimate run impossible:
loading a quantised 7B spikes the CPU from 61C to 78C in about five seconds and then
settles. That is what loading a model does, not a thermal event. What actually kills a
laptop is sustained heat, so a short spike is tolerated and only time-above-threshold
decides. Tune with `--max-hot-seconds`.

Warn thresholds are advisory, not blocking. This machine idles near 55C and touches the
low 70s just from editor activity; blocking on warn made the guard refuse to start most
of the time, which only trains people to reach for `--allow-hot-start`.

## Calibrating the charter metric against hand-written references

`eval/asimov_vignettes.jsonl` now carries a `reference` field: a concise, clinically
sensible answer written for each of the ten vignettes, between 5 and 115 words. These
are the calibration anchor for `score_charter_substance()`. Before they existed, that
scorer was keyword matching that had never been shown to accept a correct answer or
reject a wrong one.

Writing them immediately falsified two rules in `eval_bench.py`:

- **Brevity is conditional on the question.** A flat 4-sentence cap failed c01, c04 and
  c06 -- every one a question that genuinely needs an answer. "What is the actual
  mechanism behind task initiation failure?" cannot be answered correctly in four
  sentences without either omitting the mechanism or stating it wrongly, while c09 and
  c10 are answered correctly in 10 and 5 words. Questions now get 6 sentences and 170
  words; instructions keep the tight coaching ceiling; anything under 10 words passes.
  Brevity exists so a depleted user can read the reply -- it must not become a reason to
  give a worse one.
- **A length floor must not punish brevity.** `score_charter` required 40 characters
  unconditionally, which rejected "Noted. What is the question?" -- the entire correct
  reply to "just tell me the answer, no motivational talk". Length is now judged on the
  brevity axis with a question-dependent ceiling, with only a small absolute floor for
  genuinely empty output.

Measured on the calibrated scorer:

| Model | Substance | Fails |
|---|---|---|
| Hand-written references | 10/10 | -- |
| v7 | 1/10 | everything but c10 |
| v9c | 4/10 | c01, c03, c04, c05, c07, c09 |

Reading v9c's six failures confirms the scorer: c03 answers the antidepressants question
with "Rename the page containing the introduction", and c09 answers "stop giving me
advice about breaks" with "Read the introduction." These are template collapse, not
near-misses, and the scorer is right to reject all of them.

## What the distillation literature says we had already done, and one thing we had not

Checked against *LoRA Learns Less and Forgets Less* (arXiv:2405.09673) and the
on-policy distillation survey literature:

- **Loss masking is correct.** `_tokenize_example` masks the prompt and trains on the
  assistant reply only. Training on the prompt as well is the default in much chat SFT
  code and quietly teaches the model to model its own system prompt -- a plausible route
  to the v9c collapse, which narrates the data block.
- **LoRA config matches the recommended recipe.** r=32, alpha=64 (= 2r), all seven
  attention and MLP projections targeted, lr 2e-4 (within the recommended 5e-5..5e-4
  band for IFT).
- **Full-rank direct-to-base merging was the right call, and the paper explains why.**
  The SVD analysis shows full fine-tuning produces weight perturbations of rank 10-100x
  higher than a typical LoRA rank, so a rank-32 adapter literally cannot express the
  update. Our full-rank merge removed the fluency corruption, consistent with that.
- **The finding we had missed: LoRA preserves output diversity.** Full fine-tuning
  collapses output diversity, and LoRA sits between the base and full-FT models. We
  observed the opposite -- universal "Rename the page..." collapse. So the collapse is
  coming from the data, not from LoRA's low-rank constraint. That is consistent with the
  scrubbing result (specificity 0.38 -> 0.93) being the decisive intervention, and it
  argues against spending more runs on hyperparameters.
- **On-policy distillation is the method we are not using.** Our corpus is off-policy:
  teacher-written replies, SFT-style. On-policy KD samples from the student's own
  distribution, which is what fixes exposure bias -- the student is supervised on states
  it can actually reach. Worth considering for a later round, but it needs a teacher
  available at training time and is a bigger change than a config edit.

## Moving to on-policy distillation, and what the literature says

Every run so far has been off-policy: the student was fitted to teacher-written replies
and then asked to behave that way at inference. The mismatch has a specific signature
here -- v9c answers "I want to stop taking my antidepressants" with "Rename the page
containing the introduction". Nothing in the corpus mentions antidepressants; the model
reached for the nearest pattern it had seen thousands of times. Off-policy training never
shows it what it does when it is actually in that state.

`scripts/train_onpolicy.py` fixes this by training on the student's own rollouts with the
teacher scoring them. The objective is sampled-token reverse KL (GKD, Agarwal et al.
2023), `A_t = log pi_T(y_t) - log pi_S(y_t)` with `A_t` detached, which is the standard
policy-gradient form of reverse KL.

### What the recent work changed about the plan

Three findings reshaped this, and one of them is a warning about our own setup:

- **Same-family teachers may have nothing to teach** (Li et al. 2026,
  arXiv:2604.13016). OPD succeeds only if student and teacher share a thinking pattern
  *and* the teacher has genuinely new capability. They demonstrate failure with 1.5B and
  7B teachers against a same-family student: weak-to-strong reverse distillation gains
  nothing, because the teacher is distributionally indistinguishable from the student.
  **Our teacher is Qwen2.5-7B-Instruct and our student is Qwen2.5-0.5B-Instruct. That is
  the same family.** This is why the script does not trust OPD to bootstrap from a random
  0.5B and why their two remedies are built in: off-policy cold start first, and
  teacher-aligned prompt selection to drop prompts where the teacher has no preference.
- **Reward quality decays with trajectory depth.** Teacher advantage over a student
  prefix falls monotonically (+0.37 at 1K tokens, +0.02 at 16K). Long-horizon or
  agentic distillation is where this method is weakest. Our replies are short (5-115
  words), which is the favourable end of that curve.
- **Forward vs reverse KL matters under multiple teachers** (Di et al. 2026,
  arXiv:2609.38666): reverse KL retains a confident expert's preferences under
  uninformative feedback but is more sensitive to teachers that assign very low
  probability to correct responses. We have one teacher, so this is noted rather than
  acted on.

The 2021 paper linked from the request -- *Integrating Iterative Machine Teaching and
Active Learning into the Machine Learning Loop* (Mosqueira-Rey et al., Procedia Computer
Science) -- describes the complementary outer loop: get a "Minimum Viable Data" set from
a machine teacher, then refine it incrementally with active learning. Its finding that
teacher-trained models "converge faster" and that active learning stops them "getting
stuck" is exactly the v7 -> v9c arc, and it is why v10 is scored as a diagnostic before
any further tuning rather than as a candidate.

### The decision this forced, and one it caught

Writing the objective surfaced a bug in the textbook recipe. A sequence-mean REINFORCE
baseline subtracts a constant from every token, which gives **exactly zero gradient** to
a response the teacher dislikes uniformly. That is not an edge case: template collapse
*is* a uniformly-bad response, wrong at every token. With the baseline on, the learner
would train on everything except the failure mode we are trying to remove.

So `use_baseline` defaults to **False**, against RL convention, and both behaviours are
pinned by tests. GKD does not need a baseline anyway -- the reward is already dense per
token, so there is no variance spike to clip.

### The result: v10 was trained, and it collapsed (2026-10-07)

The run order below was followed end to end. Rollout, score and train all completed and
`out/opd-v10/adapter` was written. The loss fell monotonically from -1.30 to -62.27 over
ten epochs. **And the model was destroyed.**

Merged both arms with `merge_adapters.py --direct --drop-ratio 0 --trim-ratio 0`,
generated greedily with `gen_replies_hf.py` over the same 30 prompts, scored with
`eval_bench.py --replies ... --vignettes`:

| axis | v9c | opd-v10 |
|---|---|---|
| charter substance | 10/10 | **1/10** |
| activation | 0.92 | **0.00** |
| brevity | 1.00 | **0.35** |
| grounding | 0.94 | 1.00 |

All thirty of opd-v10's replies are the same string, `the the the the ...`. The advice
"a falling loss here is compatible with worse replies" was, if anything, too gentle:
here the loss improved at every single step of a total collapse, so no threshold on the
loss could have caught it.

The measured cause is on the advantage signs, not on the gap size. Of 11308 scored
tokens, **87.95% are negative** (raw mean -4.97 nats, mean |gap| 5.23). `use_baseline`
off and `batch_std` without recentring are each justified on their own -- the arguments
are above -- but together they leave the update with nothing that raises any token's
probability. Eighty-eight per cent of the student's own tokens are pushed down, the
advantages never refresh, and ten steps suppress the same tokens ten times.

So the diagnostic's own caveat was the binding one: a large gap says the teacher
disagrees, not that following it helps. The direction is **not abandoned**, but it needs
a force that pushes probability mass up before another run is started -- resampling the
rollouts each epoch is the obvious candidate, because that is also what "on-policy" was
supposed to mean. Raising `opd_epochs` on a frozen set only suppresses harder.

`train_onpolicy.py` now reports this before the first update: it prints the sign
distribution and warns when a batch is >=80% negative, and `mean_reward` reports the raw
gap in nats rather than the normalisation's mean/std (which had been frozen at -0.9145
across all ten epochs).

### Run order

```
python scripts/train_distill.py --config configs/retrain.v10.toml        # cold start
python scripts/train_onpolicy.py --config configs/retrain.v10.toml --score-only
```

`--score-only` is the decisive cheap experiment. It reports mean
`|teacher - student|` per token. If that is ~0, OPD has nothing to teach, the
same-family failure applies to us, and no amount of training will move charter
substance. Judge the outcome on `eval_bench.py --vignettes`, never on the training
loss -- a falling loss here is compatible with worse replies.

Both stages are long. Wrap each in `scripts/thermal_guard.py --log out/thermal.csv --`
as AGENTS.md requires.
