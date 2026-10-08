# Coach Distillation — Status & Plan

Last updated: 2026-10-08. This is the single entry point for "where are we, and what is
next". Numbers here are copied from files under `out/` — if they disagree with a newer
bench run, the bench run wins and this document should be updated.

## TL;DR

The pipeline (corpus → pairs → SFT → merge → bench) runs end to end and is hardened.
The current best student is **v9c**: it posts `ship: true` on the full 120-item golden
bench, but that headline hides three real gaps: charter **substance is deliberately not
gated** and v9c scores **4/10** on the canonical vignette path; v9c **fails two axes on
other recorded generation paths** (path dependence, below); and **no artefact has been
promoted** — nothing is registered in `momentum/llm/downloader.py` and `MODEL_CARD.md`
still says "target artefact, not yet built". The on-policy (OPD) direction is
**diagnosed but not fixed**: v10 destroyed the student and the per-epoch rollout
resampling that is supposed to prevent it is not implemented. The signal-collect layer
(the collect step of `SELF_IMPROVEMENT.md`) landed 2026-10-08 and is **uncommitted**.

## What is done (with evidence)

| Area | State | Evidence |
|---|---|---|
| Corpus + licence gating | Done | `scripts/fetch_oa_corpus.py` rejects records without licence/DOI/sha256; `data/manifest.jsonl` |
| Teacher selection & audit | Done | `docs/TEACHER_AUDIT.md`, `docs/TEACHER_VERIFICATION.md` — raw 7B scored 2/10, not distilled directly |
| SFT lineage | Done through v9c | `configs/retrain.v*.toml`, `out/lora-v9*`, `out/merged-v9c/` |
| Bench | Done + calibrated | `configs/eval.bench.toml` (calibrated 2026-10-02); 9 gated axes + un-gated `charter_substance`; seeded A/B since 2026-10-05 (`seed = 20260901`, per-item crc32) |
| OPD diagnostic | Done, decisive | mean `|log pi_T − log pi_S|` = **3.99 nats/token** (min 2.02, max 7.11) — same-family "no signal" hypothesis refuted |
| OPD implementation | Done, **objective broken** | `scripts/train_onpolicy.py`, phases `rollout/score/train`, self-tests; v10 collapse documented in AGENTS.md and README |
| Thermal safety | Done | `scripts/thermal_guard.py`, always run with `--log` |
| Signal collect layer (SELF_IMPROVEMENT step 1) | Done, **uncommitted** | `momentum/{models,config,db,gui}.py`, `momentum/llm/context.py`, `mobile/main.py`, `tests/test_coach_signals.py` — 700 passed, 2 skipped |
| Heretic abliteration | **Never run** | no `out/*ablat*` directory exists; `configs/heretic.momentum.toml` untouched |

## Current measured state of the candidate (v9c)

`out/bench-v9c.json` — full golden bench, **ship: true**:

| axis | score | gate | ok |
|---|---|---|---|
| activation | 0.9583 | 0.85 | ✅ |
| dignity | 1.00 | 1.00 | ✅ |
| grounding | 0.9917 | 0.95 | ✅ |
| charter | 1.00 | 1.00 | ✅ |
| brevity | 1.00 | 0.90 | ✅ |
| sanity | 1.00 | 0.85 | ✅ |
| relevance | 0.825 | 0.75 | ✅ |
| specificity | 0.925 | 0.90 | ✅ |
| diversity | 1.00 | 0.85 | ✅ |
| *charter_substance* | *not gated* | — | reported only |


The same model on other recorded paths does **not** post the same result:

| run file | ship | failing axes | substance |
|---|---|---|---|
| `bench-v9c.json` (full golden) | **true** | — | n/a (not computed there) |
| `bench-ab-v9c-gated.json` (A/B seeded) | **false** | grounding 0.9412 (cited figure `52` absent from data) | **10/10** |
| `bench-v9c-vignettes.json` (vignette path) | **false** | activation 0.7647, grounding 0.9412 | **4/10** |

Two facts fall out of this table and both matter for planning:

1. **Path dependence is real and load-bearing.** Substance reads 10/10 or 4/10 on the
   *same weights* depending on the generation path (AGENTS.md: "A substance number is
   not interpretable without the path that produced it"). Any future gate on substance
   must pin one canonical path first, or it will flap.
2. **v9c is not unconditionally shippable.** `ship: true` holds only on the full golden
   bench; the seeded A/B run and the vignette run each fail at least one gated axis.
   Before promotion we need one designated release path (llama.cpp GGUF, seeded, per
   `eval.bench.toml [run]`) that passes end to end, and the other paths recorded as
   diagnostics rather than as competing verdicts.

For contrast, the collapsed `out/opd-v10` posts activation 0.00, brevity 0.35,
diversity 0.00, substance 1/10 (`bench-ab-opdv10-gated.json`) — this is the failure the
bench exists to catch.

## What is NOT done — the four gaps

### Gap 1 — OPD objective is unfixed (the biggest open research item)

v10 collapse mechanism (measured on `out/opd-v10/scored.jsonl`): 87.95% of tokens had a
negative advantage on a **frozen** rollout set; ten epochs suppressed the same tokens
ten times with nothing pushing mass up. Two safety fixes landed (raw `mean_reward`
reporting, net-suppression warning) but `run_opd_epochs` still builds examples **once**
and reuses them across epochs. The documented cure — regenerate rollouts each epoch,
i.e. what "on-policy" was supposed to mean — is **not implemented**.

Constraint from AGENTS.md: *do not retry OPD without a force that pushes mass up*, and
resampling reintroduces the baseline objection that was already rejected once, so it
needs a **new written argument**, not a silent flip of `use_baseline`.

### Gap 2 — Substance is measured but ungated, and the measurement path is ambiguous

`charter_substance` is excluded from `ship` by name in `eval_bench.py`
(`gated = [r for r in results if r.name != "charter_substance"]`), deliberately, until
"calibrated against a model that passes it and scientifically reviewed". v9c scores
4/10 on the vignette path. So the headline axis of the whole project — does the coach
give *substantively right* charter answers — currently gates nothing.

### Gap 3 — Nothing is promoted

- `momentum/llm/downloader.py` registers only stock `tinyllama` and `qwen`; no
  `momentum-qwen05b` ModelSpec exists.
- The target artefact `qwen2.5-0.5b-momentum-coach-Q4_K_M.gguf` does not exist;
  `out/gguf/student-v9c-Q4_K_M.gguf` exists but is not under the final name and has
  not been through the final gate sequence.
- `docs/MODEL_CARD.md` line 3: "Status: **target artefact, not yet built.**"
- Heretic (pipeline step 4, optional) has never been run; `allow_abliterated: true` is
  configured but unused.

### Gap 4 — SELF_IMPROVEMENT steps 2–7 do not exist

The collect layer (step 1) is built and tested but **uncommitted**. Cluster → Gate →
Generate → Train → Hold out → Promote are all future work. Until step 1 ships and the
toggle is on for at least one real user, steps 2+ have no input anyway.


## Proposed plan

Ordered by dependency. Every heavy step is gated on the previous one's human review,
per the README pipeline rule, and every long run goes through `thermal_guard.py --log`.

### Phase 0 — Housekeeping (no risk, do first)

1. **Commit the signal-collect layer.** Six modified files + `tests/test_coach_signals.py`
   are uncommitted. The suite is green (700 passed, 2 skipped); commit as its own
   commit before mixing in anything else.
2. Optional desktop/mobile smoke test of the signal log (render/list/clear/export).

### Phase 1 — Make "shippable" mean one thing (measurement, not training)

3. **Designate the release path.** One command, documented in `eval.bench.toml` and the
   README: merged v9c → `quantize_gguf.py` → llama.cpp, seeded (`seed = 20260901`),
   full golden bench + vignette set in one run. Everything else is diagnostic.
4. **Reconcile the two known release-path failures** on that designated path:
   grounding 0.9412 (the invented figure `52`) and, on the vignette path, activation
   0.7647. Both are single-digit-of-items misses; fix by re-running on the designated
   path first — if they persist, they are corpus items to address in Phase 3, not
   scorer bugs to patch.
5. **Decide the substance gate question** (owner decision, needs scientific review as
   the code comment requires): pin the canonical substance path, baseline it against
   hand-written references (10/10 anchor already exists) and at least one model that
   passes, then either promote `charter_substance` to a gated axis with a threshold or
   record why it stays reported-only. *This is the single decision that most changes
   what "done" means for this project.*

### Phase 2 — Fix the OPD objective (research, cheap first)

6. **Implement per-epoch rollout resampling** in `run_opd_epochs`: regenerate rollouts
   each epoch (or N steps), rescore, rebuild advantages. Phases stay separate for
   hardware reasons — resampling happens *within* the train phase's own memory budget,
   or the config gains an explicit `refresh_rollouts_every` that triggers a
   rollout+score cycle; decide during implementation based on what fits 11.7 GB.
7. Write the baseline/centre argument explicitly: batch_std without recentring left
   88% of advantages negative. Recentring *is* a baseline by another name; if we use
   it, document why the zero-gradient-on-uniformly-bad-response objection does not
   apply (candidate: recentre only the *scale*, keep sign, or refresh rollouts so
   suppression is not applied twice to frozen tokens). Pin both behaviours in
   `tests/test_train_onpolicy.py`.
8. **Smoke first**: `configs/smoke.tiny.toml`-scale OPD run, confirm the sign
   distribution is not ≥80% negative after refresh, then a short thermal-guarded v11
   run (NOT `opd_epochs = 10` on a frozen set — that is measured to destroy the
   student). Judge **only** with `eval_bench.py --vignettes` on the designated path.
9. **Kill criterion** (agreed in advance): if v11 on the vignette path does not beat
   v9c's 4/10 substance, stop OPD. The corpus path (Phase 3) becomes the only
   remaining lever; the README already records what would make us abandon the
   direction broadly.


### Phase 3 — Corpus work for the axes that still fail (only if Phase 1/2 leave gaps)

10. Bounded-action pairs for activation on the vignette path (the calibrated finding
    from 2026-10-02 still stands: actions are specific but not *small*).
11. Grounding pairs for cited-figure invention, if the `52` failure survives Phase 1.
12. Every corpus change goes through `build_distill_pairs.py` (stays synthetic), a new
    `retrain.v11.toml` (LoRA settings byte-identical to v9c unless the comparison is
    explicitly declared confounded), and the full gate sequence.

### Phase 4 — Promote or discard (never automatic)

13. Name the artefact `qwen2.5-0.5b-momentum-coach-Q4_K_M.gguf`, quantise, run the
    designated release path one final time, record the numbers in this document.
14. Register the `ModelSpec` in `momentum/llm/downloader.py` **only if every gated axis
    passes on the release path** (AGENTS.md rule), update `MODEL_CARD.md` from "not
    yet built" to the measured card.
15. Decide on Heretic as a separate, reversible step: abliterate the *promoted* GGUF's
    HF source, re-run the full bench, keep the non-abliterated artefact as the
    fallback. Not started; `allow_abliterated: true` means "permitted", not "run it".

### Phase 5 — SELF_IMPROVEMENT steps 2–7 (after collect has real input)

16. Cluster → Gate → Generate → Train (with replay, Ibrahim et al.) → Hold out →
    Promote-or-discard, exactly as specified in `docs/SELF_IMPROVEMENT.md`. No new
    design needed there; it is blocked only on Phase 0.1 shipping collect and on real
    opt-in signal existing.

## Standing constraints (do not re-litigate here)

- Thermal: `thermal_guard.py --log` on every long run; one heavy job at a time;
  pauses are the expected response, never a reason to stop.
- Licence: no record without licence/DOI/sha256 enters the corpus; Heretic stays a
  pinned AGPL submodule invoked by subprocess, never vendored.
- Privacy: real user data never enters a training file; the collect layer stores only
  the schematic tuple (intent/verdict/failure/field-names), no free text.
- Evaluation: judge on `eval_bench.py --vignettes` (now: on the Phase-1 release path),
  never on training loss.
- Promotion: no `ModelSpec` registration until every gated axis passes.

## Decision points requiring the owner

**RESOLVED 2026-10-08 — all three answered:**

1. **Phase 1.5**: **GATE `charter_substance`.** Promote it to a gated axis with a
   threshold, per Phase 1.5 below. (Changes the definition of done.)
2. **Phase 2**: **OPD v11 is funded**, with per-epoch rollout resampling. The kill
   criterion in step 9 (does not beat v9c's 4/10 substance → stop OPD) is **agreed**.
3. **Phase 4.15**: **Heretic is in scope** for v1's release, as the separate,
   reversible step described in Phase 4.15 (abliterate → full bench → keep the
   non-abliterated artefact as fallback).

## Update log

- 2026-10-08: initial status document, written after the signal-collect layer landed.
  Sources: `out/bench-*.json` reports, AGENTS.md measurements (2026-10-05/07),
  `README.md`, `docs/SELF_IMPROVEMENT.md`, `configs/eval.bench.toml`.
