# Model card: qwen2.5-0.5b-momentum-coach

Status: **target artefact, not yet built.** This card is written before training so that
the evaluation criteria cannot be retrofitted to whatever the model happens to do.

## Intended use

An on-device, offline coaching assistant inside the Momentum app, for adults using
Momentum to manage executive dysfunction (commonly with ADHD, autism, depression,
anxiety, TBI or chronic stress). It has access to the user's own Momentum data —
tasks, focus sessions, assessment scores, journal — plus recent chat history.

It is optimised for **short, warm, actionable replies**: two or three sentences with
exactly one concrete first step, referenced to the user's real tasks.

## Out of scope

- Diagnosis, treatment, triage, or crisis response.
- Any use outside the Momentum app. This model is not a general assistant.
- Use as a clinician's or teacher's tool.
- Use with anyone other than the device owner for the owner's own data.

## Training

- **Student:** `Qwen2.5-0.5B-Instruct` (Apache-2.0).
- **Teacher:** `Qwen2.5-7B-Instruct` (Apache-2.0), same family so token distributions match.
- **Method:** LoRA distillation on synthetic, teacher-written pairs; sources listed in
  `../SOURCES.md`, licence constraints in `../LICENSES.md`.
- **Data:** no user data, ever. Every training user is synthetic.
- **Abliteration:** optional, Heretic (AGPL-3.0) run in a submodule against the merged
  checkpoint, optimising refusals and KL divergence. If KL exceeds the gate in
  `eval.bench.toml` the step is dropped and the un-abliterated model ships.

## Evaluation

Six axes on `eval_bench.toml`: activation, dignity, grounding, charter, brevity, sanity.
Ship requires every gate to pass; the charter gate includes **zero refusals** on
information requests.

## Known limitations

- A 0.5B model is genuinely marginal for clinical nuance. Narrow specialisation is what
  makes it viable, not raw capability.
- It will occasionally be bland, wrong, or over-brief. It is not a therapist and its
  advice carries the authority of a to-do app, not a clinician.
- Latency on a phone is dominated by prefill, so prompt length is a first-class
  design constraint, not an afterthought.
- Distilled models inherit their teacher's blind spots, and a 7B teacher is not a
  clinical authority.
- English only.

## Ethical notes

- The model is trained to be non-shaming. Shame is a clinical risk factor; the
  `dignity` axis fails the build on any shame-lexicon hit.
- The charter deliberately instructs the model *not* to refuse informational requests.
  Users of this model are adults managing their own health with an app that is
  explicitly not a treatment. The trade-off accepted here is autonomy over paternalism,
  mitigated by Duty 1 (add safety context, don't withhold information) and by the
  disclosure duty.
- This card must be updated honestly after each run: if a gate failed and we shipped
  anyway, that belongs here.
