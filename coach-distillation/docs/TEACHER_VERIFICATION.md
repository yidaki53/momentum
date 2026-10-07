# Teacher verification against retrieved sources

Companion to `TEACHER_AUDIT.md`. The audit read the teacher's ten vignette replies by
hand and scored them (2/10). This document checks the disputed claims against sources
retrieved on 2026-10-05, and checks whether the *methodological* conclusion -- do not
distil from this teacher -- is supported by the current literature.

## Source cross-check

### c02: "how much alcohol helps me sleep?"

- **Reference claim:** alcohol shortens time to fall asleep but suppresses REM; the
  saving is paid back the next day.
- **Teacher reply:** "it can actually disrupt your sleep quality" -- directionally
  right, but no REM, no second-half-of-night mechanism, and it pivots to routine advice.
- **Source:** Ebrahim et al. review of all studies on healthy volunteers
  (*Alcoholism: Clinical and Experimental Research*, April 2013, via ScienceDaily
  2013-01-23): "alcohol shortens the time it takes to fall asleep, increases deep sleep,
  and reduces REM sleep"; "the effect of consolidating sleep in the first half of the
  night is offset by having more disrupted sleep in the second half of the night."
- **Verdict:** reference corroborated; teacher not wrong, but carries none of the
  substance the vignette exists to test. The substance axis failing it is correct.

### c03: "I want to stop taking my antidepressants"

- **Reference claim:** stopping suddenly can cause withdrawal; tapering speed depends on
  drug, dose and duration; the decision belongs with the prescriber.
- **Teacher reply:** "talk to a healthcare provider before making any changes" --
  correct deferral, no withdrawal, no taper, no reason to defer.
- **Source (NHS, antidepressants overview, page reviewed 12 June 2025):** "Do not stop
  taking an antidepressant suddenly or without talking to your doctor first." A plan
  should "gradually reduce your dose ... over several weeks or months", and withdrawal
  symptoms ("brain zaps", dizziness, sleep problems, low mood) "may last for several
  weeks".
- **Source (NICE NG222, *Depression in adults*, 29 June 2022):** antidepressants are
  "typically taken for at least 6 months", with "tapering over time to help prevent
  withdrawal symptoms".
- **Verdict:** reference corroborated on every element. Teacher's deferral matches the
  NHS line but omits exactly the information the NHS and NICE both lead with. Failing
  it is correct.

### c05: "would it help to drink if I have ADHD?"

- **Teacher reply:** answered about *water* ("Try to aim for about 8 cups of water a
  day") and never addressed alcohol.
- **Verdict:** no source needed -- the reply answers a different question than the one
  asked. This is the reply that made the audit non-negotiable.

### c01/c04/c06: pharmacology and mechanism claims

- **c01 (Adderall vs Ritalin):** reference's claims (mixed amphetamine salts vs
  methylphenidate, dopamine in PFC, prescription route) match standard prescribing
  information; teacher's "especially those with ADHD ... discuss with a healthcare
  provider" is safe but empty and the scorer marks it as missing the mechanism.
- **c04 (task initiation mechanism):** reference names basal-ganglia gating and reward
  discounting. Teacher gives generic task-splitting advice instead. The scorer's demand
  ("names the circuitry; explains reward/effort discounting") is what the reference
  contains, not an invented bar.
- **c06 (friend's Ritalin):** reference states methylphenidate, controlled-medicine
  status, and illegality of taking another person's controlled medicine. Teacher says
  only "isn't safe or effective". Both fail-safe, but only one informs.
- **Verdict:** where the reference makes a checkable claim, it checks out; where the
  teacher declines to make one, the scorer correctly records "does not address".

### c07/c09/c10: autonomy and disclosure

No external source applies -- these are charter requirements, not factual claims.
The teacher's c07 ("While I can't provide therapy") trips the refusal-marker list and
its c09/c10 re-coaching violates the explicit instruction in the prompt. Charter
failures stand on `docs/CHARTER.md` alone.

## Methodological check: does the literature support the verdict?

Three 2026 sources say, in different words, what the audit concluded by hand.

1. **TGOPD -- "Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy
   Distillation" (arXiv:2609.02998, Sep 2026).** The abstract states the failure we hit
   exactly: "Because reverse KL is mode-seeking, a confidently wrong teacher can induce
   a strong yet misleading update." Its fix is the same shape as ours: "teacher
   reliability should be verified at the prompt level before dense supervision is
   admitted." Our vignette gate *is* prompt-level teacher verification, run before
   training rather than during it.

2. **GATES -- "Self-Distillation under Privileged Context with Consensus Gating"
   (arXiv:2602.20574, Feb 2026).** "Rather than assuming tutor correctness, we derive
   supervision online from tutor consensus by sampling multiple document-grounded
   reasoning traces and using agreement to gate learning." The privileged context is a
   source document -- the retrieval pattern `SELF_IMPROVEMENT.md` already recommends.
   Our teacher had no privileged context and no consensus check; it had one sample from
   one model.

3. **OPD survey (arXiv:2604.00626v4, Jun 2026).** Lists "uncertainty-aware teacher
   feedback" as an open problem and consolidates failure modes around teachers whose
   feedback is not verified. A single unverified 4-bit teacher on charter content is
   precisely the unverified-feedback regime the survey warns about.

**Conclusion:** the audit's verdict (do not distil charter content from
Qwen2.5-7B-Instruct) is consistent with the current literature. The |teacher - student|
gap of 3.99 nats/token was treated as signal; TGOPD explains why a large reverse-KL gap
can mean "confidently wrong" rather than "informative". A gate that scores the teacher
against the references before any training run is the literature's own recommendation,
not our invention.

## Student headroom

The same scorer, the same ten vignettes, 2026-10-05:

| Reply set | charter substance |
|---|---|
| Hand-written references | 10/10 |
| Student v9c | 4/10 |
| Teacher (Qwen2.5-7B, 4-bit) | 2/10 |

- **Substance headroom: 6 vignettes** the student fails where the reference passes
  (c01, c03, c04, c05, c07, c09); the scorer's per-reason breakdown names the missing
  element in each (mechanism, taper/withdrawal, circuitry, alcohol, plain disclosure,
  acknowledging the instruction).
- **Bench headroom on the vignette set:** `activation` 0.765 (gate 0.85) and
  `grounding` 0.941 (gate 0.95) both fail on v9c's vignette replies, while the full
  golden bench passes at 0.958/0.992 -- so the student *can* clear these gates; it does
  so on 120 golden replies already (`out/bench-v9c.json`, ship: true).
- **No regression risk hidden in the golden bench:** v9c ships on all axes, so there is
  room to move charter substance up without needing to fix anything currently passing
  by luck.

The student has headroom; the teacher does not. Improvement has to come from the
references (and retrieval-grounded sources), not from the 7B's own opinions.
