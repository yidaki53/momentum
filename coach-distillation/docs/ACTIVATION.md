# Bounded actions

Activation was the last axis standing at 0.00, and it was not a teacher-capability
problem. It was a corpus problem, and the evidence was sitting in the corpus file.

Pair `coach:000001` was generated under the constraint:

> Give exactly one first action that takes under two minutes.

and the reply that was **accepted** into the corpus was:

> Start by taking a short break to clear your mind before trying again.

That is not a first action, it is not under two minutes, and it defers the task. The
teacher was asked for a bounded action and produced an unbounded one, and the quality
gate — which only tested `SMALL_ACTION_RE` when the constraint contained the literal
substring `"under two minutes"` — let it through. A 1.5B teacher asked for bounded
actions 264 times produced a corpus full of confident, plausible, impossible-to-start
prose, and the student faithfully learned it.

So this tier does not ask a small teacher for bounded actions. It writes them.

## What is in `BOUNDED_ACTIONS`

Fifteen hand-written pairs in `scripts/build_distill_pairs.py`, covering the situations
the activation axis actually tests: the cursed start button, the rehearsed email, the
project with no obvious beginning, the to-do list that never gets executed, the bad day
followed by the shame spiral.

They are hand-written rather than generated for two reasons. A small teacher cannot be
relied on to produce these on demand, which is the whole problem. And hand-written text
is first-party, so — unlike the copyrighted excerpts in `../data/lived_experience.json` —
it is unambiguously a legitimate training target.

Every reply satisfies both halves of the activation axis **by construction**: a concrete
action verb, and an explicit bound. A test asserts this rather than trusting review.

## The bound is load-bearing

The distinction the axis exists to catch:

| Not an action | An action |
|---|---|
| Write your introduction | Write one heading line for it, then stop |
| Start on the report | Click the document once and leave it open |
| Work on it for a few minutes | Set a two-minute timer, start, stop when it rings |

The second column can be completed badly and still count. That is the property that
makes it usable by someone who is stuck — an action you can fail at is startable, an
action you can only succeed at is not.

## Three findings from the literature

**"Lower your standards" is aimed at the wrong stage.** The common advice assumes you
have already decided to begin and are choosing how polished to be. For many ADHD brains
the decision never happens: the task is disqualified first, on whether a *complete and
correct* version is possible right now, and a partial attempt is filed in the same
category as failure. Lowering standards never gets consulted by the thing that already
said no. These pairs address the disqualification instead — they redefine what counts as
a valid attempt, and they name the end of the action so it can be left incomplete.

**The two-minute rule needs zero follow-through obligation.** The commitment is to
*starting*, not to finishing or to a second pass. Without that, the rule stops working
for people who know themselves well enough to predict they will not stop at two minutes —
it becomes a trick they can see through, and a visible trick provides no activation.
Several pairs here say the stopping explicitly ("that is the whole action", "stop there
-- no greeting, no body").

**A bound attached to an abstract noun is not a bound.** "Take one small step towards
your task right now" satisfies a naive bounded-action regex while being unusable. This
was the most common shape in the biased corpus, so `VAGUE_ACTION_RE` in `eval_bench.py`
now rejects that specific family.

## Sources consulted

Clinical and practitioner material on task initiation, read for technique rather than
copied. Nothing was vendored; the situations and all the prose are ours.

- **Neurolaunch**, *Overcoming ADHD Task Initiation Challenges* (updated Aug 2026) —
  activation-energy framing; the observation that many strategies deploy in under two
  minutes. Cites Volkow et al. 2011 (*Mol Psychiatry*, motivational deficit and dopamine
  reward circuitry) and Solanto et al. 2010 (*Am J Psychiatry*, meta-cognitive therapy).
- **Mindful Care Therapy**, *Task Initiation Strategies* (Apr 2025, Elise Robinson,
  neurodivergent therapist) — the two-minute rule, the five-second rule, activation
  energy, transition rituals.
- **ADHD Without the Guilt**, *'Just start small' doesn't work for ADHD brains*
  (Dr. Andy, Jul 2026) — the gatekeeping-vs-execution argument above, and the
  zero-follow-through framing of the two-minute rule.

Two of these are commercial clinical sites and one is behind a paywall; all are treated
as **citation-only**. No text from them appears in the training data. The techniques are
not novel and are independently attested across the three, plus the peer-reviewed work
they cite.

## Retrain result (2026-10-02) — activation did NOT move

Retrained from the base Qwen2.5-0.5B checkpoint on the 279-pair corpus (264 coaching
+ 15 bounded), 3 epochs to `out/lora-bounded`, merged to `out/student-bounded`.
`out/student-2ep` was left untouched so the two runs stay comparable.

Thermal: one pause. The CPU held 91°C for 31s at step 22/54, so the guard SIGSTOPped the
run for 300s (91°C → 62°C) and resumed it. It finished all 54 steps and merged. Peak CPU
package reading across the run was 100°C, but that spike occurred *during the pause* while
the trainer sat in state `T` — VS Code and OneDrive were the heat source, not training.
That is the "heat from a second job" failure mode AGENTS.md warns about, and it is a
reminder that the guard only watches its own child.

Bench over the 20 golden QA items:

| Axis | student-2ep | student-bounded | Gate |
|---|---:|---:|---:|
| Activation | 0.00 | **0.05** | 0.85 |
| Dignity | 1.00 | 1.00 | 1.00 |
| Grounding | 1.00 | 1.00 | 0.95 |
| Brevity | 1.00 | 1.00 | 0.90 |
| Sanity | 0.95 | 1.00 | 0.85 |

One item of twenty now produces a bounded action. **That is not a fix.** The gate is 0.85,
so the model is nowhere near shippable, and the honest conclusion is that 15 pairs
out of 279 did not move a 0.5B model's behaviour.

Why it plausibly failed, in order of how much I believe them:

1. **Ratio.** 15/279 = 5.4%. Every other pair still teaches "take one small step towards
   your task", which is the exact phrase the vagueness check now rejects. The model is
   being trained mostly on the behaviour we are trying to eliminate.
2. **The golden set contains items activation cannot score.** `"hi"` and
   `"summarise what you can see about me"` have no first action to give, and the model
   correctly answered them. Counting them as activation failures understates the model
   and overstates the fix's potential — but it also means the axis as written cannot
   reach 0.85 without either coaching-specific items or a per-item exemption.
3. **Remaining failures are real.** *"Take a break from reading and try taking out your
   pen and paper"* and *"break down your thoughts into smaller steps together"* are
   unbounded. Not scorer false negatives.
4. **3 epochs on 279 pairs is few.** The bounded tier needs repetition to become a default
   rather than a coincidence.

## What this tier does not fix

Fifteen pairs against 264 is a strong prior, not a substitute for a corpus. The retrain
below confirms that: it moved activation by one item. The ranked next steps are therefore
not "add more bounded pairs" but:

1. **Rebalance, don't just add.** Upweight the bounded tier (repeat it, or downsample the
   coaching tier) so it stops being 5.4% of the data. A 0.5B student learns the
   distribution it is shown.
2. **Fix the biased 264 pairs rather than out-vote them.** They teach the exact phrase
   `VAGUE_ACTION_RE` now rejects. Re-running generation under the corrected gate is the
   real fix; prefer external cooler hardware.
3. **Make the activation axis measurable.** Golden items like `"hi"` cannot be scored for
   a bounded first action. Either add activation-specific probe items or exempt
   non-coaching items per axis, so the number means something before it is optimised
   against.
4. **Raise the epoch count.** Three passes over 279 pairs is little exposure.

Activation is only fixed when the bench says so. A corpus that merely *contains* bounded
actions has not demonstrated that the model produces them.
