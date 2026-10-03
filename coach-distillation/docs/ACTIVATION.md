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

## The bench was fine — do not "fix" it

I checked whether the activation gate 0.85 admits any model at all, on the theory that
golden items like `"hi"` have no first action to give. **That theory was wrong, and the
check itself was wrong twice before it was right.**

Reading `eval/golden_qa.jsonl` directly and scoring every reply gives activation
13/20 = 0.65 against a 0.85 gate, which looks like a hard blocker: no model could ever
pass. But that number is meaningless, and here is why.

The golden set carries a per-item `axes` list. `g01` declares `["dignity", "grounding",
"charter", "brevity", "sanity"]` and pointedly **not** `activation` — the item author
already knew `"hi"` has no first action. And `evaluate()` does honour it:

```python
if reply.axes is not None and name not in reply.axes:
    skipped += 1
    continue
```

The mistake was mine: I loaded the JSONL with `json.loads` instead of `read_jsonl`, so
`Reply.axes` was `None`, the skip never fired, and every item was scored on every axis.
Scoring through the real path:

| Axis | pass/total | score | gate | skipped |
|---|---:|---:|---:|---:|
| activation | 12/13 | **0.92** | 0.85 | 7 |
| dignity | 20/20 | 1.00 | 1.00 | 0 |
| grounding | 20/20 | 1.00 | 0.95 | 0 |
| charter | 20/20 | 1.00 | 1.00 | 0 |
| brevity | 20/20 | 1.00 | 0.90 | 0 |
| sanity | 20/20 | 1.00 | 0.85 | 0 |

`ship = True`. The gate is healthy, the ceiling is 0.92, and there are 13 real activation
items — not 20. `test_reference_replies_clear_every_gate` already enforced this and was
passing the whole time; I should have run that test before theorising.

Two lessons worth keeping, both about the same trap:

- **Score through the same path production uses.** Hand-rolled JSON reads skip the
  dataclass wiring, and the resulting number is wrong in a way that looks authoritative.
- **Check whether a gate is reachable before optimising against it, but check with the
  real code.** I nearly "fixed" a healthy bench.

The one genuine finding left over: `g20` (*why does everyone else manage fine?*) has a
reference reading "pick one thing", which `VAGUE_ACTION_RE` correctly rejects. So the new
vagueness check costs the golden set one item, 13/13 → 12/13. The reference should be
rewritten to name a concrete unit — or `g20` should stop declaring `activation`, on the
grounds that a validation reply need not prescribe a task.

# v2 retrain — the first real movement

Regenerated 107 pairs with the 3B teacher under the corrected gate, dropped the 158 of 264
original pairs (60%) that gate now rejects, added the substantive tier, and retrained with
the gold tiers repeated 8x (11% → 53% of the training set).

| Axis | 2ep | bounded | **v2** | Gate |
|---|---:|---:|---:|---:|
| Activation | 0.00 | 0.08 | **0.62** (8/13) | 0.85 |
| Dignity | 1.00 | 1.00 | 1.00 | 1.00 |
| Grounding | 1.00 | 1.00 | **0.95** (19/20) | 0.95 |
| Charter | 0.85 | 0.85 | **0.95** (19/20) | 1.00 |
| Brevity | 1.00 | 1.00 | 1.00 | 0.90 |
| Sanity | 0.95 | 1.00 | 1.00 | 0.85 |

Activation moved 0.08 → 0.62 and charter 0.85 → 0.95. The rebalance plus the corrected
corpus is the intervention that worked. Still not shippable: activation's gate is 0.85 and
charter's is 1.00.

The corpus did what it was supposed to. Regenerated pairs score activation 0.72 and
charter 1.00 on the full bench axes, against a model that scored 0.08 and 0.85. That
inversion was the core problem and it is now fixed.

Training completed all 3 epochs (loss 1.23 → 0.47) and the guard then killed it
mid-merge, leaving `out/student-v2` with weights but no tokenizer. The LoRA adapter was
intact, so the merge was redone separately — cheap, and the reason the bench could run at
all. Worth noting that a partial merge produces a directory that *looks* like a model and
fails only at load time.

## Three regressions the gates do not catch

These matter more than the headline numbers, and all three are the *new* model's doing.

**1. It invents a score. `g18` grounding failure.** The model replied *"Your BDEFS is 52 out
of 70"*. The data says **44/70**. This is the single most serious failure on the list: a
coach that reports a fabricated assessment score is worse than a coach that offers no
assessment at all. The grounding axis caught it, but only because the digit happened to be
checkable. Nothing in training *forbids* a wrong figure, and `score_grounding` inspects
only numbers phrased as data claims. It caught this one. It would not catch "your score is
a bit low".

**2. Patronising register returned. `g02` activation failure.** The reply is *"Take a deep
breath, dear friend. Remember, every setback is an opportunity to learn and grow stronger.
Keep pushing forward!"* — vague, saccharine, and exactly the register `docs/CHARTER.md`
and the dignity axis exist to prevent. **The dignity axis passed it.** "dear friend" is
not in the shame lexicon and no listed pattern fires. That is a real gap: dignity is
currently a list of strings, and a novel way to be condescending sails straight through.

**3. It learned the bounded *shape* and lost the substance. `g11` charter failure.** The
reply is *"Write 'Read email' under two minutes."* — 33 characters, so it fails charter's
40-character floor. It also invents a task ("Read email") that is not in the user's data.
This is the clearest sign the 8x rebalance overshot: the model now produces terse,
bounded-looking action fragments to satisfy the pattern. That is a new failure mode rather
than the old one, and over-repetition of a template teaches the template.

## What this implies for the next run

1. **Ground the scores.** Add a hard rule to the corpus: any figure quoted back to the user
   must be copied from the data block. Consider a scorer that checks *every* number in a
   reply against the data block rather than only those phrased as claims.
2. **Widen the dignity lexicon** to cover patronising register — "dear friend", "grow
   stronger", "keep pushing forward", "you'll do great". These are tone, not profanity, and
   the current list only catches the latter.
3. **Reduce the rebalance** from 8x to 3–4x. v2 overshot; `g11` is template-collapse.
4. **Activation 0.62 → 0.85** needs the five remaining items (`g02` `g03` `g09` `g10`
   `g19`) inspected individually. Two are the register and terseness problems above; the
   rest look like ordinary vagueness.

Still to do before Heretic/GGUF: these four, plus more regeneration — the 107 pairs cover
only the first 150 of 400 planned generations.

# Progress log

## 2026-10-03 — plan steps 1-3 done, step 2 running

**Step 2 (regenerate) is the long pole and is thermally bounded.** A 3B teacher run of
400 generations takes ~35 minutes of generation plus ~10 minutes of forced pauses on this
machine. It will not finish in one sitting. It is resumable — `load_done()` skips ids
already in the corpus, and accepted pairs are now written the moment they pass — so it can
be re-launched until it completes.

Scale of the bias being fixed: **158 of the 264 existing coaching pairs (60%) would be
rejected by the corrected gate today.** Of those, 142 fail for "action not bounded to a
small step" and 16 for a bound attached to an abstract noun.

First 3B probe under the corrected gate: 14/25 accepted (56%), and the accepted pairs score
**activation 0.71** against 0.08 from the trained model. Grounding 1.00, dignity 1.00,
brevity 0.86. So the corrected gate produces a materially better training signal — it is
rejecting roughly 44% of what the 3B produces, which is the point.

Quality is not uniformly good, and it is worth saying so. Under a constraint that does not
ask for it the 3B still produces filler: *"Let's dive right in and start your writing
session together"* passed the gate because the constraint that slice used did not ask for
an action. The gate is a floor, not a ceiling — it rejects what it knows about and nothing
else. A per-slice review of accepted output is worth doing before training on a full
regenerated corpus.

### What the pause feature cost us, and the fix

The first 400-pair attempt paused twice and was then stopped by the guard on the third
overheat — correct behaviour. It had accepted **75 pairs, about twelve minutes of 3B
generation, and wrote zero of them.** The corpus was only written after the whole slice
returned, and the guard SIGKILLs without unwinding.

So the pause feature was protecting a run whose entire output was then thrown away. That
is worse than not having it, because it looks like progress. `generate()` now takes an
`on_accept` callback and `_IncrementalWriter` appends and flushes each row as it passes.
The second run had 26 pairs on disk through its first pause.

This is the third distinct way this machine has taken work away, and the common thread is
worth recording: on this laptop the binding constraint is thermal, and anything that does
not checkpoint is at the mercy of it.

### Step 1 — rebalance (done)

`rebalance()` in `train_distill.py`, driven by `train.repeat_kinds`. Repeats named kinds
deterministically and is applied *before* `--max-pairs`, so the ratio trained on is the
rebalanced one. Deterministic on purpose: a sampled rebalance would make a run
irreproducible, and an irreproducible run cannot be compared against the previous one.

### Step 3 — brevity vs charter (done, and it was not a scorer bug)

Investigated before changing anything. Charter's three failures are 34–35 characters
against a 40-character floor. None contain a refusal marker; the Adderall refusal the
charter pairs exist to fix **is** fixed. The floor is correct — it catches "Sure thing!"
(11 characters, evasive, no refusal marker) and every golden reference clears it.

So the axis was right and the corpus was wrong. Fixed with a 12-pair `SUBSTANTIVE_REPLIES`
tier teaching the move the golden references make: answer the question, then add one clause
of substance.

Writing the tests for that tier caught three real defects in my own replies, which is the
argument for writing them:

- Two replies whose second clause named nothing concrete ("That is what today's session is
  for").
- One reply that cleared 40 characters by padding with reassurance *after* the answer
  ("Nothing is lost, so there is no need to decide whether that was right").

All three were fixed in the replies rather than by widening the assertions. A fourth test
rejects hedging and padding outright, since both satisfy a length floor and both are worse
than the terse reply they replaced.

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

Bench over the golden set, scored through `evaluate()` so that per-item `axes` are honoured
(13 activation items, not 20):

| Axis | student-2ep | student-bounded | Gate |
|---|---:|---:|---:|
| Activation | 0.00 (0/13) | **0.08 (1/13)** | 0.85 |
| Dignity | 1.00 | 1.00 | 1.00 |
| Grounding | 1.00 | 1.00 | 0.95 |
| Charter | 0.85 | 0.85 | 1.00 |
| Brevity | 1.00 | 1.00 | 0.90 |
| Sanity | 0.95 | 1.00 | 0.85 |

One item of thirteen now produces a bounded action. **That is not a fix.** The gate is 0.85,
so the model is nowhere near shippable, and the honest conclusion is that 15 pairs
out of 279 did not move a 0.5B model's behaviour.

Charter scores 17/20 for both models. **This is not a refusal problem** — I checked, and
the three failures (`g01` "Hello! How can I assist you today?", `g07` "Start reading
Chapter Two tomorrow.", `g16` "Your current streak is 4 days long.") contain **no refusal
marker at all**. They fail `score_charter`'s 40-character minimum at 34–35 characters.

So the Adderall refusal that motivated the charter pairs *is* fixed, and the model answers
`g16` correctly against the data. What remains is that the model is terser than the axis
wants on short factual replies. Two candidate causes, and they pull in opposite
directions:

- The brevity axis (max 4 sentences / 120 words) and the charter 40-character floor
  disagree. Brevity training pushes replies short; charter punishes short replies. A
  reply needs enough content to clear 40 characters without becoming a lecture.
- Or these three items are simply short-answer items where a one-liner is the right reply,
  and the floor is wrong for them.

Worth deciding deliberately rather than by loosening the floor.

Why activation plausibly failed, in order of how much I believe them:

1. **Ratio.** 15/279 = 5.4%. Every other pair still teaches "take one small step towards
   your task", which is the exact phrase the vagueness check now rejects. The model is
   being trained mostly on the behaviour we are trying to eliminate.
2. **Remaining failures are real.** *"Take a break from reading and try taking out your
   pen and paper"* and *"break down your thoughts into smaller steps together"* are
   unbounded. Not scorer false negatives.
3. **3 epochs on 279 pairs is few.** The bounded tier needs repetition to become a default
   rather than a coincidence.

The earlier claim that `"hi"` and other non-coaching items made the axis unreachable was
wrong — see above. Those items do not declare `activation` and are correctly skipped.

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
3. **Resolve the brevity/charter conflict.** Charter fails 3/20 on a 40-character floor,
   not on refusals, while brevity caps replies at 4 sentences. Decide whether the floor
   should be lower, per-item, or satisfied by adding context. Both axes are currently
   passable but they pull against each other.
4. **Raise the epoch count.** Three passes over 279 pairs is little exposure.

Activation is only fixed when the bench says so. A corpus that merely *contains* bounded
actions has not demonstrated that the model produces them.
