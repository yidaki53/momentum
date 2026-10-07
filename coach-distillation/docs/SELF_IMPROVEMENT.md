# Could the student keep learning by itself?

Short answer: **yes in principle, no as a direct fine-tune on user conversations, and
the interesting version is a verifier loop rather than a self-training loop.** This note
sets out why, and what the version we could actually ship looks like.

## The three sources of signal, ranked by how much we should trust them

1. **The user telling us the reply was wrong.** Strongest. An explicit "that was not
   what I asked" or an edit is a real, free label. Expensive in attention, though: most
   users just leave, and the users who leave teach us nothing.
2. **What the user did next.** Weak but free. If the coach suggests "Rename the page"
   and the user instead opens the settings dialog, that is evidence — but the app
   cannot currently observe it, and inferring intent from clickstreams is exactly the
   kind of thing that goes wrong invisibly.
3. **A teacher scoring its own reply.** This is what `train_onpolicy.py` already does.
   No user involvement, no privacy exposure, and it is the only signal we can collect
   at scale today.

Signals 1 and 2 are the interesting ones and neither is wired up.

## The hard constraint: user text must not enter a training file

`AGENTS.md` is explicit — *"Real Momentum user data must never enter a training file"* —
and the reason is not squeamishness. Our users are discussing depression, ADHD and
medication. That text is the most sensitive data this app will ever hold, and a training
file is exactly the place it would leak from: into git, into an adapter checkpoint, into
a GGUF, into a bug report, into a laptop that gets stolen.

This rules out the obvious design. "Log the conversation, fine-tune on it" is not a
stretch goal we might reach later; it is not permitted, and the licence-gating in
`fetch_oa_corpus.py` exists for the same reason on the web side.

What *is* permitted is a **schematic** signal: reduce each interaction to a category
plus a verdict, the way `build_distill_pairs.py` already generates synthetic users
rather than harvesting them.

```
{ "intent": "stop-medication", "verdict": "non-responsive",
  "failure": "answered with an unrelated task action",
  "user_data_fields_used": ["active_tasks"] }        <- no free text, ever
```

From that shape a teacher can regenerate a synthetic user in the same situation and a
synthetic correct reply. The pipeline that already exists does exactly this. The new
part is only the trigger and the aggregation.

## Web search: the right use is grounding the teacher, not training on the web

We already have client-side retrieval (PubMed for clinical, DuckDuckGo otherwise).
There is a tempting move here — scrape search results and train on them — and it is
blocked twice over: the licence gate rejects anything without a provable licence, and
raw web text is a distribution nothing like our output format.

The useful version is narrower and better. Retrieval already happens *at inference*,
so the student gets grounded context it did not have at training time. Extend that one
step:

- The student answers using retrieved context.
- The teacher **verifies the answer against the retrieved document** and scores it.
- Where the teacher disagrees, that is a high-value training signal, because the
  disagreement is anchored to a source rather than to the teacher's memory.

This is the `PBSD` / `GATES` pattern in the survey's method table: the teacher's
document-grounded trajectory is filtered by an agreement gate before being distilled,
so unreliable retrieval-conditioned answers never reach the student. It converts web
search from a source of training text into a source of *verifiable* supervision, which
sidesteps licensing entirely because we never copy the document into a training file —
we only record whether the answer was consistent with it.

The urgency has a document behind it: `TEACHER_AUDIT.md` scored the raw 7B teacher at
2/10 on the charter vignettes against the student's 4/10, and `TEACHER_VERIFICATION.md`
cross-checked that verdict against retrieved sources rather than the scorer's opinion.
A teacher that cannot hold the charter unaided is exactly the teacher whose disagreements
must be anchored to a document before they count as signal.

## The failure mode that kills naive self-training

Self-training on your own outputs is a random walk. Without a signal that is
*independent* of the model, every round reinforces whatever the model already believed,
and variance shrinks each time. That is textbook mode collapse, and it is the same
thing that went wrong with v9c — just faster and without anyone noticing.

So the signal must be external to the student. Three things in the literature are worth
copying:

- **EGRSD** — entropy-gate the update: only train where the student was uncertain.
  Uncertain is where learning is possible; confident-and-correct needs no nudge.
- **UniSD** — require **multi-teacher agreement** plus an **EMA** of the student. Two
  independent teachers agreeing is evidence; one teacher agreeing with itself is not.
  The EMA is what stops the walk from taking huge steps.
- **Local Collapse** (Liu et al. 2026) — the most directly relevant result. It studies
  OPD from a strong to a weak member of the *same* Qwen3 family and reports collapse,
  using BIC-truncated OPD to control it. Same-family, same-size-gap, same collapse we
  are trying to fix.

## What we would actually build

Not "the model retrains itself". A bounded, auditable, opt-in loop:

1. **Collect.** On device, locally, behind an explicit per-user toggle. Store the
   schematic tuple above — never free text. A visible, exportable, deletable log.
2. **Cluster.** Group failures by intent category. A single odd interaction is noise;
   eleven interactions in the same category are a real gap.
3. **Gate.** Discard the cluster if it does not clear a threshold, or if the teacher
   cannot produce a confident answer for it. Most clusters will be dropped, and that is
   the design working.
4. **Generate.** Synthesise a small number of new pairs from the cluster using the
   existing `build_distill_pairs.py` machinery, so the corpus stays synthetic.
5. **Train.** Ordinary SFT plus a few on-policy epochs, with replay of the existing
   corpus to hold the charter axes steady — Ibrahim et al. (arXiv:2403.08763) show
   LR re-warm/decay plus replay alone matches full retraining.
6. **Hold out.** Evaluate against the vignette set and the golden set, including
   prompts the new cluster does not cover. A change that fixes `c03` while breaking
   `c07` is not progress.
7. **Promote or discard.** Never ship automatically. The bench gates apply exactly as
   they do now, and manual review still applies.

## What would make me change my mind about the whole direction

If the teacher diagnostic comes back at ~0, the honest conclusion is that a 0.5B
student on this hardware has no headroom, and the right response is a better base model
or a smaller product, not a more elaborate learning loop. A self-improvement pipeline
built on a model that cannot hold the charter is a way to automate the production of
mediocre coaching.

That is the order of operations, and it is why `pick_teacher.py` runs first.
