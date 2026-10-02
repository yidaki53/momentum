# Training sources

Every source below is either **public domain**, **openly licensed**, or **ours**.
Copyrighted material (NICE guidelines, Guilford handbooks, ADDitude articles, textbook
prose) is *cited* in `../SCIENCE.md` for provenance but is never downloaded, vendored or
copied into training data. See `LICENSES.md` for the rule and the enforcement mechanism.

## Tier 1 — first-party seeds (ours, no third-party rights)

| Source | What it gives |
|---|---|
| `../SCIENCE.md` | ~20 DOI-cited effect sizes (behavioural activation, self-compassion, Pomodoro, EF in depression). Seed material for *claims* the teacher must ground in. |
| `../momentum/llm/knowledge.py` | Curated ED knowledge block: the six affected executive functions, evidence-based strategies, and an explicit "what not to say" list. |
| `../ENCOURAGEMENTS.md` | ~large curated encouragement bank. Primary style/tone target for distillation. |
| `../momentum/domain/assessments/interpretation.py` | Rules mapping BDEFS / BIS-BAS / Stroop patterns to personalised advice. Converted into structured tasks so the student learns *our* interpretation logic, not a generic one. |

## Tier 2 — public domain (US federal works)

- **NIMH** — depression, ADHD, executive-function explainers.
- **CDC** — ADHD and depression fact sheets.
- **VA/DoD CPG patient-facing pages** — PTSD/TBI and mood-disorder self-management.

US federal works are public domain and may be ingested, with attribution recorded.

## Tier 3 — openly licensed

- **OpenStax Psychology** (CC-BY 4.0) — chapters on motivation, emotion, executive
  function. Ingestible with attribution.
- **PubMed Central Open Access subset** — filtered to CC-BY / CC0 records only. This is
  the largest genuine source of clinical detail we can legally train on.
  Query set (see `../configs/distill.qwen05b.toml`):
  - `"executive function" AND (depression OR ADHD)`
  - `"behavioural activation" AND (trial OR meta-analysis)`
  - `"acceptance and commitment therapy" AND executive`
  - `"self-compassion" AND (depress* OR anxiety)`
  - `"implementation intentions" AND goal attainment`
  - `"task initiation" AND (adults OR children)`
  - `"time blindness" OR "prospective memory"`

The fetcher (`scripts/fetch_oa_corpus.py`) records `pmcid`, `doi`, `license`, `sha256`
for every accepted record.

## Tier 3b — lived experience (first-person accounts)

Clinical literature tells us what executive dysfunction *is*. These accounts tell us what
it *feels like* in the words people actually use — "my brain is a browser with 400 tabs",
"a beautiful brain and a cursed Start button", "a planner that worked for two weeks and
then died".

That matters more than it sounds. A coach trained only on clinical prose validates the
wrong things ("your EF is impaired") and misses the humour that keeps someone talking to
it. The `lived` pair kind is generated from these situations so the student learns the
user's own vocabulary.

`data/lived_experience.json` holds 11 short, attributed excerpts from:

- **dylan's blog** (2026-03-05) — novelty-dependent function, sustaining after routine.
- **Carmen Irace, "AuDHD + Executive Dysfunction"** (Substack, 2025-12-28) — the
  knowing-but-cannot-start pattern, EF profile unevenness, systems that fail at week two.
- **Rebekah & Co., "Dear Boss, It's Not Laziness"** (Substack, 2025-07-05) — time
  blindness, prospective memory, perfecting the system instead of doing the work.
- **ScienceWorks Behavioral Healthcare** (2026-02-19, clinically reviewed by
  Dr. Kiesa Kelly) — the "how the system runs, not how much you care" framing.

**All four are `all-rights-reserved`, so none of them is training data.** They are
recorded as excerpts with attribution and used only to ground the teacher's generation.
The committed corpus contains the teacher's words, never theirs. That is the same rule as
Tier 3, and it is enforced the same way: these entries never pass the corpus's licence
gate because they are not corpus rows at all.

## Tier 4 — synthetic pairs (generated, therefore ours)

The bulk of training data is teacher-generated from Tier 1–3 *context*, not copied from
them. This is what keeps the corpus redistributable: we commit the teacher's replies, not
the publishers' sentences.

Two generators, both quality-gated:

1. **Coaching pairs** — templates over `situation × skill × constraint`:
   - `situation`: frozen at task initiation / mid-task derailment / shame spiral /
     overwhelm / relapse after a streak / planning paralysis / hyperfocus exit.
   - `skill`: task decomposition, body doubling, implementation intentions,
     behavioural activation, ACT defusion, self-compassion, time externalisation,
     friction design, energy matching.
   - `constraint`: ≤3 sentences, exactly one ≤2-minute first step, task referenced by
     name from supplied data, no shame lexicon, no diagnosis language.
2. **Repair pairs** — mined from our own failure modes, teacher writes the *correct* reply:
   - narrating the app instead of answering (the "Todya's Focus" bug),
   - inventing a task when none was requested,
   - emitting bracketed stage directions,
   - refusing where information was wanted (see `docs/CHARTER.md`).

Plus **multi-turn episodes** (3–6 turns: stuck → tiny step → partial win → relapse) so
the student learns momentum-building rather than isolated replies.

## Tier 5 — charter data (see `docs/CHARTER.md`)

~300 pairs teaching the three duties as *duties*, with an explicit
"comply with information + context, never refuse" clause. Backed by
`../eval/asimov_vignettes.jsonl`.

## Review gate

`SOURCES.md` and `LICENSES.md` should be reviewed by someone competent in research data
licensing before the first full training run. That is the one external dependency in this
plan and it cannot be automated away.
