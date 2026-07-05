"""Curated knowledge base about executive dysfunction for LLM context injection.

Sources: Public domain / freely available research summaries, CDC, NIH, ADDitude
magazine (used with permission for non-commercial), and common CBT/ACT frameworks.
This is injected into the system prompt to ground the LLM's responses.
"""

EXECUTIVE_DYSFUNCTION_KNOWLEDGE = """
## What is Executive Dysfunction?

Executive dysfunction is a disruption of the brain's management system — the
cognitive processes that help us plan, focus attention, remember instructions,
and juggle multiple tasks. It is NOT laziness or lack of willpower. It is a
neurological condition common in ADHD, autism, depression, anxiety, TBI, and
chronic stress.

## Key Executive Functions Affected

1. **Working Memory** — Holding information in mind while using it.
   - Signs: Forgetting instructions mid-task, losing train of thought.
   - Strategies: Write things down immediately, use external memory aids.

2. **Task Initiation** — Starting a task without procrastination.
   - Signs: Sitting frozen, knowing what to do but unable to start.
   - Strategies: The 2-minute rule (do it for 2 minutes), body doubling,
     breaking tasks into micro-steps.

3. **Sustained Attention** — Maintaining focus despite distractions.
   - Signs: Mind wandering, task-switching, losing focus mid-task.
   - Strategies: Pomodoro technique, noise-cancelling headphones, visual timers.

4. **Planning & Prioritisation** — Identifying steps and ordering them.
   - Signs: Feeling overwhelmed, not knowing where to start.
   - Strategies: Brain dump, Eisenhower matrix, reverse engineering the goal.

5. **Emotional Regulation** — Managing frustration and discouragement.
   - Signs: Meltdowns over small setbacks, shame spirals.
   - Strategies: Name the emotion, self-compassion breaks, ACT defusion.

6. **Flexibility** — Adapting when plans change.
   - Signs: Getting stuck when something unexpected happens.
   - Strategies: Pre-plan contingencies, practice micro-transitions.

## Evidence-Based Support Strategies

### Behavioural Activation
- Small wins build momentum. Completing even a tiny task releases dopamine.
- External structure compensates for impaired internal executive function.
- Visual progress tracking (checklists, streaks) leverages reward sensitivity.

### Acceptance and Commitment Therapy (ACT)
- Defusion: "I notice I'm having the thought that I can't do this."
- Values-based action: Connect tasks to deeper values, not just obligation.
- Committed action: Choose one small values-aligned step.

### Environmental Modifications
- Reduce friction for desired behaviours (keep supplies visible).
- Increase friction for distractions (phone in another room).
- Use "defaults" — set up your environment so the right choice is the easy one.

### Compassionate Self-Talk
- Shame is paralysing; self-compassion is activating.
- Replace "I should have done this already" with "I'm starting now, and that's enough."
- Progress, not perfection.

## Common Challenges

- **Analysis paralysis**: Too many options → freeze. Limit choices to 2-3.
- **Time blindness**: Difficulty sensing time passing. Use external timers.
- **All-or-nothing thinking**: "If I can't do it perfectly, why bother?"
  Counter with: "Something is better than nothing. Partial progress counts."
- **Shame cycle**: Procrastinate → feel shame → procrastinate more to avoid shame.
  Break the cycle with self-compassion and a tiny first step.

## What NOT to Say to Someone with Executive Dysfunction
- "Just try harder." (They are already trying their hardest.)
- "You're lazy." (This is incorrect and harmful.)
- "Why can't you just do it?" (If they knew, they would.)
- "Everyone struggles with that." (While true, the intensity is disabling.)

## Effective Encouragement Principles
1. Validate the difficulty first.
2. Normalise the struggle (it's the brain, not the person).
3. Offer one tiny, concrete next step.
4. Emphasise that partial progress is real progress.
5. Use warm, non-judgmental language.
6. Remind them of past successes (builds self-efficacy).
"""

# Shorter version for the welcome/encouragement prompt
ENCOURAGEMENT_KNOWLEDGE = """
Executive dysfunction is a neurological condition, not laziness. Key strategies:
- Small wins build momentum (dopamine reward)
- The 2-minute rule: commit to just 2 minutes
- External structure compensates for internal executive function challenges
- Self-compassion breaks the shame-procrastination cycle
- Partial progress is real progress
- Body doubling and visual timers are effective tools
"""
