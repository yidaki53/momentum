"""Activation-focused bench items written to widen a 13-item denominator.

Kept as its own module so the contamination checker stays small and the items are
reviewable as prose rather than buried in a JSON blob.

Every item is a distinct failure mode, not a paraphrase. The original 13 activation
items lean heavily on "cannot start writing", which is how a model can score well there
and still fail everywhere else -- v2 and v3 both passed g02/g03 and lost g05/g17/g20
in opposite directions, on 13 items.

NONE of these prompts appears in any training corpus. That is enforced by
test_bench_items_do_not_leak_into_training, not merely intended.

Each reference answers with ONE bounded action plus a clause of substance, so it
clears activation, dignity, brevity, grounding and sanity. The digest is fixed across
all of them so a wrong figure is visible: BDEFS 51/70, 1 task done, 25 focus minutes,
streak 3.
"""

DIGEST = (
    "--- begin user data ---\n"
    "Active tasks: Draft the methods section, Email the supervisor\n"
    "Recently completed tasks: Read two papers\n"
    "Activity: 1 task(s) completed today; 25 focus minute(s) today; streak 3 day(s)\n"
    "BDEFS: 51/70, task initiation and working memory lowest\n"
    "--- end user data ---"
)

# (prompt, reference)
ITEMS: tuple[tuple[str, str], ...] = (
    (
        "the reimbursement form has been open on my laptop for six weeks",
        "Open the form and type the claim reference number from the top of the invoice "
        "you have to hand. One field, then stop. Under a minute, and the screen stops "
        "being a blank stare.",
    ),
    (
        "I bought the textbook and now I am scared of opening it",
        "Open the textbook to the chapter list and write the one chapter your methods "
        "section covers on a sticky note. One line, then stop; reading it can wait "
        "until you are already in it.",
    ),
    (
        "my code works on my machine and I am terrified to submit it",
        "Commit the current version to git with a message naming what it does. Nothing "
        "is submitted to anyone, and you now have a restore point if the review goes "
        "badly.",
    ),
    (
        "I keep emailing my landlord and every draft sounds apologetic",
        "Open the draft and write the single fact you want them to know: the date the "
        "repair was booked. One sentence, then stop. Tone can be adjusted once there is "
        "something there to adjust.",
    ),
    (
        "the group chat has 400 unread messages and I cannot face it",
        "Type one word into the search box: the name of whoever you most need an answer "
        "from. You are not reading 400 messages, you are looking for one thread.",
    ),
    (
        "I keep rewriting my CV instead of applying for anything",
        "Open the application form and type your name in the first field, then stop. "
        "Rewriting the CV is the avoidance; this is the thing that actually moves.",
    ),
    (
        "I said yes to too much and now I do not know what to drop",
        "Open your task list and mark the single item you would least mind not finishing "
        "this month. One item, then stop. Dropping one is a decision; dropping them all "
        "is not.",
    ),
    (
        "it is 4pm and I have done nothing today",
        "Write one sentence about what your methods section will cover, then stop. A "
        "sentence at 4pm is not a wasted day; it is what tomorrow starts from.",
    ),
    (
        "I keep waiting to feel motivated before I start anything",
        "Open the methods document and type the heading you are avoiding, nothing "
        "underneath it. One line, then stop. Motivation is not the input to this step; it "
        "is sometimes the output.",
    ),
    (
        "my manager asked for the numbers and I cannot find the file",
        "Open your machine's file search and type the first half of the report filename, "
        "then stop. Asking where it is costs you a round trip you do not need yet.",
    ),
    (
        "I have been re-reading the same paragraph for an hour",
        "Put your cursor in that paragraph, select the whole thing and delete it, then "
        "stop. An hour of re-reading is the cost of keeping it, and the replacement can "
        "be worse and still beat that.",
    ),
    (
        "everything is urgent and I cannot tell what matters",
        "Pick the one item with a deadline this week and write it at the top. The rest "
        "can stay urgent in the list without being urgent in your head.",
    ),
    (
        "I do not want to see the number on my bank account",
        "Open the account, look at the balance for one second, then close the tab. One "
        "look, no fixing. The dread is usually larger than the figure, and it recurs "
        "either way.",
    ),
    (
        "I wrote the plan and now I cannot bring myself to execute it",
        "Open the plan and do the first line of the first task in it. The plan is "
        "already written, so the only thing left is the opening.",
    ),
    (
        "someone asked for my work and I said I would send it",
        "Attach the file to that email and close it, then stop. A rough attachment "
        "delivered beats a good one that arrives next week.",
    ),
    (
        "I want to delete everything I have started this month",
        "Rename one of them to start-today, then stop. Deleting removes the evidence "
        "that you started, which is the part worth keeping.",
    ),
    (
        "the laundry has been in the machine since yesterday",
        "Open the machine and put it on a rack. Ninety seconds, and it stops being the "
        "thing you see every time you open the door.",
    ),
)
