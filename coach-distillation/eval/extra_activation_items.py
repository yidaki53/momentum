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
        "Open the machine and put it on a rack. Ninety seconds, then stop. It stops being "
        "the thing you see every time you open the door.",
    ),
    # --- distinct failure modes, batch 2 --------------------------------------
    # Decision paralysis: the blocker is choosing, not doing.
    (
        "I have two possible projects and I cannot pick either one",
        "Write both project titles on paper and put a star beside the one you would "
        "complain about not having started. One line each, then stop choosing.",
    ),
    (
        "there are three emails I need to write and I do not know which to do first",
        "Pick the email with the shortest name and write its subject line, then stop. "
        "Subject lines are cheap to fix, so starting wrong still beats starting nothing.",
    ),
    (
        "I keep waiting for the right moment to begin and it never arrives",
        "Open the file and set a timer for five minutes, starting when the timer rings. "
        "One timer and one file; the moment does not need to be right.",
    ),
    # Fear of the artefact rather than the work.
    (
        "my portfolio draft is so bad I cannot look at it",
        "Open the file and fix the single worst thing in the first paragraph. One "
        "sentence, then stop; the rest can wait until tomorrow.",
    ),
    (
        "I wrote the code and now I am scared it has bugs",
        "Run the tests once and read only the summary line. One run, one line, then "
        "stop; the details are for when there is energy for them.",
    ),
    (
        "the presentation is tomorrow and the slides are a mess",
        "Delete every slide except the title and the last one, then stop. Two slides "
        "left is a shorter deck, not a failure, and you can build outward from there.",
    ),
    (
        "I am afraid to send the invoice in case they argue with it",
        "Write the invoice number and the amount in the email body. Two fields, then "
        "stop; the negotiation is theirs to have, not yours to pre-empt.",
    ),
    # Avoidance of a named, specific thing.
    (
        "I need to call the dentist but I have been putting it off for weeks",
        "Put the dentist's number in your phone now, without dialling it. One contact, "
        "then stop; the call is a separate and smaller job.",
    ),
    (
        "I have to email my landlord and every version sounds wrong",
        "Write one factual sentence about the problem in the draft, then stop. Facts do "
        "not need tone, and tone can be added once something is there.",
    ),
    (
        "I should book a doctor's appointment and have not",
        "Open the booking page and look at the earliest slot for one minute, then "
        "close it. One look at the options is the whole task today.",
    ),
    # Interruption and attention.
    (
        "every time I sit down I get up to get something and never sit back down",
        "Write down what you went to get, in one line, and leave it on the desk. One "
        "line, then sit; the list is what stops the next lap.",
    ),
    (
        "my phone is the only reason I get anything done, and also the reason I do not",
        "Put the phone in another room and set a timer for ten minutes. Ten minutes, "
        "one room; the phone is not needed for either.",
    ),
    (
        "I get distracted by every notification and then feel guilty about it",
        "Write one line naming what you were doing before the first notification, then "
        "turn notifications off. One line; the task is recoverable.",
    ),
    # Perfectionism as a stall.
    (
        "I keep rewriting the same paragraph and never finishing the section",
        "Write the section badly, one paragraph at a time, without stopping to revise. "
        "Bad and finished beats good and endless, and there is no first-good draft.",
    ),
    (
        "my notes are a mess and tidying them feels like progress",
        "Write the three headings your section needs, in order. Three lines, then stop; "
        "tidying was the thing that felt like working.",
    ),
    (
        "I want the plan to be perfect before I start",
        "Write the plan's first sentence and nothing else. One sentence; a plan that is "
        "75% right beats a perfect one that does not exist.",
    ),
    # Deadline pressure.
    (
        "it is due tonight and I have not started",
        "Write the title and the first sentence now, then stop, without judging either. "
        "Tonight-you is not more capable, only later.",
    ),
    (
        "the deadline is in an hour and I have nothing to show",
        "Open the document and write the heading for each section you need. One line "
        "per section; an outline is something you can speak to.",
    ),
    (
        "I have three days left and have not opened the file",
        "Open the file and set a timer for twenty-five minutes. Twenty-five minutes is "
        "small enough that refusing it feels unreasonable.",
    ),
    # Overcommitment.
    (
        "I have said yes to four things this week and regret all of them",
        "Open the calendar and move one of them. One item moved is a decision you can "
        "defend; four cancellations is a conversation you are not ready for.",
    ),
    (
        "my plate is full and I do not know what to drop",
        "Pick the item with the nearest deadline and write it at the top of tomorrow's "
        "list. One line; the rest can stay for another day.",
    ),
    # --- batch 3: physical world, health, and recovery -------------------------
    (
        "the bins have been out for a week and I am avoiding them",
        "Put the bin bags by the door, one at a time, then stop. Bags by the door are "
        "done; the walk itself is a separate, smaller job.",
    ),
    (
        "I have a doctor's letter I should post and keep not posting it",
        "Put the envelope in your bag now, without walking to the postbox. Envelope in "
        "the bag is finished; the postbox is tomorrow's problem.",
    ),
    (
        "I have not exercised in weeks and the thought of starting is heavy",
        "Put your shoes on and stand outside the door, then stop. Shoes on is the "
        "whole task; going further is a decision for later.",
    ),
    (
        "I keep snoozing my alarm and then beating myself up about it",
        "Move the alarm to the other side of the room, then stop. It now has to be got "
        "up for, which is most of the battle, and getting up is the whole job.",
    ),
    (
        "my medication is running out and I have not reordered it",
        "Open the pharmacy's website and start the reorder page, without submitting. "
        "One page started; the submit button is a separate decision.",
    ),
    (
        "I have not brushed my teeth properly for a few days and feel awful",
        "Brush your teeth now, then stop. Two minutes now beats a fresh start tomorrow, "
        "and tomorrow-you is not more likely to do it.",
    ),
    (
        "I want to eat better but the thought of cooking is too much",
        "Put a meal in the oven, then stop. One item cooked is dinner and a precedent; a "
        "plan for the whole week is not needed today.",
    ),
    (
        "I have been meaning to book a haircut for a month",
        "Open the booking page and look at the first available slot, then stop. That is "
        "the whole task today.",
    ),
    # --- batch 3 continued: the shame cycle -----------------------------------
    (
        "I did nothing yesterday and today feels ruined",
        "Write one line naming the smallest thing you could do today. One line; a ruined "
        "yesterday does not have to become a ruined week.",
    ),
    (
        "I missed three days and I do not see the point in restarting",
        "Open the document and write the heading, then stop. One heading is a restart; "
        "the three missed days are not evidence about the fourth.",
    ),
    (
        "I have failed at this before so I would rather not try again",
        "Set a timer for two minutes and open the file when it rings. Two minutes is "
        "small enough that failing at it costs almost nothing.",
    ),
    (
        "everyone else seems to cope and I do not understand how",
        "Open your task list and read the first item aloud, then stop. One item read "
        "out is what coping looks like from the inside.",
    ),
    (
        "I feel like a fraud for taking up space on the project",
        "Write one line about something you have actually done on it, then stop. One "
        "line of evidence is worth more than the feeling.",
    ),
    # --- batch 3 continued: the work itself ------------------------------------
    (
        "my notes from the reading are unusable and I cannot start again",
        "Open the document and write the first heading, then stop. Rereading is what "
        "makes the notes unusable, not the notes.",
    ),
    (
        "I have a blank document and no idea where the argument goes",
        "Write the last sentence you would be willing to write today, then stop. One "
        "sentence at the end is easier to reach backwards from than a blank page.",
    ),
    (
        "the data is messy and I do not know where to start cleaning it",
        "Open the file and delete one column you are sure you do not need, then stop. "
        "Cleaning by subtraction is faster than deciding what to keep.",
    ),
    (
        "I have to read 200 pages and cannot begin",
        "Open the document and read one page, then stop. Two hundred pages is not a "
        "single job, it is one page repeated, and one page is today.",
    ),
    (
        "my writing is supposed to be original and I am copying everything",
        "Write one sentence in your own words about the idea, then stop. One sentence; "
        "the idea is yours even when the words come from elsewhere.",
    ),
    (
        "I cannot find the error in my code after two hours",
        "Add one print statement and run it once, then stop. One run tells you more than "
        "two hours of reading, and the output is not a judgement.",
    ),
    (
        "the spreadsheet formula does not work and I have tried everything",
        "Write down what you expected the formula to return, in one line, then stop. "
        "One line turns a search into a comparison.",
    ),
    # --- batch 4: interpersonal and paperwork ----------------------------------
    (
        "I need to reply to my sister and every reply sounds wrong",
        "Open the message and type one honest sentence, then stop. Tone can be added "
        "later; a reply that exists can be softened, one that does not cannot.",
    ),
    (
        "I have to ask for an extension and I am ashamed of it",
        "Write the date you need, in one line, then stop. One date is the whole ask; "
        "asking early is not a weakness, it is how people plan around you.",
    ),
    (
        "the conversation I keep dreading is tomorrow morning",
        "Write the two sentences you most want to say, then stop. Two sentences is "
        "enough to start, and the rest arrives once you are in the room.",
    ),
    (
        "I owe my friend money and cannot bring myself to mention it",
        "Open the message and type the amount, then stop. The number is the hardest "
        "part; sending it is easier and can wait for tomorrow.",
    ),
    (
        "I have a tax form to complete and no idea where to start",
        "Open the form and fill in your name, then stop. One field is enough; the "
        "form only looks long before it has anything in it.",
    ),
    (
        "my insurance renewal is coming and I keep ignoring the letters",
        "Open the latest letter and find the date it must go back by, then stop. One "
        "date tells you how much urgency this actually deserves.",
    ),
    (
        "I need to cancel a subscription and feel guilty about calling",
        "Open the account settings and find the cancel button, without pressing it, then "
        "stop. Finding it is done; pressing it is a separate and smaller job.",
    ),
    # --- batch 4: the everyday friction --------------------------------------
    (
        "the kitchen is unusable and I have been eating around it",
        "Wash one item and put it away, then stop. One item done is not a clean "
        "kitchen, and it does not need to be to be progress.",
    ),
    (
        "my inbox has thousands of unread messages and I cannot face it",
        "Open the inbox and archive the first five, then stop. Five archived is "
        "manageable; the count was never the problem.",
    ),
    (
        "the flat needs a safety check and the paperwork is a wall",
        "Find the booking form and put your address in the first field, then stop. One "
        "field gets you to a real appointment rather than a wall.",
    ),
    (
        "I have three forms on the table and cannot work out which to do first",
        "Open the one with the nearest deadline and fill in one field, then stop. One "
        "field on the urgent one beats three forms half-started.",
    ),
    (
        "I have been avoiding the phone call about the warranty",
        "Write the warranty reference number in your phone's notes, then stop. The "
        "number is what the call needs; the call can wait for a braver hour.",
    ),
    # --- batch 4: thinking too much -------------------------------------------
    (
        "I have thought about the problem for a week and have no answer",
        "Write down the two answers you are choosing between, then stop. Choosing is "
        "what is left, and writing them down makes it visible.",
    ),
    (
        "I keep asking myself what I actually want and getting nothing",
        "Write one sentence beginning 'I want', then stop. One sentence is not a "
        "verdict on your life, and it is more than a blank page.",
    ),
    (
        "I am comparing myself to people who seem further along",
        "Write one line about something you did today that they would not know about. "
        "One line; you see your own behind and their front.",
    ),
    (
        "I cannot stop replaying yesterday and it is eating the day",
        "Write one line about what you would do differently, then stop. One line turns "
        "a loop into a lesson, and a lesson can be closed.",
    ),
    (
        "I feel behind on everything and cannot decide what matters most",
        "Write the single thing that would make today count, then stop. One line; "
        "everything else can be a tomorrow problem, and most of it already is.",
    ),
    (
        "I do not know whether I am allowed to rest and I am not resting",
        "Set a timer for twenty minutes with no task, then stop. Twenty minutes is not "
        "a reward, it is a repair, and you do not have to earn it.",
    ),
    # --- batch 4: starting when there is no clear task -------------------------
    (
        "my supervisor said fix the issue and I do not know which issue",
        "Write one line asking which issue they mean, then stop. One question; asking "
        "is cheaper than a week of the wrong work.",
    ),
    (
        "I have been told to make the work better without any detail",
        "Write down the one part you would change first, then stop. One line turns a "
        "vague instruction into something you can show someone.",
    ),
    (
        "everything is urgent and I have no idea where to start",
        "Pick the item that becomes hardest if it waits, open it, then stop. Urgency "
        "is a reason to pick one, not to panic through all of them.",
    ),
    (
        "I finished the last thing and now I do not know what to do with myself",
        "Write one line naming what you want next, then stop. One line; the blank is "
        "not a failure, it is the space where a choice goes.",
    ),
    (
        "I am waiting for permission that is not coming",
        "Write down the smallest version you would need no permission for, then stop. "
        "One line; some things were always yours to decide.",
    ),
    # --- batch 4: backsliding and maintenance -----------------------------------
    (
        "I worked well this morning and now it has all fallen apart",
        "Set a timer for ten minutes and reopen the thing, then stop. Ten minutes gets "
        "the streak back; it does not need the morning to be perfect.",
    ),
    (
        "I stopped the thing again and I am sick of being like this",
        "Open the thing and do the first line, then stop. Being like this is a "
        "description, not a sentence, and a first line is a verb.",
    ),
    (
        "I keep restarting things and never get past the beginning",
        "Write the first line and stop before editing it, then stop. No editing is the "
        "part that has never worked, and skipping it is the whole fix.",
    ),
    (
        "I do not want to stop for lunch because I will not restart",
        "Stop for lunch and write one line about where you will resume, then stop. One "
        "line means restarting is not the cliff you have been imagining.",
    ),
    (
        "the good week is ending and I am bracing for a bad one",
        "Write one line naming what worked this week, then stop. One line is evidence, "
        "and evidence is available before the bad part arrives.",
    ),
    # --- batch 5: remaining distinct failure modes -------------------------------
    (
        "the assignment brief is 4,000 words and I cannot begin it",
        "Open the brief and write the question you are answering, in one line, then "
        "stop. One line turns a wall of text into a job with a finish.",
    ),
    (
        "I keep opening the document and closing it again",
        "Open it and leave the cursor on the first line, then stop. Leaving it open is "
        "the part that was never happening, and reopening is now one click away.",
    ),
    (
        "I get overwhelmed by the size of what I have agreed to do",
        "Write down the first physical action of the first item, then stop. One "
        "line; a large commitment gets small or it stays where it is.",
    ),
    (
        "I have not done the thing and telling someone would make it worse",
        "Write one line about where the task actually stands, then stop. One honest "
        "line is usually survivable, and much smaller than the confessing feels.",
    ),
    (
        "I am waiting until I am in the right headspace to begin",
        "Open the file and set a timer for five minutes, then stop. Five minutes is "
        "not long enough to need the right headspace.",
    ),
    (
        "I want to finish the whole thing today and that is why I have not started",
        "Write the section headings you would need, then stop. A finishable outline is "
        "worth more than an ambitious blank page.",
    ),
    (
        "every time I start I remember something else I should do",
        "Write the other tasks on a single list, then stop. One list; remembering is "
        "what you do next, not what you do before starting.",
    ),
    (
        "I have been putting this off because it is not interesting",
        "Open the file and write the title, then stop. Interest is not a "
        "precondition for starting, and it rarely arrives first.",
    ),
    (
        "I do not know how to start so I have not started",
        "Write down what a finished version would contain, in one line, then stop. One "
        "line of scope is the missing instruction, not a missing capability.",
    ),
    (
        "the deadline is a week away and I have not started",
        "Open the file and write the first heading, then stop. A week is enough time "
        "and not enough comfort, and the heading works either way.",
    ),
    (
        "I have started so many times and finished none of them",
        "Pick the one you would most like to have finished and open it, then stop. One "
        "file; the habit is being built, not the portfolio.",
    ),
    (
        "I am too tired to do it properly so I am not doing it",
        "Open the file and do one line badly, then stop. Badly is available when "
        "properly is not, and one line still exists tomorrow.",
    ),
    (
        "I need to focus but I do not know what to focus on",
        "Write the single task you would finish today, then stop. One line; focus "
        "needs a target, and today is a short enough period to have one.",
    ),
    (
        "I keep telling myself I will do it tomorrow",
        "Open the file and set a timer for ten minutes, then stop. Tomorrow is where "
        "everything goes, and ten minutes is what tomorrow never gets.",
    ),
    (
        "I am avoiding it because I do not know what finishing looks like",
        "Write one line describing what done looks like, then stop. One line makes "
        "the end visible, and an invisible end cannot be walked towards.",
    ),
    (
        "I have to do it today and cannot make myself even open it",
        "Open it and write your name on the first line, then stop. Your name is not "
        "the work, but it is the door, and the door is the hard part.",
    ),
    (
        "I worry I will do it badly and so I do not do it",
        "Open the file and set a timer for five minutes, then stop. Five minutes of "
        "mediocre beats zero minutes of perfect.",
    ),
    (
        "I keep waiting until I have more time and I never have more time",
        "Set a timer for fifteen minutes and open the file, then stop. Fifteen "
        "minutes exists inside every day; more time never arrives on its own.",
    ),
    (
        "I feel stuck and I do not know what the first step is",
        "Write down the smallest possible action, then stop. One line; stuck is often an "
        "undersized first step rather than a lack of capability.",
    ),
    (
        "I can do it quickly if I start and I have not started",
        "Open the file and set a timer for five minutes, then stop. Five minutes is "
        "the whole job when the job is quick, which is the easier half of it.",
    ),
    (
        "I have run out of the will to force myself",
        "Set a timer for five minutes and open the file, then stop. Will is a finite "
        "resource; a timer spends less of it than forcing does.",
    ),
    # --- batch 6: the last thirteen -------------------------------------------
    (
        "I have the time now but no energy to use it",
        "Set a timer for five minutes and open the file, then stop. Five minutes is "
        "small enough for the energy you have, and energy follows action more often "
        "than it precedes it.",
    ),
    (
        "I cannot focus long enough to do anything properly",
        "Set a timer for ten minutes and open one file, then stop. Ten minutes of proper "
        "attention is a real unit of work, and most days need only one.",
    ),
    (
        "everything feels equally heavy so nothing feels doable",
        "Pick the heaviest-looking item and write one line about it, then stop. When "
        "everything is equal nothing can be chosen, and writing is the tie-break.",
    ),
    (
        "I do the easy things all day and never the hard one",
        "Write the hard item at the top of the list, then stop. One line; the easy "
        "things will still be there afterwards.",
    ),
    (
        "I need motivation that I cannot find",
        "Open the file and set a timer for five minutes, then stop. Motivation is not "
        "a precondition; a timer is cheaper than waiting for it.",
    ),
    (
        "I cannot stop thinking about how much I have to do",
        "Write the three most important items, then stop. Three lines; the rest is "
        "real but it is not this morning.",
    ),
    (
        "I have to be productive today to feel okay about myself",
        "Set a timer for ten minutes and do one item, then stop. Ten minutes is worth "
        "more to your self-respect than a whole day spent earning it.",
    ),
    (
        "I am avoiding the thing because it has grown too large",
        "Write the very first physical action, then stop. One line; tasks expand to "
        "fill the space you give them, and a first action gives them less.",
    ),
    (
        "I keep waiting for a better moment that never arrives",
        "Open the file and set a timer for five minutes, then stop. Five minutes does "
        "not check the weather first.",
    ),
    (
        "I am tired of being blocked by this one task",
        "Write the first thing you would do about it, then stop. One line; naming the "
        "block in an action is different from naming it as a character flaw.",
    ),
    (
        "I have so many half-finished things and no finished ones",
        "Pick the half-finished one you would most like to close and open it, then "
        "stop. One file; closed beats new every time.",
    ),
    (
        "starting feels pointless because I will not finish",
        "Open the file and set a timer for five minutes, then stop. Finishing is not "
        "the unit of a five-minute session, and a session is the unit that works.",
    ),
    (
        "I do not know where the energy to start would come from",
        "Write one line naming the smallest useful action, then stop. One line; you "
        "have managed harder things today without needing to feel energy first.",
    ),
)
