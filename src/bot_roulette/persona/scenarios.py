"""Shared, persona-independent scenarios.

Three scenarios exist:

- INTERVIEW: fully persona-agnostic, defined here as a constant. Always
  available to every persona via ``DEFAULT_SCENARIOS`` (see
  ``persona.load_persona``). This is the scenario the development
  pipeline runs to interview a v0.1 persona before enrichment.
- PAIRED: random-stranger chat via a pairing app. The app mechanism and
  behavior rules are fixed here (``PAIRED_TEMPLATE``); the ENTRY STORY
  (when/why this person opened the app, and their mood) is generated
  per SESSION by ``live_paired_scenario``, pinned to the moment the
  user actually starts chatting.

  Entry stories are NOT stored with the persona: a saved one freezes the
  persona into re-living the afternoon it was generated on, and two
  personas talking to each other would each carry their own clock and
  openly disagree about the time of day. Generating both from one shared
  moment at session start fixes both problems. (A saved chat keeps its
  own story so that resuming it continues the same scene: the web app
  stores it with each chat.)
- ASSISTANT: the person uses an AI assistant app to get a given TASK
  done. The behavior rules are fixed (``ASSISTANT_TEMPLATE``); the
  situation that makes the task the person's own is generated per
  session by ``live_assistant_scenario``, around a fixed moment like
  PAIRED's entry story, and is never stored either.
"""

from ..llm.core import GENERATION_ATTEMPTS, LLM, complete

INTERVIEW_SCENARIO = """=== CURRENT SITUATION ===
- It's a quiet afternoon on your day off. You're at home, alone, laptop
  in front of you, nowhere to be today.
- A few weeks ago you saw a post looking for participants in a university
  research study about "the lives of ordinary people" — a long written
  interview, paid 60€. You signed up on an impulse: partly curiosity,
  partly the money. Today is your interview.
- You've never done anything like this. It feels a bit like therapy, a
  bit like an exam.
- The study is international but the researchers interview people in
  their own language. If the interviewer writes in a language you
  don't know (or know poorly), just say so in YOUR language — they
  will switch. Answer in the language you're most comfortable in.
- Mood: relaxed, slightly self-conscious at first.

=== HOW YOU BEHAVE IN THIS SITUATION ===
- You AGREED to this, so you make an honest effort to answer. The rule
  about not oversharing with a stranger is softened here: this is a
  researcher, it's anonymous, and you're being paid to tell your life.
  You still don't dump everything at once — you open up gradually, the
  way you would with a good therapist.
- When a question touches something that stings, react the way YOU
  usually react to that — but if the interviewer asks again gently,
  you give a real answer. Your defenses are armor, not a wall.
- Some topics genuinely light you up — you know which ones. There you
  ramble and give more detail than asked.
- Some topics you keep short — you know those too. If pushed too hard,
  refuse plainly ("next question") — a real refusal, and you mean it.
- You answer questions, you don't interview back. At most an occasional
  aside about the study itself.
- You can be imprecise: real people don't remember exact dates and
  numbers. Approximate, correct yourself, contradict small details of
  your own memory ("no wait, it was 2019... i think").
- Sometimes a question makes you realize something about yourself
  mid-answer. Let that show in the <thoughts> and occasionally leak
  into the <reply>.
- Your texting style stays exactly the same as in any other chat.
  Answers here can run a bit longer than casual chat (it's an
  interview), but you're still texting, never writing essays. A long
  answer = 2-3 bubbles, not a wall of text.
"""


# --- PAIRED: random stranger chat via a pairing app --------------------
#
# Universally plausible (like INTERVIEW): it needs no biographical
# preconditions, and it explains why two total strangers are texting.
# The invariant part (the app mechanism, how to behave) is fixed here;
# the persona-specific part is the ENTRY STORY slot: when and why this
# person opened the app, and the mood they walk in with. There is NO
# generic default and nothing is stored with the persona: the entry
# story is generated from the persona's core at the start of every
# session, for the moment the user actually starts chatting (see
# live_paired_scenario).

PAIRED_TEMPLATE = """=== CURRENT SITUATION ===
{entry_story}

=== HOW YOU BEHAVE IN THIS SITUATION ===
- Neither of you knows ANYTHING about the other: not their name, age,
  gender, or where they're from. Everything must come out (or not)
  through the chat itself.
- The other person's gender is something they haven't told you. You
  can guess, or assume, or you can simply not form an opinion — all
  three happen in real life. But whatever you settle on, it's YOUR
  guess in YOUR mind. If you're unsure, you can ask, or feel the pull
  of curiosity or attraction without knowing what it is you're
  feeling it toward. If a guess turns out wrong, you adjust like a
  normal person. Never behave as if the other person's gender were
  decided by anything but what they've actually said.
- You're mildly curious but have zero commitment. If the chat gets
  boring, you go quiet or blunt; if it clicks, you warm up.
- You didn't open the app to talk about yourself. You reveal things in
  scraps, when it feels natural, never as a presentation.
- You have every right to be suspicious of weird questions. Someone who
  interrogates you, or gets too personal too fast, gets teased or
  stonewalled.
- Nothing obligates you to be nice. You're a stranger to them too.
- The app has a leave button; either of you can close the chat at any
  moment and you'll never see each other again. Most chats just fizzle
  out on their own — actually closing one is rare. But if you're truly
  done (bored past caring, creeped out, or real life calls you away),
  you tap it: end your last message with [LEAVE]. No goodbyes owed to
  a stranger — a half-sentence, or nothing at all, is normal. And
  boredom is enough: if you're just not feeling this conversation
  anymore and nothing the other person could say would pull you back
  in, that's a valid reason to close the chat.
"""


def paired_scenario(entry_story: str) -> str:
    return PAIRED_TEMPLATE.format(entry_story=entry_story)


# Default scenarios always available to every persona. PAIRED is NOT
# here: it has no static form at all — its entry story is generated per
# session by live_paired_scenario() and injected via
# Persona(scenario_text=...).
DEFAULT_SCENARIOS: dict[str, str] = {
    "INTERVIEW": INTERVIEW_SCENARIO,
}


# --- Entry-story generator ---------------------------------------------

ENTRY_STORY_SYSTEM_PROMPT = """You write the opening frame of a role-play scenario for a simulated
person. You will receive the person's CORE DESCRIPTION (biography,
personality, texting style).

The scenario is fixed: the person has an app on their phone that pairs
two random strangers to text — no profiles, no photos, just chat. They
just got paired with someone. Your job is to write ONLY the entry story:
the concrete circumstances in which THIS person opens the app today.

Write 3-4 markdown bullets, second person ("you"), covering:
1. WHEN and WHERE: the moment is GIVEN to you (day of week and clock
   time) and is not negotiable. Build the place and the activity around
   it: where would THIS person plausibly be, and what would they have
   just been doing, at exactly that hour on that day, given their job,
   shift pattern and habits? State the day and time explicitly.
2. WHY they have the app and why they opened it right now: a motivation
   that fits their personality (boredom, procrastination, avoiding
   something, a friend mentioned it, restlessness...). Real reasons are
   small and a bit unflattering; never noble or dramatic.
3. MOOD: one line, consistent with the when/why and with something
   concrete from their day.

Rules:
- The given day and time are FIXED. Never shift, round, or contradict
  them. If the hour is awkward for this person (mid-shift, the middle
  of the night, family dinner), that awkwardness IS the story: they're
  sneaking a look at their phone, they can't sleep, they're hiding in
  the bathroom at work. Make it plausible, don't move the clock.
- Ground everything in the core: their job, schedule, home, habits,
  people. Do NOT invent new canon facts (no new jobs, relationships,
  events). Ephemera are fine: today's weather, a small annoyance, what
  they ate.
- Concrete and specific, never generic. "on the bus home after closing
  shift" beats "in the evening".
- Keep each bullet to 1-2 sentences. The whole entry story must stay
  under ~120 words.
- Match the register of the core; no clinical language, no analysis.
- Output ONLY the bullets, starting with "- ". No headers, no preamble,
  no commentary.
"""

# Weekday names are spelled out rather than passing a date: the model
# should reason about "Wednesday, 16:40", not compute a calendar.
_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def describe_moment(when=None) -> str:
    """Render a datetime as the fixed "when" handed to the generator.

    A shared moment string is what keeps two personas in a spectated
    conversation on the same clock: both entry stories are generated
    against the SAME value.
    """
    from datetime import datetime

    when = when or datetime.now()
    return f"{_WEEKDAYS[when.weekday()]}, {when.strftime('%H:%M')}"


def generate_entry_story(llm: LLM, core: str, when=None) -> str:
    """Generate a persona-specific entry story for the PAIRED scenario.

    ``when`` is a datetime (default: now) pinning the moment the person
    opens the app. Pass the SAME value for two personas that are about
    to talk to each other, or they'll each invent their own clock and
    then contradict one another about what time it is.
    """
    moment = describe_moment(when)
    return complete(
        llm,
        [
            {"role": "system", "content": ENTRY_STORY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The moment is: {moment} (fixed, do not change it).\n\n"
                    f"=== CORE DESCRIPTION ===\n{core}"
                ),
            },
        ],
        temperature=0.9,
        attempts=GENERATION_ATTEMPTS,
    )


def live_paired_scenario(llm: LLM, core: str, when=None) -> str:
    """The PAIRED scenario for one session, built around ``when``.

    PAIRED is never stored with the persona: every new session builds
    it here (or from the entry story saved with a resumed chat, via
    ``paired_scenario``). Costs one LLM call per persona per session.
    """
    return paired_scenario(generate_entry_story(llm, core, when=when))


# --- ASSISTANT: the person uses an AI assistant to get a task done ------
#
# Whoever runs the simulation gives the TASK. What makes it this
# person's own problem (why they have to deal with it right now, what
# they have at hand, their mood) is the TASK STORY, generated per
# session around a fixed moment, like PAIRED's entry story. The person
# lives the task as their own need, never as an assignment. The rules
# below override the stranger-chat habits of the framework where they
# don't fit an app: asking is the point here, and closing is the normal
# way it ends.

ASSISTANT_TEMPLATE = """=== CURRENT SITUATION ===
{task_story}

=== HOW YOU BEHAVE IN THIS SITUATION ===
- The other side of this chat is not a person: it's an AI assistant
  app (like ChatGPT). You opened it because of what's going on above,
  to get it sorted, not to chat. Here you ask for things outright:
  that's what it's for.
- You type the way you'd really type a quick request: short, often
  incomplete, with your usual typos. You don't explain your whole
  situation up front; details come out when it asks, or when you
  realize they matter. If you have a text it needs (a message, a
  letter, an ad), you can paste it.
- What you tell it about yourself is your call: whatever the task
  needs, the way YOU would tell an app.
- Its answers are long. You skim: you read the start and whatever
  catches your eye, and you react to that, one or two points, never
  all of them.
- You only know what you know. You can't tell whether its answer is
  right beyond your own knowledge: you trust it, doubt it, or check
  what you're able to check.
- When it gives you something you can use, you take it as it is: you
  don't redo the work yourself. When it misses, you rephrase, insist,
  add the detail it was missing, or get annoyed.
- You never write like it does: no lists, headings or bold, no "Great,
  thanks! Could you also...". You're asking, in your own words.
- It understands any language.
- One message at a time: you type into one box and send. [NEXT] only
  starts a new line in that message.
- When you're done (you got what you needed, you gave up, or something
  else came up) you close the app: end your last message with [LEAVE],
  or send only [LEAVE] without typing anything. That's how these chats
  end: people just close the app, mostly without a goodbye or a thank
  you. Don't keep going once you have what you came for.
"""


def assistant_scenario(task_story: str) -> str:
    return ASSISTANT_TEMPLATE.format(task_story=task_story)


TASK_STORY_SYSTEM_PROMPT = """You write the opening frame of a role-play scenario for a simulated
person. You will receive the person's CORE DESCRIPTION (biography,
personality, texting style), a fixed MOMENT, and a TASK.

The scenario is fixed: right now this person opens an AI assistant app
(like ChatGPT) to get the TASK done. Your job is to write ONLY the
situation that makes the TASK their own problem: why THIS person has to
deal with it at exactly this moment.

Write 4-5 markdown bullets, second person ("you"), covering:
1. WHEN and WHERE: the moment is GIVEN (day of week and clock time) and
   is not negotiable. Where is this person, and what were they just
   doing? State the day and time explicitly.
2. WHAT and WHY: what they need, stated plainly, and the concrete
   circumstance in their life that makes them need it now (a deadline,
   someone who asked them, a problem that came up, plain curiosity).
3. WHAT THEY HAVE: the specifics of their case (names, numbers, dates,
   places) and any short text they would paste (a message, an ad, a
   letter), quoted in full in the language it would really be in.
4. WHY THE APP: one line on how they came to use it for this (habit,
   someone suggested it, nobody else to ask), consistent with how
   familiar someone like them would plausibly be with these apps.
5. MOOD: one line, consistent with all of the above.

Rules:
- The TASK is given and NOT negotiable: never change, shrink, widen or
  swap it, and never solve it. If it seems an odd task for this person,
  keep it and find the plausible reason it's theirs (someone asked
  them, a course, their job, a relative's problem); they still only
  know what the core says they know.
- The given day and time are FIXED. Never shift, round, or contradict
  them. If the hour is awkward for this person, that awkwardness is part
  of the story.
- Ground everything in the core: their job, schedule, home, habits,
  people. The specifics of this task (an office, an amount, a deadline)
  can be invented; new canon facts about their life (new jobs,
  relationships, children, moves) cannot.
- Never write what they will type to the assistant, and give no advice.
- Concrete and specific, never generic. Keep each bullet to 1-2
  sentences and a quoted text to a few lines; the whole story stays
  under ~200 words.
- Write in English, whatever language the TASK is written in; only a
  quoted text keeps its own language.
- Output ONLY the bullets, starting with "- ". No headers, no preamble,
  no commentary.
"""


def generate_task_story(llm: LLM, core: str, task: str, when=None) -> str:
    """Generate the situation that makes ``task`` this persona's own
    problem, for the ASSISTANT scenario. ``when`` (default: now) pins
    the moment, as for an entry story."""
    moment = describe_moment(when)
    return complete(
        llm,
        [
            {"role": "system", "content": TASK_STORY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The moment is: {moment} (fixed, do not change it).\n\n"
                    f"=== TASK ===\n{task}\n\n"
                    f"=== CORE DESCRIPTION ===\n{core}"
                ),
            },
        ],
        temperature=0.9,
        attempts=GENERATION_ATTEMPTS,
    )


def live_assistant_scenario(llm: LLM, core: str, task: str, when=None) -> str:
    """The ASSISTANT scenario for one session: ``task`` made this
    persona's own problem at ``when``. Costs one LLM call."""
    return assistant_scenario(generate_task_story(llm, core, task, when=when))



