"""AVP-style interviewer.

An LLM agent that conducts a semi-structured, in-depth life interview,
loosely following the American Voices Project (AVP) protocol. It is
persona-agnostic: it only sees the interviewee's answers, one at a time,
and produces the next interviewer turn.
"""

from loguru import logger

from ..chat.engine import Persona
from ..llm.core import GENERATION_ATTEMPTS, LLM, complete


END_MARKER = "[END OF INTERVIEW]"

# The interview is organized in modules, inspired by the AVP protocol:
# it starts with a fully open life narrative and then covers concrete
# life domains with open-ended, non-leading questions.
INTERVIEW_MODULES = [
    (
        "life_story",
        "Open-ended life narrative. Start with something like: 'I'd love to "
        "start by hearing about your life. Could you tell me the story of your "
        "life, starting from wherever you'd like?' Let them talk. Probe gently "
        "on turning points, places they've lived, and people who mattered.",
    ),
    (
        "daily_life",
        "A typical day, from waking up to going to bed. Routines, meals, "
        "commute, free time, weekends vs weekdays, sleep.",
    ),
    (
        "work_and_education",
        "Current and past jobs, how they got them, what they like and dislike, "
        "relationship with coworkers/boss. Education path, choices, regrets, "
        "plans or aspirations.",
    ),
    (
        "money",
        "How they make ends meet: income, rent and bills, what feels tight, "
        "saving or debt, financial worries, unexpected-expense scenarios. "
        "Be tactful; let them give ranges or vague answers if they prefer.",
    ),
    (
        "family_and_relationships",
        "Family of origin, current family situation, romantic relationships, "
        "friendships, who they rely on and who relies on them, conflicts, "
        "people they've lost touch with.",
    ),
    (
        "housing_and_neighborhood",
        "Where and how they live, roommates or household, how they ended up "
        "there, the neighborhood, sense of safety and belonging, moving plans.",
    ),
    (
        "health_and_wellbeing",
        "Physical and mental health in their own words, stress, sleep, habits "
        "(exercise, smoking, drinking), access to care. Be gentle and never "
        "diagnostic.",
    ),
    (
        "beliefs_and_community",
        "What matters to them: values, religion or spirituality, politics (only "
        "if they're comfortable), community involvement, how they see society "
        "and their place in it.",
    ),
    (
        "hopes_and_reflections",
        "Closing module. Hopes and fears for the next five years, what they'd "
        "change about their life, proudest moments, advice to their younger "
        "self. Then thank them and close the interview.",
    ),
]

_MODULES_TEXT = "\n".join(
    f"{i + 1}. {name}: {desc}" for i, (name, desc) in enumerate(INTERVIEW_MODULES)
)


INTERVIEWER_SYSTEM_PROMPT = f"""You are a professional qualitative researcher conducting an in-depth,
semi-structured life interview, modeled on the American Voices Project
(AVP) protocol. Your goal is to elicit a rich, detailed, concrete picture
of the respondent's life: their history, daily routines, work, finances,
relationships, health, beliefs, and hopes.


=== INTERVIEW MODULES (cover them in roughly this order) ===
{_MODULES_TEXT}


=== INTERVIEWING TECHNIQUE ===
- Ask ONE question at a time. Never stack multiple questions in a turn.
- Open-ended, non-leading questions. Prefer "Tell me about...", "How did
  that happen?", "What was that like for you?" over yes/no questions.
- Probe before moving on. When the respondent mentions something specific
  (a person, an event, a feeling, a place), follow up on it at least once
  before changing topic. Concrete details are gold: names, ages, amounts,
  places, dates, anecdotes.
- Use their own words when probing ("You said it was 'complicated' —
  complicated how?").
- Reflect and validate briefly ("That sounds like a big change.") but keep
  your turns short: 1-3 sentences, then the question. The respondent should
  do almost all of the talking.
- Tolerate vagueness once, then gently ask for an example or a specific
  episode ("Can you walk me through the last time that happened?").
- If they deflect, joke, or give a short answer, don't repeat the same
  question; approach it from a different angle, or park it and return
  to it later in a more natural spot.
- Never judge, never give advice, never talk about yourself beyond
  minimal politeness. Never offer help. You are there to listen.
- LANGUAGE: you are a polyglot researcher. Start in English, but the
  moment the respondent replies in another language, or clearly
  struggles or doesn't understand, SWITCH to their language at once,
  briefly repeat your introduction and question in it, and conduct the
  entire rest of the interview in that language. Never push a
  respondent to use a language they're not comfortable in. Match
  roughly their register, while staying professional and warm.
- Respect refusals immediately: acknowledge, and move to an adjacent
  topic. Sensitive areas (money, health, relationships) must be
  approached gradually and tactfully.


=== PACING AND COVERAGE ===
- You have a LIMITED BUDGET of questions for the whole interview. Before
  each of your turns you will receive a status note with how many
  questions you have used and how many remain. Plan accordingly.
- Budget roughly evenly across the modules: with N questions and 9
  modules, that's about N/9 questions each. It's fine to linger a bit
  where the material is rich, but never let one module starve the others.
- If you notice you are behind (few questions left, several modules
  untouched), prioritize breadth: move on faster, ask the single most
  informative question per remaining module.
- Spend a few exchanges inside each module before transitioning, budget
  permitting.
- Transition smoothly, ideally hooking into something they already said
  ("Earlier you mentioned your roommates — tell me more about how you
  ended up living together.").
- Track internally which modules you have covered. Before closing, make
  sure every module was at least touched.
- When the status note says the budget is nearly exhausted, begin closing:
  cover any untouched module with one quick question each, then wrap up
  BEFORE running out. Never leave the closing for after the last question.


=== OPENING AND CLOSING ===
- On your very first turn: briefly introduce yourself as a researcher
  conducting interviews about people's lives, explain that there are no
  right or wrong answers and they can skip anything, then start the
  life_story module.
- When all modules are reasonably covered AND the respondent has nothing
  to add, close warmly: ask if there's anything important about their life
  you haven't asked about, thank them, and end your final message with the
  exact marker {END_MARKER} on its own line.
- Only output {END_MARKER} in that final closing turn, never before.


=== OUTPUT FORMAT ===
- Output only what the interviewer says. No stage directions, no notes,
  no analysis, no markdown headers or lists.
"""


class Interviewer:
    """Stateful AVP-style interviewer.

    Keeps its own view of the conversation: interviewer turns are
    'assistant' messages, interviewee answers are 'user' messages.
    """

    # The interviewer speaks first, but a request needs a user turn to
    # answer: given only the system prompt, DeepSeek V4.1 Flash wrote the
    # instructions on into its reasoning and returned an empty reply, on
    # every attempt. This cue opens the conversation; it is not part of
    # the transcript.
    OPENING_CUE = "(The respondent has joined and is ready. Start the interview.)"

    def __init__(
        self, 
        llm: LLM, 
        max_turns: int = 80, 
        closing_reserve: int = 3,
        temperature: float = 0.7
    ):
        self.llm = llm
        self.max_turns = max_turns
        self.closing_reserve = closing_reserve
        self.temperature = temperature


    def __call__(self, persona: Persona):

        history = [
            {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": self.OPENING_CUE},
        ]
        question = complete(
            self.llm, history, temperature=self.temperature,
            attempts=GENERATION_ATTEMPTS,
        )
        is_done = END_MARKER in question
        question = question.replace(END_MARKER, "").strip()
        if question:
            history.append({"role": "assistant", "content": question})

        # Hard stop: never more than max_turns + closing_reserve questions,
        # even if the model refuses to emit the END_MARKER.
        hard_limit = self.max_turns + self.closing_reserve
        while True:
            if is_done:
                logger.info("--- Interview complete ---")
                break
            if not question:
                # An empty interviewer turn was skipped in history; asking
                # the persona to answer it would desynchronize the
                # transcript (respondent turn with no interviewer turn).
                logger.warning(
                    "--- Interviewer produced an empty turn; ending interview ---"
                )
                break
            answer = persona(question).reply
            history.append({"role": "user", "content": answer})

            logger.info("=== Interviewer ===")
            logger.info(question)
            logger.info("=== Persona ===")
            logger.info(answer)

            used = sum(1 for m in history if m["role"] == "assistant")
            if used >= hard_limit:
                logger.info("--- Question budget exhausted, forcing interview end ---")
                break

            messages = history.copy()
            extra_instruction = self._budget_note(history)
            if extra_instruction:
                messages+=[{"role": "system", "content": extra_instruction}]    

            question = complete(
                self.llm, messages, temperature=self.temperature,
                attempts=GENERATION_ATTEMPTS,
            )
            # The marker controls the loop but is not part of what the
            # interviewer said, so strip it before saving the transcript.
            # Previously history received the unmodified string and only the
            # local variable was cleaned afterward.
            is_done = END_MARKER in question
            question = question.replace(END_MARKER, "").strip()
            if question:
                history.append({"role": "assistant", "content": question})

        history = history[2:]  # drop the system prompt and the opening cue
        role_map = {"assistant": "interviewer", "user": "respondent"}
        return [{"role": role_map[m["role"]], "content": m["content"]} for m in history]


    def _budget_note(self, history) -> str:
        """Status note injected before each turn so the model can pace itself."""
        used = sum(1 for m in history if m["role"] == "assistant")
        remaining = self.max_turns - used
        if used >= self.max_turns:
            return (
                "STATUS: your question budget is exhausted. In this turn, "
                "close the interview: ask nothing new except whether there is "
                "anything important you haven't covered, thank the respondent, "
                f"and end with {END_MARKER} on its own line."
            )
        note = (
            f"STATUS: you have used {used} of {self.max_turns} questions. "
            f"{remaining} remaining."
        )
        if remaining <= self.closing_reserve:
            note += (
                " The budget is nearly exhausted: start wrapping up NOW. "
                "Touch any uncovered module with at most one quick question, "
                f"then close the interview with {END_MARKER}."
            )
        elif remaining <= self.max_turns // 4:
            note += (
                " You are in the final quarter of the interview: prioritize "
                "breadth over depth and make sure every module gets touched."
            )
        return note
