"""The AI assistant a persona uses in the ASSISTANT scenario (see
persona/scenarios.py): a plain model with a minimal system prompt and
its own history, which searches the web when it decides to, where the
server offers web search (see LLM.web_search). No persona, no state or
thoughts, no judge: it only ever sees what the persona typed."""

from ..llm.core import GENERATION_ATTEMPTS, LLM, complete

ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."

# The persona speaks first, into an empty chat box: without a nudge the
# model waits for a message that will never come.
ASSISTANT_OPENING_NUDGE = (
    "(system note: you've just opened the AI assistant app. the chat box "
    "is empty: type your first message.)"
)


class Assistant:
    """One assistant chat: answers one message at a time, keeping the
    conversation in its own history."""

    def __init__(
        self,
        llm: LLM,
        system_prompt: str = ASSISTANT_SYSTEM_PROMPT,
        *,
        web_search: bool = True,
    ):
        self.llm = llm
        self.web_search = web_search
        self._history = [{"role": "system", "content": system_prompt}]

    def __call__(self, message: str) -> str:
        messages = self._history + [{"role": "user", "content": message}]
        answer = complete(
            self.llm, messages, web_search=self.web_search,
            attempts=GENERATION_ATTEMPTS,
        )
        self._history = messages + [{"role": "assistant", "content": answer}]
        return answer
