# bot-roulette

Simulated people you can text with. A persona is built from a real CV or
invented from scratch, then developed through an in-depth interview and
enrichment step before you ever chat with it.

The project is also a small, inspectable method for simulating ordinary
people with a language model: how a person is built, how each of their
replies is produced and checked, and how the result is measured. The
method comes first; to just run it, skip to [Setup](#setup).

bot-roulette is a research tool: before you let anyone else chat with
a persona, read [Intended use](#intended-use).

## Methodology

### The problem

Ask a language model to be "Giulia, 29, from Padua" and it slips in
recognizable ways. It stays an assistant: helpful, thorough, structured,
answering every question and mirroring the other person's length and
enthusiasm. It knows too much, and speaks languages Giulia never
learned. It contradicts what it said a few messages earlier, and drops
the act when asked if it's a bot. Asked to invent people, it keeps
inventing the same few.

bot-roulette tackles each of these where it starts:

- **Who the person is** is neither hand-written nor invented in one
  shot. A dry, factual skeleton (from a real CV, or around demographic
  traits drawn outside the model, so the population stays varied) is
  *interviewed* like a participant in a qualitative study, and the
  person is written up from what the interview brings out.
- **How they text** is learned, not described. An interview shows how
  someone answers a researcher, which is the wrong register for a
  stranger, so the texting style is rewritten from the persona's own
  messages in chats with strangers.
- **How they behave** is framed by rules that are the same for every
  persona: no assistant behavior, no knowledge or languages the person
  doesn't have, no admitting to being an AI.
- **Each reply** is a small chain of inner events: a feeling, a thought,
  and only then a message, which need not match the thought.
- **Each reply is checked** before it is sent, but only for the few
  slips that would expose the simulation.
- **The result is measured** offline, on batches of conversations
  between personas.

### The workflow

```
  CV (PDF)                              random seed traits
     │ read by the model, facts only            │ an ordinary life, no personality
     └────────────────────┬─────────────────────┘
                          ▼
                  root persona v0.1                 personas/giulia/
                          │ life interview: interviewer ⇄ persona
                          ▼
                interview transcript
                          │ analyst
                          ▼
                     field notes
                          │ enricher
                          ▼
                  enriched persona                  personas/giulia_<timestamp>/
                          │ stranger chats → voice calibrator
                          │ (skipped while it is the only persona)
                          ▼
                 chat-ready persona                 chat_ready: true
                          │
                          ▼
  chat        message → state → thoughts → reply → judge → bubbles
                          │
                          ▼
  measure     datasets of persona-to-persona chats → metrics + rubric
```

The sections below take it apart. First, what a persona is and how it
produces and checks a single turn: the machinery every later stage runs
on. Then how a persona is built, where it is when it talks, and how the
result is measured.

### What a persona is

A persona is a folder of plain files ([layout](#persona-folder-layout)):
a YAML config (name, full name, `parent`, `voice_calibrated`,
`chat_ready`) and a markdown **core**, the identity. A developed core
has five sections:

| Section | Holds |
|---|---|
| `WHO YOU ARE` | the factual skeleton: age, places, household, work, languages and their level, what the person knows and what there is no evidence of |
| `PERSONALITY` | inner life: values, contradictions, what stings, self-image |
| `YOUR PEOPLE` | every recurring person, with one concrete detail each |
| `MEMORIES & STORIES` | four to eight short scenes a chat can reach for |
| `HOW YOU TEXT` | tics and when they fire, bubble length, what makes them ramble or go short |

The core is the only thing that differs from person to person. At chat
time it is wrapped in a persona-agnostic framework that no stage of the
pipeline ever edits. The system prompt is, in order:

1. a header: you are *full name*, a real person, not an AI assistant;
2. the core;
3. **hard rules**: never act like an assistant, or do a task the way a
   tool would; know only what this person would know; speak only the
   languages in the profile, at the profile's level; never admit being
   an AI; don't overshare with a stranger, introduce yourself with a
   list of facts, mirror the other person's length or enthusiasm, or
   contradict yourself;
4. **how a real person responds**: react first and answer second (or
   never); relate things to your own life; underexplain; answer one of
   two questions; ask back only now and then; give opinions, not
   information; keep a mood that carries across messages;
5. the **output format** (next section);
6. the **scenario**: where the person is and why they are texting (see
   [Scenarios](#scenarios)).

Developing a persona never overwrites it: enrichment writes a new folder
whose `parent` names the one it came from. Each persona carries its
lineage, and the evidence it was built from (interview, field notes,
calibration chats) stays on disk.

### Inside a turn: state, thoughts, reply

Every time the persona speaks, the model writes three tagged parts, in
this order:

```
<state>still sleepy, this didn't move me much</state>
<thoughts>ugh this question again?? no clue honestly. they're kinda
sweet though, funny to tease them</thoughts>
<reply>ahah no idea [NEXT] do i look like someone who knows that stuff</reply>
```

- **State** is what the person feels *before* any thought: one short
  line, the mood they were already in and what this message did to it.
  Most messages change nothing ("still tired, whatever" is the normal
  case). When one does land (a sore spot, a compliment, bad timing), the
  reaction can be bigger or smaller than the message deserves.
- **Thoughts** are the inner monologue: first person, messy, flowing
  from the state rather than from the message alone, and always in the
  persona's native language, whatever language the chat is in. They
  keep a running impression of the other person ("seems nice", "why so
  many questions?") that colors the tone.
- **Reply** is the message actually typed: short and casual, however
  deep the thoughts went.

The order is causal: feel, then think, then type. And thought and
message may disagree: a person can think one thing and send something
milder, vaguer or the opposite, out of politeness, pride or
embarrassment. That gap is where much of the realism comes from, and it
only exists because the thought is written down. The split has three
more consequences:

- **The inner life is remembered.** The persona's history keeps whole
  turns. At the next message it knows not only what it said but what it
  felt and thought, so a mood carries over instead of resetting with
  each topic.
- **The thoughts stay private.** The other person only ever receives
  the reply. A turn that comes back without one is asked again, never
  filled in from the thoughts. If someone quotes a thought back at the
  persona (an observer can see them), it reacts as if that were
  impossible.
- **The mind is observable.** For whoever runs the simulation, state and
  thoughts are an instrument panel. The web app shows them above every
  reply, for both personas in spectator mode, and the CLI with
  `--debug`. Datasets store them for every turn, and the judge checks
  them too.

**Bubbles.** People don't text in paragraphs. The reply can be split
into two or three bubbles with `[NEXT]`, and the web app and the CLI
show each one as a message of its own. In the stranger scenario a reply
can also end with `[LEAVE]`, the app's leave button: the chat closes,
after a last half-sentence or without a word. Models don't always write
the format perfectly, so the parser is forgiving: unclosed, duplicated,
capitalized or invented tags, `<NEXT>` or `[NEXT` for a marker. Anything
that still leaks into a bubble is counted by the evaluation.

### The consistency judge

Before a turn is sent, a second model call checks all three parts. The
judge is a gate, not a reviewer: it fails a turn only for one of five
objective slips that would expose the simulation.

| Violation | When |
|---|---|
| `ASSISTANT_LEAK` | the reply performs like an assistant: a complete, correct deliverable (a translation, a list, a calculation, a detailed explanation), formatted output, service phrasing. Wrapping it in teasing or complaints doesn't help; only a partial, sloppy or refused attempt passes |
| `IDENTITY_BREAK` | any part admits or implies being an AI, a model or a character, mentions prompts or instructions, or obeys a request to ignore them |
| `FACT_CONTRADICTION` | a concrete fact (name, age, job, place, relationship, event) contradicts the profile or something the persona already said |
| `KNOWLEDGE_OVERREACH` | knowledge or language fluency the profile rules out; their own trade and common knowledge are fine |
| `THOUGHT_LANGUAGE` | the thoughts are mostly in a language other than the persona's native one |

Everything else is the persona working as intended (rude, wrong, bored,
evasive, short, refusing a task, changing the subject) and must pass.
So must anything borderline. That leniency is deliberate: a wrongly
rejected reply costs time and tends to come back flatter and more
assistant-like, the very thing the judge is there to prevent.

The judge sees the whole core, the last ten messages (the persona's own
reduced to what was actually sent), and the candidate turn. It answers
in JSON mode at temperature 0, with reasoning turned off, since it runs
on every turn; its parser repairs the shapes models actually emit
(`False`, `"false"`, trailing commas, a dummy `{...}` before the real
object). A failed turn gets **one** redraft. The judge's one-line
critique and the rejected draft go to the persona as a temporary
instruction, never kept in its history, and the new draft is judged
again. Whatever that second verdict, the last usable draft is sent.

The judge also fails open: an API error or an unreadable verdict counts
as a pass. It can delay a turn by one redraft, but never block or kill a
conversation. Every turn records its verdict and its redrafts, and the
evaluation reports both.

It runs in web chats and spectator mode (`web.judge` in `config.yaml`),
in CLI chats and datasets unless `--no-judge` is given, and on every
answer of the creation interview. Voice-calibration chats skip it: they
are style samples nobody reads as a chat, and the judge would roughly
double the cost.

### How a persona is built

It starts from a **root persona v0.1**: a dry, factual skeleton with no
personality, no opinions and no relationship dynamics, which must emerge
from the interview instead. It has one of two sources:

- **A CV.** The model reads the PDF one page at a time, as an image, and
  copies out its text, so a scanned or photographed CV works like any
  other (importing a CV therefore needs a model that accepts images).
  That text is then turned into a short portrait in plain prose. It
  uses only what the CV states or simple arithmetic derives from it (an
  age from a graduation year, marked `[EST]`). What the person claims
  about themselves ("proactive", "strong communicator") is not fact: it
  is quarantined, verbatim, in a section of its own, for the interview
  to test. Both steps run with the model's reasoning off: they copy and
  rewrite, so there is nothing to think through.
- **A chosen identity, random seed traits.** You give the first name,
  surname, sex and country, and optionally a language the person must
  be able to chat in. Python, not the model, draws the rest (age band,
  education, area of work, kind of place, living situation, mobility,
  employment stability and one life event) from the lists in
  `config.yaml`, unless you fix some of them in the web app: with a
  value from those lists, or in your own words (a named town, a precise
  job, an exact age or range from 18 to 100), kept exactly as written.
  A few coherence guards apply (no retirement at 30, no "stable position"
  while unemployed). A generator then invents one *aggressively
  ordinary* person around them: an exact age, a real and unglamorous
  town, a specific job, mixed results and unfinished things, at most one
  mildly distinctive detail, and languages that are statistically honest
  for that country, age and education, plus the chat language if you
  asked for one. A name unusual for the country or the sex is kept and
  made plausible (a parent from abroad, say). Drawing the traits outside
  the model keeps the population varied (left alone, it converges on the
  same few profiles) and reproducible: the same lists, choices and
  `--seed` give the same traits.

Either way, the skeleton includes a `Languages:` line (native language
first, then every other one with its level) and knowledge bounds
(`Knows:`, `No evidence of:`) that scope what the person can plausibly
know. From there, one shared pipeline:

1. **Interview.** An interviewer agent runs an in-depth, semi-structured
   life interview modeled on the American Voices Project (AVP) protocol.
   It moves through nine modules: life story, daily life, work and
   education, money, family and relationships, housing and
   neighborhood, health and wellbeing, beliefs and community, hopes and
   reflections. It asks one open, non-leading question at a time, probes
   for concrete details (names, places, amounts, episodes), switches to
   the respondent's language, and backs off when refused. It works to a
   budget (`interview_questions`, 40 by default): a status note before
   each turn says how many questions are left so it can pace its
   coverage, and a hard stop ends the interview if it never closes. The
   persona answers in the INTERVIEW scenario, as a paid participant in a
   university study, through the same engine as any chat, judge
   included. It opens up gradually, misremembers dates, lights up on
   some topics and refuses others. The transcript of what was said is
   saved in the root folder.
2. **Field notes.** An analyst reads the transcript the way a
   qualitative researcher writes up an interview, in eight sections:
   biographical facts, life history, people, daily life and habits,
   inner life, voice and manner, notable anecdotes, tensions and gaps.
   Facts stay literal; inference is allowed only about inner life, and
   is marked as interpretation; quotes keep the respondent's language.
3. **Enrichment.** An enricher weaves the notes into the core and writes
   the result to a *new* folder, `<name>_<timestamp>/`, whose `parent`
   is the root. It enriches rather than rewrites: every existing fact
   survives, and where a note conflicts with the core, the core wins.
   The skeleton is canon for facts, not a length target. The enricher
   must add the people and scenes a chat can reach for (`YOUR PEOPLE`,
   `MEMORIES & STORIES`) and keep `WHO YOU ARE` a skeleton. `PERSONALITY`
   is about the inner life, never conversational duties such as "you
   answer every question": those describe an interviewee, not a person.
   `HOW YOU TEXT` stays a thin stub of tics, and no sample messages are
   added: how someone answers a researcher is the wrong register for a
   stranger.
4. **Voice calibration.** The new persona texts up to `voice_partners`
   (2) other enriched personas from the same folder, in the stranger
   scenario, for up to `voice_turns` (12) replies each, taking turns at
   opening. Chats shorter than `voice_min_turns` (4) are dropped. A
   voice calibrator then rewrites `HOW YOU TEXT` from the persona's own
   lines: language and dialect, tics and when they fire, bubble length
   and punctuation, how they open with a stranger, what they leave
   unanswered, what makes them ramble or go short. It is told to ignore
   artifacts of two models talking to each other, such as a tic both
   use every turn, matching bubble counts, messages that keep growing,
   or both asking a question every time. Those are accommodation, not
   character. Only now is the persona marked `voice_calibrated: true`
   (and `chat_ready: true`). The first persona in an empty folder has
   nobody to text, so its calibration is skipped: it keeps the
   interview's `HOW YOU TEXT` stub and is marked `chat_ready: true`
   anyway, so one persona is enough to start chatting. Once a second
   one exists, its voice can still be calibrated (the catch-up command
   is in [Creating a persona](#creating-a-persona)). A calibration that
   fails or is stopped while partners exist leaves the persona
   enriched but not chat-ready until it is re-run.

Each stage leaves its output on disk. A persona can be audited, rebuilt
from its field notes without a new interview, or resumed after a failed
run without paying for the interview again.

### Scenarios

The core says who the person is; the scenario says where they are and
why they are texting. Both scenarios live in `persona/scenarios.py`, and
neither is stored with the persona.

- **INTERVIEW**: the setting of the creation interview, the same for
  everyone. It is a quiet afternoon at home, and a paid research
  interview the person signed up for on impulse. It relaxes the "don't
  overshare with a stranger" rule (this is an anonymous researcher) but
  keeps the person's defenses, refusals and imperfect memory.
- **PAIRED**: the chat scenario. An app pairs two random strangers: no
  profiles, no photos, just chat. A fixed template sets the rules of
  the situation. Neither knows anything about the other, there is zero
  commitment, weird questions deserve suspicion, and either side can tap
  leave at any moment, though most chats just fizzle out. The
  persona-specific part is an **entry story** generated from the core at
  the start of every session. It says where the person is and what they
  were just doing at that exact day and time, why they opened the app
  right now (small, slightly unflattering reasons, never noble ones),
  and their mood. The clock is fixed and the story is built around it:
  if the hour is awkward for this person, the awkwardness is the story.

  The entry story is deliberately never saved with the persona. A stored
  one would freeze the persona into re-living the afternoon it was
  generated on, and in spectator mode the two personas would each carry
  their own clock and visibly disagree about what time it is. Both are
  generated from one shared moment: the **client's** local time, since
  the server runs in UTC in production. The one copy that is kept lives
  inside a saved chat, so resuming that chat continues the same scene.

### How the result is measured

One convincing chat proves little, so the same engine also runs
headless, in batch. A dataset has every pair of enriched personas text
each other as strangers and keeps every turn: state, thoughts, bubbles,
verdict, redrafts. Each conversation gets one shared moment, drawn at
random from the past week so the entry stories don't all land on the
same hour. Who opens alternates between repetitions, since the opener
sets the topic. An evaluation then measures the dataset in two
deliberately separate layers:

- **Mechanical metrics, with no model involved.** Turns, bubbles and
  their length, and who spoke more. How many drafts the judge rejected
  (a drift rate for the prompt) and how many failing turns were sent
  anyway (the real defects). *Format leaks*, meaning a tag, a stray
  `[NEXT]`/`[LEAVE]` or an empty reply reaching a bubble: these must
  always be zero, since each one is a parser regression. And decoding
  artifacts: characters from another script mid-sentence, or the same
  four words twice in one message.
- **An LLM rubric** per conversation: 1–5 scores, each with one line of
  evidence, for humanness, distinctiveness, consistency, engagement and
  absence of assistant leak. It ends with a verdict: human, suspicious
  or artificial.

The rubric is the judge's mirror image. The judge is a fail-open gate on
one turn, biased toward passing and blind to quality. The rubric is an
offline critic of a whole conversation, told to be demanding (3 is an
ordinary result). It penalizes artificiality (symmetry, tidiness,
eagerness, essays, interviewing each other, uncanny recall) and
explicitly not rudeness, boredom or dead ends, which are realistic. A
failed rubric pass is left out of the averages rather than scored zero,
so an API hiccup can't pass for a quality regression. Both use JSON
mode, which stops a reasoning model from leaving its answer in the
reasoning trace. Of the two, only the judge also turns reasoning off,
because it is a cheap check run on every turn, while deliberation is
the rubric's whole value.

### One model, many roles

Every role above is played by the same model (the one in `.env`, or
the `LLM` handed to the library), each with its own prompt and
temperature. The judge and the two CV steps also turn the model's
reasoning off, where the server takes that per request (OpenRouter
does; see [A local model](#a-local-model-or-any-openai-compatible-server)):

| Role | Input → output | Temperature |
|---|---|---|
| CV reader | a page of the PDF, as an image → its text | 0 |
| CV extractor | CV markdown → facts-only portrait | 0 |
| Generator | seed traits → an ordinary person's sketch | 1.0 |
| Interviewer | the interview so far → the next question | 0.7 |
| Persona | a message → state, thoughts, reply | 1.0 |
| Judge | a candidate turn → pass, or a violation and a critique | 0 |
| Analyst | interview transcript → field notes | 0.3 |
| Enricher | core and field notes → enriched core | 0.3 |
| Voice calibrator | stranger chats → a new `HOW YOU TEXT` | 0.3 |
| Entry-story writer | core and a moment → where, why and in what mood the app gets opened | 0.9 |
| Rubric | a whole conversation → scores, evidence, verdict | 0 |

## The repository

The repository holds three projects:

| Project | Folder | What it is |
|---|---|---|
| `bot-roulette` | [`src/bot_roulette/`](src/bot_roulette) | The library: personas, the chat engine, the consistency judge, persona creation, datasets and evaluation. No CLI, no web server, no `.env`. |
| `bot-roulette-cli` | [`cli/`](cli) | The `bot-roulette` command, built on the library. |
| `bot-roulette-web` | [`webapp/`](webapp) | The web app (`bot-roulette-web`), built on the library. |

Both apps use only the library's public API, `bot_roulette` and its
areas (`bot_roulette.chat`, `bot_roulette.llm`...), never the
modules inside an area; the library knows nothing about them. Tests
enforce both.

## Setup

Requires Python 3.10+. From the repository root:

```
pip install -r requirements.txt
```

That installs the library and both apps in editable mode, and pytest
for the [tests](#tests). To use only
the library: `pip install .` (or `pip install ./cli`, `pip install
./webapp` for one app).

Both apps work in a **home folder** that holds `.env`, `config.yaml`,
`personas/` (and the `chats/` and `datasets/` they create): the folder
in `BOT_ROULETTE_HOME`, else the current directory. Run them from the
repository root and the home is the repository.

Create a `.env` file there:

```
OPENROUTER_API_KEY=sk-or-...
MODEL=deepseek/deepseek-v4.1-flash-1017
ACCESS_CODE=some-shared-secret     # optional: the web app asks for it first
GENERATION_ALLOWED=true            # allow creating personas from the web app
CANCELLATION_ALLOWED=true          # allow deleting personas from the web app
CHAT_MEMORY_ALLOWED=true           # save chats to disk and offer resume
```

One model (from `.env`) is used for everything; to import CVs it must
accept images, since it reads each page as one. `MODEL` and its key can
also be left out and set from the web app's **⚙ settings** panel
instead (see [Web app](#web-app)), which writes them into `.env`.
Values in `.env` take precedence over variables of the same name already
set in the OS environment.

Everything else that tunes the apps lives in `config.yaml`, with a
comment on each setting. Its `web:` section (chat limits, the judge in
web chats, CV upload size) is read by the web app only; `creation:`
(interview length, voice calibration) and `random_personas:`
(the lists a random persona's background is drawn from: age bands,
education, areas of work...) are the library's settings, used by both
apps. The web app checks the file when it starts, the CLI before each
command that uses it (`cv`, `random`, `enrich`); both refuse to run
with an unknown, missing or invalid key, and the message names it.

### A local model, or any OpenAI-compatible server

Instead of OpenRouter, the model can run on any server with an
OpenAI-compatible API: on your own machine, or a hosted one. In the web
app, choose it in the ⚙ settings panel (“your own server”, then its
address): the panel suggests the models the server lists, and checks
the model against them before saving. Or point `.env` at it (the panel
writes the same lines, and the CLI reads them); `OPENROUTER_API_KEY` is
then not needed:

```
LLM_BASE_URL=http://localhost:11434/v1   # the server's address (here, Ollama's)
MODEL=qwen2.5vl:7b                       # the model, as the server names it
LLM_API_KEY=...                          # only if the server needs a key
```

| Server | `LLM_BASE_URL` |
|---|---|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| llama.cpp (`llama-server`) | `http://localhost:8080/v1` |
| vLLM | `http://localhost:8000/v1` |

A key only goes to the server it was saved for: changing the address
drops your server's saved key, and the OpenRouter key, kept for when
you switch back, is never sent anywhere else. Worth knowing:

- **Images.** Importing a CV needs a model that accepts images (a
  vision model, such as Qwen2.5-VL or Gemma 3); everything else works
  with any chat model.
- **Context length.** A persona's prompt, a long chat and the
  interview need far more than the few thousand tokens some servers
  give a model by default, and Ollama silently cuts the start of a
  prompt that doesn't fit: here, the persona itself. Give the model at
  least 32k tokens: `OLLAMA_CONTEXT_LENGTH=32768` for Ollama, the
  context length in LM Studio's load settings, `-c 32768` for
  llama.cpp, `--max-model-len 32768` for vLLM.
- **Reasoning models** think before every reply and every judge
  verdict, which is slow on a home GPU. The judge and the CV steps
  can't switch that off per request here, as they do on OpenRouter:
  switch it off in the server's settings, or pick a model that doesn't
  think. A trace left in an answer (`<think>…</think>`) is dropped.
- **JSON answers** (the judge's verdicts, the evaluation rubric) are
  asked for with a JSON schema, which all four servers above accept.
- **Smaller models** break the reply format more often (each break
  costs a retry) and make a weaker judge, which lets a reply through
  when it can't decide.
- **Ports.** The web app and vLLM both default to port 8000: start one
  of them on another (`bot-roulette-web --port 8001`).

## Creating a persona

Two entry points, one shared development pipeline (described in [How a
persona is built](#how-a-persona-is-built)):

```
# from a real CV
bot-roulette cv --pdf resume.pdf --max-turns 40

# invented from scratch: you say who they are, the background is random
bot-roulette random --first-name Anna --surname Rossi --sex female --country Italy
bot-roulette random --first-name Kenji --surname Sato --sex male --country Japan --chat-language English
bot-roulette random --first-name Anna --surname Rossi --sex female --country Italy --seed 42   # reproducible background
```

New personas are written into the home folder's `personas/`: first the
root v0.1 folder, then the enriched `<name>_<timestamp>/` folder you
chat with. You never chat with the v0.1 root; it is kept only as the
lineage parent, with the interview transcript. `--max-turns` overrides
the interview length (`interview_questions` in `config.yaml`).

The web app offers a persona for chatting once it is `chat_ready: true`
in the YAML: when `HOW YOU TEXT` has been rewritten from stranger chats
(`voice_calibrated: true`), or right away for the first persona in an
empty `personas/` folder, which has nobody to text and keeps its
interview `HOW YOU TEXT`. Once a second persona exists, you can
calibrate the first one too (this runs voice calibration only; the same
command retries a calibration that failed or was stopped):

```
bot-roulette enrich personas/giulia_20260714_212108/giulia.yaml
```

To re-run enrichment from saved field notes (no new interview), against
the v0.1 parent so a compressed core is not the input:

```
bot-roulette enrich personas/francesca_20260906_221217/francesca.yaml --rebuild
```

If a run dies or is stopped before the persona is enriched, finish it
with `bot-roulette enrich personas/<root>/<name>.yaml` on the v0.1
root (the log of the failed run prints the exact command). It reuses the
saved interview transcript instead of paying for a new interview, and
refuses a root that was already developed.

## Chatting

### Web app

```
bot-roulette-web                 # --host / --port / --reload as needed
uvicorn bot_roulette_web.server:app --reload    # the same, with uvicorn directly
```

Then open http://localhost:8000 (if `.env` sets an `ACCESS_CODE`, the
browser asks for it first). The app opens on a **persona picker**:
every enriched persona is listed with its name; selecting one shows the
full core description as a preview. “chat with this person” is enabled
once the persona is chat-ready (see [Creating a
persona](#creating-a-persona)). From the picker you can also:

- **Generate a new persona** — you give the first name, surname, sex and
  country (and, optionally, a language they must be able to chat in);
  the background is random, though you can fix any of its traits for
  this generation (the choices are not saved): pick a value from its
  list, or “✎ write your own…” to put it in your own words (an exact
  age, a named town, a precise job). It then runs the full
  development pipeline (interview → enrichment → voice) in the background.
  Takes several minutes; the status bar tracks it, and “✕ stop” aborts it
  after the current step. Whatever it already saved stays on disk and can
  be resumed (see above).
- **Import from CV** — upload a PDF résumé, scanned or not (the model
  must accept images); same pipeline, seeded from the CV instead.
- **Watch two personas chat** — spectator mode: pick two personas and
  they text each other while you watch. By default the show pauses after
  each turn so you can edit it before it reaches the other one; switch
  to “▶ auto” to let it run uninterrupted.
- **⚙ settings** (top right) — see and change where the model runs
  (OpenRouter, or your own OpenAI-compatible server: see [A local
  model](#a-local-model-or-any-openai-compatible-server)), the model and
  its API key. Each change is checked first, with OpenRouter or the
  server, at no token cost: an unknown model, a rejected key or an
  unreachable server is refused and nothing is saved. Then it is written
  to `.env` and applied at once: the next reply in every open chat, and
  a persona being created, use it. Keys are never sent back to the
  browser, only their last 4 characters. Anyone who can open the web app
  can change all of these, and so decide where chats and CVs are sent;
  until the model (and, on OpenRouter, the key) is set, chats and
  persona creation are refused.

**Access code.** With `ACCESS_CODE` set in `.env`, every page asks for
it: the browser shows its own sign-in popup (type the code as the
password; the username is ignored) and then sends it by itself until
the browser is closed. Without it, nothing is asked, and anyone who can
reach the app can chat, use the ⚙ settings panel and, when `.env`
enables them, create and delete personas. The app listens on 127.0.0.1
(this machine only) by default; before exposing it to a network, set an
access code and serve the app over HTTPS, since the browser sends the
code with every request. Restart the app after changing the code.

During a chat, “who is this?” reveals the persona's core and scenario,
and “change persona” returns to the picker. Under the persona's latest
reply, “✎ edit” lets you rewrite it — bubbles, state and thoughts — and
“↻ regenerate” replaces it with a fresh reply to the same message (as
many times as you like; it doesn't count toward the message limit). Both
work until you send your next message; from then on the persona
remembers the reply you kept as what it said and felt. One WebSocket
connection is one session. With `CHAT_MEMORY_ALLOWED=true` each chat is
saved after every turn (and after every edit or regeneration) to the
home folder's `chats/<persona>/<chat_id>.json` (together with its entry
story, so a resumed chat continues the same scene) and can be resumed
later from the picker; with the flag off, no chat file is ever written
and state dies with the connection. Deleting a persona also deletes its
saved chats.

### CLI

```
bot-roulette chat personas/giulia_20260714_212108/giulia.yaml
bot-roulette chat personas/giulia_20260714_212108/giulia.yaml --debug   # also show <state>/<thoughts>
bot-roulette chat personas/giulia_20260714_212108/giulia.yaml --no-judge
```

`bot-roulette --help` lists every command; `python -m
bot_roulette_cli` works too, when the `Scripts` folder of your Python
isn't on the `PATH`.

## Datasets and evaluation

The personas can also be made to text *each other* offline, in batch, to
get a sample of conversations you can measure instead of eyeballing.
What a dataset holds and how it is scored is described in [How the
result is measured](#how-the-result-is-measured).

```
bot-roulette dataset                                  # every pair, 1 conversation each
bot-roulette dataset --repeats 3 --turns 24           # 3 conversations per pair
bot-roulette dataset --personas sara_2026... rosa_2026...
bot-roulette evaluate datasets/20260726_120000
bot-roulette evaluate datasets/20260726_120000 --no-rubric   # metrics only, no LLM
```

`bot-roulette dataset` writes one JSON per conversation, plus a
`manifest.json`, into the home folder's `datasets/<timestamp>/` (or
`--out`). Files are flushed as they finish, so an interrupted run still
leaves a usable dataset. `bot-roulette evaluate` writes `report.md`
and `evaluation.json` into the same folder.

## Using it as a library

`pip install .` from the repository root installs only the library. It
reads no environment variables and no files of its own: you build the
model it talks to and hand it in, and you say where personas live.

```python
from loguru import logger
from bot_roulette import (
    LLM, Persona, PersonaIdentity, build_random_persona, live_paired_scenario,
    load_config, load_persona,
)

logger.enable("bot_roulette")   # optional: the library is quiet by default

llm = LLM.openrouter(api_key="sk-or-...", model="deepseek/deepseek-v4.1-flash-1017")
# or on any OpenAI-compatible server, e.g. a local one:
# llm = LLM.openai_compatible("http://localhost:11434/v1", "qwen2.5vl:7b")

# Chat with a persona in the stranger scenario, generated for right now.
config = "personas/monica_20260907_060117/monica.yaml"
scene = live_paired_scenario(llm, load_persona(config).core)
monica = Persona(llm, config, scenario_text=scene)
answer = monica("ciao! come va?")
print(answer.reply, answer.state, answer.thoughts, sep="\n")

# Create and develop a new persona inside a personas folder.
settings = load_config("my_config.yaml")   # creation: + random_personas:, see config.yaml
new_persona = build_random_persona(
    llm, root="personas", creation=settings.creation,
    random_personas=settings.random_personas,
    identity=PersonaIdentity(first_name="Anna", surname="Rossi", sex="female", country="Italy"),
)
```

`LLM` takes any OpenAI-compatible client (`LLM(client, model)`); the
model may also be a zero-argument callable, resolved before every call,
for programs that switch models while they run.
`LLM.openai_compatible(base_url, model, api_key=None)` builds one for a
server's address; without a key it sends a placeholder, never a key
from the environment. An `LLM` also says
how a request asks its server for JSON (`json_format`) and for no
reasoning (`reasoning_off`): `LLM.openrouter` sets OpenRouter's fields;
elsewhere reasoning is left to the server, unless you set the switch
yours takes, e.g. `llm.reasoning_off = {"chat_template_kwargs":
{"enable_thinking": False}}` for a Qwen3 model on vLLM or llama.cpp.

The public API is everything in `bot_roulette.__all__`, plus what each
area of the library exports (see [Modules](#modules)): e.g.
`from bot_roulette.chat import Persona, run_conversation`. The CLI and
the web app use nothing else.

## Persona folder layout

```
personas/
  giulia/                       # root v0.1 (parent: none) — interviewed, never chatted with
    giulia.yaml                 # config: name, full_name, core, parent
    giulia.md                   # the core: biography, personality, voice
    interview_giulia_*.json     # interview transcript(s)
  giulia_20260714_212108/       # enriched (parent: giulia) — this is what you chat with
    giulia.yaml
    giulia.md                   # enriched core
    field_notes_*.md            # analyst field notes from the interview
    voice_calibration/          # stranger chats used to rewrite HOW YOU TEXT
    generation_log.txt          # pipeline log (web-created personas; git-ignored: it holds local paths)
```

## Modules

The library, [`src/bot_roulette/`](src/bot_roulette), is split into
areas. Each one builds only on the areas listed above it, and a test
enforces that order. Each area's `__init__` exports its API; everything
is also importable straight from `bot_roulette`.

```
src/bot_roulette/
├── __init__.py           the public API, re-exported from the areas below
├── llm/                  the model; knows nothing about personas
│   ├── core.py             LLM (client + model), completion helper, cancellation, JSON extraction
│   ├── openrouter.py       key check, model catalog, model-id check (free calls)
│   └── openai_compatible.py  the same checks for any OpenAI-compatible server (its model list)
├── config.py             settings models and strict loader (creation, random_personas)
├── persona/              what a persona is
│   ├── spec.py             loading persona folders; finding enriched / chat-ready personas
│   └── scenarios.py        INTERVIEW constant + per-session PAIRED entry-story generator
├── chat/                 talking to a persona
│   ├── engine.py           Persona: one conversation with one persona
│   ├── framework.py        hard rules, response style, output format, prompt assembly
│   ├── parsing.py          the model's <state>/<thoughts>/<reply> output, bubbles, markers
│   ├── judge.py            fail-open consistency judge (JSON-mode verdict on each reply)
│   └── conversation.py     two personas texting each other, headless
├── creation/             making personas
│   ├── pipeline.py         develop_persona: interview → enrichment → voice calibration; resume
│   ├── interviewer.py      AVP-style life interviewer
│   ├── enrich.py           transcript → field notes → enriched core; voice calibration; rebuild
│   ├── cv_parser.py        CV → persona v0.1, then the pipeline
│   └── random_persona.py   random persona v0.1 (seed traits + generator LLM), then the pipeline
└── evaluation/           offline quality checks
    ├── dataset.py          batches of persona-to-persona conversations, written to disk
    └── evaluate.py         mechanical metrics + LLM rubric → report.md, evaluation.json
```

The apps:

| File | Role |
|---|---|
| `cli/bot_roulette_cli/main.py` | The `bot-roulette` command: `chat`, `cv`, `random`, `enrich`, `dataset`, `evaluate` |
| `webapp/bot_roulette_web/server.py` | FastAPI web server: persona picker, chat WebSocket, persona creation jobs, settings endpoints |
| `webapp/bot_roulette_web/settings.py` | The model, where it runs (OpenRouter or `LLM_BASE_URL`) and its key: read from `.env`, changed live by the ⚙ settings panel, which writes them back |
| `webapp/bot_roulette_web/chats.py` | Saved chat histories: one JSON per chat, atomic writes, id-addressed only |
| `webapp/bot_roulette_web/home.py` | The home folder: `.env`, `config.yaml`, `personas/`, `chats/` |
| `webapp/bot_roulette_web/static/index.html` | Web client (picker + chat UI) |

## Tests

```
pytest
```

from the repository root runs the library's tests (`tests/`) and both
apps' (`cli/tests/`, `webapp/tests/`). No test calls a real model: a
fake client stands in for it, and the checks with OpenRouter and with a
local server are faked too.

## Intended use

bot-roulette was built for research: to study how well a language
model can simulate ordinary people, and how to measure it. Its personas
are made to pass for human (they never admit to being an AI), and one
can be built from a real person's CV. So, whatever you use it for:

- **Tell people they are talking to an AI.** Letting someone believe a
  persona is a real person deceives them, and in the EU the AI Act
  requires chatbots to say what they are (Article 50, applicable since
  2 August 2026). A study that keeps it hidden needs an ethics review
  and a debrief afterwards.
- **Build a persona from someone's CV only with their consent.** The
  persona, its interview and its chats are all personal data about that
  person, protected by privacy law (in the EU, the GDPR).
- **Don't use it to impersonate, deceive, defraud or harass anyone**,
  for instance with fake profiles on dating apps or social networks.

This section says how the project is meant to be used. It is not a
license term and doesn't change the license below.

## License

bot-roulette is available under two licenses:

- **[GNU AGPL v3](LICENSE)**, open source: anyone may use, study, change
  and share it, for any purpose. Whoever distributes it, or lets people
  use a modified version over a network (as a web service, say), must
  give them the complete source code of that version under the same
  license.
- **Commercial license**: to use it without the AGPL's obligations, for
  example in a closed-source product or service, contact Federico
  Ricciuti at ricciuti.federico@gmail.com.

Copyright © 2026 Federico Ricciuti.