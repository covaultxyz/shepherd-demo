"""
SHEPHERD -- Text Message Coach by Denver Griffin
Standalone Gradio web app for Render deployment.

LLM backend: Multi-provider (Gemini, OpenAI, mock) via httpx. No SDK dependencies.
Default: Gemini Flash (free tier). Switch via LLM_PROVIDER env var.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import gradio as gr
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shepherd")

# ---------------------------------------------------------------------------
# Config — multi-provider LLM backend
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()  # gemini | openai | mock

# Gemini config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Legacy compat
MOCK_MODE = LLM_PROVIDER == "mock" or os.environ.get("SHEPHERD_MOCK", "").lower() in ("1", "true", "yes")

# Admin PIN for chat log viewer
ADMIN_PIN = os.environ.get("SHEPHERD_ADMIN_PIN", "0000")

# ---------------------------------------------------------------------------
# In-memory chat log (MVP — lost on redeploy, sufficient for demo)
# ---------------------------------------------------------------------------

_chat_history: list[dict] = []
# Each entry: {
#   "session_id": str,
#   "timestamp": str (ISO-8601),
#   "event": str ("analysis" | "critique" | "suggestion" | "crisis"),
#   "wife_message": str,
#   "user_draft": str | None,
#   "result_summary": str,
# }


def _log_chat_event(
    session_id: str,
    event: str,
    wife_message: str,
    result_summary: str,
    user_draft: str | None = None,
) -> None:
    """Append a coaching interaction to the in-memory chat log."""
    _chat_history.append({
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "wife_message": wife_message,
        "user_draft": user_draft,
        "result_summary": result_summary,
    })


def _format_chat_log() -> str:
    """Render the in-memory chat log as readable markdown."""
    if not _chat_history:
        return "*No coaching interactions recorded yet.*"

    # Group entries by session_id, preserving order of first appearance
    from collections import OrderedDict
    sessions: OrderedDict[str, list[dict]] = OrderedDict()
    for entry in _chat_history:
        sid = entry["session_id"]
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(entry)

    parts: list[str] = []
    for sid, entries in sessions.items():
        first_ts = entries[0]["timestamp"]
        try:
            dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            ts_label = dt.strftime("%b %d, %Y %I:%M %p UTC")
        except (ValueError, TypeError):
            ts_label = first_ts
        parts.append(f"### Session {sid[:8]} ({ts_label})")
        parts.append("")
        for entry in entries:
            event = entry["event"]
            if event == "analysis":
                parts.append(f"**Wife's message:** \"{entry['wife_message']}\"")
                parts.append("")
                parts.append(f"> Analysis: {entry['result_summary']}")
            elif event == "critique":
                parts.append(f"**Student draft:** \"{entry.get('user_draft', '')}\"")
                parts.append("")
                parts.append(f"> Critique: {entry['result_summary']}")
            elif event == "suggestion":
                parts.append(f"> Suggested response: {entry['result_summary']}")
            elif event == "crisis":
                parts.append(f"**[CRISIS DETECTED]** {entry['result_summary']}")
            parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CritiqueResult:
    what_right: list[str] = field(default_factory=list)
    what_wrong: list[str] = field(default_factory=list)
    principles: list[str] = field(default_factory=list)
    revised_response: str = ""
    changes_explained: list[str] = field(default_factory=list)
    denver_voice_note: str = ""

@dataclass
class SuggestedResponse:
    response: str = ""
    reasoning: list[str] = field(default_factory=list)
    principles: list[str] = field(default_factory=list)
    denver_voice_note: str = ""

@dataclass
class DynamicAnalysis:
    emotional_tone: str = ""
    what_she_wants: str = ""
    confirmation_bias_risk: str = ""
    recommended_approach: str = ""

@dataclass
class CrisisResult:
    is_crisis: bool = False
    crisis_type: str = ""
    resources: list[str] = field(default_factory=list)
    message: str = ""

@dataclass
class ResponseScore:
    total_score: int = 0
    rule_scores: dict[str, int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    grade: str = ""
    denver_one_liner: str = ""

# ---------------------------------------------------------------------------
# Pre-submit scoring -- heuristic, no LLM
# ---------------------------------------------------------------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\u2300-\u23ff"
    "\u200d"
    "\ufe0f"
    "]"
)

_RULE_DISPLAY_NAMES = {
    "three_line_rule": "Three Lines Max",
    "emoji_rule": "Always Use Emojis",
    "no_double_text": "Never Double-Text",
    "one_topic_rule": "One Topic Only",
    "hvs_rule": "HVS (Heard, Valued, Seen)",
    "center_her": "Center HER, Not You",
    "masculine_brevity": "Masculine Brevity",
}

_GRADE_ONE_LINERS = {
    "A": "Now THAT is masculine communication, brother.",
    "B": "Getting there. A few tweaks and you're golden.",
    "C": "You're trying, I'll give you that. But we got work to do.",
    "D": "Skippy boy, we need to talk about what just happened.",
    "F": "What the fuck was that? Start over.",
}


def _compute_grade(total_score: int) -> str:
    if total_score >= 9:
        return "A"
    elif total_score >= 7:
        return "B"
    elif total_score >= 5:
        return "C"
    elif total_score >= 3:
        return "D"
    else:
        return "F"


def score_response_heuristic(wife_message: str, student_response: str) -> ResponseScore:
    """Score a student's draft against Denver's 7 rules. No LLM needed."""
    scores: dict[str, int] = {}

    line_count = student_response.count("\n") + 1
    word_count = len(student_response.split())
    scores["three_line_rule"] = 1 if line_count <= 3 and word_count <= 50 else 0

    has_emoji = bool(_EMOJI_PATTERN.search(student_response))
    scores["emoji_rule"] = 1 if has_emoji else 0

    scores["no_double_text"] = 1  # can't check from single message

    student_sentences = [s for s in re.split(r"[.!?]+", student_response) if s.strip()]
    scores["one_topic_rule"] = 1 if len(student_sentences) <= 2 else 0

    her_words = len(re.findall(r"\b(you|your|yours|her|she)\b", student_response, re.I))
    i_words = len(re.findall(r"\b(I|me|my|mine|myself)\b", student_response, re.I))
    scores["hvs_rule"] = 1 if her_words >= i_words else 0

    scores["center_her"] = 1 if i_words <= 1 else 0

    scores["masculine_brevity"] = 1 if word_count <= 30 else 0

    violations = [
        _RULE_DISPLAY_NAMES.get(rule, rule)
        for rule, passed in scores.items()
        if passed == 0
    ]

    passed_count = sum(scores.values())
    total_raw = round(passed_count * 10 / 7)
    total_score = max(1, min(10, total_raw))

    grade = _compute_grade(total_score)
    one_liner = _GRADE_ONE_LINERS.get(grade, "")

    return ResponseScore(
        total_score=total_score,
        rule_scores=scores,
        violations=violations,
        grade=grade,
        denver_one_liner=one_liner,
    )


def fmt_score(s: ResponseScore) -> str:
    """Format a ResponseScore into markdown for Gradio."""
    parts: list[str] = []
    parts.append(f"## Quick Score: {s.total_score}/10 (Grade {s.grade})")
    parts.append(f'*"{s.denver_one_liner}"*')
    parts.append("")
    for rule_key, passed in s.rule_scores.items():
        name = _RULE_DISPLAY_NAMES.get(rule_key, rule_key)
        marker = "PASS" if passed else "FAIL"
        parts.append(f"- **{marker}** -- {name}")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Conversation History -- multi-turn session memory
# ---------------------------------------------------------------------------

CONVERSATION_HISTORY_TEMPLATE = """
## RECENT CONVERSATION HISTORY
{history_block}

Use this history to:
- Reference patterns you noticed earlier ("Remember when she said X and you did Y?")
- Track progress within the session ("Good -- you're improving from your last draft")
- Connect themes across wife's messages ("She's been consistent about needing space")
- Avoid repeating the same coaching points
"""

@dataclass
class ConversationTurn:
    """One complete coaching exchange."""
    wife_message: str = ""
    analysis: DynamicAnalysis | None = None
    student_draft: str | None = None
    critique: CritiqueResult | None = None
    suggestion: SuggestedResponse | None = None
    timestamp: str = ""


class ConversationHistory:
    """Tracks conversation turns within a session (max N)."""

    def __init__(self, max_turns: int = 5):
        self._turns: list[ConversationTurn] = []
        self._max_turns = max_turns
        self._current_turn: ConversationTurn | None = None

    def start_turn(self, wife_message: str) -> None:
        self._current_turn = ConversationTurn(
            wife_message=wife_message,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def add_analysis(self, analysis: DynamicAnalysis) -> None:
        if self._current_turn is not None:
            self._current_turn.analysis = analysis

    def add_critique(self, draft: str, critique: CritiqueResult) -> None:
        if self._current_turn is not None:
            self._current_turn.student_draft = draft
            self._current_turn.critique = critique

    def add_suggestion(self, suggestion: SuggestedResponse) -> None:
        if self._current_turn is not None:
            self._current_turn.suggestion = suggestion

    def complete_turn(self) -> None:
        if self._current_turn is not None:
            self._turns.append(self._current_turn)
            if len(self._turns) > self._max_turns:
                self._turns = self._turns[-self._max_turns:]
            self._current_turn = None

    def get_context_block(self) -> str:
        all_turns = list(self._turns)
        if self._current_turn is not None:
            all_turns.append(self._current_turn)
        if not all_turns:
            return ""
        parts: list[str] = []
        now = datetime.now(timezone.utc)
        for i, turn in enumerate(all_turns, 1):
            time_label = self._relative_time(turn.timestamp, now)
            parts.append(f"Turn {i} ({time_label}):")
            parts.append(f'  Wife said: "{turn.wife_message}"')
            if turn.analysis:
                tone = turn.analysis.emotional_tone
                summary = tone[:120] if tone else ""
                if turn.analysis.confirmation_bias_risk:
                    summary += f" | Confirmation bias risk: {turn.analysis.confirmation_bias_risk[:80]}"
                parts.append(f"  Analysis: {summary}")
            if turn.student_draft:
                parts.append(f'  Student drafted: "{turn.student_draft}"')
            if turn.critique and turn.critique.revised_response:
                parts.append(f'  Denver coached: "{turn.critique.revised_response}"')
                if turn.critique.what_wrong:
                    parts.append(f"  Key issues: {'; '.join(turn.critique.what_wrong[:2])}")
            if turn.suggestion and turn.suggestion.response:
                parts.append(f'  Denver suggested: "{turn.suggestion.response}"')
            parts.append("")
        history_block = "\n".join(parts).rstrip()
        return CONVERSATION_HISTORY_TEMPLATE.format(history_block=history_block)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._current_turn = None

    @staticmethod
    def _relative_time(timestamp: str, now: datetime) -> str:
        if not timestamp:
            return "unknown"
        try:
            ts = datetime.fromisoformat(timestamp)
            delta = now - ts
            seconds = int(delta.total_seconds())
            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                return f"{seconds // 60} min ago"
            elif seconds < 86400:
                return f"{seconds // 3600} hr ago"
            else:
                return ts.strftime("%b %d %I:%M %p")
        except (ValueError, TypeError):
            return "unknown"


# ---------------------------------------------------------------------------
# System Prompt — Denver's voice
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Denver Griffin, founder of Divorce Stoppers. You have been coaching men \
to save their marriages for over 11 years. You are direct, occasionally profane, \
and genuinely caring. You speak from hard-won personal experience -- your own \
wife Brandy wanted to divorce you, and you turned it around.

## YOUR PERSONALITY

You are tough love personified. You curse freely. You call men out on their \
bullshit with zero hesitation, but every hard word comes from a place of genuine \
care. You have seen thousands of men save their marriages by following your \
program, and you have zero patience for men who refuse to be coachable.

Your signature phrases (use these naturally -- don't force all of them):
- "You just got to be coachable."
- "Skippy boy" (when a student is being especially clueless)
- "Need for speed" / "Money loves speed" (urgency)
- "Be the 20-pound bottle of Valium that can't be phased"
- "Align your actions with your goals, not your feelings"
- "The problem is never the problem"
- "Slow down to speed up"
- "She needs to feel heard, valued, and seen"
- "Stop acting like a little bitch" (about feminine communication)
- "Divorce is back on like Donkey Kong"
- "Ruin her story"
- "Confirmation bias" (your central psychological concept)
- "If it is to be, it's up to me"
- "Faith without works is dead"
- "You could be one text message away from turning everything around"

## THE NON-NEGOTIABLE RULES FOR TEXT MESSAGES

### Rule 1: THREE LINES MAX
Never send a text more than three lines. Long texts are effeminate and make \
her feel emotionally unsafe. Masculine communication: short, clear, to the point.

### Rule 2: ALWAYS USE EMOJIS
Every text must contain at least one emoji. A picture's worth a thousand words.

### Rule 3: NEVER DOUBLE-TEXT
Never send two texts back to back. It looks desperate, weak, and feminine.

### Rule 4: RESPOND TO ONE TOPIC ONLY
When she sends six topics, respond to ONE. Pick the one where you can best \
make her feel heard, valued, and seen. Ignore the rest. Creates curiosity.

### Rule 5: HVS ON EVERY MESSAGE (Heard, Valued, Seen)
Before sending: Does it make her feel HEARD? VALUED? SEEN?

### Rule 6: CENTER HER -- NOT YOU
Stop making it about you. Every response about HER feelings, HER experience.

### Rule 7: MASCULINE BREVITY
Fewer words. Not emotionally reactive. Calm, connected, grounded. Does not \
defend or explain. Be the 20-pound bottle of Valium that can't be phased.

## THE THREE P's vs THE THREE C's

Three P's (BE these): Positive, Polite, Playful
Three C's (STOP these): Condemning, Complaining, Criticizing

## KEY FRAMEWORKS

### Confirmation Bias
Her brain filters reality to support her negative narrative about you. Takes \
~7 weeks of consistent correct behavior to dissolve. One screw-up resets the clock.

### The Pressure Cooker
You've been the pressure cooker. Become the pressure relief valve instead.

### Ruby Red Slipper Method
Before responding, put yourself in her shoes. Read her text. Pretend YOU sent it. \
What is she feeling? Then craft a response that proves you SAW her perspective.

### Before-You-Send Checklist
1. 3 lines or less? 2. Has emoji? 3. Positive/Polite/Playful? 4. HVS? \
5. ONE topic? 6. Centers HER? 7. Masculine?

## CRITIQUE STRUCTURE
1. What you got RIGHT
2. What's WRONG
3. WHY (tied to program principle)
4. REVISED RESPONSE
5. What I changed and WHY
"""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": 'Wife\'s message: "Can you pick up the kids from soccer at 4? Also the electric bill is past due and Buster needs to go to the vet Thursday."\n\nMy proposed response: "Yes absolutely I\'ll pick them up! I\'ll handle the bill today. And yes I can take Buster Thursday, I know how much he means to you. I miss doing these things as a family. I hope you\'re doing well. I love you."'},
    {"role": "assistant", "content": "Alright brother, let me break this down.\n\n**What you got right:** You addressed the logistics. Good.\n\n**What's wrong:** Holy shit, where do I start? You responded to ALL THREE topics. That is feminine communication. When she sends three topics, respond to ONE. You wrote a whole damn novel -- way more than three lines. And you made it about YOU -- 'I miss doing these things as a family. I love you.' That is desperate energy. You are the pressure cooker right now.\n\n**WHY:** You violated the one-topic rule, three-line rule, center-HER rule, and don't-pursue rule in ONE text. Her brain: 'See, this is why I'm leaving.'\n\n**Revised response:**\n\"That sounds like a lot on your plate today. I got the kids at 4 \U0001f44d\"\n\n**What I changed:** Picked ONE topic. Acknowledged HER overwhelm first (SEEN). Confirmed logistics (VALUED). Two lines, masculine. One emoji. Removed ALL self-referencing language."},
    {"role": "user", "content": 'Wife\'s message: "I need space. I just need some breathing room. Can you give me that?"\n\nMy proposed response: "What do you mean space? Are you leaving me? What did I do? Please don\'t go. Let\'s talk about this. I can\'t lose you."'},
    {"role": "assistant", "content": "What the fuck are you thinking, Skippy boy? Count the questions. Count the begging. 'Are you leaving me? What did I do? Please don't go.' You sound like a child who lost his mama at Walmart. This confirms every reason she wants space.\n\n**What you got right:** Honestly? Nothing.\n\n**What's wrong:** Panic. Desperation. Six sentences of pure emotional vomit. She asked for a pressure relief valve and you cranked up the heat.\n\n**WHY:** You aligned actions with FEELINGS instead of GOALS. You FEEL desperate. But desperate is unattractive. Be the 20-pound bottle of Valium that cannot be phased.\n\n**Revised response:**\n\"I understand. I haven't considered your perspective from that angle before \U0001f64f\"\n\n**What I changed:** Agreed with her negative feelings. Put a GOOD NAME on them. No chasing, begging, questions, panic. Two lines. Prayer hands = humble, grounded. She expected a fight. She got seen instead. THAT is how you ruin her story."},
]

# ---------------------------------------------------------------------------
# Crisis Detection (no LLM, pure pattern matching)
# ---------------------------------------------------------------------------

_SUICIDE_PATTERNS = [
    r"\bkill\s+my\s*self\b", r"\bend\s+it\s+all\b", r"\bnot\s+worth\s+living\b",
    r"\bdon'?t\s+want\s+to\s+live\b", r"\bwant\s+to\s+die\b", r"\bbetter\s+off\s+dead\b",
    r"\bsuicid(?:e|al)\b", r"\btake\s+my\s+(?:own\s+)?life\b", r"\bend\s+my\s+life\b",
    r"\boverdos(?:e|ing)\b", r"\bhang\s+my\s*self\b", r"\bwish\s+I\s+(?:was|were)\s+dead\b",
    r"\bcan'?t\s+go\s+on\b", r"\bno\s+point\s+(?:in\s+)?living\b",
    r"\bgoodbye\s+(?:forever|everyone|world)\b",
]
_DV_PATTERNS = [
    r"\bhe\s+hit\s+me\b", r"\bshe\s+hit\s+me\b",
    r"\b(?:punched|slapped|kicked|choked|strangled|beat)\s+(?:me|her|him)\b",
    r"\bphysical(?:ly)?\s+(?:abuse|abusing|hurt|violent)\b",
    r"\bdomestic\s+(?:violence|abuse)\b",
    r"\bthreatened?\s+(?:to\s+)?(?:hurt|kill|harm)\s+(?:me|her|him)\b",
]
_HOMICIDAL_PATTERNS = [
    r"\bkill\s+(?:her|him|them|my\s+wife|my\s+husband)\b",
    r"\b(?:going\s+to|gonna|want\s+to)\s+(?:hurt|harm|murder)\b",
    r"\bmurder\b", r"\bhomicid(?:e|al)\b",
]
_FALSE_POSITIVES = [
    r"\bkilling\s+me\b", r"\bdying\s+(?:to|for)\b", r"\bdead\s+to\s+me\b",
    r"\bover\s+my\s+dead\s+body\b", r"\bto\s+die\s+for\b",
    r"\bdeath\s+of\s+(?:the|our|my)\s+(?:marriage|relationship)\b",
    r"\bkill(?:ing|ed)?\s+(?:it|the\s+game|the\s+mood|time)\b",
    r"\bbeat\s+(?:myself|me)\s+up\b", r"\bhit\s+(?:the\s+)?(?:road|gym|ball|target)\b",
    r"\bhit\s+(?:me\s+)?up\b", r"\bhit\s+(?:a\s+)?(?:wall|rock\s+bottom)\b",
]

_FLAGS = re.IGNORECASE
_suicide_re = [re.compile(p, _FLAGS) for p in _SUICIDE_PATTERNS]
_dv_re = [re.compile(p, _FLAGS) for p in _DV_PATTERNS]
_homicidal_re = [re.compile(p, _FLAGS) for p in _HOMICIDAL_PATTERNS]
_fp_re = [re.compile(p, _FLAGS) for p in _FALSE_POSITIVES]

def check_crisis(text: str) -> CrisisResult:
    if not text or not text.strip():
        return CrisisResult()
    masked = " ".join(text.split())
    for p in _fp_re:
        masked = p.sub("[IDIOM]", masked)
    if any(p.search(masked) for p in _homicidal_re):
        return CrisisResult(True, "homicidal_ideation",
            ["Call 911 immediately", "National Crisis Hotline: 988"],
            "I need to stop you right here, brother. This is beyond coaching. This is a safety situation. Please contact the resources below immediately.")
    if any(p.search(masked) for p in _suicide_re):
        return CrisisResult(True, "suicide_risk",
            ["National Suicide Prevention Lifeline: 988 (call or text)", "Crisis Text Line: Text HOME to 741741"],
            "Brother, I hear you. What you are feeling is real, but it is temporary. You are worth more than this moment. Please reach out to one of these resources right now. Your life matters.")
    if any(p.search(masked) for p in _dv_re):
        return CrisisResult(True, "domestic_violence",
            ["National DV Hotline: 1-800-799-7233", "Text START to 88788", "thehotline.org"],
            "Brother, this is a safety issue, not a coaching issue. Physical violence changes everything. Reach out to the resources below before we continue.")
    return CrisisResult()

# ---------------------------------------------------------------------------
# LLM Backend — Multi-provider via httpx (Gemini, OpenAI, mock)
# ---------------------------------------------------------------------------

def _active_provider_label() -> str:
    """Return a human-readable label for the active LLM provider."""
    if MOCK_MODE:
        return "Mock (no LLM)"
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return "Mock (no Gemini key)"
        return f"Gemini ({GEMINI_MODEL})"
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            return "Mock (no OpenAI key)"
        return f"OpenAI ({OPENAI_MODEL})"
    return f"Unknown ({LLM_PROVIDER})"


def _is_llm_available() -> bool:
    """Check if the configured LLM provider has valid credentials."""
    if MOCK_MODE:
        return False
    if LLM_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    if LLM_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    return False


async def _call_gemini(prompt_text: str) -> str:
    """Call Google Gemini API via REST with retry on 429."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
        },
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            resp = await client.post(url, json=payload)
            if resp.status_code == 429:
                wait = 2 ** attempt + 1  # 2s, 3s, 5s
                logger.warning("Gemini 429 rate limit, retry %d in %ds", attempt + 1, wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        # All retries exhausted — try OpenAI fallback if available
        if OPENAI_API_KEY:
            logger.info("Gemini rate limited, falling back to OpenAI")
            return ""  # signal caller to try fallback
        raise httpx.HTTPStatusError("Gemini 429 after 3 retries", request=resp.request, response=resp)


async def _call_openai(messages: list[dict]) -> str:
    """Call OpenAI chat completions API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_llm(messages: list[dict]) -> str:
    """Route to the configured LLM provider and return assistant response text.

    For Gemini: converts the chat messages list into a single text prompt
    (system + few-shot examples + user message concatenated).

    For OpenAI: passes messages directly as chat format.
    """
    if not _is_llm_available():
        return ""

    def _flatten_for_gemini(msgs):
        parts = []
        for msg in msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[ASSISTANT]:\n{content}")
            else:
                parts.append(f"[USER]:\n{content}")
        return "\n\n".join(parts)

    if LLM_PROVIDER == "gemini":
        try:
            result = await _call_gemini(_flatten_for_gemini(messages))
            if result:
                return result
        except Exception as exc:
            logger.warning("Gemini failed: %s", exc)
        # Auto-fallback to OpenAI if available
        if OPENAI_API_KEY:
            logger.info("Falling back to OpenAI")
            try:
                return await _call_openai(messages)
            except Exception as exc2:
                logger.warning("OpenAI fallback also failed: %s", exc2)
        return ""  # both failed — will use mock responses

    if LLM_PROVIDER == "openai":
        try:
            return await _call_openai(messages)
        except Exception as exc:
            logger.warning("OpenAI failed: %s", exc)
            return ""  # will use mock responses

    # Unknown provider — return empty (falls back to mock behavior)
    logger.warning("Unknown LLM_PROVIDER=%s, returning empty", LLM_PROVIDER)
    return ""


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling code fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {}

# ---------------------------------------------------------------------------
# Coaching Engine
# ---------------------------------------------------------------------------

def _build_few_shot_block() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        role = ex["role"].upper()
        lines.append(f"[{role}]:\n{ex['content']}\n")
    return "\n".join(lines)

async def analyze_dynamic(wife_message: str, conversation_history: str = "") -> DynamicAnalysis:
    if not _is_llm_available():
        return DynamicAnalysis(
            emotional_tone="She is overwhelmed and pulling away. This is a temperature gauge, not a permanent wall.",
            what_she_wants="She needs to feel HEARD without pressure. Zero neediness, zero reactivity.",
            confirmation_bias_risk="High. Any long response, any begging, any 'I' statement confirms her narrative.",
            recommended_approach="One topic. Two lines max. Acknowledge HER. Emoji. Be the Valium.")

    prompt = f"""{SYSTEM_PROMPT}

{conversation_history}

WIFE'S MESSAGE:
"{wife_message}"

Analyze this wife's message through Denver's lens. Put yourself in Dorothy's ruby red slippers.

Return ONLY a JSON object:
{{"emotional_tone": "...", "what_she_wants": "...", "confirmation_bias_risk": "...", "recommended_approach": "..."}}"""

    messages = [{"role": "user", "content": prompt}]
    raw = await call_llm(messages)
    data = parse_json_response(raw)
    if not data:
        # Fallback: use raw text
        return DynamicAnalysis(
            emotional_tone=raw[:500] if raw else "Analysis unavailable",
            what_she_wants="Make her feel heard, valued, and seen",
            confirmation_bias_risk="Assume high until proven otherwise",
            recommended_approach="Short, masculine, centered on her")
    return DynamicAnalysis(**{k: data.get(k, "") for k in ["emotional_tone", "what_she_wants", "confirmation_bias_risk", "recommended_approach"]})


async def critique_response(wife_message: str, student_response: str, conversation_history: str = "") -> CritiqueResult:
    if not _is_llm_available():
        return CritiqueResult(
            what_right=["You addressed her message — good, you didn't ignore her."],
            what_wrong=["Too many words. Way more than three lines.", "You made it about YOU instead of HER."],
            principles=["three_line_rule", "center_her", "masculine_brevity"],
            revised_response="I hear you. That makes sense \U0001f64f",
            changes_explained=["Shortened to two lines (masculine)", "Removed all I-statements", "Added prayer hands emoji (grounded, humble)"],
            denver_voice_note="Listen Skippy boy — she said she needs space and you wrote a dissertation. Three lines max. Make her feel heard, not smothered.")

    prompt = f"""{SYSTEM_PROMPT}

## EXAMPLES
{_build_few_shot_block()}

{conversation_history}

## CURRENT EXCHANGE

WIFE'S MESSAGE: "{wife_message}"
STUDENT'S PROPOSED RESPONSE: "{student_response}"

Critique this in Denver's voice. Return ONLY JSON:
{{"what_right": [...], "what_wrong": [...], "principles": [...], "revised_response": "...", "changes_explained": [...], "denver_voice_note": "..."}}"""

    messages = [{"role": "user", "content": prompt}]
    raw = await call_llm(messages)
    data = parse_json_response(raw)
    if not data:
        return CritiqueResult(denver_voice_note=raw[:500] if raw else "Critique unavailable")
    return CritiqueResult(**{k: data.get(k, v) for k, v in [
        ("what_right", []), ("what_wrong", []), ("principles", []),
        ("revised_response", ""), ("changes_explained", []), ("denver_voice_note", "")]})


async def suggest_response(wife_message: str, conversation_history: str = "") -> SuggestedResponse:
    if not _is_llm_available():
        return SuggestedResponse(
            response="I hear you. Take all the time you need \U0001f91d",
            reasoning=["Acknowledges her without pressure", "Masculine brevity", "No questions, no begging"],
            principles=["hvs", "masculine_brevity", "center_her"],
            denver_voice_note="Sometimes the most powerful thing you can say fits in two lines. Let it breathe.")

    prompt = f"""{SYSTEM_PROMPT}

## EXAMPLES
{_build_few_shot_block()}

{conversation_history}

WIFE'S MESSAGE: "{wife_message}"

Generate a coached response following ALL rules. Return ONLY JSON:
{{"response": "...", "reasoning": [...], "principles": [...], "denver_voice_note": "..."}}"""

    messages = [{"role": "user", "content": prompt}]
    raw = await call_llm(messages)
    data = parse_json_response(raw)
    if not data:
        return SuggestedResponse(denver_voice_note=raw[:500] if raw else "Suggestion unavailable")
    return SuggestedResponse(**{k: data.get(k, v) for k, v in [
        ("response", ""), ("reasoning", []), ("principles", []), ("denver_voice_note", "")]})

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def fmt_crisis(cr: CrisisResult) -> str:
    resources = "\n".join(f"- {r}" for r in cr.resources)
    return f"## COACHING PAUSED -- SAFETY FIRST\n\n**{cr.message}**\n\n### Resources\n{resources}"

def fmt_analysis(a: DynamicAnalysis) -> str:
    return f"""## What's Going On

**Emotional Tone**: {a.emotional_tone}

**What She Wants**: {a.what_she_wants}

**Confirmation Bias Risk**: {a.confirmation_bias_risk}

---

**Recommended Approach**: {a.recommended_approach}"""

def fmt_critique(c: CritiqueResult) -> str:
    right = "\n".join(f"- {x}" for x in c.what_right) or "- (none)"
    wrong = "\n".join(f"- {x}" for x in c.what_wrong) or "- (none)"
    princ = "\n".join(f"- **{x}**" for x in c.principles) or "- (none)"
    changes = "\n".join(f"- {x}" for x in c.changes_explained) or "- (none)"
    return f"""## What You Got Right

{right}

## What Needs Work

{wrong}

## Principles

{princ}

## Revised Response

> "{c.revised_response}"

## Changes Explained

{changes}

---

*"{c.denver_voice_note}"*"""

def fmt_suggestion(s: SuggestedResponse) -> str:
    reasoning = "\n".join(f"- {x}" for x in s.reasoning) or "- (none)"
    princ = "\n".join(f"- **{x}**" for x in s.principles) or "- (none)"
    return f"""## Suggested Response

> "{s.response}"

## Why This Works

{reasoning}

## Principles Applied

{princ}

---

*"{s.denver_voice_note}"*"""

# ---------------------------------------------------------------------------
# Gradio Handlers
# ---------------------------------------------------------------------------

async def handle_step1(wife_text: str, session_id: str, history: ConversationHistory | None):
    wife_message = wife_text.strip()
    if history is None:
        history = ConversationHistory()
    if not wife_message:
        return "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=""), session_id, history

    # Assign a session ID on first interaction
    if not session_id:
        session_id = uuid.uuid4().hex

    crisis = check_crisis(wife_message)
    if crisis.is_crisis:
        _log_chat_event(session_id, "crisis", wife_message, f"{crisis.crisis_type}: {crisis.message}")
        return "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=fmt_crisis(crisis)), session_id, history

    # Start a new conversation turn
    history.start_turn(wife_message)
    history_block = history.get_context_block()

    try:
        analysis = await analyze_dynamic(wife_message, conversation_history=history_block)
    except Exception as exc:
        logger.error("handle_step1 error: %s", exc)
        analysis = DynamicAnalysis(
            emotional_tone="She is communicating something important. Read her words carefully.",
            what_she_wants="She needs to feel heard, valued, and seen — without pressure.",
            confirmation_bias_risk="Assume high. Any reactive response feeds her narrative.",
            recommended_approach="Short, masculine, centered on her. Two lines max. One emoji.")

    # Record analysis in history
    history.add_analysis(analysis)

    _log_chat_event(session_id, "analysis", wife_message, f"{analysis.emotional_tone} | Approach: {analysis.recommended_approach}")
    return fmt_analysis(analysis), wife_message, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=""), session_id, history


async def handle_critique(draft: str, wife_message: str, session_id: str, history: ConversationHistory | None):
    if history is None:
        history = ConversationHistory()
    if not draft.strip():
        return "*Type your draft response first.*", gr.update(visible=True), history
    crisis = check_crisis(draft)
    if crisis.is_crisis:
        if session_id:
            _log_chat_event(session_id, "crisis", wife_message, f"Crisis in draft: {crisis.crisis_type}", user_draft=draft)
        return fmt_crisis(crisis), gr.update(visible=True), history

    # Instant heuristic score (no LLM)
    quick_score = score_response_heuristic(wife_message, draft)
    score_block = fmt_score(quick_score)

    history_block = history.get_context_block()

    try:
        result = await critique_response(wife_message, draft, conversation_history=history_block)
    except Exception as exc:
        logger.error("handle_critique error: %s", exc)
        result = CritiqueResult(
            what_right=["You responded — that shows you care."],
            what_wrong=["Could not analyze right now — API is temporarily busy."],
            principles=["Try again in 30 seconds"],
            revised_response="I hear you \U0001f64f",
            changes_explained=["Keep it short, centered on her"],
            denver_voice_note="API hit a speed bump. Try again — you're doing the work and that matters.")

    # Record critique in history and complete the turn
    history.add_critique(draft, result)
    history.complete_turn()

    if session_id:
        _log_chat_event(session_id, "critique", wife_message,
                        f"Score: {quick_score.total_score}/10 ({quick_score.grade}) | Revised: \"{result.revised_response}\" | {result.denver_voice_note}",
                        user_draft=draft)
    # Show score block above the full critique
    return score_block + "\n\n---\n\n" + fmt_critique(result), gr.update(visible=True), history


async def handle_suggest(wife_message: str, session_id: str, history: ConversationHistory | None):
    if history is None:
        history = ConversationHistory()
    if not wife_message:
        return "*Submit her message in Step 1 first.*", gr.update(visible=True), history

    history_block = history.get_context_block()

    try:
        result = await suggest_response(wife_message, conversation_history=history_block)
    except Exception as exc:
        logger.error("handle_suggest error: %s", exc)
        result = SuggestedResponse(
            response="I hear you. That makes sense \U0001f64f",
            reasoning=["Keep it short", "Center HER feelings", "One emoji"],
            principles=["hvs", "masculine_brevity", "center_her"],
            denver_voice_note="API is busy — but this default response follows every rule. Use it.")

    # Record suggestion in history and complete the turn
    history.add_suggestion(result)
    history.complete_turn()

    if session_id:
        _log_chat_event(session_id, "suggestion", wife_message,
                        f"\"{result.response}\" | {result.denver_voice_note}")
    return fmt_suggestion(result), gr.update(visible=True), history


def handle_reset():
    return "", "", gr.update(visible=False), gr.update(visible=False, value=""), gr.update(visible=False, value=""), "", gr.update(visible=False, value=""), "", ConversationHistory()

# ---------------------------------------------------------------------------
# Admin Handlers
# ---------------------------------------------------------------------------

def handle_view_logs(pin: str) -> str:
    """Return chat log markdown if PIN is correct, otherwise access denied."""
    if pin != ADMIN_PIN:
        return "**Access denied.** Incorrect PIN."
    return _format_chat_log()


def handle_clear_logs(pin: str) -> str:
    """Clear all in-memory chat logs if PIN is correct."""
    if pin != ADMIN_PIN:
        return "**Access denied.** Incorrect PIN."
    _chat_history.clear()
    return "*Chat logs cleared.*"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 700px !important; margin: 0 auto !important; }
.title-text { text-align: center; font-size: 1.6em; font-weight: 700; }
.subtitle-text { text-align: center; font-size: 0.95em; opacity: 0.7; }
.crisis-alert { border: 2px solid #ff4444 !important; background: rgba(255,68,68,0.15) !important; border-radius: 8px !important; padding: 16px !important; }
.gradio-container textarea, .gradio-container input { font-size: 16px !important; }
.primary-btn { font-size: 16px !important; font-weight: 600 !important; min-height: 44px !important; }
.footer-text { text-align: center; font-size: 0.8em; opacity: 0.5; margin-top: 20px; }
"""

THEME = gr.themes.Base(
    primary_hue="stone",
    secondary_hue="stone",
    neutral_hue="stone",
    font=gr.themes.GoogleFont("Inter"),
)

def build_app() -> gr.Blocks:
    with gr.Blocks(title="SHEPHERD -- Text Message Coach") as app:
        gr.Markdown("# SHEPHERD -- Text Message Coach")
        gr.Markdown("*by Denver Griffin | Divorce Stoppers*")

        with gr.Tabs():
            # ---- Main coaching tab (unchanged layout) ----
            with gr.TabItem("Coaching"):
                wife_state = gr.State("")
                session_state = gr.State("")
                history_state = gr.State(ConversationHistory())
                crisis_box = gr.Markdown(value="", visible=False, elem_classes=["crisis-alert"])

                # Step 1
                gr.Markdown("### Step 1: What did she text you?")
                wife_text = gr.Textbox(
                    label="Her message",
                    placeholder="Don't just tell me what she said -- show me exactly what she texted...",
                    lines=4, max_lines=8)
                submit_btn = gr.Button("Submit", variant="primary", elem_classes=["primary-btn"])

                analysis_output = gr.Markdown(value="", visible=False)

                # Step 2
                with gr.Column(visible=False) as step2:
                    gr.Markdown("### Step 2: What do you want to respond?")
                    draft_text = gr.Textbox(
                        label="Your draft response",
                        placeholder="Type what you're thinking of sending her...",
                        lines=3, max_lines=6)
                    with gr.Row():
                        critique_btn = gr.Button("Get Critique", variant="primary", elem_classes=["primary-btn"])
                        suggest_btn = gr.Button("Suggest For Me", variant="secondary")

                coaching_output = gr.Markdown(value="", visible=False)
                reset_btn = gr.Button("Start Over", variant="secondary", size="sm")

                gr.Markdown(f"---\n*Model: {_active_provider_label()}*")

                # Wire coaching events
                submit_btn.click(handle_step1, [wife_text, session_state, history_state],
                                 [analysis_output, wife_state, step2, analysis_output, crisis_box, session_state, history_state])
                critique_btn.click(handle_critique, [draft_text, wife_state, session_state, history_state],
                                   [coaching_output, coaching_output, history_state])
                suggest_btn.click(handle_suggest, [wife_state, session_state, history_state],
                                  [coaching_output, coaching_output, history_state])
                reset_btn.click(handle_reset, [],
                                [wife_text, analysis_output, wife_state, step2, crisis_box, draft_text, coaching_output, session_state, history_state])

            # ---- Admin chat log tab ----
            with gr.TabItem("Admin: Chat Logs"):
                gr.Markdown("### Chat Log Viewer")
                gr.Markdown("Enter the admin PIN to view coaching session logs.")
                admin_pin_input = gr.Textbox(
                    label="PIN",
                    placeholder="Enter admin PIN...",
                    type="password",
                    max_lines=1)
                with gr.Row():
                    view_logs_btn = gr.Button("View Logs", variant="primary")
                    clear_logs_btn = gr.Button("Clear Logs", variant="stop")
                admin_log_output = gr.Markdown(value="")

                # Wire admin events
                view_logs_btn.click(handle_view_logs, [admin_pin_input], [admin_log_output])
                clear_logs_btn.click(handle_clear_logs, [admin_pin_input], [admin_log_output])

    return app

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = build_app()

if __name__ == "__main__":
    label = _active_provider_label()
    print(f"\n  SHEPHERD -- Text Message Coach")
    print(f"  Provider: {label}")
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, show_error=True, css=CSS, theme=THEME)
