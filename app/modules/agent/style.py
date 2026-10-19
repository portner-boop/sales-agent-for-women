import re

from app.modules.agent.resources import response_text

DAY_PARTS = {
    "morning": r"\b(?:утр[оау]|утром|утрен\w*|morning|buenos días|por la mañana|bom dia)\b",
    "evening": r"\b(?:вечер\w*|evening|tonight)\b",
    "afternoon": r"\b(?:afternoon|buenas tardes|por la tarde|boa tarde)\b",
    "night": r"\b(?:ноч[ьи]|ночью|ночн\w*|night|noche|noite)\b",
}


def formulaic_capability_reply(text: str) -> bool:
    normalized = text.casefold().translate(str.maketrans("’‘‐‑–—", "''----"))
    normalized = " ".join(normalized.split())
    return bool(
        re.search(
            r"\bi (?:don't|do not) (?:really )?have (?:any |real )?weekends\b|"
            r"\bi have no (?:real )?weekends\b|"
            r"\b(?:this (?:place|chat)|it|here) is (?:simple[,;:]? )?(?:a )?(?:text[- ]only|only text)\b|"
            r"\bthis is (?:just )?a text[- ]only chat\b|"
            r"\b(?:here we|we here) can only (?:text|exchange text messages)\b|"
            r"\bi (?:don't|do not) (?:do|have) (?:real[- ]life|in[- ]person) meetings\b|"
            r"\bу меня (?:здесь |тут )?нет (?:настоящих |реальных )?выходных\b|"
            r"\b(?:здесь|тут|в этом чате) (?:есть |доступен |возможен )?только текст\b|"
            r"\b(?:этот чат|это место) (?:просто |только )?текстов\w*\b|"
            r"\bя не (?:встречаюсь|провожу встречи) в реальной жизни\b|"
            r"\bi don't share (?:any |personal )?contacts?\b|"
            r"\bi'm marko[.!]?\s+let's just talk\b|"
            r"\bя марко[.!]?\s+давай просто поговорим\b",
            normalized,
        )
    )


def memory_profile_readout(text: str, user_text: str) -> bool:
    if re.search(
        r"^\s*(?:so far,?\s+)?i remember what you(?:['’]ve| have)? told me\s*[:;,]|"
        r"^\s*я помню о тебе следующее\s*[:;,]|"
        r"^\s*из того,? что ты рассказала,? я помню\s*[:;,]",
        text,
        re.I,
    ):
        return True
    return bool(
        re.search(
            r"what else[^.!?\n]{0,60}(?:keep in mind|remember|stor\w*)|что ещ[её][^.!?\n]{0,60}(?:запомн\w*|сохран\w*)",
            text,
            re.I,
        )
    )


def unsolicited_day_part(
    text: str, user_text: str, time_reference_allowed: bool = False
) -> bool:
    future_event = bool(
        re.search(
            r"\b(?:tonight|this evening|tomorrow|later|завтра|сегодня вечером|вечером|demain|ce soir|mañana|esta noche)\b|غدًا|غدا|هذا المساء",
            user_text,
            re.I,
        )
    )
    current_question = bool(
        re.search(
            r"how(?:'s|’s| is).*?(?:morning|evening|night).*?(?:going|so far)|как.*?проходит.*?(?:вечер|утро|ночь)",
            text,
            re.I,
        )
    )
    if future_event and current_question:
        return True
    if time_reference_allowed or re.search(
        r"\b(?:когда|во сколько|when|what time|cuándo|cuando|quando)\b", user_text, re.I
    ):
        return False
    mentions_unconfirmed = any(
        re.search(pattern, text, re.I) and not re.search(pattern, user_text, re.I)
        for pattern in DAY_PARTS.values()
    )
    addresses_current_time = bool(
        re.search(
            r"\b(?:good morning|good afternoon|good evening|buenos días|buenas tardes|buenas noches|bom dia|boa tarde|boa noite|доброе утро|добрый вечер|доброй ночи)\b|how(?:'s|’s| is).*?(?:your |the )?(?:morning|evening|night)|как.*?(?:вечер|утро|ночь).*?(?:проходит|проводишь)|как.*?проходит.*?(?:вечер|утро|ночь)|(?:it is|it's|сейчас|у тебя|for you).*?(?:morning|evening|night|утро|вечер|ночь)",
            text,
            re.I,
        )
    )
    return mentions_unconfirmed and addresses_current_time


def refine_day_question(text: str, user_text: str) -> tuple[str, bool]:
    original = text
    if unsolicited_day_part(text, user_text):
        text = re.sub(
            r"\b(?:доброе утро|добрый вечер|доброй ночи)\b",
            response_text("greeting", "ru"),
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:good morning|good afternoon|good evening)\b",
            response_text("greeting", "en"),
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:buenos días|buenas tardes|buenas noches)\b",
            response_text("greeting", "es"),
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:bom dia|boa tarde|boa noite)\b",
            response_text("greeting", "pt"),
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bкак (?:у тебя )?проходит (?:вечер|утро|ночь)\s*\?",
            response_text("day_question", "ru"),
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bкак (?:ты )?(?:вечер|утро|ночь) проводишь\s*\?",
            response_text("day_question", "ru"),
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bhow(?:’s|'s| is) (?:your |the )?(?:evening|morning|night)(?: going)?(?: so far)?\s*\?",
            response_text("day_question", "en"),
            text,
            flags=re.I,
        )
    text = re.sub(
        r"\bкак (?:у тебя )?проходит день\s*[,;:]\s*в целом норм(?:ально)? (?:или|и) уже вымотал[ао]?\s*\?",
        response_text("day_question", "ru"),
        text,
        flags=re.I,
    )
    question_changed = text != original and bool(
        re.search(r"Как проходит день\?|How is your day going\?", text, re.I)
    )
    return text, question_changed


def strip_terminal_period(text: str) -> str:
    text = text.strip()
    if not text.endswith(".") or text.endswith(".."):
        return text
    if re.search(r"(?:\b[^\W\d_]\.\s*){2,}$", text):
        return text
    return text[:-1]
