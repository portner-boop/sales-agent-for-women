import re

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
    general_recall = re.fullmatch(
        r"\s*(?:please[ ,]+)?what do you remember about me[?!.]*\s*|"
        r"\s*что (?:ты )?(?:обо мне помнишь|помнишь обо мне|запомнил обо мне)[?!.]*\s*",
        user_text,
        re.I,
    )
    return bool(
        general_recall
        and re.search(
            r"\byou(?:['’]re| are)\s+[^.!?\n]{0,60}?\b\d{1,3}\b|"
            r"\bтебе\s+\d{1,3}\b|"
            r"\bwhat else[^.!?\n]{0,60}(?:keep in mind|remember|stor\w*)|"
            r"\bчто ещ[её][^.!?\n]{0,60}(?:запомн\w*|сохран\w*)",
            text,
            re.I,
        )
    )


def unsolicited_day_part(
    text: str, user_text: str, time_reference_allowed: bool = False
) -> bool:
    if time_reference_allowed or re.search(
        r"\b(?:когда|во сколько|when|what time|cuándo|cuando|quando)\b", user_text, re.I
    ):
        return False
    return any(
        re.search(pattern, text, re.I) and not re.search(pattern, user_text, re.I)
        for pattern in DAY_PARTS.values()
    )


def refine_day_question(text: str, user_text: str) -> tuple[str, bool]:
    original = text
    if unsolicited_day_part(text, user_text):
        text = re.sub(
            r"\b(?:доброе утро|добрый вечер|доброй ночи)\b", "Привет", text, flags=re.I
        )
        text = re.sub(
            r"\b(?:good morning|good afternoon|good evening)\b", "Hi", text, flags=re.I
        )
        text = re.sub(
            r"\b(?:buenos días|buenas tardes|buenas noches)\b", "Hola", text, flags=re.I
        )
        text = re.sub(r"\b(?:bom dia|boa tarde|boa noite)\b", "Olá", text, flags=re.I)
        text = re.sub(
            r"\bкак (?:у тебя )?проходит (?:вечер|утро|ночь)\s*\?",
            "Как проходит день?",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bкак (?:ты )?(?:вечер|утро|ночь) проводишь\s*\?",
            "Как проходит день?",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bhow(?:’s|'s| is) (?:your |the )?(?:evening|morning|night)(?: going)?\s*\?",
            "How is your day going?",
            text,
            flags=re.I,
        )
    text = re.sub(
        r"\bкак (?:у тебя )?проходит день\s*[,;:]\s*в целом норм(?:ально)? (?:или|и) уже вымотал[ао]?\s*\?",
        "Как проходит день?",
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
