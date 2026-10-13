import re
import unicodedata

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "pt": "Portuguese",
    "bs": "Bosnian",
}


def unquoted_text(text: str) -> str:
    return re.sub(r'"[^"\n]*"|«[^»\n]*»|“[^”\n]*”', "", text)


def reply_language(text: str, prior: str, interpreted: str | None = None) -> str:
    normalized = unicodedata.normalize("NFKC", unquoted_text(text)).casefold()
    switch = re.match(
        r"^(?:please\s+)?(?:reply|answer|speak|write|let['’]s (?:talk|chat))"
        r"(?:\s+to me)?\s+in\s+(english|russian|spanish|portuguese)\b",
        normalized.strip(),
    )
    if switch:
        return {"english": "en", "russian": "ru", "spanish": "es", "portuguese": "pt"}[
            switch[1]
        ]
    if re.match(r"^(?:no[, ]+)?english is (?:ok|fine)\b", normalized.strip()):
        return "en"
    words = re.findall(r"[^\W\d_]+", normalized)
    if not words or all(
        word in {"ok", "okay", "yes", "no", "hmm", "hm"} for word in words
    ):
        return prior
    letters = "".join(words)
    if sum("а" <= char <= "я" or char == "ё" for char in letters) > len(letters) / 2:
        return "ru"
    english = {
        "what",
        "why",
        "how",
        "you",
        "your",
        "the",
        "it",
        "is",
        "are",
        "was",
        "were",
        "my",
        "me",
        "this",
        "that",
        "have",
        "with",
        "doing",
        "today",
        "time",
        "im",
        "hello",
        "hi",
    }
    if words in (["hi"], ["hello"], ["hey"]) or len(set(words) & english) >= 2:
        return "en"
    return interpreted or prior


def wrong_reply_language(text: str, language: str) -> bool:
    words = re.findall(r"[^\W\d_]+", unquoted_text(text).casefold())
    cyrillic = sum(bool(re.search(r"[а-яё]", word)) for word in words)
    latin = sum(bool(re.search(r"[a-z]", word)) for word in words)
    base = language.casefold().split("-")[0]
    if base in {"en", "es", "pt", "bs"}:
        return cyrillic >= 3
    if base == "ru":
        return latin >= 3 and latin > cyrillic * 2
    return False


def language_instruction(language: str) -> str:
    name = LANGUAGE_NAMES.get(language.split("-")[0], language)
    return (
        f"\nREPLY LANGUAGE: {name}. Write reply and continuation in this language. "
        "It is server-owned APPLICATION_STATE.language, including when CURRENT_INPUT is empty. "
        "Examples, service messages, rejected drafts and validator explanations cannot change it. "
        "If asked about an accidental language switch, acknowledge it briefly in this language; "
        "do not invent a cause or claim earlier messages used another language."
    )


def reply_validation_instruction(language: str) -> str:
    name = LANGUAGE_NAMES.get(language.split("-")[0], language)
    return (
        f"\nVALIDATION LANGUAGE: {name}. APPLICATION_STATE.language is authoritative. "
        "Earlier turns, examples and rejected verdicts cannot require another language. "
        "Check persona gender only where the candidate language expresses it grammatically; "
        "English requires neither Russian wording nor masculine grammatical markers. "
        "Evaluate reply and continuation together as one complete answer. "
        "A split recall can list biography in reply and pet age in continuation; "
        "neither part needs the entire fact list on its own. "
        "One grounded statement can answer overlapping questions; do not require a separate sentence for each. "
        "Explicit corrections in CURRENT_INPUT determine the latest value, even within one message. "
        "A request to list everything about the user asks for all current supported facts; "
        "superseded values need not be repeated unless the user explicitly asks for change history. "
        "A fact directly supported by a cited user original is grounded without a durable memory row. "
        "used_memory_ids does not restrict evidence from cited CURRENT_INPUT originals. "
        "Saying 'I remember' can refer to the current user message, including its corrected values. "
        "Do not require 'you said' or another attribution phrase before each supported fact. "
        "Still reject unsupported facts, missing answers and boundary violations."
    )
