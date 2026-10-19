import re
import unicodedata

from app.modules.agent.resources import prompt_contract

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
    if interpreted:
        return interpreted
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
    return prompt_contract("language_generate").format(language=name)


def reply_validation_instruction(language: str) -> str:
    name = LANGUAGE_NAMES.get(language.split("-")[0], language)
    return prompt_contract("language_validate").format(language=name)
