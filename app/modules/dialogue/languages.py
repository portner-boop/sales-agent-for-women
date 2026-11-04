import re
from functools import cache, lru_cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder


@cache
def language_aliases() -> dict[str, str]:
    aliases = {
        language.name.casefold(): language.iso_code_639_1.name.lower()
        for language in Language.all()
    }
    aliases.update(
        {
            "русский": "ru",
            "español": "es",
            "português": "pt",
            "français": "fr",
            "deutsch": "de",
            "العربية": "ar",
            "中文": "zh",
        }
    )
    return aliases


def canonical_language(value: str) -> str:
    parts = value.strip().replace("_", "-").split("-")
    parts[0] = language_aliases().get(parts[0].casefold(), parts[0].casefold())
    return "-".join(parts)


@cache
def language_detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


def language_prose(text: str) -> str:
    return re.sub(r'"[^"\n]*"|«[^»\n]*»|“[^”\n]*”', "", text)


def confident_language(text: str) -> str | None:
    prose = language_prose(text)
    if len(re.findall(r"[^\W\d_]", prose)) < 35 or len(prose.split()) < 6:
        return None
    detector = language_detector()
    scores = detector.compute_language_confidence_values(prose)
    if (
        len(scores) < 2
        or scores[0].value < 0.85
        or scores[0].value - scores[1].value < 0.25
    ):
        return None
    code = scores[0].language.iso_code_639_1
    return code.name.lower() if code else None


SCRIPT_DEFAULTS = {
    "cyrillic": "ru",
    "latin": "en",
    "arabic": "ar",
    "hebrew": "he",
    "greek": "el",
    "cjk": "zh",
}
LANGUAGE_SCRIPTS = {
    **dict.fromkeys(
        ("ru", "uk", "be", "bg", "sr", "mk", "kk", "ky", "tg", "mn"), "cyrillic"
    ),
    **dict.fromkeys(("ar", "fa", "ur"), "arabic"),
    "he": "hebrew",
    "el": "greek",
    **dict.fromkeys(("zh", "ja", "ko"), "cjk"),
}


def script_of(text: str) -> str | None:
    counts: dict[str, int] = {}
    for character in language_prose(text):
        if not character.isalpha():
            continue
        code = ord(character)
        if 0x0400 <= code <= 0x04FF:
            script = "cyrillic"
        elif 0x0600 <= code <= 0x06FF:
            script = "arabic"
        elif 0x0590 <= code <= 0x05FF:
            script = "hebrew"
        elif 0x0370 <= code <= 0x03FF:
            script = "greek"
        elif code >= 0x3040:
            script = "cjk"
        elif code < 0x0250:
            script = "latin"
        else:
            continue
        counts[script] = counts.get(script, 0) + 1
    if sum(counts.values()) < 2:
        return None
    return max(counts, key=lambda key: counts[key])


def language_script(language: str) -> str:
    return LANGUAGE_SCRIPTS.get(language.casefold().split("-")[0], "latin")


def likely_language(text: str) -> str | None:
    confident = confident_language(text)
    if confident is not None:
        return confident
    prose = language_prose(text)
    if len(re.findall(r"[^\W\d_]", prose)) < 8:
        return None
    scores = language_detector().compute_language_confidence_values(prose)
    if (
        len(scores) < 2
        or scores[0].value < 0.9
        or scores[0].value - scores[1].value < 0.7
    ):
        return None
    code = scores[0].language.iso_code_639_1
    return code.name.lower() if code else None


def message_language(text: str, fallback: str) -> str:
    """Her language for this message: detected words, else script, else fallback."""
    detected = likely_language(text)
    if detected is not None:
        return detected
    script = script_of(text)
    if script is None or language_script(fallback) == script:
        return fallback
    return SCRIPT_DEFAULTS.get(script, fallback)


@lru_cache(maxsize=16)
def contrast_detector(first: Language, second: Language) -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(first, second).build()


def confirmed_language_mismatch(text: str, language: str) -> bool:
    detected = confident_language(text)
    expected = language.casefold().split("-")[0]
    if detected is None or detected == expected:
        return False
    languages = {item.iso_code_639_1.name.lower(): item for item in Language.all()}
    if expected not in languages or detected not in languages:
        return False
    scores = contrast_detector(
        languages[expected], languages[detected]
    ).compute_language_confidence_values(language_prose(text))
    return (
        len(scores) == 2
        and scores[0].language == languages[detected]
        and scores[0].value >= 0.85
        and scores[0].value - scores[1].value >= 0.25
    )
