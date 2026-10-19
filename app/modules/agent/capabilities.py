import re
import unicodedata

from app.modules.agent.resources import response_text

PHOTO_WORD = r"(?:фото(?:графи[яиюйи])?|фотк[аиуио]|фоточк[аиуио]|сним(?:ок|ки|ку)|photos?|pictures?|pics?|fotos?|fotograf[ií]as?|imagens?)"
PHOTO_REQUEST = re.compile(
    rf"(?:^|[.!?]\s*|\bу тебя\s+)(?:а\s+)?(?:есть|имеются)\b[^.!?]*\b{PHOTO_WORD}\b|"
    rf"\b(?:покажи|пришли|скинь|отправь|поделись|можно|можешь)\b[^.!?]*\b{PHOTO_WORD}\b|"
    rf"\b(?:do you have|have you got|can you|could you|send|show|any)\b[^.!?]*\b{PHOTO_WORD}\b|"
    rf"\b(?:tienes|tens|tem|puedes|pode|manda|env[ií]a|mostra|muestra)\b[^.!?]*\b{PHOTO_WORD}\b|"
    rf"\b{PHOTO_WORD}\b\s+(?:есть|у тебя|please|por favor)\s*[?!.]*$",
    re.I,
)
PHOTO_QUERY_WORD = re.compile(
    rf"{PHOTO_WORD}|а|и|ну|привет|здравствуй|есть|имеются|у|тебя|твой|твои|твое|"
    r"покажи|пришли|скинь|отправь|поделись|можно|можешь|пожалуйста|мне|свои|себя|"
    r"hi|hello|do|you|have|got|any|a|an|some|can|could|send|show|me|your|please|of|yourself|"
    r"hola|tienes|puedes|enviarme|mandarme|manda|envía|muestra|tus|una|un|por|favor|"
    r"olá|ola|você|voce|tem|tens|pode|enviar|mandar|mostra|suas|uma|um|te|ver",
    re.I,
)


def normalized_photo_input(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def is_photo_request(text: str) -> bool:
    normalized = normalized_photo_input(text)
    normalized = re.sub(
        r"^(?:привет|здравствуй|hello|hi|hola|olá)[,.!?\s]+", "", normalized
    )
    return bool(
        PHOTO_REQUEST.search(normalized)
        or re.search(
            rf"\b(?:твои|твое|твоё|your|tus|suas)\s+{PHOTO_WORD}\s*[?!.]*$",
            normalized,
            re.I,
        )
        or re.fullmatch(rf"{PHOTO_WORD}\s*[?!.]*", normalized, re.I)
    )


def is_standalone_photo_request(text: str) -> bool:
    return is_photo_request(text) and all(
        PHOTO_QUERY_WORD.fullmatch(word)
        for word in re.findall(r"\w+", normalized_photo_input(text))
    )


def photo_refusal(language: str) -> str:
    return response_text("photo_refusal", language)


def photo_language(text: str, prior_language: str) -> str:
    normalized = normalized_photo_input(text)
    if re.search(r"[а-яё]", normalized):
        return "ru"
    if re.search(r"\b(?:tienes|puedes|envía|muestra|hola|tus)\b", normalized):
        return "es"
    if re.search(r"\b(?:você|voce|tem|pode|enviar|mandar|olá|suas)\b", normalized):
        return "pt"
    if re.search(r"\b(?:photos?|pictures?|pics?)\b", normalized):
        return "en"
    return prior_language
