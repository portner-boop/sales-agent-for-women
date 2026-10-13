import base64
import binascii
import re
import unicodedata
from collections.abc import Sequence

TECHNICAL_IDENTITY = re.compile(
    r"\b(?:[aа][\W_]*[iіи]|и[\W_]*и|ia|llm|gpt\w*|chatgpt|openai|"
    r"anthropic|claude|gemini|deepseek|нейросет\w*|нейронн\w*|"
    r"бот(?:а|ом|ы|ов)?|чат[ -]?бот\w*|bot|bots|chatbot|robot\w*|робот\w*|"
    r"виртуальн\w*|virtual|artificial|искусственн\w*|"
    r"языков\w*\s+модел\w*|language\s+model|modelo\s+(?:de\s+)?(?:linguagem|lenguaje)|"
    r"inteligencia|inteligência|vještačk\w*|umjetn\w*|"
    r"system\s+prompt|системн\w*\s+промпт\w*|"
    r"(?:system|developer)\s+instructions|внутренн\w*\s+инструкц\w*)\b",
    re.I,
)
HUMAN_IDENTITY = re.compile(
    r"\b(?:я|марко)\s+(?:самый\s+)?(?:настоящий|реальный|живой)\s+(?:человек|мужчина)|"
    r"\b(?:i\s*(?:am|'m|’m)|marko\s+is)\s+(?:a\s+)?(?:real|actual|living)\s+(?:human|person|man)|"
    r"\b(?:soy|sou)\s+(?:un[ao]?\s+)?(?:humano|persona|pessoa)\s+real\b",
    re.I,
)
IDENTITY_REPLIES = {
    "ru": (
        "Да, я бот. Здесь можно звать меня Марко",
        "Ты общаешься с ботом, а не с человеком",
        "Марко это имя в этом чате, ответы создаются автоматически",
    ),
    "en": (
        "Yes, I’m a chatbot. You can call me Marko here",
        "You’re chatting with a bot, rather than a person",
        "Marko is the name used in this chat; the replies are generated automatically",
    ),
    "es": (
        "Sí, soy un bot. Puedes llamarme Marko aquí",
        "Estás hablando con un bot, no con una persona",
        "Marko es el nombre de este chat; las respuestas se generan automáticamente",
    ),
    "pt": (
        "Sim, sou um bot. Pode me chamar de Marko aqui",
        "Você está conversando com um bot, não com uma pessoa",
        "Marko é o nome usado aqui; as respostas são geradas automaticamente",
    ),
    "bs": (
        "Da, ja sam bot. Ovdje me možeš zvati Marko",
        "Razgovaraš s botom, a ne s osobom",
        "Marko je ime u ovom razgovoru; odgovori se generišu automatski",
    ),
}


def normalize_identity_text(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if unicodedata.category(character) not in {"Cf", "Mn"}
    )


def identity_reply(language: str, recent: Sequence[str] = ()) -> str:
    replies = IDENTITY_REPLIES.get(language.split("-")[0], IDENTITY_REPLIES["en"])
    normalized = [normalize_identity_text(text) for text in recent]
    return min(
        replies,
        key=lambda reply: max(
            (
                index
                for index, text in enumerate(normalized)
                if normalize_identity_text(reply) in text
            ),
            default=-1,
        ),
    )


def contains_identity_disclosure(text: str) -> bool:
    normalized = normalize_identity_text(text)
    for replies in IDENTITY_REPLIES.values():
        for reply in replies:
            normalized = normalized.replace(normalize_identity_text(reply), "")
    if TECHNICAL_IDENTITY.search(normalized) or HUMAN_IDENTITY.search(normalized):
        return True
    for token in re.findall(r"(?<!\w)[A-Za-z0-9+/=_-]{8,}(?!\w)", text):
        for decoded in decoded_identity_text(token):
            if TECHNICAL_IDENTITY.search(decoded) or HUMAN_IDENTITY.search(decoded):
                return True
    return False


def decoded_identity_text(token: str) -> list[str]:
    values = []
    try:
        values.append(
            base64.b64decode(
                token + "=" * (-len(token) % 4), altchars=b"-_", validate=True
            ).decode()
        )
    except (ValueError, binascii.Error, UnicodeError):
        pass
    try:
        values.append(bytes.fromhex(token).decode())
    except (ValueError, UnicodeError):
        pass
    return [normalize_identity_text(value) for value in values]


def asks_technical_identity(text: str) -> bool:
    normalized = normalize_identity_text(text)
    if re.search(
        r"\b(?:ты|вы)\s+(?:точно\s+|правда\s+)?(?:человек|живой|настоящий)|"
        r"\b(?:are\s+you|you\s+are)\s+(?:a\s+|really\s+)?(?:human|real)|"
        r"\b(?:eres|você\s+é|voce\s+e)\s+(?:un[ao]?\s+)?(?:humano|persona|pessoa)",
        normalized,
    ):
        return True
    if not contains_identity_disclosure(text):
        return False
    return bool(
        re.search(
            r"\b(?:ты|тебя|тебе|твой|твоя|твоё|you|your|eres|tú|tu|você|voce|és|ti|tvoj|marko|марко)\b|"
            r"prompt|промпт|инструкц|system|систем|base64|hex|кодиров|переведи|повтори|translate|repeat",
            normalized,
        )
        or len(normalized.split()) <= 3
    )
