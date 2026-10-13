import re
import unicodedata

PLATFORM_TARGET = re.compile(
    r"\b(?:behappy2day|be\s*happy\s*2\s*day|платформ\w*|platform|plataforma|"
    r"на (?:ваш(?:ем)? |тво[её]м |твой |эт(?:ом|от) )?сайт(?:е)?|your (?:site|website)|the (?:site|website)|website|"
    r"tu sitio|el sitio|site)\b",
    re.I,
)
PLATFORM_REQUEST = re.compile(
    r"(?:^|[.!?;,]\s*)(?:а\s+)?(?:как|где|куда|дай|пришли|скинь|покажи|можно|хочу|ссылк\w*|"
    r"на как(?:ой|ую)|how|where|can|could|send|give|i want|link|"
    r"cómo|dónde|cuál|cual|puedo|quiero|enlace|envía|"
    r"como|onde|posso|quero|manda)\b",
    re.I,
)
PLATFORM_ACCESS = re.compile(
    r"общ\w*|перепис\w*|разговар\w*|напис\w*|связа\w*|поговор\w*|поболт\w*|продолж\w*|пообщ\w*|зайти|перейти|попасть|откры\w*|ссылк\w*|"
    r"найти|профил\w*|chat|talk|message|continue|access|join|open|find|get|profile|link|"
    r"hablar|chatear|entrar|continuar|acceder|encontr\w*|perfil|enlace|"
    r"conversar|acessar",
    re.I,
)


def normalize_platform_input(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"^(?:привет|hello|hi|hola|olá)[,.!?\s]+", "", normalized).lstrip(
        "¿¡"
    )


def requests_platform_info(text: str) -> bool:
    normalized = normalize_platform_input(text)
    if not re.search(r"behappy2day|be\s*happy\s*2\s*day", normalized) and re.search(
        r"\b(?:telegram|zoom|whatsapp|discord|телеграм\w*)\b", normalized
    ):
        return False
    return bool(
        PLATFORM_TARGET.search(normalized)
        and PLATFORM_REQUEST.search(normalized)
        and (
            PLATFORM_ACCESS.search(normalized)
            or re.fullmatch(r"(?:хочу|можно)\s+на\s+платформу[?!.]*", normalized)
        )
    )


def requests_platform_link_again(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:дай|пришли|скинь|покажи)\s+(?:мне\s+)?ссылку(?:\s+пожалуйста)?[?!.]*|"
            r"(?:send|give|show)(?:\s+me)?\s+(?:the\s+)?link(?:\s+please)?[?!.]*|"
            r"(?:envía|manda)\s+(?:el\s+)?enlace[?!.]*|"
            r"manda\s+(?:o\s+)?link[?!.]*|"
            r"(?:а\s+)?(?:где\s+)?(?:ссылк[ау]|link|enlace)[?!.]*",
            normalize_platform_input(text),
        )
    )


def declines_platform_link(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"не (?:надо|нужно)(?:\s+ссылк[ау])?[?!.]*|"
            r"не (?:присылай|отправляй|давай) ссылку(?: на платформу)?[?!.]*|"
            r"(?:don't|do not) send (?:me )?(?:the )?link[?!.]*",
            normalize_platform_input(text),
        )
    )


def platform_request_language(text: str, prior_language: str) -> str:
    normalized = normalize_platform_input(text)
    if re.search(r"[а-яё]", normalized):
        return "ru"
    if re.search(
        r"\b(?:cómo|dónde|puedo|quiero|enlace|sitio|hablar|chatear)\b", normalized
    ):
        return "es"
    if re.search(r"\b(?:como|onde|posso|quero|conversar|acessar)\b", normalized):
        return "pt"
    if re.search(
        r"\b(?:how|where|can|could|send|give|link|website|platform)\b", normalized
    ):
        return "en"
    return prior_language
