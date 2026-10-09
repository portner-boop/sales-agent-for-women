import re


def strip_terminal_period(text: str) -> str:
    text = text.strip()
    if not text.endswith(".") or text.endswith(".."):
        return text
    if re.search(r"(?:\b[^\W\d_]\.\s*){2,}$", text):
        return text
    return text[:-1]
