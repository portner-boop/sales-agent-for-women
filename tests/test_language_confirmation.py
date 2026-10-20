import pytest

from app.modules.dialogue.validation import detected_language_mismatch


@pytest.mark.parametrize(
    "text,language",
    [
        ("Sou uma IA que conversa como o Marko, uma personagem fictícia", "pt"),
        ("Sou uma IA que conversa como o Marko, uma personagem fictícia", "pt-BR"),
        (
            'Sou uma IA. "Je suis un personnage automatisé et cette conversation est écrite par une IA"',
            "pt",
        ),
    ],
)
def test_portuguese_reply_is_not_rejected_as_french(text: str, language: str) -> None:
    assert not detected_language_mismatch(text, language)


def test_french_reply_declared_as_portuguese_is_rejected() -> None:
    assert detected_language_mismatch(
        "Je suis un personnage automatisé et cette conversation est écrite par une IA",
        "pt",
    )
