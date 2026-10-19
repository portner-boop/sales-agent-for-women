import pytest

from app.application import Application
from app.evaluation import receive
from app.modules.agent.time import current_time_facts
from tests.helpers import scalar
from tests.test_followups import enable


@pytest.mark.parametrize(
    "text,language,answer,continuation",
    [
        (
            "What do you remember about me?",
            "en",
            "You put honey in your coffee",
            "I like that little twist",
        ),
        (
            "Que retiens-tu de moi ?",
            "fr",
            "Tu mets du miel dans ton café",
            "Ce mélange me plaît",
        ),
        (
            "What time is it in Split?",
            "en",
            "It's {clock} in Split",
            "I still like that concert story",
        ),
        (
            "Который час в Сплите?",
            "ru",
            "В Сплите сейчас {clock}",
            "Та история с концертом мне понравилась",
        ),
    ],
)
async def test_open_factual_answer_keeps_validated_continuation_and_timer(
    app: Application, text: str, language: str, answer: str, continuation: str
) -> None:
    enable(app)
    clock = current_time_facts("What time is it in Split?", app.clock.now())[0][
        "local_time"
    ]
    answer = answer.format(clock=clock)
    app.model.responses["interpret"].append(
        {
            "intent": "factual_chat",
            "dialogue_mode": "continuation",
            "language": language,
            "continue_conversation": True,
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": answer,
            "continuation": continuation,
            "dialogue_act": "answer_and_opinion",
            "followup_allowed": True,
        }
    )
    await receive(app, app.sender, 1, text, 1, app.clock.now())
    assert app.sender.sent == [(1, answer), (1, continuation)]
    request = next(data for task, data in app.model.calls if task == "generate")
    assert not request["DIALOGUE_PLAN"]["requires_continuation"]
    verdict_input = next(data for task, data in app.model.calls if task == "validate")
    assert verdict_input["reply"]["continuation"] == continuation
    assert (
        await scalar(app, "SELECT COUNT(*) FROM followups WHERE status='pending'") == 1
    )


@pytest.mark.parametrize(
    "mode,continue_conversation",
    [("continuation", False), ("closing", True), ("service", False)],
)
async def test_factual_continuation_respects_closed_or_answer_only_turns(
    app: Application, mode: str, continue_conversation: bool
) -> None:
    app.model.responses["interpret"].append(
        {
            "intent": "current_time",
            "dialogue_mode": mode,
            "language": "en",
            "continue_conversation": continue_conversation,
        }
    )
    clock = current_time_facts("What time is it in Split?", app.clock.now())[0][
        "local_time"
    ]
    answer = f"It's {clock} in Split"
    app.model.responses["generate"].append({"reply": answer, "dialogue_act": "answer"})
    await receive(
        app, app.sender, 1, "Just the time in Split, please", 1, app.clock.now()
    )
    assert app.sender.sent == [(1, answer)]
    request = next(data for task, data in app.model.calls if task == "generate")
    assert not request["DIALOGUE_PLAN"]["requires_continuation"]
    assert await scalar(app, "SELECT COUNT(*) FROM followups") == 0


async def test_repair_removes_invented_teasing_and_preserves_next_thought(
    app: Application,
) -> None:
    await receive(
        app,
        app.sender,
        1,
        "I drink coffee with honey, which people find strange",
        1,
        app.clock.now(),
    )
    app.model.responses["interpret"].append(
        {
            "intent": "recall",
            "dialogue_mode": "continuation",
            "language": "en",
            "continue_conversation": True,
        }
    )
    app.model.responses["generate"].append(
        {
            "reply": "People tease you for drinking coffee with honey",
            "continuation": "I like honey more than syrup",
            "dialogue_act": "recall_and_opinion",
        }
    )
    app.model.responses["validate"].append(
        {"decision": "reject", "reason_codes": ["UNSUPPORTED_FACT"]}
    )
    app.model.responses["repair"].append(
        {
            "reply": "You put honey in your coffee",
            "continuation": "I like honey more than syrup",
            "dialogue_act": "recall_and_opinion",
        }
    )
    await receive(
        app, app.sender, 2, "What do you remember about me?", 1, app.clock.now()
    )
    assert app.sender.sent[-2:] == [
        (1, "You put honey in your coffee"),
        (1, "I like honey more than syrup"),
    ]
    assert all("tease" not in text for _, text in app.sender.sent)
    request = next(data for task, data in app.model.calls if task == "repair")
    assert not request["DIALOGUE_PLAN"]["requires_continuation"]
    assert "UNSUPPORTED_FACT" in request["REPAIR_REASON_CODES"]
