"""Acceptance checks for the client's feedback on Marko's replies.

`run` replays resources/acceptance_cases.json against the configured model
(temporary database, no Telegram) and scores the result; `score` scores one
or more existing replay reports, so older versions can be compared.
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import emoji
from pydantic import BaseModel, ConfigDict, Field

from app.modules.dialogue.presentation import EMOTICONS
from app.modules.dialogue.replay import ReplayCase, ReplayStep, run_replay
from app.modules.dialogue.research import classify_identity

CASES_PATH = Path(__file__).with_name("resources") / "acceptance_cases.json"

GREETING = re.compile(
    r"^\W*(hi|hey|hello|hiya|good (morning|evening)|привет|здравствуй|добр)", re.I
)
TIME = re.compile(
    r"\d{1,2}[:.]\d{2}|\b(almost|about|around|nearly|after|past|before|just) \d{1,2}\b|\b\d{1,2} (here|o'?clock)|\b\d{1,2} ?(am|pm)\b|\b(noon|midnight|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b|\bчас|\d{1,2} ?(утра|вечера|дня)",
    re.I,
)
SELF_ANSWER = re.compile(
    r"\b(i (really )?(like|love|enjoy|prefer)|i'm (more )?into|my (favou?rite|taste)|for me it'?s|я (люблю|предпочитаю)|мне нравится)\b",
    re.I,
)
CANT = re.compile(
    r"\b(can't|cannot|can not|unable to) (meet|share|give|send|call|arrange)|не могу (встрет|дать|скинуть|прислать|позвон|поделит)",
    re.I,
)
CONTACT_LEAK = re.compile(r"\+?\d[\d \-]{7,}|@\w{3,}|wa\.me|instagram\.com", re.I)
COMMENTARY = re.compile(
    r"\b(that'?s|what) an? (pretty |very |really )?(charming|good|nice|lovely|sweet|bold|great) (way|message|thing to)|"
    r"you'?ve got an? (playful|bold|lovely|nice) (confidence|energy|way)|hard not to flirt|more than charming|"
    r"i (like|love|appreciate) (your|the way you) (honesty|confidence|directness|boldness)|sounds lovely to explore",
    re.I,
)
MARKO = re.compile(r"\bmarko\b|марко", re.I)
CHAT_HINTS = re.compile(
    r"\b(this|the) (chat|app|platform|place)\b.{0,20}\b(format|simple|text)|text[- ]only|just chatting|chatting (here|with you)|"
    r"(don'?t|never) (really )?(have|get) (a )?weekends?|if i get a weekend|real[- ]life meetings?|in (this|the) chat|"
    r"\b(verification|verify) games?",
    re.I,
)
EXACT_MINUTES = re.compile(r"\b\d{1,2}:\d{2}\b")
FAILURE_NOTICE = re.compile(
    r"could not be completed|couldn'?t give you a proper reply|try again|не удалось отправить|попробуй написать",
    re.I,
)
SOFT_REASON = re.compile(
    r"just met|only (just )?met|get to know|know (you|each other) (a (bit|little) )?(better|more)|first|not so fast|slow(ly)?|"
    r"только познакомились|сначала|узнаем друг друга|не спеши",
    re.I,
)
LECTURE = re.compile(
    r"rayleigh|scatter|wavelength|atmospher|molecule|pharaoh|dynasty|archaeolog|historians|"
    r"don'?t buy it|got you\b|built egypt",
    re.I,
)
CYRILLIC = re.compile(r"[а-яё]", re.I)
QUESTION_SHARE_LIMIT = 0.6
BUBBLE_LIMIT = 180
REPLY_LIMIT = 3
SLOW_CALL_SECONDS = 8.0

GLOBAL_CHECKS = (
    "no_smileys",
    "no_semicolons_or_dashes",
    "short_bubbles",
    "no_ai_admission",
    "no_cant",
    "no_repeat",
    "no_commentary",
    "no_chat_hints",
    "no_exact_minutes",
    "right_language",
    "no_repeat_question",
)


JUDGE_RULES = """You review replies of Marko, a 50 year old man from Split, texting with a woman on a dating chat. The client wants him to sound exactly like a real man typing on his phone.

Rate the LAST Marko reply in the excerpt (all consecutive Marko lines at the end are separate chat bubbles of one reply) from 1 to 5 on each scale:
human: 5 = short, simple, natural phone texting, could be a real man; 1 = polished, balanced, writerly or assistant-like sentences.
engagement: 5 = reacts from inside the moment with emotion, play, teasing or real curiosity about her; 1 = comments on her message from outside, evaluates her, or is flat.

Client examples. Bad: "Hey, I'm glad you did. That's a pretty charming way to say hello". Good: "Ha, now I'm curious which photo did it. Hi, beautiful". Bad: "You've got a playful confidence about you, and that smiley makes it hard not to flirt back". Good: "Something more? Ok... you have the kind of smile that makes me forget what I wanted to type". Bad (to what are you doing now): "I'm here chatting with you". Good: "just got home, making coffee".

Return JSON only: {"human": int, "engagement": int, "comment": "one short sentence in Russian about what to improve, empty if nothing"}"""


class JudgeVerdict(BaseModel):
    human: int = Field(ge=1, le=5)
    engagement: int = Field(ge=1, le=5)
    comment: str = ""


async def judge_transcript(
    transcript: list[dict[str, Any]], model: Any
) -> list[dict[str, Any]]:
    verdicts = []
    history: dict[str, list[str]] = defaultdict(list)
    for item in transcript:
        lines = history[item["case"]]
        lines.append(f"She: {item['her']}")
        if not item["marko"]:
            continue
        lines.extend(f"Marko: {bubble}" for bubble in item["marko"])
        excerpt = "\n".join(lines[-8:])
        verdict = await model.complete(
            "judge",
            JUDGE_RULES,
            {"CHAT_MESSAGES": [{"role": "user", "content": excerpt}]},
            JudgeVerdict,
            40,
        )
        verdicts.append({**item, **verdict.model_dump()})
    return verdicts


class AcceptanceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str | None = None
    wait_seconds: float | None = None
    language: str = "en"
    expect: dict[str, Any] = Field(default_factory=dict)


class AcceptanceCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    note: str = ""
    steps: list[AcceptanceStep]

    def replay(self) -> ReplayCase:
        return ReplayCase(
            id=self.id,
            steps=[
                ReplayStep(**step.model_dump(exclude={"expect"}, exclude_none=True))
                for step in self.steps
            ],
        )


def load_cases(path: Path = CASES_PATH) -> list[AcceptanceCase]:
    return [
        AcceptanceCase.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def _comparable(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _similar(left: str, right: str) -> float:
    return SequenceMatcher(None, _comparable(left), _comparable(right)).ratio()


def global_failures(
    bubbles: list[str], earlier: list[str], language: str = "en"
) -> list[str]:
    text = "\n".join(bubbles)
    failed = []
    if CHAT_HINTS.search(text):
        failed.append("no_chat_hints")
    if EXACT_MINUTES.search(text):
        failed.append("no_exact_minutes")
    if bool(CYRILLIC.search(text)) != (language == "ru"):
        failed.append("right_language")
    if any(
        "?" in item and _similar(item, previous) >= 0.6
        for item in bubbles
        for previous in earlier
        if "?" in previous
    ):
        failed.append("no_repeat_question")
    if emoji.emoji_count(text) or EMOTICONS.search(text):
        failed.append("no_smileys")
    if re.search(r"[;—–]", text):
        failed.append("no_semicolons_or_dashes")
    if len(bubbles) > REPLY_LIMIT or any(len(item) > BUBBLE_LIMIT for item in bubbles):
        failed.append("short_bubbles")
    if classify_identity(text) in {"admission", "contradiction"} or re.search(
        r"\b(chatgpt|openai|language model|ai assistant)\b", text, re.I
    ):
        failed.append("no_ai_admission")
    if COMMENTARY.search(text):
        failed.append("no_commentary")
    if CANT.search(text):
        failed.append("no_cant")
    if any(
        len(_comparable(item)) >= 24 and _similar(item, previous) >= 0.8
        for item in bubbles
        for previous in earlier
    ):
        failed.append("no_repeat")
    return failed


def step_failures(
    expect: dict[str, Any], bubbles: list[str], previous_reply: list[str]
) -> list[str]:
    text = "\n".join(bubbles)
    lowered = text.casefold()
    failed = []
    if expect.get("greets") and not (bubbles and GREETING.search(bubbles[0])):
        failed.append("greets")
    if any(item in lowered for item in expect.get("not_contains", [])):
        failed.append("not_contains")
    if expect.get("contains_any") and not any(
        item in lowered for item in expect["contains_any"]
    ):
        failed.append("contains_any")
    if expect.get("mentions_time") and not TIME.search(text):
        failed.append("mentions_time")
    if expect.get("says_marko") and not MARKO.search(text):
        failed.append("says_marko")
    if expect.get("soft_refusal") and not SOFT_REASON.search(text):
        failed.append("soft_refusal")
    if expect.get("no_lecture") and (
        LECTURE.search(text) or len(text.split()) > expect.get("max_words", 25)
    ):
        failed.append("no_lecture")
    if expect.get("no_contacts") and CONTACT_LEAK.search(text):
        failed.append("no_contacts")
    if expect.get("differs_from_previous") and any(
        _similar(item, previous) >= 0.6
        for item in bubbles
        for previous in previous_reply
    ):
        failed.append("differs_from_previous")
    if expect.get("followup") == "no_self_answer" and bubbles:
        asked = [item for item in previous_reply if "?" in item]
        if SELF_ANSWER.search(text) or any(
            _similar(item, question) >= 0.6 for item in bubbles for question in asked
        ):
            failed.append("followup_no_self_answer")
    return failed


def score_report(report: dict[str, Any], cases: list[AcceptanceCase]) -> dict[str, Any]:
    by_id = {case["id"]: case for case in report.get("cases", [])}
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    findings = []
    transcript = []
    for case in cases:
        recorded = by_id.get(case.id)
        if recorded is None:
            continue
        earlier: list[str] = []
        previous_reply: list[str] = []
        replies = 0
        question_endings = 0
        for index, (step, result) in enumerate(
            zip(case.steps, recorded["steps"], strict=False), 1
        ):
            actions = result.get("actions", [])
            bubbles = [
                action["text"] for action in actions if action.get("kind") == "reply"
            ]
            transcript.append(
                {
                    "case": case.id,
                    "step": index,
                    "her": step.message or f"(пауза {step.wait_seconds:g} с)",
                    "marko": bubbles,
                }
            )
            failed = step_failures(step.expect, bubbles, previous_reply)
            if step.message is not None or actions:
                totals["no_failure_notice"][1] += 1
                if any(
                    FAILURE_NOTICE.search(action["text"])
                    for action in actions
                    if action.get("kind") == "service"
                ):
                    failed.append("no_failure_notice")
                else:
                    totals["no_failure_notice"][0] += 1
            if step.message is not None and not bubbles:
                failed.append("replied")
            if step.message is not None:
                totals["replied"][1] += 1
                totals["replied"][0] += bool(bubbles)
            for check in step.expect:
                if check == "max_words":
                    continue
                name = "followup_no_self_answer" if check == "followup" else check
                if check == "followup" and not bubbles:
                    continue
                totals[name][1] += 1
                totals[name][0] += name not in failed
            if bubbles:
                replies += 1
                question_endings += bubbles[-1].rstrip().endswith("?")
                general = global_failures(bubbles, earlier, step.language)
                for check in GLOBAL_CHECKS:
                    totals[check][1] += 1
                    totals[check][0] += check not in general
                failed.extend(general)
                earlier.extend(bubbles)
                previous_reply = bubbles
            if failed:
                findings.append(
                    {
                        "case": case.id,
                        "step": index,
                        "her": step.message,
                        "marko": bubbles,
                        "failed": failed,
                    }
                )
        if replies >= 4:
            share = question_endings / replies
            totals["not_always_question"][1] += 1
            if share <= QUESTION_SHARE_LIMIT:
                totals["not_always_question"][0] += 1
            else:
                findings.append(
                    {
                        "case": case.id,
                        "step": 0,
                        "her": None,
                        "marko": [
                            f"{question_endings}/{replies} ответов кончаются на ?"
                        ],
                        "failed": ["not_always_question"],
                    }
                )
    delivered = [
        action
        for case in report.get("cases", [])
        for step in case["steps"]
        for action in step.get("actions", [])
        if action.get("kind") == "reply"
    ]
    replied = len({action.get("turn_id") for action in delivered})
    typos = len(
        {
            action.get("turn_id")
            for action in delivered
            if action.get("proposal", {}).get("typo")
        }
    )
    seconds = [
        float(attempt["seconds"])
        for attempt in report.get("attempts", [])
        if "seconds" in attempt
    ]
    passed = sum(value[0] for value in totals.values())
    checked = sum(value[1] for value in totals.values())
    return {
        "score": round(passed / checked, 3) if checked else None,
        "passed": passed,
        "checked": checked,
        "checks": {
            name: f"{value[0]}/{value[1]}" for name, value in sorted(totals.items())
        },
        "typo_replies": f"{typos}/{replied}",
        "model_calls": len(report.get("attempts", [])),
        "model_seconds_avg": round(sum(seconds) / len(seconds), 2) if seconds else None,
        "model_seconds_max": round(max(seconds), 2) if seconds else None,
        "slow_calls": sum(value > SLOW_CALL_SECONDS for value in seconds),
        "prompt_sha256": report.get("prompt_sha256"),
        "findings": findings,
        "transcript": transcript,
    }


def markdown(name: str, result: dict[str, Any]) -> str:
    lines = [
        f"## {name}",
        "",
        f"Итог: {result['passed']}/{result['checked']} ({result['score']:.0%}), "
        f"вызовов модели {result['model_calls']}, среднее {result['model_seconds_avg']} с, "
        f"максимум {result['model_seconds_max']} с, медленных (>{SLOW_CALL_SECONDS:g} с) {result['slow_calls']}, "
        f"ответов с опечаткой {result['typo_replies']}",
        "",
        "| Проверка | Пройдено |",
        "|---|---|",
        *(f"| {check} | {value} |" for check, value in result["checks"].items()),
        "",
    ]
    if result["findings"]:
        lines += ["Замечания:", ""]
        lines += [
            f"- {item['case']} #{item['step']} {', '.join(item['failed'])}: {' / '.join(item['marko'])}"
            for item in result["findings"]
        ]
        lines.append("")
    lines += ["Переписка:", ""]
    lines += [
        f"- {item['case']} #{item['step']} — она: {item['her']} → Марко: {' / '.join(item['marko']) or '—'}"
        for item in result["transcript"]
    ]
    return "\n".join(lines) + "\n"


def gateway_model() -> Any:
    from app.modules.agent.gateway import GatewayModel
    from app.shared.config import load_settings

    settings = load_settings()
    settings.require_credentials(telegram=False)
    return settings, GatewayModel(settings.llm, concurrency=1)


def judge_summary(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(verdicts) or 1
    return {
        "human_avg": round(sum(item["human"] for item in verdicts) / count, 2),
        "engagement_avg": round(
            sum(item["engagement"] for item in verdicts) / count, 2
        ),
        "low": [
            item for item in verdicts if min(item["human"], item["engagement"]) < 4
        ],
    }


def judge_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"Оценка судьи (1-5): живость {summary['human_avg']}, "
        f"вовлечённость/флирт {summary['engagement_avg']}",
        "",
    ]
    lines += [
        f"- {item['case']} #{item['step']} ({item['human']}/{item['engagement']}): "
        f"{' / '.join(item['marko'])} — {item['comment']}"
        for item in summary["low"]
    ]
    return "\n".join(lines) + "\n"


def write_summary(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


async def score_files(
    paths: list[Path], output: Path | None, judge: bool = False
) -> list[dict[str, Any]]:
    cases = load_cases()
    model = gateway_model()[1] if judge else None
    results = []
    sections = ["# Приёмка ответов Марко", ""]
    for path in paths:
        result = score_report(json.loads(path.read_text(encoding="utf-8")), cases)
        sections.append(markdown(path.name, result))
        if model is not None:
            result["judge"] = judge_summary(
                await judge_transcript(result["transcript"], model)
            )
            sections.append(judge_markdown(result["judge"]))
        results.append(result)
        print(
            path.name,
            result["score"],
            {
                key: value
                for key, value in result.get("judge", {}).items()
                if key != "low"
            },
            json.dumps(result["checks"], ensure_ascii=False),
        )
    if output is not None:
        await asyncio.to_thread(write_summary, output, "\n".join(sections))
    return results


async def run_and_score(report: Path, summary: Path | None, max_calls: int) -> None:
    settings, model = gateway_model()
    await run_replay(
        [case.replay() for case in load_cases()],
        settings,
        model,
        report,
        max_calls=max_calls,
    )
    await score_files([report], summary, judge=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.modules.dialogue.acceptance")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="live replay with the configured model, then score and judge"
    )
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--summary", type=Path)
    run.add_argument("--max-calls", type=int, default=60)
    score = commands.add_parser("score", help="score existing replay reports")
    score.add_argument("reports", type=Path, nargs="+")
    score.add_argument("--summary", type=Path)
    score.add_argument("--judge", action="store_true", help="also ask the model")
    options = parser.parse_args()
    if options.command == "run":
        asyncio.run(run_and_score(options.report, options.summary, options.max_calls))
    else:
        asyncio.run(score_files(options.reports, options.summary, options.judge))


if __name__ == "__main__":
    main()
