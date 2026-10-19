import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def local_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, ZoneInfo("Europe/Moscow")).isoformat(
        timespec="milliseconds"
    )


def replay_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def finish() -> None:
        if pending:
            steps.append(
                {
                    "text": "\n".join(item["text"] for item in pending),
                    "sequences": [item["sequence"] for item in pending],
                    "received_at": pending[0]["received_at"],
                }
            )
            pending.clear()

    for message in messages:
        if message["role"] == "user" and not message["text"].startswith("/"):
            pending.append(message)
        else:
            finish()
    finish()
    return steps


def render_export(data: dict[str, Any], source_hash: str) -> str:
    lines = [
        "# Серверные диалоги для регрессии Марко",
        "",
        f"Экспорт: {data['exported_at']}. SHA256 исходного JSON: `{source_hash}`.",
        "Время ниже — Europe/Moscow (UTC+03:00). Реплики сохранены дословно, включая ошибки и повторы. Сервисные сообщения явно отделены. Это локальная копия исходной переписки с личными данными; не публичная обезличенная фикстура.",
        "",
        "Для replay соседние сообщения пользователя до следующего ответа объединяются. Исходная гонка редактирования/доставки этим не воспроизводится. Повторный диалог использует новые ответы: короткое «да» может ссылаться на формулировку исходного бота, поэтому оценки требуют просмотра контекста.",
    ]
    for case in data["conversations"]:
        lines += [
            "",
            f"## {case['label']}",
            "",
            f"Источник: `{case.get('source', data['source'])}`; scope `{case['scope_id']}`; epoch `{case['messages'][0]['epoch']}`.",
            "",
            "### Подтверждённые наблюдения",
            "",
        ]
        for incident in case.get("incidents", []):
            lines.append(
                f"- {incident['description']} Реплики: {', '.join('#' + str(number) for number in incident['sequences'])}."
            )
        lines += ["", "### Переписка", ""]
        pending_at = None
        for message in case["messages"]:
            role = {"user": "Клиент", "assistant": "Марко", "service": "Сервис"}[
                message["role"]
            ]
            suffix = ""
            if message["role"] == "user":
                if pending_at is None:
                    pending_at = message["received_at"]
            elif message["role"] == "assistant" and pending_at is not None:
                suffix = (
                    f" · первый ответ через {message['received_at'] - pending_at:.2f} с"
                )
                pending_at = None
            lines += [
                f"#### #{message['sequence']} · {role} · {local_timestamp(message['received_at'])}{suffix}",
                "",
                *("> " + line for line in message["text"].split("\n")),
                "",
            ]
        lines += [
            "### Диагностика выполнения",
            "",
            "| Начало (Москва) | Статус | Ошибка |",
            "|---|---|---|",
        ]
        for run in case["runs"]:
            lines.append(
                f"| {local_timestamp(run['started_at'])} | {run['status']} | {run['error_code'] or '—'} |"
            )
        steps = replay_steps(case["messages"])
        lines += [
            "",
            f"Пользовательских сообщений: {sum(message['role'] == 'user' for message in case['messages'])}; входов replay после объединения: {len(steps)}.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    data = json.loads(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    cases = {
        "source_sha256": checksum,
        "source": str(args.source),
        "cases": [
            {
                "id": case["case_id"],
                "label": case["label"],
                "language": "en",
                "requirements": case["requirements"],
                "steps": replay_steps(case["messages"]),
            }
            for case in data["conversations"]
        ],
    }
    for path, text in (
        (args.markdown, render_export(data, checksum)),
        (args.cases, json.dumps(cases, ensure_ascii=False, indent=2) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
    print(
        json.dumps(
            {
                "markdown": str(args.markdown),
                "cases": str(args.cases),
                "conversations": len(cases["cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
