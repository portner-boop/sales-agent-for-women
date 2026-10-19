import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> str:
    if not values:
        return "—"
    ordered = sorted(values)
    return f"{ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]:.2f}"


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Сравнение режимов Марко",
        "",
        f"Каталог: `{report['catalog']['version']}`, SHA256 `{report['catalog']['sha256']}`.",
        f"Модель: `{report.get('model', 'без live-прогона')}`. QMD: {report['qmd_version']}, {report['qmd_mode']}.",
        "",
        f"Остановка/ограничение: {report.get("interruption", "не зафиксировано")}",
        "",
        "## Подбор правил",
        "",
        "Python выбирает карточки по уже полученным policy_topics Interpreter и серверным признакам recall/time/followup/commerce. Добавляет обязательное ядро и зависимости. Для неизвестной ситуации возвращает полный компактный каталог. Дополнительного запроса к LLM для выбора нет. Проверяющий всегда получает полный каталог той же версии.",
        "",
        "| Режим | p50 / p95, мс | Средние токены правил | Попадание до fallback | Покрытие после fallback | Fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, data in report["retrieval"]["summary"].items():
        latency = data["selection_seconds"]
        lines.append(
            f"| {mode} | {latency['p50'] * 1000:.3f} / {latency['p95'] * 1000:.3f} | {data['mean_policy_tokens']:.0f} | {data.get('raw_retrieval_hit_rate', 0):.1%} | {data['rule_coverage']:.1%} | {data['fallback_count']} |"
        )
    retrieval = report["retrieval"]
    cold = retrieval.get("cold_qmd_seconds")
    if cold is not None:
        lines += [
            "",
            f"Первый запуск QMD: {cold * 1000:.1f} мс. Пиковый RSS дочерних процессов QMD на macOS: {retrieval['peak_qmd_child_rss_bytes_macos'] / 1024**2:.1f} МиБ. Это процессы CLI, не постоянно работающий сервер. Прогретое измерение использует кеш ОС/индекса и новый CLI-процесс на каждый запрос. GGUF-модели, embeddings, expansion и reranker не загружались.",
        ]
    lines += [
        "",
        "## Диалоги",
        "",
        "| Режим | Диалоги / незавершённые | Ответы / входы | Обработка p50 / p95, с | Входные / выходные токены | Попытки с repair / все попытки |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    grades = []
    for mode in report["retrieval"]["summary"]:
        cases = [case for case in report["dialogues"] if case["mode"] == mode]
        completed = [case for case in cases if "transcript" in case]
        turns = [
            turn for case in completed for turn in case["transcript"] if "user" in turn
        ]
        runs = [run for case in completed for run in case["runs"]]
        events = [
            event for run in runs for event in run["metrics"].get("llm_events", [])
        ]
        times = [
            float(run["metrics"]["elapsed_seconds"])
            for run in runs
            if "elapsed_seconds" in run["metrics"]
        ]
        repaired = sum(
            any(
                event["task"] in {"repair", "recover"}
                for event in run["metrics"].get("llm_events", [])
            )
            for run in runs
        )
        lines.append(
            f"| {mode} | {len(completed)} / {len(cases) - len(completed)} | {sum(bool(turn['replies']) for turn in turns)} / {len(turns)} | {percentile(times, .5)} / {percentile(times, .95)} | {sum(event.get('input_tokens', 0) for event in events):,} / {sum(event.get('output_tokens', 0) for event in events):,} | {repaired} / {len(runs)} |"
        )
        if completed:
            graded = [case for case in completed if isinstance(case.get("grade"), dict)]
            counts = {
                key: sum(case["grade"][key] is False for case in graded)
                for key in (graded[0]["grade"] if graded else {})
                if key != "reasons"
            }
            grades.append(
                f"- {mode}: "
                + "; ".join(
                    f"{key} — {value}/{len(graded)}" for key, value in counts.items()
                )
            )
            lengths = [len(reply) for turn in turns for reply in turn["replies"]]
            silence = [
                turn
                for case in completed
                for turn in case["transcript"]
                if turn.get("trigger") == "silence"
            ]
            grades.append(
                f"  Длина части: медиана {statistics.median(lengths) if lengths else 0:.0f} символов; follow-up доставлен в {sum(bool(turn['replies']) for turn in silence)}/{len(silence)} проверенных silence-событий. Отсутствие follow-up допустимо при незавершённом вопросе или границе."
            )
    lines += [
        "",
        "Задержка обработки выше — одна попытка модели с проверками, включая неудачные попытки и follow-up. Она не является временем получения ответа клиентом. Паузы набора и ожидание retry ускорены в тесте; время LLM реально. Telegram заменён тестовым отправителем. Токены относятся к обработке диалога, без финального LLM-оценщика; денежная стоимость не рассчитана.",
        "",
        "## Замечания автоматического оценщика",
        "",
        "Числа показывают диалоги с отрицательным флагом по критерию, а не измеренную точность. Оценщик использует ту же модель и может ошибаться; оригинальные причины и переписки сохранены в JSON.",
        "",
        *grades,
        "",
        "## Ограничения",
        "",
        "Один последовательный сценарий на язык, EN/RU/ES/FR/AR, до трёх повторов на режим. Это небольшая выборка, без статистического доказательства превосходства. Режимы могут отличаться случайными ответами и нагрузкой провайдера. Дополнительные проверки рестарта, доставки, отмены, границ и памяти выполняются тестами приложения; live-матрица не заменяет их.",
        "",
        "После установки новая версия собирает фазовые задержки, токены, ошибки, выбранные ID/версию правил и события проверок в agent_runs. Пауза набора хранится отдельно от транспорта. Личные сообщения не нужны для агрегированного отчёта метрик. QMD остаётся изолированным прототипом; преимуществ для выпуска этот поиск не доказывает.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(json.loads(args.report.read_text())), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
