# Восстановление Luna: Responses API

## Подтверждённая проблема

23 октября текущий GatewayModel явно выбирает Chat Completions. Синтетический запрос к настроенному шлюзу с `verbosity=low` возвращает HTTP 400: параметр перенесён в `text.verbosity` Responses API. Запрос Chat Completions без verbosity также отклонён: шлюз сообщает об отсутствии слова JSON во входе, хотя системное сообщение содержит JSON. Это указывает на несовместимость преобразования протоколов; исходники провайдера недоступны.

Прямой запрос к `/responses` с `text.verbosity=low`, `text.format.type=json_object`, `reasoning.effort=low`, `store=false` успешно завершился и вернул ожидаемый JSON. Все диагностические входы синтетические, данные клиентов и Telegram не используются. Исходные результаты: `reports/server-incident-20261023/verbosity-baseline.json`, `chat_without_verbosity.json`, `responses_text_verbosity.json`.

## Решение и контракты

- GatewayModel сохраняет интерфейс `complete`, ограничение параллелизма, таймауты, исходные роли полной истории, метрики, псевдонимы ссылок и три существующих режима структурированного результата.
- ChatOpenAI явно использует Responses API. Установленная версия адаптера переносит verbosity в text.verbosity, reasoning_effort в reasoning.effort, лимит токенов в max_output_tokens и response_format в text.format. Ручное смешение полей двух протоколов не требуется.
- `output_version=responses/v1` сохраняет блоки Responses. JSON-парсер получает их исходный текст через AIMessage.text без редактирования; диагностическое raw_response остаётся строкой, а raw_response_blocks сохраняет исходные блоки, включая отказ и reasoning. `store=false` отключает сохранение Response у провайдера. Каждый ход отправляет полную историю; previous_response_id и серверное продолжение не используются. Проверка установленного адаптера показала, что v0 тоже возвращает список текстовых блоков и не решает эту задачу.
- Успех определяется статусом Responses `completed`. `incomplete` с причиной `max_output_tokens` становится TRUNCATED_OUTPUT; остальные незавершённые статусы — INCOMPLETE_OUTPUT. Они не доходят до доставки как успешный ответ. Существующие ограниченные повторы и сервисное уведомление об ошибке сохраняются.
- Настройки модели, verbosity, reasoning, промпты, память, миграции, Telegram и Xray не меняются. Дополнительных модельных вызовов или скрытого fallback между API нет.

## Последовательность

1. Изменить транспорт и проверку завершённости GatewayModel.
2. Перевести HTTP-mock тесты на реальный формат Responses, проверять URL, вложенность параметров, роли, необработанный текст, usage и три режима JSON. Проверить обрезанный и незавершённый ответы, отказ и повреждённый JSON.
3. Запустить целевые и полные тесты, Ruff, Black, mypy, diff-check.
4. Выполнить ограниченный живой прогон GatewayModel на синтетических данных без Telegram и рабочей БД.
5. Зафиксировать результат и подготовить минимальный patch для серверного релиза. Применение на сервере требует восстановления SSH-авторизации: сервер отвечает через разрешённый прокси, но текущий пароль root отклонён. Не заявлять о серверном исправлении до реального применения и проверки.

## Источники

- [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create): text.verbosity, text.format, status, incomplete_details, store.
- [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create): top-level verbosity допустим в этом протоколе.
- Установленные langchain-openai 1.7.0 и openai 3.26.1: локальная реализация `_construct_responses_api_payload` и `_construct_lc_result_from_responses_api`.
