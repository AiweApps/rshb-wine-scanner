# Сканер винных этикеток

Сервис находит вино по фотографии бутылки или этикетки и возвращает карточку
из каталога «Своё вино»: название, характеристики, эталонное фото и ссылку на товар.
На снимке с несколькими бутылками можно выбрать нужную.

**[Открыть сервис](https://wines.aiweapps.com)** ·
[Мобильное приложение](https://github.com/AiweApps/rshb-hack-flutter)

## Попробовать в браузере

Откройте сервис, загрузите фотографию и выберите найденную бутылку.
Регистрация не нужна. Поддерживаются JPEG, PNG и WebP до 20 МиБ и 24 Мп.

## HTTP API

Адрес: `https://wines.aiweapps.com`.
Для приложения или скрипта получите гостевой токен, затем отправьте фото:

```bash
TOKEN=$(curl --fail -sS -X POST https://wines.aiweapps.com/auth/guest/token | jq -r .access_token)

curl --fail -H "Authorization: Bearer $TOKEN" -F 'image=@photo.jpg' \
  https://wines.aiweapps.com/api/recognize
```

В ответе `view.bottles` находятся карточки распознанных бутылок и альтернативы.
У карточки есть `slug`, `title`, `producer`, `page_url` и `reference`.
`page_url` — ссылка на товар в «Своём вине», `reference` — путь к эталонному фото.
Токен действует 24 часа.

Для проверки с одним ответом используйте `POST /v1/eval/predict` с тем же
полем `image` и токеном. Ответ: `{"slug":"идентификатор-товара"}` или
`{"slug":null}`, если сервис не выбрал одну главную цель.

[Полное описание API, авторизации и ошибок](docs/API.md).

## Оценщик организатора

Из корня репозитория запустите проверку своей подборки:

```bash
bash evaluation/organizer/participant_remote_test.sh \
  --images-dir /path/to/queries \
  --manifest /path/to/queries.tsv \
  --output predictions.jsonl
```

Скрипт сам получает токен и обращается к `https://wines.aiweapps.com`.
Нужны bash, curl, jq и awk.

В `queries.tsv` две колонки через табуляцию. Пути к фото указаны относительно
`--images-dir`:

```text
query_id	image_path
q-001	photo-001.jpg
q-002	photo-002.webp
```

Результат — JSONL: одна строка на фото с полями `query_id`, `image_path`,
`image_sha256`, `predicted_slug`, `latency_ms`. Посмотреть его:

```bash
jq . predictions.jsonl
```

Скрипт сохраняет ответы и время; точность считают отдельно по правильным ответам.
Лимит — 10 секунд на фото. При ошибке или таймауте ответ также будет `null`.
Существующий выходной файл не перезаписывается.

## Документация

- [Архитектура](ARCHITECTURE.md) — как система находит этикетку и выбирает товар.
- [Результаты проверки](docs/QUALITY.md) — качество, скорость и условия замеров.
- [Самостоятельный запуск](docs/DEPLOYMENT.md) — установка своей копии с отдельным пакетом моделей и каталога.
- [Проверка сервиса](docs/RELEASE.md) — короткий маршрут демонстрации.
