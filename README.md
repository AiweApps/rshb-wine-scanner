# Сканер винных этикеток

Фото бутылки → карточка вина из каталога «Своё вино», эталонное изображение и ссылка
на товар. Распознавание локальное: компьютерное зрение, OCR и ранжировщик по каталогу,
без внешних OCR/LLM API.

- Сервис: **https://vines.aiweapps.com** — веб-интерфейс и HTTP API, гостевой вход без регистрации.
- Мобильное приложение: https://github.com/AiweApps/rshb-hack-flutter.
- Исходники: https://github.com/AiweApps/rshb-wine-scanner.

## Варианты

| Вариант | OCR | Где работает | Что нужно |
|---|---|---|---|
| Сервис vines.aiweapps.com | Apple Vision | Нативный распознаватель на Mac | Только браузер или HTTP-клиент |
| Самостоятельный Docker (Linux CPU) | PP-OCRv6 Medium | Контейнеры `core` + `gateway-linux` | Этот репозиторий и пакет активов |
| Шлюз к Mac-распознавателю | Apple Vision | Docker-шлюз + нативный процесс на Mac | Нативный распознаватель и пакет карточек |

Детектор, визуальный поиск B3 и ранжировщик одинаковы во всех вариантах, отличается OCR.
На повторном контроле из 69 фото Apple Vision дал 65/69 верных первых карточек,
PP-OCRv6 — 63/69. Это небольшое измеренное преимущество на одном наборе, а не
гарантия для любых фото. Подробнее — [качество](docs/QUALITY.md).

## Использование сервиса

В браузере откройте https://vines.aiweapps.com: загрузите фото, выберите бутылку,
получите карточку, эталон и ссылку. Гостевая сессия создаётся автоматически.

Из приложения или скрипта:

```bash
TOKEN=$(curl -s -X POST https://vines.aiweapps.com/auth/guest/token | jq -r .access_token)

curl -H "Authorization: Bearer $TOKEN" -F 'image=@photo.jpg' \
  https://vines.aiweapps.com/api/recognize        # цели, карточки, альтернативы

curl -H "Authorization: Bearer $TOKEN" -F 'image=@photo.jpg' \
  https://vines.aiweapps.com/v1/eval/predict      # {"slug":"…"} или {"slug":null}
```

Токен действует 24 часа, логин и пароль не нужны. Маршруты, ошибки и лимиты —
[API](docs/API.md).

### Оценщик организатора

Официальный `evaluation/organizer/participant_test.sh` не передаёт заголовок
авторизации. Обёртка сама получает один гостевой токен и запускает этот же скрипт
против публичного сервиса:

```bash
bash evaluation/organizer/participant_remote_test.sh \
  --images-dir /path/to/queries \
  --manifest /path/to/queries.tsv \
  --output predictions-remote.jsonl
```

Нужны bash, curl, jq и awk. Адрес по умолчанию — `https://vines.aiweapps.com`,
другой задаётся `--origin https://HOST[:PORT]`. Токен хранится только во временном
файле с правами 0600 и не попадает в аргументы, вывод и URL. Существующий `--output`
не перезаписывается.

`queries.tsv` — две колонки через табуляцию, пути относительно `--images-dir`:

```text
query_id	image_path
q-001	photo-001.jpg
q-002	photo-002.webp
```

Фото отправляются по одному multipart-полем `image`, лимит 10 с на фото и 5 с на
соединение, без повторов. В `predictions.jsonl` — `query_id`, `image_path`,
`image_sha256`, `predicted_slug`, `latency_ms`. При таймауте, ошибке или отсутствии
главной цели `predicted_slug` равен `null`. Скрипт не считает точность.

Для локального запуска официальный скрипт вызывается напрямую, без авторизации:

```bash
bash evaluation/organizer/participant_test.sh \
  --images-dir /path/to/queries --manifest /path/to/queries.tsv \
  --endpoint http://127.0.0.1:8387/v1/eval/predict \
  --output predictions-local.jsonl
```

Порт 8387 — `gateway-linux`, 8287 — шлюз к Mac.

## Самостоятельный запуск в Docker (Linux, PP-OCRv6)

Код ядра и распознавателя входит в репозиторий. **Веса моделей, галерея эталонов,
индекс и каталог в Git и образ не входят** — без пакета активов распознавание не
запустится. Пакет выдаётся отдельно (около 5,3 ГБ) вместе со значениями для `.env`.

Требования: Docker с Compose, для контейнеров не меньше 2 CPU и больше 12 ГБ памяти
(проверено с 16 ГБ в Docker Desktop). Проверено на Linux/ARM64; x86_64 не проверялся.

Структура пакета на хосте:

```text
rshb-wine-scanner-assets/
  core/                       # веса, индекс, каталог, профили
    manifest.json
    tree/{config,data,models,runs}/
  web-core/                   # карточки и миниатюры для интерфейса
    manifest.json
    cards.json
    thumbnails/
```

```bash
cp .env.example .env
```

В `.env` заполните блок Linux:

```dotenv
WINE_SCANNER_CORE_ASSETS_HOST_DIR=/path/to/rshb-wine-scanner-assets/core
WINE_SCANNER_CORE_ASSETS_SHA256=<sha256 core/manifest.json>
WINE_SCANNER_CORE_EXPECTED=<дескриптор ядра>
WINE_SCANNER_LINUX_ASSETS_HOST_DIR=/path/to/rshb-wine-scanner-assets/web-core
WINE_SCANNER_LINUX_ASSETS_SHA256=<sha256 web-core/manifest.json>
```

Compose проверяет обязательные переменные всех сервисов, включая неиспользуемый
здесь `gateway`. Если Mac-вариант не нужен, продублируйте в них значения Linux:

```dotenv
WINE_SCANNER_ASSETS_HOST_DIR=/path/to/rshb-wine-scanner-assets/web-core
WINE_SCANNER_ASSETS_SHA256=<то же, что WINE_SCANNER_LINUX_ASSETS_SHA256>
WINE_SCANNER_EXPECTED_PROFILE=<то же, что WINE_SCANNER_CORE_EXPECTED>
```

```bash
docker compose --profile linux build core gateway-linux
docker compose --profile linux up -d core gateway-linux
curl --fail http://127.0.0.1:8375/health/ready   # ядро; первая готовность 3,5–5 мин
curl --fail http://127.0.0.1:8387/health/ready   # шлюз
```

Интерфейс: http://127.0.0.1:8387/. Проверка одним фото:

```bash
curl --fail -F 'image=@photo.jpg' http://127.0.0.1:8387/v1/eval/predict
```

Остановка: `docker compose --profile linux down`. Порты публикуются только на
127.0.0.1 и меняются через `WINE_SCANNER_CORE_PUBLISH_PORT` и
`WINE_SCANNER_LINUX_PUBLISH_PORT`. Дескриптор ядра можно пересчитать без загрузки
моделей: `docker compose --profile linux run --rm --no-deps core describe`.
Устройство ядра и проверки пакета — [CORE](docs/CORE.md).

## Шлюз к Mac-распознавателю (Apple Vision)

Так работает сервис vines.aiweapps.com. Docker-контейнер `gateway` отдаёт интерфейс
и API, а распознавание выполняет нативный процесс на macOS с Apple Vision на порту
8175 (`/health/ready`, `/v1/recognize`). Этот процесс и его активы в репозитории не
запускаются одной командой: нужна подготовленная Mac-среда распознавателя.

При доступном распознавателе заполните в `.env` основной блок:

```dotenv
WINE_SCANNER_ASSETS_HOST_DIR=/path/to/web-assets
WINE_SCANNER_ASSETS_SHA256=<sha256 manifest.json>
WINE_SCANNER_EXPECTED_PROFILE=<контрольная сумма профиля распознавателя>
WINE_SCANNER_BACKEND=http://host.docker.internal:8175
```

```bash
docker compose up -d --build gateway
curl --fail http://127.0.0.1:8287/health/ready
```

Без Docker, Python 3.12+:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export WINE_SCANNER_ASSETS_DIR=/path/to/web-assets
export WINE_SCANNER_ASSETS_SHA256='MANIFEST_SHA256'
export WINE_SCANNER_EXPECTED_PROFILE='PROFILE_CHECKSUM'
.venv/bin/python -m wine_scanner          # распознаватель по умолчанию http://127.0.0.1:8175
```

Пакет карточек создаётся из исходной среды распознавателя через
`tools/export_web_assets.py`; для ядра Linux он привязывается к дескриптору через
`tools/bind_web_assets.py`. Получателю готового пакета эти шаги не нужны.

## Возможности и ограничения

- JPEG, PNG, WebP до 20 МиБ и 24 Мп. HEIC/MPO сначала преобразуйте в JPEG.
- Несколько бутылок на фото: рамки, выбор нужной или своей области (`target_roi`).
- Основная карточка и альтернативы с эталоном и ссылкой на каталог.
- Распознаются вина каталога; для неизвестного вина надёжный отказ не гарантирован.
- Сходство не является вероятностью: `probability` всегда `null`, уверенность не откалибрована.
- Год урожая на карточке не всегда совпадает с годом на бутылке.
- Загруженные фото не сохраняются как коллекция и не используются для обучения.

## Если что-то не работает

| Симптом | Причина и действие |
|---|---|
| `required variable … is missing a value` | Заполните и блок `gateway` в `.env`, см. запуск Linux выше |
| `ready:false`, 503 после старта | Идёт проверка пакета и загрузка моделей; ядро Linux готовится несколько минут |
| Контейнер не стартует, ошибка SHA/профиля | Значения `.env` не от этого пакета; повреждённый или изменённый пакет не запускается |
| `core` падает по памяти | Выделите Docker больше 12 ГБ; лимит контейнера — 12 ГБ |
| 503 с `Retry-After` | Очередь занята: один запрос в работе и до двух ожидают; повторите позже |
| 504, `null` в оценщике | Отдельные сложные сцены дольше 10 с, особенно на 2 CPU |
| 413 / 415 / 422 | Слишком большое фото, неподдерживаемый формат или неверная `target_roi` |
| 401 на сервисе | Токен истёк — получите новый; 429 — подождите по `Retry-After` |

## Документация

[Архитектура](ARCHITECTURE.md) · [API](docs/API.md) · [Ядро Linux](docs/CORE.md) ·
[Качество](docs/QUALITY.md) · [Выпуск и демонстрация](docs/RELEASE.md) ·
[История](docs/DEVELOPMENT.md)

| Путь | Назначение |
|---|---|
| `wine_scanner/` | HTTP-шлюз: API, очередь, проверка входа, карточки, интерфейс (`static/`) |
| `wine_scanner_core/` | Ядро Linux: проверка пакета, CPU/PP-OCRv6-адаптер, HTTP |
| `recognizer/` | Serving-код распознавателя и журнал происхождения `SOURCES.json` |
| `docker/`, `Dockerfile`, `compose.yaml` | Образы и запуск с закреплёнными версиями |
| `tools/` | Экспорт пакета карточек и привязка к ядру |
| `evaluation/organizer/` | Скрипт организатора и обёртка для публичного сервиса |
