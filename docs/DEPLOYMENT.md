# Самостоятельное развёртывание

Для использования готового сервиса достаточно [README](../README.md).
Этот документ предназначен для установки собственной копии. Команды выполняются
из корня репозитория.

## Варианты запуска

| Вариант | OCR | Где работает | Что нужно |
|---|---|---|---|
| Сервис vines.aiweapps.com | Apple Vision | Нативный распознаватель на Mac | Только браузер или HTTP-клиент |
| Самостоятельный Docker (Linux CPU) | PP-OCRv6 Medium | Контейнеры `core` + `gateway-linux` | Этот репозиторий и пакет активов |
| Шлюз к Mac-распознавателю | Apple Vision | Docker-шлюз + нативный процесс на Mac | Нативный распознаватель и пакет карточек |

Модели поиска бутылок, сравнения этикеток и выбора товара одинаковы во всех
вариантах. Отличается компонент чтения текста.
На повторном контроле из 69 фото Apple Vision дал 65/69 верных первых карточек,
PP-OCRv6 — 63/69. Условия этого сравнения и остальные результаты — в [отчёте о качестве](QUALITY.md).

## Самостоятельный запуск в Docker (Linux, PP-OCRv6)

Для самостоятельного запуска нужны этот репозиторий и **отдельный пакет моделей
и каталога** (около 5,3 ГБ). Запросите пакет у команды AiweApps вместе со значениями
для `.env`. Веса и фотографии не включены в Git и Docker-образ. Для использования
готового сервиса пакет не требуется.

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
Устройство ядра и проверки пакета — [CORE](CORE.md).

## Шлюз к Mac-распознавателю (Apple Vision)

Этот вариант подключает Docker-шлюз к уже подготовленному распознавателю на macOS.
Шлюз отдаёт интерфейс и API, распознаватель с Apple Vision принимает запросы на
порту 8175 (`/health/ready`, `/v1/recognize`). Для самостоятельного запуска целиком
из этого репозитория используйте Linux-вариант выше. Готовый публичный сервис
также использует Apple Vision.

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

## Проверка локального API

Для локального запуска официальный скрипт вызывается напрямую, без авторизации:

```bash
bash evaluation/organizer/participant_test.sh \
  --images-dir /path/to/queries --manifest /path/to/queries.tsv \
  --endpoint http://127.0.0.1:8387/v1/eval/predict \
  --output predictions-local.jsonl
```

Порт 8387 — `gateway-linux`, 8287 — шлюз к Mac.

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

