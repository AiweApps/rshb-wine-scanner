"""Public request limits, validation of uploads and target frames, and the error shape of every refusal."""
from io import BytesIO
import json
import math
import re

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000
MULTIPART_SLACK = 256 * 1024
FORMATS = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}
HEIF_BRANDS = (b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis', b'mif1', b'msf1', b'avif', b'avis')
DECISIONS = {
    'unknown': 'Не удалось выбрать совпадение',
    'insufficient_evidence': 'Недостаточно признаков для ответа',
    'ambiguous_target': 'На фото несколько бутылок — выберите нужную',
    'no_target': 'Бутылка с этикеткой не найдена',
}
CODE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
SLUG = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,200}$')


class Rejected(Exception):
    def __init__(self, status, code, message, retry_after=None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after

    def body(self):
        body = {'error': self.code, 'message': self.message}
        if self.retry_after is not None:
            body['retry_after'] = self.retry_after
        return body


def code(value):
    """A backend enum value as-is when it is a plain code; anything else never reaches a public answer."""
    return value if isinstance(value, str) and CODE.match(value) else None


def codes(values):
    return [v for v in values[:32] if code(v)] if isinstance(values, list) else []


def slug(value):
    return value if isinstance(value, str) and SLUG.match(value) else None


def box(value):
    if (isinstance(value, list) and len(value) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        return value
    return None


def inspect_image(data, filename):
    if len(data) > MAX_BYTES:
        raise Rejected(413, 'too_large', 'Файл больше 20 МиБ')
    if not data:
        raise Rejected(422, 'empty', 'Пустой файл')
    if data[4:8] == b'ftyp' and data[8:12] in HEIF_BRANDS or (filename or '').lower().endswith(('.heic', '.heif')):
        raise Rejected(415, 'heic_unsupported', 'HEIC/HEIF в первом выпуске не поддерживается: '
                       'сохраните фото как JPEG (на iPhone: Настройки → Камера → Форматы → «Наиболее совместимый»)')
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(BytesIO(data)) as image:
            fmt, (width, height) = image.format, image.size
            orientation = image.getexif().get(0x0112, 1)
    except Image.DecompressionBombError:
        raise Rejected(413, 'too_many_pixels', 'Изображение больше 24 Мп')
    except (UnidentifiedImageError, OSError, ValueError):
        raise Rejected(415, 'unsupported_format', 'Не удалось прочитать изображение: нужен JPEG, PNG или WebP')
    if fmt not in FORMATS:
        raise Rejected(415, 'unsupported_format', f'Формат {fmt} не поддерживается: нужен JPEG, PNG или WebP')
    if width * height > MAX_PIXELS:
        raise Rejected(413, 'too_many_pixels', f'Изображение {width}×{height} больше 24 Мп')
    size = [height, width] if orientation in (5, 6, 7, 8) else [width, height]
    return {'format': fmt, 'mime': FORMATS[fmt], 'size': size, 'bytes': len(data)}


def parse_roi(text, size):
    """[x1, y1, x2, y2] in EXIF-oriented original pixels, forwarded unchanged; outside the frame is refused."""
    if text in (None, ''):
        return None
    try:
        roi = json.loads(text)
    except ValueError:
        raise Rejected(422, 'bad_roi', 'Рамка должна быть JSON-массивом [x1, y1, x2, y2]')
    if box(roi) is None:
        raise Rejected(422, 'bad_roi', 'Рамка должна содержать четыре числа')
    width, height = size
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise Rejected(422, 'bad_roi', f'Рамка должна лежать внутри изображения {width}×{height} и иметь x2 > x1, y2 > y1')
    return roi
