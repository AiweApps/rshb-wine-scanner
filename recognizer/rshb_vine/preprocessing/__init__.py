"""Coordinate-safe input decoding and conservative target selection."""
import io
import math
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


def decode(data, max_bytes=20 * 1024 * 1024, max_pixels=24_000_000):
    if not data or len(data) > max_bytes:
        raise InvalidImage('Empty image or upload exceeds 20 MiB')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > max_pixels or im.format not in ('JPEG', 'PNG', 'WEBP'):
                raise InvalidImage('Unsupported format or decoded pixel limit exceeded')
            orientation = im.getexif().get(274, 1)
            original_size = list(im.size)
            rgb = ImageOps.exif_transpose(im).convert('RGBA')
            bg = Image.new('RGBA', rgb.size, 'white')
            bg.alpha_composite(rgb)
            return bg.convert('RGB'), {'exif_orientation': orientation, 'encoded_size': original_size,
                                       'coordinate_space': 'EXIF-oriented original pixels', 'oriented_size': list(rgb.size)}
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InvalidImage(str(exc)) from exc


def checked_box(box, size):
    if len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
        raise InvalidImage('ROI requires four finite coordinates')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise InvalidImage('ROI outside oriented image')
    return [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]


def select_target(image, detections, user_roi=None):
    if user_roi is not None:
        return checked_box(user_roi, image.size), False, 'user_roi'
    if not detections:
        return [0, 0, *image.size], False, 'detector_empty_full_frame'
    w, h = image.size
    def priority(d):
        x1, y1, x2, y2 = d['bbox']
        distance = ((x1 + x2) / 2 / w - .5) ** 2 + ((y1 + y2) / 2 / h - .5) ** 2
        return distance - .1 * (x2 - x1) * (y2 - y1) / (w * h)
    ranked = sorted(detections, key=priority)
    ambiguous = len(ranked) > 1 and abs(priority(ranked[0]) - priority(ranked[1])) < .04
    return checked_box(ranked[0]['bbox'], image.size), ambiguous, 'detector_center_area'


def label_box(observations, size):
    points = [p for r in observations if r['score'] >= .5 for p in r['polygon']]
    if not points:
        return None
    xs, ys = zip(*points)
    margin = max(size) * .025
    box = [max(0, min(xs)-margin), max(0, min(ys)-margin),
           min(size[0], max(xs)+margin), min(size[1], max(ys)+margin)]
    if box[2]-box[0] < 4 or box[3]-box[1] < 4:
        return None
    return checked_box(box, size)


def translate_observations(rows, origin):
    return [dict(r, polygon=[[x + origin[0], y + origin[1]] for x, y in r['polygon']]) for r in rows]
