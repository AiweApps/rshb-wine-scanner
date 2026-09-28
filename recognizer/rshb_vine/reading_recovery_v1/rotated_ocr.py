"""O: one bounded rotated Vision reread of the single target's own label view (90 then 270 CCW, no retry).

Eligible only when the response has exactly one target, its Vision lines (primary + conditional rereads, the
input the frozen SparseAlternativeOCR counts) hold at most one confident line by that same count, and no packet
line of any reader contains a producer form: catalogue brand forms with verified aliases, the derived prefix
aliases of the frozen alternative engine and the admitted B/D short producer forms. The crop is the frozen
label view (``views[0]`` of the target, rotated by ``retrieval_pixel_rotation_ccw``) of that target only.

New lines whose duplicate key (``ocr_provenance_v1`` ``_norm``) already occurs in the packet are dropped, so
every existing reading stays and no reading is replaced. Kept lines are inserted after the conditional rereads
and before the Paddle block, and each read is recorded as an extra ``conditional_label_ocr.reads`` entry with
its total rotation, so the frozen provenance contract traces them as Vision rereads of the label view crop.
Packet lines carry polygon [] as every appended block; the read keeps polygons in rotated view pixels and
``polygon_original`` in EXIF-oriented original pixels. When lines were kept, ``conditional_label_ocr`` reports
``performed`` true with the frozen stage's value in ``baseline_performed`` and per-source read/line counts.

Declared bypass: the new lines skip the legacy LineEvidenceVerifier / AlternateEvidence reranks (the profile
verifier is not rerun and Paddle is not reordered); only the final typed ranker and resolver downstream see them.
"""
import copy
import io
import math
import re
import time

from PIL import Image, ImageOps

from rshb_vine.io import read_json, sha256
from rshb_vine.ocr_provenance_v1.contract import _norm
from rshb_vine.positive_variant_text import contains
from rshb_vine.typed_catalog_lexicon import normalize

POLICY = 'reading-recovery-rotated-ocr-v1'
EXTRA_ROTATIONS = (90, 270)
LABEL_VIEWS = ('detected_label', 'front_label', 'partial_label')
SHORT_FORMS = 'runs/text-evidence-improve-v1/identity/short-producer-forms-v1.json'
SHORT_FORMS_CHECKSUM = '22a4e4f8eb605b4a164d3da5b2139456e9554e7a813c59fb11fcf51db2de1566'


def confident_lines(observations):
    return sum(o['score'] >= .85 and len(re.sub(r'[^a-zа-яё]', '', o['raw_text'].lower())) >= 3 for o in observations)


def producer_forms(lexicon, root):
    table = read_json(root / SHORT_FORMS)
    if table.get('checksum') != SHORT_FORMS_CHECKSUM:
        raise ValueError('Short producer form table differs from the admitted B/D table')
    forms = {f for item in lexicon.items.values() for f in item['brand_forms'] if f}
    forms |= {normalize(' '.join(a['short_tokens'])) for a in table['admitted']}
    return sorted(f for f in forms if f)


def producer_hits(observations, forms):
    hits = []
    for i, o in enumerate(observations):
        line = normalize(o.get('raw_text', ''))
        found = [f for f in forms if contains(line, f)]
        if found:
            hits.append({'index': i, 'raw_text': o['raw_text'], 'score': o.get('score'), 'forms': found[:5]})
    return hits


def split_packet(result):
    """(Vision lines, Paddle block) of the single-target packet, or None when the Paddle suffix does not trace."""
    packet = (result.get('variant_text') or {}).get('observations', [])
    paddle = (result.get('alternative_ocr') or {}).get('observations') or []
    if not paddle:
        return packet, []
    cut = len(packet) - len(paddle)
    tail = packet[cut:] if cut >= 0 else []
    if cut < 0 or [(o['raw_text'], o['score']) for o in tail] != [(o['raw_text'], o['score']) for o in paddle] \
            or any(o.get('polygon') != [] for o in tail):
        return None
    return packet[:cut], tail


def plan(result, forms):
    targets = result.get('targets') or []
    if result.get('decision') == 'invalid_image':
        return {'eligible': False, 'reason': 'invalid_image'}
    if len(targets) != 1:
        return {'eligible': False, 'reason': 'not_single_target', 'targets': len(targets)}
    if not (result.get('variant_text') or {}).get('performed'):
        return {'eligible': False, 'reason': 'no_variant_text'}
    split = split_packet(result)
    if split is None:
        return {'eligible': False, 'reason': 'paddle_block_not_traced'}
    vision, paddle = split
    count = confident_lines(vision)
    out = {'instance_id': str(targets[0]['instance_id']), 'vision_lines': len(vision), 'paddle_lines': len(paddle),
           'confident_vision_lines': count}
    if count > 1:
        return {**out, 'eligible': False, 'reason': 'not_sparse'}
    hits = producer_hits(vision + paddle, forms)
    if hits:
        return {**out, 'eligible': False, 'reason': 'producer_form_read', 'producer_hits': hits}
    target = targets[0]
    views = [v for v in (target.get('retrieval') or {}).get('views', []) if v.get('kind') in LABEL_VIEWS]
    if not views:
        return {**out, 'eligible': False, 'reason': 'no_label_view'}
    angle = target.get('retrieval_pixel_rotation_ccw', 0) or 0
    if angle % 90:
        return {**out, 'eligible': False, 'reason': 'non_right_angle_view_rotation'}
    return {**out, 'eligible': True, 'reason': 'sparse_no_producer', 'bbox': [int(v) for v in views[0]['bbox']],
            'view_rotation': angle % 360}


def to_view_crop(point, rotation, size):
    """Point in the crop rotated CCW by ``rotation`` (PIL expand) back to the unrotated crop of ``size``."""
    x, y = point
    w, h = size
    if rotation == 0:
        return x, y
    if rotation == 90:
        return w - y, x
    if rotation == 180:
        return w - x, h - y
    if rotation == 270:
        return y, h - x
    raise ValueError('rotation must be a right angle')


class RotatedSparseReread:
    """Delegate for the frozen SparseAlternativeOCR stage of ExpandedCatalogPipeline; inert unless ``plan`` holds."""

    def __init__(self, parent, reader, lexicon, root):
        self.parent = parent
        self.reader = reader
        self.forms = producer_forms(lexicon, root)
        self.manifest = {'policy': POLICY, 'extra_rotations': list(EXTRA_ROTATIONS), 'reader': 'Vision revision3',
                         'sparse': 'SparseAlternativeOCR count (score>=.85, >=3 letters) <= 1 over Vision lines',
                         'producer_forms': len(self.forms), 'short_forms_checksum': SHORT_FORMS_CHECKSUM,
                         'retries': 0, 'source_sha256': sha256(__file__), 'calibrated': False}

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def apply(self, data, baseline):
        result = self.parent.apply(data, baseline)
        decision = plan(result, self.forms)
        trace = {'policy': POLICY, **decision, 'performed': False, 'reads': [], 'added_lines': 0}
        if not decision['eligible']:
            result['rotated_label_ocr'] = trace
            return result
        started = time.perf_counter()
        packet = result['variant_text']['observations']
        _, paddle = split_packet(result)
        seen = {_norm(o.get('raw_text', '')) for o in packet}
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
        box = decision['bbox']
        crop = image.crop(tuple(box))
        reads, added = [], []
        try:
            for extra in EXTRA_ROTATIONS:
                total = (decision['view_rotation'] + extra) % 360
                view = crop.rotate(total, expand=True) if total else crop
                t = time.perf_counter()
                rows = self.reader.read(view)
                ms = (time.perf_counter() - t) * 1000
                raw, kept = [], []
                for o in rows:
                    if not isinstance(o.get('score'), (int, float)) or not math.isfinite(o['score']) or not 0 <= o['score'] <= 1:
                        raise ValueError('Invalid OCR score')
                    points = [to_view_crop(p, total, crop.size) for p in o.get('polygon') or []]
                    row = {**o, 'polygon_original': [[float(x) + box[0], float(y) + box[1]] for x, y in points],
                           'polygon_coordinate_space': 'rotated label view pixels',
                           'polygon_original_coordinate_space': 'EXIF-oriented original pixels'}
                    raw.append(row)
                    key = _norm(o.get('raw_text', ''))
                    if key and key not in seen:
                        seen.add(key)
                        kept.append(row)
                reads.append({'instance_id': decision['instance_id'], 'bbox': box, 'rotation': total, 'observations': kept,
                              'ms': ms, 'source': POLICY})
                trace['reads'].append({'extra_rotation': extra, 'rotation': total, 'view_size': list(view.size),
                                       'raw_observations': raw, 'kept': len(kept), 'duplicates': len(raw) - len(kept), 'ms': ms})
                added += [{'raw_text': o['raw_text'], 'score': o['score'], 'polygon': []} for o in kept]
        except Exception as exc:
            trace.update(error_type=type(exc).__name__, reason='rotated_read_failed',
                         ms=(time.perf_counter() - started) * 1000)
            result['rotated_label_ocr'] = trace
            return result
        cut = len(packet) - len(paddle)
        result['variant_text'] = {**result['variant_text'], 'observations': packet[:cut] + added + packet[cut:]}
        conditional = result.setdefault('conditional_label_ocr', {'performed': False, 'reads': [], 'reader': 'Vision revision3'})
        extra = [copy.deepcopy(r) for r in reads if r['observations']]
        if extra:
            before = list(conditional.get('reads', []))
            conditional.update(baseline_performed=conditional.get('performed', False), performed=True,
                               reads=before + extra, sources={'conditional_line_profile': len(before), POLICY: len(extra)},
                               added_lines={POLICY: len(added)})
        trace.update(performed=True, added_lines=len(added), ms=(time.perf_counter() - started) * 1000)
        result['rotated_label_ocr'] = trace
        timing = result.setdefault('timing_ms', {})
        timing['rotated_label_ocr'] = trace['ms']
        return result


def install(release, root):
    """Wrap the one SparseAlternativeOCR instance of the loaded graph's ExpandedCatalogPipeline; nothing global."""
    from rshb_vine.catalog_additions import ExpandedCatalogPipeline
    from rshb_vine.conditional_line_profile import ConditionalLineProfile
    from rshb_vine.sparse_alternative_ocr_v2 import SparseAlternativeOCR
    expanded = release.base.target.inner.parent.runtime.control.expanded
    if (type(expanded) is not ExpandedCatalogPipeline or type(expanded.alternative) is not SparseAlternativeOCR
            or type(expanded.lines) is not ConditionalLineProfile):
        raise ValueError('Unexpected expanded pipeline graph for the rotated reread')
    stage = RotatedSparseReread(expanded.alternative, expanded.lines.reader, expanded.alternative.engine.lexicon, root)
    expanded.alternative = stage
    return {'policy': POLICY, 'path': 'base.target.inner.parent.runtime.control.expanded.alternative',
            'reader': 'expanded.lines.reader', 'manifest': stage.manifest}
