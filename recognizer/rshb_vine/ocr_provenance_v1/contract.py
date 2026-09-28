"""Additive OCR line provenance: reader, crop source, native score scale; a side-car, never a rewrite.

Legacy observation dicts (raw_text, score, polygon, order) are consumed by old103/v4 features and
digested; they are never modified here. Provenance lives in a parallel list aligned by index.

A target's merged observation packet is, in pipeline order (rshb_vine/catalog_additions.py:93):
  primary Vision read of the packet crop (bottle context crop for one target, label crop per instance)
  + ConditionalLineProfile Vision rereads of the label view (conditional_line_profile.py:18)
  + SparseAlternativeOCR PaddleOCR reads of the label view (sparse_alternative_ocr_v2.py:51).
Appended blocks carry polygon [] and are copied from their stage records, so the packet is traced by
matching those blocks as exact suffixes; anything that does not match is reported, never guessed.
"""
from copy import deepcopy
import hashlib
import json
import re

CONTRACT = 'ocr-line-provenance-v1'
VISION = {'reader': 'apple_vision', 'revision': 3, 'recognition_level': 'accurate', 'languages': ['ru-RU', 'en-US'],
          'score_scale': 'vision_accurate_native_confidence_quantized'}
PADDLE = {'reader': 'paddle_ppocrv5_cpu', 'score_scale': 'paddle_native_continuous_uncalibrated'}


def _key(o):
    return (o.get('raw_text'), o.get('score'))


def _norm(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', str(text).casefold())).strip()


def packet_digest(observations):
    return hashlib.sha256(json.dumps(observations, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def trace(observations, primary, supplements):
    """Provenance list for ``observations``.

    ``primary``: {'crop_source', 'crop_bbox', 'rotation', 'coordinate_space'} of the packet read.
    ``supplements``: ordered [{'stage', 'path', 'descriptor', 'crop_source', 'crop_bbox', 'rotation', 'observations'}]
    appended after the primary read. Returns (records, report); records[i] describes observations[i].
    """
    n = len(observations)
    end, blocks = n, []
    for block in reversed(supplements):
        rows = block['observations']
        start = end - len(rows)
        if rows and start >= 0 and [_key(o) for o in observations[start:end]] == [_key(o) for o in rows] \
                and all(o.get('polygon') == [] for o in observations[start:end]):
            blocks.append((start, end, block))
            end = start
        elif rows:
            blocks.append((None, None, block))
    matched = [(s, e, b) for s, e, b in blocks if s is not None]
    unmatched = [b['path'] for s, e, b in blocks if s is None]
    records = [None] * n
    for i in range(end):
        records[i] = {'index': i, 'stage': 'primary_packet_read', 'path': primary.get('path'), **VISION,
                      'crop_source': primary['crop_source'], 'crop_bbox': primary.get('crop_bbox'),
                      'rotation': primary.get('rotation'), 'coordinate_space': primary.get('coordinate_space')}
    for start, stop, block in matched:
        for i in range(start, stop):
            records[i] = {'index': i, 'stage': block['stage'], 'path': block['path'], **block['descriptor'],
                          'crop_source': block['crop_source'], 'crop_bbox': block.get('crop_bbox'),
                          'rotation': block.get('rotation'), 'coordinate_space': 'label_crop_polygon_omitted'}
    groups = {}
    for i, o in enumerate(observations):
        records[i]['native_score'] = o.get('score')
        records[i]['raw_text'] = o.get('raw_text')
        records[i]['duplicate_group'] = _norm(o.get('raw_text', ''))
        groups.setdefault(records[i]['duplicate_group'], []).append(i)
    for i, record in enumerate(records):
        members = groups[record['duplicate_group']]
        record['duplicate_count'] = len(members)
        record['duplicate_readers'] = sorted({records[j]['reader'] + ':' + records[j]['stage'] for j in members})
    polygon_empty_in_primary = sum(1 for i in range(end) if observations[i].get('polygon') == [])
    report = {'lines': n, 'primary_lines': end, 'supplement_blocks_matched': len(matched),
              'supplement_blocks_unmatched': unmatched, 'primary_lines_with_empty_polygon': polygon_empty_in_primary,
              'packet_digest': packet_digest(observations)}
    return records, report


def from_runtime_result(result, instance_id):
    """Primary/supplement descriptions from a saved runtime result (receipt ``result``)."""
    sid = str(instance_id)
    targets = result.get('targets', [])
    if len(targets) == 1:
        packet = result.get('variant_text') or {}
        primary = {'path': 'variant_text', 'crop_source': 'bottle_context_crop', 'crop_bbox': packet.get('bbox_original'),
                   'rotation': packet.get('rotation_ccw'), 'coordinate_space': packet.get('polygon_coordinate_space')}
    else:
        packet = next((p for p in (result.get('instance_text') or {}).get('targets', []) if str(p['instance_id']) == sid), {})
        primary = {'path': 'instance_text.targets', 'crop_source': 'instance_label_crop', 'crop_bbox': packet.get('bbox'),
                   'rotation': None, 'coordinate_space': 'instance_label_crop_pixels'}
    supplements = []
    for i, read in enumerate((result.get('conditional_label_ocr') or {}).get('reads', [])):
        if str(read.get('instance_id')) == sid:
            supplements.append({'stage': 'conditional_label_reread', 'path': f'conditional_label_ocr.reads.{i}', 'descriptor': VISION,
                                'crop_source': 'label_view_crop', 'crop_bbox': read.get('bbox'), 'rotation': read.get('rotation'),
                                'observations': read.get('observations', [])})
    alternative = result.get('alternative_ocr') or {}
    if len(targets) == 1 and alternative.get('observations'):
        supplements.append({'stage': 'sparse_alternative_read', 'path': 'alternative_ocr', 'descriptor': PADDLE,
                            'crop_source': 'label_view_crop', 'crop_bbox': alternative.get('bbox'), 'rotation': None,
                            'observations': alternative['observations']})
    return primary, supplements


def from_evidence_packet(packet):
    """Primary/supplement descriptions from a selector_evidence target packet (historical evidence files)."""
    ocr = packet.get('ocr_packet') or {}
    primary = {'path': 'ocr_packet', 'crop_source': 'bottle_context_crop' if ocr.get('bbox_original') else 'instance_label_crop',
               'crop_bbox': ocr.get('bbox_original') or ocr.get('bbox'), 'rotation': ocr.get('rotation_ccw'),
               'coordinate_space': ocr.get('polygon_coordinate_space')}
    supplements = []
    for read in packet.get('supplemental_reads', []):
        paddle = read['path'] == 'alternative_ocr'
        supplements.append({'stage': 'sparse_alternative_read' if paddle else 'conditional_label_reread', 'path': read['path'],
                            'descriptor': PADDLE if paddle else VISION, 'crop_source': 'label_view_crop',
                            'crop_bbox': read.get('bbox'), 'rotation': read.get('rotation'), 'observations': read.get('observations', [])})
    return primary, supplements


class ProvenancedReader:
    """Opt-in successor wrapper for an OCR reader: identical rows plus a side-car provenance list."""

    def __init__(self, reader, descriptor):
        self.reader = reader
        self.descriptor = dict(descriptor)
        info = getattr(reader, 'info', None)
        if info:
            self.descriptor['reader_info'] = deepcopy(info)
        digest = getattr(reader, 'executable_sha256', None)
        if digest:
            self.descriptor['binary_sha256'] = digest

    def read(self, image, *, crop_source, crop_bbox=None, rotation=None):
        rows = self.reader.read(image)
        provenance = [{'index': i, **self.descriptor, 'crop_source': crop_source, 'crop_bbox': crop_bbox,
                       'rotation': rotation, 'crop_size': list(image.size), 'native_score': row.get('score'),
                       'raw_text': row.get('raw_text')} for i, row in enumerate(rows)]
        return rows, provenance
