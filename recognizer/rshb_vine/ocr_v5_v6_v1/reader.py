"""Shared-detector PaddleOCR line readers (PP-OCRv5 Cyrillic, PP-OCRv6 medium) and their fusion.

Native recognizer scores are reader-local; they are never compared across models.
"""
import hashlib
import math
import os
from pathlib import Path
import threading
import time
import unicodedata

import numpy as np

from rshb_vine.io import sha256

ROOT = Path(__file__).resolve().parents[2]

DETECTOR = {
    'model_name': 'PP-OCRv5_mobile_det', 'dir': 'models/ocr_detector',
    'sha256': {'inference.pdiparams': 'afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b',
               'inference.json': '05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c',
               'inference.yml': '98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699'}}
# Parameters the existing PaddleOCR pipeline (rshb_vine.models.OCR) resolves for this detector.
DETECTOR_PARAMS = {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'box_thresh': 0.6,
                   'unclip_ratio': 1.5}
DETECTOR_MAX_SIDE_LIMIT = 4000
# PP-OCRv6 medium uses the PaddleOCR 3.7 default OCR.yaml pipeline parameters, identical to the above.
DETECTORS = {
    'v5_mobile': {**DETECTOR, 'params': DETECTOR_PARAMS, 'max_side_limit': DETECTOR_MAX_SIDE_LIMIT},
    'v6_medium': {
        'model_name': 'PP-OCRv6_medium_det', 'dir': 'models/ocr-v5-v6-v1/PP-OCRv6_medium_det_infer',
        'source_url': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_det_infer.tar',
        'archive_sha256': '144d0621e059566e5086e228829171591c144c2deb07b2dad4962214fbabfcf7',
        'sha256': {'inference.pdiparams': '85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960',
                   'inference.json': '0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5',
                   'inference.yml': '7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871'},
        'params': DETECTOR_PARAMS, 'max_side_limit': DETECTOR_MAX_SIDE_LIMIT,
        'params_source': 'paddlex/configs/pipelines/OCR.yaml (model file PostProcess .2/.45/1.4 not used)',
        'status': 'config-deviating historical diagnostic; not the admitted model defaults'},
}
DETECTORS['v6_medium_native'] = {
    **{key: DETECTORS['v6_medium'][key] for key in ('model_name', 'dir', 'source_url', 'archive_sha256', 'sha256')},
    'params': {}, 'max_side_limit': DETECTOR_MAX_SIDE_LIMIT,
    'effective_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.2, 'box_thresh': 0.45,
                         'unclip_ratio': 1.4, 'max_candidates': 3000},
    'params_source': 'no overrides: installed PaddleX predictor resize default + model inference.yml PostProcess'}
RECOGNIZERS = {
    'v5_cyrillic': {
        'model_name': 'cyrillic_PP-OCRv5_mobile_rec', 'dir': 'models/ocr-v5-v6-v1/cyrillic_PP-OCRv5_mobile_rec_infer',
        'source_url': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv5_mobile_rec_infer.tar',
        'archive_sha256': 'ef0c66f19ebe6849daf9aa7485e9253d2a8d851e165e208bf08ba68a620ae645',
        'sha256': {'inference.pdiparams': '434dc9fa2a99fa3653e08f8cf793ae56be7dd41c35c4980e6255147cc02bbc80',
                   'inference.json': '5d90f1bfca52d80c01de176c5238fae2459995a99ff1dbfe5319ab4ed1735df2',
                   'inference.yml': '5c76cc91fa98410178a09f498db10050d0ec1634a660053d3005ab7be581f501'}},
    'v6_medium': {
        'model_name': 'PP-OCRv6_medium_rec', 'dir': 'models/ocr-v5-v6-v1/PP-OCRv6_medium_rec_infer',
        'source_url': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_infer.tar',
        'archive_sha256': '4eecc1c6a4623765042e6fc15446da0da110b7d875b6b72b2d351d2b2dbd4da6',
        'sha256': {'inference.pdiparams': '1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319',
                   'inference.json': '0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8',
                   'inference.yml': '991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129'}},
}
# Predeclared operational ordinal bridge per reader; not a calibration and not Vision confidence.
ORDINAL_BANDS = ((0.90, 1.0), (0.70, 0.5))
ORDINAL_FLOOR = 0.3
FUSION_RULE = 'script-route-v1'
RUNTIME = {'device': 'cpu', 'enable_mkldnn': False, 'cpu_threads': 4}


def verify_model(spec, root=ROOT):
    directory = Path(root) / spec['dir']
    actual = {name: sha256(directory / name) for name in spec['sha256']}
    if actual != spec['sha256']:
        raise ValueError(f"Model files changed: {spec['model_name']}")
    return {'model_name': spec['model_name'], 'dir': spec['dir'], 'sha256': actual,
            **{key: spec[key] for key in ('source_url', 'archive_sha256') if key in spec}}


def contributes(text, score, error):
    """Errors, non-finite or out-of-range scores and normalized-empty text never become admitted text."""
    return error is None and math.isfinite(score) and 0.0 <= score <= 1.0 and bool(normalize(text))


def ordinal_band(score):
    for bound, band in ORDINAL_BANDS:
        if score >= bound:
            return band
    return ORDINAL_FLOOR


def script_class(text):
    cyrillic = latin = other = False
    for char in unicodedata.normalize('NFKC', text):
        if not char.isalpha():
            continue
        name = unicodedata.name(char, '')
        if name.startswith('CYRILLIC'):
            cyrillic = True
        elif name.startswith('LATIN'):
            latin = True
        else:
            other = True
    if cyrillic:
        return 'CYR'
    if other:
        return 'OTHER'
    return 'LAT' if latin else 'NONE'


def fuse_line(v5, v6):
    """Select exactly one reading of one physical line without comparing native scores."""
    s5, s6 = script_class(v5['raw_text']), script_class(v6['raw_text'])
    v5_empty, v6_empty = not normalize(v5['raw_text']), not normalize(v6['raw_text'])
    if s5 == 'CYR':
        return 'v5_cyrillic', 'v5_has_cyrillic'
    if s5 == 'LAT' and s6 == 'LAT' and not v6_empty:
        return 'v6_medium', 'both_latin'
    if v5_empty and not v6_empty and s6 == 'LAT':
        return 'v6_medium', 'v5_empty'
    return 'v5_cyrillic', 'default_v5'


def detector_input_size(size, params=DETECTOR_PARAMS):
    width, height = size
    side, kind = params['limit_side_len'], params['limit_type']
    if kind == 'max':
        ratio = side / max(width, height) if max(width, height) > side else 1.0
    else:
        ratio = side / min(width, height) if min(width, height) < side else 1.0
    width, height = int(width * ratio), int(height * ratio)
    if max(width, height) > DETECTOR_MAX_SIDE_LIMIT:
        ratio = DETECTOR_MAX_SIDE_LIMIT / max(width, height)
        width, height = int(width * ratio), int(height * ratio)
    return [max(int(round(width / 32) * 32), 32), max(int(round(height / 32) * 32), 32)]


def _crop_digest(crop):
    header = f"{crop.shape}|{crop.dtype}".encode()
    return hashlib.sha256(header + np.ascontiguousarray(crop).tobytes()).hexdigest()


class LineReaders:
    """Frozen PP-OCRv5 mobile detector → shared line crops → one or both recognizers."""

    def __init__(self, root=ROOT, recognizers=('v5_cyrillic', 'v6_medium'), detector='v5_mobile'):
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        from paddleocr import TextDetection, TextRecognition
        from paddlex.inference.pipelines.components import CropByPolys, SortQuadBoxes
        self.root = Path(root)
        self.lock = threading.Lock()
        spec = DETECTORS[detector]
        self.detector_id, self.max_side_limit = detector, spec['max_side_limit']
        self.models = {'detector': {'id': detector, **verify_model(spec, self.root)}}
        self.init_seconds = {}
        started = time.perf_counter()
        self.detector = TextDetection(model_name=spec['model_name'], model_dir=str(self.root / spec['dir']),
                                      **spec['params'], **RUNTIME)
        self.effective_params = self._effective_detector_params()
        if 'effective_params' in spec and self.effective_params != spec['effective_params']:
            raise ValueError(f'Effective detector parameters differ from pinned: {self.effective_params}')
        self.init_seconds['detector'] = time.perf_counter() - started
        self.sort_boxes, self.crop_by_polys = SortQuadBoxes(), CropByPolys(det_box_type='quad')
        self.recognizers = {}
        for name in recognizers:
            spec = RECOGNIZERS[name]
            self.models[name] = verify_model(spec, self.root)
            started = time.perf_counter()
            self.recognizers[name] = TextRecognition(model_name=spec['model_name'], model_dir=str(self.root / spec['dir']), **RUNTIME)
            self.init_seconds[name] = time.perf_counter() - started

    def _effective_detector_params(self):
        predictor = self.detector.paddlex_predictor
        resize, post = predictor.pre_tfs['Resize'], predictor.post_op
        return {'limit_side_len': predictor.limit_side_len or resize.limit_side_len,
                'limit_type': predictor.limit_type or resize.limit_type,
                'thresh': predictor.thresh or post.thresh, 'box_thresh': predictor.box_thresh or post.box_thresh,
                'unclip_ratio': predictor.unclip_ratio or post.unclip_ratio, 'max_candidates': post.max_candidates}

    def info(self):
        return {'models': self.models, 'detector_id': self.detector_id,
                'detector_params': {**DETECTORS[self.detector_id]['params'], 'max_side_limit': self.max_side_limit},
                'detector_effective_params': {**self.effective_params, 'max_side_limit': self.max_side_limit},
                'crop': 'SortQuadBoxes + CropByPolys(quad), perspective warp, rot90 if h/w>=1.5', 'runtime': RUNTIME,
                'recognizer_batch_size': 1, 'textline_orientation': False,
                'ordinal_bands': [list(item) for item in ORDINAL_BANDS], 'ordinal_floor': ORDINAL_FLOOR,
                'fusion_rule': FUSION_RULE}

    def detect(self, image):
        """Return shared line records with BGR crops, as the existing PaddleOCR pipeline builds them."""
        bgr = np.array(image.convert('RGB'))[:, :, ::-1].copy()
        started = time.perf_counter()
        result = next(iter(self.detector.predict(bgr, batch_size=1, max_side_limit=self.max_side_limit)))
        polys = self.sort_boxes(result['dt_polys'])
        seconds = time.perf_counter() - started
        started = time.perf_counter()
        crops = self.crop_by_polys(bgr, polys)
        self.last_crop_seconds = time.perf_counter() - started
        lines = []
        for poly, crop in zip(polys, crops):
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            points = np.asarray(poly, dtype=np.float32)
            width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
            height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
            lines.append({'line_id': len(lines), 'polygon': points.tolist(), 'crop': crop,
                          'crop_sha256': _crop_digest(crop), 'crop_shape': list(crop.shape),
                          'rotated_90': bool(int(height) >= 1.5 * max(int(width), 1))})
        return lines, seconds

    def recognize(self, name, lines):
        rows, started = [], time.perf_counter()
        for line in lines:
            error = None
            try:
                result = next(iter(self.recognizers[name].predict(line['crop'], batch_size=1)))
                text, score = str(result['rec_text']), float(result['rec_score'])
            except Exception as exc:
                text, score, error = '', 0.0, f'{type(exc).__name__}: {exc}'
            rows.append({'line_id': line['line_id'], 'reader': name, 'raw_text': text, 'native_score': score,
                         'polygon': line['polygon'], 'crop_sha256': line['crop_sha256'],
                         'contributes': contributes(text, score, error),
                         'admitted_band': ordinal_band(score) if contributes(text, score, error) else None,
                         'error': error})
        return rows, time.perf_counter() - started

    def read_with_trace(self, image):
        with self.lock:
            lines, det_seconds = self.detect(image)
            trace = {'image_size': list(image.size),
                     'detector_input_size': detector_input_size(image.size, self.effective_params),
                     'detector_id': self.detector_id, 'detector_seconds': det_seconds,
                     'crop_seconds': self.last_crop_seconds,
                     'lines': [{key: line[key] for key in ('line_id', 'polygon', 'crop_sha256', 'crop_shape', 'rotated_90')}
                               for line in lines],
                     'readers': {}}
            for name in self.recognizers:
                rows, seconds = self.recognize(name, lines)
                trace['readers'][name] = {'rows': rows, 'seconds': seconds}
            if set(self.recognizers) == {'v5_cyrillic', 'v6_medium'}:
                trace['readers']['v5_v6_fusion'] = {'rows': fuse_rows(trace['readers']['v5_cyrillic']['rows'],
                                                                      trace['readers']['v6_medium']['rows']),
                                                    'seconds': 0.0}
            return lines, trace


def fuse_rows(v5_rows, v6_rows):
    by_line = {row['line_id']: row for row in v6_rows}
    fused = []
    for v5 in v5_rows:
        v6 = by_line[v5['line_id']]
        selected, reason = fuse_line(v5, v6)
        chosen, other = (v5, v6) if selected == 'v5_cyrillic' else (v6, v5)
        fused.append({**chosen, 'reader': 'v5_v6_fusion', 'selected_reader': selected, 'selection_reason': reason,
                      'script': {'v5_cyrillic': script_class(v5['raw_text']), 'v6_medium': script_class(v6['raw_text'])},
                      'agreement': normalize(v5['raw_text']) == normalize(v6['raw_text']),
                      'alternatives': [{'reader': other['reader'], 'raw_text': other['raw_text'],
                                        'native_score': other['native_score'], 'contributes': False}]})
    return fused


def normalize(text):
    text = unicodedata.normalize('NFKC', text).upper().replace('Ё', 'Е')
    return ' '.join(''.join(char if char.isalnum() else ' ' for char in text).split())


class PaddleLineReader:
    """Consumer interface: read(PIL) -> rows for one arm ('v5_cyrillic', 'v6_medium' or 'v5_v6_fusion')."""

    def __init__(self, arm, root=ROOT, detector='v5_mobile'):
        needed = ('v5_cyrillic', 'v6_medium') if arm == 'v5_v6_fusion' else (arm,)
        self.arm = arm
        self.readers = LineReaders(root, needed, detector)
        self.last_trace = None

    def info(self):
        return {'arm': self.arm, **self.readers.info()}

    def read_with_trace(self, image):
        _, trace = self.readers.read_with_trace(image)
        self.last_trace = trace
        return trace['readers'][self.arm]['rows'], trace

    def read(self, image):
        return self.read_with_trace(image)[0]
