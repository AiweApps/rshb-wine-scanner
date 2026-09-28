"""Isolated experimental OCR arm over the frozen 3c8c2da4 runtime; Vision readers are swapped by object identity.

One process loads the unchanged active profile, locates every reachable reader object by an exhaustive graph walk,
and replaces each reference of the two VisionOCR instances (slot A: variant/instance packet reader, slot B:
ConditionalLineProfile label reread) with a traced slot proxy. The existing SparseAlternativeOCR PaddleOCR reader is
wrapped by a transparent recorder and otherwise left unchanged. In non-Vision arms the VisionOCR class is replaced by
a recording sentinel and the Vision workers are closed, so any leak is counted even where frozen code swallows it.

Compatibility bridge (root-approved, bridge msg 7): frozen consumers see Vision-shaped rows {raw_text, score,
polygon}; `score` is the reader's predeclared ordinal band (>=0.90 -> 1.0, >=0.70 -> 0.5, else 0.3), an operational
admission, not a calibration and not Vision confidence. The frozen provenance side-car keeps labelling the primary
packet 'apple_vision'; that label is the historical primary OCR slot. The authoritative reader, native score and
line id of every forwarded row are in the result's `experimental_ocr` block.
"""
from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
import functools
import hashlib
import io
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

from rshb_vine.io import read_json, seal, sha256, verify

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'config/recognition-current.json'
PROTOCOL = 'runs/ocr-v5-v6-v1/protocol.json'
PROTOCOL_CHECKSUM = '971ee88920d518bdcc36ffba7fda7cd6330dda8aa062637d332e48a5c2c0d3a3'
PROFILE_CHECKSUM = '3c8c2da4fc718e1c65c98167f84805fa7c8941425d369f07f088f29e3b6f528a'
PROFILE_FILE_SHA256 = '0dd04c9771f8ce227ba13b26c76273a303d3f4330159be4a2e38b6535b1207a9'
RUNTIME_CHECKSUM = '0837848868bed3e63825e241275db2d552f81778129a70c61be4e52b2cb77d18'
ARMS = ('vision_current', 'v5_cyrillic', 'v6_medium', 'v5_v6_fusion')
KIND = 'ocr-v5-v6-experimental-runtime-v1'
OVERRIDE = 'ocr-v5-v6-experimental'
BANDS = ((0.90, 1.0), (0.70, 0.5))
FLOOR = 0.3
SLOTS = {'A': 'variant/instance packet reader', 'B': 'ConditionalLineProfile label reread'}
SLOT_PATHS = {'A': "pipeline.next_states['N'].gate.variant_reader",
              'B': 'pipeline.base.runtime.control.expanded.lines.reader'}
SLOT_OWNERS = {'A': (None, 'variant_reader'), 'B': ('ConditionalLineProfile', 'reader')}
CALLSITES = {'A': {'rshb_vine/variant_text_pipeline.py': 'single_target_context_crop',
                   'rshb_vine/instance_text_profile.py': 'instance_label_crop'},
             'B': {'rshb_vine/conditional_line_profile.py': 'conditional_label_view'},
             'sparse': {'rshb_vine/sparse_alternative_ocr_v2.py': 'sparse_alternative_label_view'}}
SOURCES = ('rshb_vine/ocr_v5_v6_v1/pipeline.py', 'rshb_vine/ocr_v5_v6_v1/reader.py', 'rshb_vine/vision_ocr.py',
           'rshb_vine/models.py', 'rshb_vine/variant_text_pipeline.py', 'rshb_vine/instance_text_profile.py',
           'rshb_vine/conditional_line_profile.py', 'rshb_vine/sparse_alternative_ocr_v2.py',
           'rshb_vine/ocr_provenance_v1/contract.py')
FORBIDDEN_PORTS = {8175, 8187, 8287}
LEAF_MODULES = ('torch', 'numpy', 'paddle', 'paddlex', 'paddleocr', 'transformers', 'PIL', 'cv2', 'onnxruntime',
                'safetensors', 'threading', '_thread', 'subprocess', 'io', '_io', 'logging', 'fastapi', 'starlette',
                'uvicorn', 'requests', 'urllib3', 'sklearn', 'scipy', 'lightgbm', 'xgboost', 'tokenizers', 'huggingface_hub')
WALK_LIMIT = 30_000_000
DISCLOSURE = ("Frozen feature channel 'vision_r3_macos' and provenance reader 'apple_vision' denote the historical "
              "PRIMARY OCR SLOT, not actual Vision provenance, in non-Vision arms. Forwarded score is the reader's "
              "predeclared ordinal band (>=0.90->1.0, >=0.70->0.5, else 0.3): operational admission, not calibration, "
              "not Vision confidence. True reader, native score and line id are in experimental_ocr.calls.")


def band(score):
    for bound, value in BANDS:
        if score >= bound:
            return value
    return FLOOR


def drop_reason(row):
    """Structural reader-contract exclusion (root bridge msg 8); None means the row is forwarded."""
    from rshb_vine.ocr_v5_v6_v1.reader import normalize
    score = row.get('native_score')
    if row.get('error') is not None:
        return 'reader_error'
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        return 'invalid_native_score'
    if not isinstance(row.get('raw_text'), str) or not normalize(row['raw_text']):
        return 'normalized_empty_text'
    return None


def pixel_sha256(image):
    header = f'{image.mode}|{image.size[0]}x{image.size[1]}|'.encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def _is_leaf(obj):
    if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, range, memoryview,
                                       type, types.ModuleType, types.BuiltinFunctionType, types.CodeType)):
        return True
    if isinstance(obj, SlotReader):
        return True
    module = type(obj).__module__ or ''
    return module.split('.', 1)[0] in LEAF_MODULES


def _children(obj):
    """(kind, key, child) edges; kind tells how the reference could be rebound in place."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            yield 'dict', key, value
        return
    if isinstance(obj, list):
        for i, value in enumerate(list(obj)):
            yield 'list', i, value
        return
    if isinstance(obj, (tuple, set, frozenset, deque)):
        for i, value in enumerate(list(obj)):
            yield 'frozen', i, value
        return
    if isinstance(obj, types.MethodType):
        yield 'frozen', '__self__', obj.__self__
        yield 'frozen', '__func__', obj.__func__
        return
    if isinstance(obj, functools.partial):
        yield 'frozen', 'func', obj.func
        for i, value in enumerate(obj.args):
            yield 'frozen', i, value
        for key, value in (obj.keywords or {}).items():
            yield 'frozen', key, value
        return
    if isinstance(obj, types.FunctionType):
        for i, cell in enumerate(obj.__closure__ or ()):
            try:
                yield 'cell', i, cell.cell_contents
            except ValueError:
                pass
        return
    attrs = getattr(obj, '__dict__', None)
    if isinstance(attrs, dict):
        for key, value in list(attrs.items()):
            yield 'attr', key, value
    for cls in type(obj).__mro__:
        for name in cls.__dict__.get('__slots__', ()) if isinstance(cls.__dict__.get('__slots__', ()), (tuple, list)) else ():
            try:
                yield 'attr', name, object.__getattribute__(obj, name)
            except AttributeError:
                pass


def _label(path, kind, key):
    return f'{path}.{key}' if kind == 'attr' else f'{path}[{key!r}]'


def census(pipeline, classes):
    """Every reference to an instance of ``classes`` reachable from ``pipeline``; fails closed when not exhaustive."""
    seen, found, queue, visited = {id(pipeline)}, {}, deque([(pipeline, 'pipeline')]), 0
    while queue:
        obj, path = queue.popleft()
        visited += 1
        if visited > WALK_LIMIT:
            raise RuntimeError('Graph walk exceeded its limit; coverage cannot be claimed')
        for kind, key, child in _children(obj):
            if isinstance(child, classes):
                entry = found.setdefault(id(child), {'object': child, 'type': type(child).__qualname__, 'refs': []})
                entry['refs'].append({'container': obj, 'kind': kind, 'key': key, 'path': _label(path, kind, key)})
            if _is_leaf(child) or id(child) in seen:
                continue
            seen.add(id(child))
            queue.append((child, _label(path, kind, key)))
    return found, visited


def _slot_of(ref):
    for slot, (owner, key) in SLOT_OWNERS.items():
        if ref['kind'] == 'attr' and ref['key'] == key and (owner is None or type(ref['container']).__name__ == owner):
            return slot
    return None


def _rebind(ref, value):
    container, kind, key = ref['container'], ref['kind'], ref['key']
    if kind == 'attr':
        object.__setattr__(container, key, value)
    elif kind in ('dict', 'list'):
        container[key] = value
    elif kind == 'cell':
        container.__closure__[key].cell_contents = value
    else:
        raise RuntimeError('Reader referenced from an immutable container: ' + ref['path'])


def _callsite():
    frame = sys._getframe(2)
    while frame is not None:
        name = Path(frame.f_code.co_filename).resolve()
        try:
            relative = str(name.relative_to(ROOT))
        except ValueError:
            relative = str(name)
        if relative != 'rshb_vine/ocr_v5_v6_v1/pipeline.py':
            return relative, frame.f_lineno
        frame = frame.f_back
    return None, None


class RequestTrace:
    """Request-local OCR ledger; the frozen runtime serves one request at a time under its own lock."""

    def __init__(self, crop_dir=None):
        self.crop_dir = Path(crop_dir) if crop_dir else None
        self.calls, self.leaks, self.active = [], [], False

    def begin(self):
        self.calls, self.leaks, self.active = [], [], True

    def end(self):
        self.active = False
        return self.calls, self.leaks

    def crop(self, image):
        digest_ = pixel_sha256(image)
        record = {'pixel_sha256': digest_, 'size': list(image.size), 'mode': image.mode}
        if self.crop_dir is not None:
            path = self.crop_dir / f'{digest_}.png'
            if not path.exists():
                self.crop_dir.mkdir(parents=True, exist_ok=True)
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                tmp = path.with_suffix('.tmp%d' % threading.get_ident())
                tmp.write_bytes(buffer.getvalue())
                tmp.replace(path)
            record['png'] = str(path.relative_to(ROOT))
        return record


class SlotReader:
    """Traced stand-in for one reader slot; forwards Vision-shaped rows and records the authoritative trace."""

    def __init__(self, slot, arm, reader, ledger, kind):
        self.slot, self.arm, self.inner, self.ledger, self.kind = slot, arm, reader, ledger, kind

    def read(self, image):
        file, line = _callsite()
        call = {'seq': len(self.ledger.calls), 'slot': self.slot, 'callsite': f'{file}:{line}',
                'crop_context': CALLSITES.get(self.slot, {}).get(file, 'unexpected_callsite'),
                'crop': self.ledger.crop(image)}
        started = time.perf_counter()
        try:
            if self.kind == 'vision':
                rows = self.inner.read(image)
                call.update(reader='apple_vision_r3', score_scale='vision_native_quantized', forwarded=deepcopy(rows))
            elif self.kind == 'sparse_existing':
                rows = self.inner.read(image)
                call.update(reader='paddle_ppocrv5_eslav_mobile_existing', score_scale='paddle_native_continuous_existing',
                            forwarded=deepcopy(rows))
            else:
                rows, trace = self.inner.read_with_trace(image)
                forwarded, lines = [], []
                for row in rows:
                    reason = drop_reason(row)
                    if reason is None and row['admitted_band'] != band(row['native_score']):
                        raise RuntimeError('Reader band differs from the sealed adapter bands')
                    lines.append({**row, 'forwarded': reason is None,
                                  'forwarded_index': len(forwarded) if reason is None else None, 'drop_reason': reason})
                    if reason is None:
                        forwarded.append({'raw_text': row['raw_text'], 'score': band(row['native_score']),
                                          'polygon': row['polygon']})
                call.update(reader=self.arm, score_scale='ordinal_band_operational_admission',
                            lines=lines, forwarded=deepcopy(forwarded),
                            detector={'lines': len(trace['lines']), 'seconds': trace['detector_seconds'],
                                      'image_size': trace['image_size']},
                            recognizer_seconds={k: v['seconds'] for k, v in trace['readers'].items()})
                rows = forwarded
        except Exception as exc:
            call.update(error=f'{type(exc).__name__}: {exc}', ms=(time.perf_counter() - started) * 1000)
            self.ledger.calls.append(call)
            raise
        call['ms'] = (time.perf_counter() - started) * 1000
        self.ledger.calls.append(call)
        return rows


def _vision_sentinel(ledger):
    def forbidden(self, *args, **kwargs):
        file, line = _callsite()
        ledger.leaks.append({'callsite': f'{file}:{line}', 'at': time.time()})
        raise RuntimeError('Apple Vision is forbidden in this experimental arm')
    return forbidden


def _vision_children():
    out = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,command='], capture_output=True, text=True, check=True).stdout
    me = str(__import__('os').getpid())
    return [line.strip() for line in out.splitlines()
            if line.split(None, 2)[1:2] == [me] and 'vision-ocr' in line]


def detector_spec(R, detector):
    """Pinned common text detector of the Paddle arms; None is the reader's default (existing PP-OCRv5 mobile)."""
    if detector is None:
        return {'id': None, 'model_name': R.DETECTOR['model_name'], 'dir': R.DETECTOR['dir'],
                'sha256': R.DETECTOR['sha256'], 'params': getattr(R, 'DETECTOR_PARAMS', None),
                'max_side_limit': getattr(R, 'DETECTOR_MAX_SIDE_LIMIT', None)}
    registry = getattr(R, 'DETECTORS', None) or {}
    if detector not in registry:
        raise ValueError('Reader defines no detector %r' % detector)
    spec = registry[detector]
    return {'id': detector, 'model_name': spec['model_name'], 'dir': spec['dir'], 'sha256': spec['sha256'],
            **{k: v for k, v in spec.items() if k not in ('model_name', 'dir', 'sha256')}}


def adapter_descriptor(arm, detector=None):
    """Reviewable, model-free description of what this arm changes; its checksum is what root admits."""
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    if arm == 'vision_current' and detector is not None:
        raise ValueError('The Vision arm has no Paddle detector')
    from rshb_vine.ocr_v5_v6_v1 import reader as R
    if tuple(R.ORDINAL_BANDS) != BANDS or R.ORDINAL_FLOOR != FLOOR:
        raise ValueError('Reader bands differ from the root-sealed adapter bands')
    protocol = verify(read_json(ROOT / PROTOCOL))
    if protocol['checksum'] != PROTOCOL_CHECKSUM or protocol['arms'] != list(ARMS):
        raise ValueError('Protocol changed')
    swapped = arm != 'vision_current'
    return seal({
        'kind': 'ocr-v5-v6-adapter-descriptor-v1', 'arm': arm, 'protocol': PROTOCOL_CHECKSUM,
        'parent_profile': PROFILE_CHECKSUM, 'parent_profile_file_sha256': PROFILE_FILE_SHA256,
        'parent_runtime': RUNTIME_CHECKSUM, 'override': OVERRIDE,
        'slots': {slot: {'role': SLOTS[slot], 'first_path': SLOT_PATHS[slot], 'callsites': CALLSITES[slot],
                         'reader': arm if swapped else 'apple_vision_r3 (unchanged)'} for slot in SLOTS},
        'sparse_alternative': {'reader': 'models.OCR PP-OCRv5_mobile_det + eslav_PP-OCRv5_mobile_rec (existing)',
                               'change': 'none; recorded by a transparent wrapper',
                               'callsites': CALLSITES['sparse']},
        'vision_gallery': 'frozen Vision-derived gallery/reference text signatures retained unchanged',
        'adapter': None if not swapped else {
            'forwarded_row': ['raw_text', 'score', 'polygon'], 'score': 'ordinal band of the reader native score',
            'bands': [list(b) for b in BANDS], 'floor': FLOOR, 'calibrated': False,
            'drop': 'reader error, non-finite/out-of-range native score or normalized-empty text: not forwarded, kept in the trace with reason',
            'polygon': 'reader quad in crop pixels (same space as the Vision rows it replaces)',
            'fusion': R.FUSION_RULE if arm == 'v5_v6_fusion' else None,
            'native_score_max_across_models': False},
        'vision_guard': None if not swapped else 'VisionOCR.__init__/read/_start_worker replaced by a recording '
                                                 'sentinel; both workers closed; no vision-ocr child process',
        'disclosure': DISCLOSURE if swapped else 'unchanged Vision arm; readers wrapped by transparent recorders',
        'ranking': 'frozen B3/ranker/catalog/bottle-label detectors; no new ranking weights',
        'sources_sha256': {name: sha256(ROOT / name) for name in SOURCES},
        'reader_models': {k: v['sha256'] for k, v in R.RECOGNIZERS.items()},
        'detector': detector_spec(R, detector) if swapped else None})


class ExperimentalRuntime:
    """One arm per process: frozen runtime, swapped slots, per-request `experimental_ocr` side-car."""

    def __init__(self, arm, detector=None, crop_dir=None):
        started = time.perf_counter()
        self.descriptor = adapter_descriptor(arm, detector)
        self.arm, self.detector = arm, detector
        if sha256(ROOT / PROFILE) != PROFILE_FILE_SHA256:
            raise ValueError('Active profile file changed')
        from rshb_vine.models import OCR
        from rshb_vine.recognition_factory import load_runtime
        from rshb_vine.vision_ocr import VisionOCR
        parent = load_runtime(ROOT, ROOT / PROFILE)
        if parent.profile['checksum'] != PROFILE_CHECKSUM or parent.manifest['checksum'] != RUNTIME_CHECKSUM:
            raise ValueError('Loaded runtime is not the pinned 3c8c2da4/08378488')
        self.parent = parent
        self.profile = parent.profile
        self.ledger = RequestTrace(crop_dir)
        guarded = [VisionOCR, OCR]
        rotated = sys.modules.get('rshb_vine.reading_recovery_v1.rotated_ocr')
        if rotated is not None:
            guarded.append(rotated.RotatedSparseReread)
        found, visited = census(parent, tuple(guarded))
        vision = [e for e in found.values() if isinstance(e['object'], VisionOCR)]
        sparse = [e for e in found.values() if type(e['object']) is OCR]
        if len(vision) + len(sparse) != len(found):
            raise RuntimeError('An uninstalled reader stage is reachable: %s' % sorted(e['type'] for e in found.values()))
        slots, unclassified = {}, []
        for entry in vision:
            names = {_slot_of(r) for r in entry['refs']} - {None}
            if len(names) != 1 or names & set(slots):
                raise RuntimeError('VisionOCR instance does not map to exactly one slot: %s' % sorted(
                    r['path'] for r in entry['refs']))
            slot = names.pop()
            slots[slot] = entry
            unclassified += [r['path'] for r in entry['refs'] if _slot_of(r) is None]
        if set(slots) != set(SLOTS) or len(vision) != 2 or len(sparse) != 1:
            raise RuntimeError('Unexpected reader census: vision=%d sparse=%d' % (len(vision), len(sparse)))
        if arm == 'vision_current':
            readers = {slot: SlotReader(slot, arm, entry['object'], self.ledger, 'vision') for slot, entry in slots.items()}
            reader_info = {slot: entry['object'].info for slot, entry in slots.items()}
        else:
            from rshb_vine.ocr_v5_v6_v1.reader import PaddleLineReader
            shared = PaddleLineReader(arm, ROOT, **({} if detector is None else {'detector': detector}))
            reader_info = shared.info()
            if reader_info['models']['detector']['sha256'] != self.descriptor['detector']['sha256']:
                raise RuntimeError('Loaded detector differs from the admitted descriptor')
            readers = {slot: SlotReader(slot, arm, shared, self.ledger, 'paddle') for slot in slots}
        for slot, entry in slots.items():
            for ref in entry['refs']:
                _rebind(ref, readers[slot])
        for ref in sparse[0]['refs']:
            _rebind(ref, SlotReader('sparse', arm, sparse[0]['object'], self.ledger, 'sparse_existing'))
        if arm != 'vision_current':
            for entry in vision:
                entry['object'].close()
            sentinel = _vision_sentinel(self.ledger)
            for name in ('__init__', 'read', '_start_worker'):
                setattr(VisionOCR, name, sentinel)
        after, visited_after = census(parent, tuple(guarded))
        if after:
            raise RuntimeError('Reader references remain after the swap: %s' % sorted(
                r['path'] for e in after.values() for r in e['refs']))
        children = _vision_children()
        if arm != 'vision_current' and children:
            raise RuntimeError('Vision worker process still running: %s' % children)
        self.coverage = {'visited_before': visited, 'visited_after': visited_after,
                         'slots': {s: sorted(r['path'] for r in e['refs']) for s, e in slots.items()},
                         'unclassified_refs_rebound_to_slot': sorted(unclassified),
                         'sparse': sorted(r['path'] for r in sparse[0]['refs']),
                         'rotated_reread_module_loaded': rotated is not None, 'rotated_reread_instances': 0,
                         'remaining_reader_refs': 0, 'vision_children': children}
        self.manifest = seal({'kind': KIND, 'arm': arm, 'detector': detector, 'override': OVERRIDE,
                              'descriptor': self.descriptor['checksum'],
                              'parent_profile': PROFILE_CHECKSUM, 'parent_runtime': RUNTIME_CHECKSUM,
                              'reader_info': reader_info,
                              'coverage': {k: v for k, v in self.coverage.items() if k != 'vision_children'},
                              'calibrated': False, 'live_release_changed': False})
        self.load_seconds = time.perf_counter() - started

    @contextmanager
    def _request(self):
        self.ledger.begin()
        try:
            yield
        finally:
            self.ledger.end()

    def recognize(self, data, roi=None, bottles='addressed'):
        with self._request():
            started = time.perf_counter()
            result = self.parent.recognize(data, roi, bottles)
            seconds = time.perf_counter() - started
            calls, leaks = list(self.ledger.calls), list(self.ledger.leaks)
        drops = Counter(line['drop_reason'] for c in calls for line in c.get('lines') or [] if not line['forwarded'])
        result['experimental_ocr'] = {
            'kind': KIND, 'arm': self.arm, 'experimental_checksum': self.manifest['checksum'],
            'descriptor': self.descriptor['checksum'], 'parent_profile': PROFILE_CHECKSUM,
            'parent_runtime': RUNTIME_CHECKSUM, 'disclosure': self.descriptor['disclosure'],
            'detector': self.detector, 'calls': calls, 'vision_leaks': leaks,
            'reader_errors': sum(1 for c in calls if c.get('error')),
            'line_reader_errors': drops.get('reader_error', 0), 'line_drops': dict(drops),
            'forwarded_rows': sum(len(c.get('forwarded') or []) for c in calls),
            'unexpected_callsites': sum(1 for c in calls if c['crop_context'] == 'unexpected_callsite'),
            'seconds': seconds}
        return result


def create_app(runtime):
    """Loopback HTTP of one experimental arm with the frozen target-contract routes."""
    from rshb_vine.target_contract_v2.runtime import create_app as target_app
    return target_app(runtime)
