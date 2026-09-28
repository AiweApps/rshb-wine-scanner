"""Linux CPU execution of the vendored recognizer with PP-OCRv6 medium in both Apple Vision reader slots.

Everything is installed before the vendored runtime is built and recorded in the core manifest: constructors that
request MPS get an explicit CPU device, the two Vision readers become V6 slot readers (the macOS worker is never
executed), torch/torchvision CPU wheels report their public version to the two pinned version guards, and Paddle uses
the configured thread count. The vendored runtime's own source, weight, gallery and profile checks stay active.
"""
from copy import deepcopy
import functools
import gc
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import types

ARM, DETECTOR = 'v6_medium', 'v6_medium_native'
VISION_PLAN = 'runs/vision-text-v1/plan.json'
SLOT_CONSTRUCTORS = {'scripts/run_vision_text.py': 'A', 'rshb_vine/conditional_line_profile.py': 'B'}
DEVICE_CLASSES = (('rshb_vine.models', 'Encoder'), ('rshb_vine.models', 'Detector'),
                  ('rshb_vine.encoder_artifact', 'ArtifactEncoder'), ('rshb_vine.label_detector', 'LabelDetector'),
                  ('rshb_vine.instance_detector', 'InstanceDetector'))
PACKAGES = ('torch', 'torchvision', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub', 'numpy', 'scipy',
            'Pillow', 'paddlepaddle', 'paddleocr', 'paddlex', 'lightglue', 'fastapi', 'starlette', 'uvicorn')
VERSION_GUARDS = ('rshb_vine/models.py', 'rshb_vine/label_first_release.py')
LOCAL_BUILDS = {'torch': '+cpu', 'torchvision': '+cpu'}
ADAPTER_FILES = ('wine_scanner_core/linux_v6.py', 'wine_scanner_core/provenance.py')


class Ledger:
    def __init__(self):
        self.devices, self.vision_shells, self.vision_leaks, self.blocked = [], [], [], []
        self.versions, self.paddle_threads, self.shells = {}, [], {}


LEDGER = Ledger()


def _caller(root, skip=()):
    """First frame outside this adapter, as a path relative to the vendored root."""
    frame = sys._getframe(2)
    while frame is not None:
        name = Path(frame.f_code.co_filename).resolve()
        try:
            rel = name.relative_to(root).as_posix()
        except ValueError:
            rel = name.as_posix()
        if not rel.endswith(ADAPTER_FILES) and rel not in skip:
            return rel, frame.f_lineno
        frame = frame.f_back
    return None, None


def memory():
    out = {}
    if sys.platform == 'linux':
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM'):
                out[key] = int(value.split()[0]) * 1024
    return out


def _install_devices(root):
    import rshb_vine.models as M
    for module in ('rshb_vine.encoder_artifact', 'rshb_vine.label_detector', 'rshb_vine.instance_detector'):
        if module in sys.modules:
            raise RuntimeError('Device-bearing module imported before the adapter: ' + module)
    vendored_checked = M.device_checked

    def device_checked(name):
        if name != 'cpu':
            raise RuntimeError('Linux CPU adapter: unmapped device request %r' % (name,))
        return vendored_checked(name)
    M.device_checked = device_checked
    for module_name, class_name in DEVICE_CLASSES:
        cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        original = cls.__init__
        signature = inspect.signature(original)

        @functools.wraps(original)
        def init(self, *args, __original=original, __signature=signature, __name=class_name, **kwargs):
            bound = __signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            requested = bound.arguments['device']
            if requested not in ('cpu', 'mps'):
                raise RuntimeError('Linux CPU adapter: unsupported device %r for %s' % (requested, __name))
            bound.arguments['device'] = 'cpu'
            file, line = _caller(root)
            LEDGER.devices.append({'class': __name, 'callsite': f'{file}:{line}', 'requested': requested,
                                   'actual': 'cpu'})
            __original(*bound.args, **bound.kwargs)
        cls.__init__ = init


def _install_paddle_threads(root, threads):
    from rshb_vine.ocr_v5_v6_v1 import reader as R
    LEDGER.paddle_threads.append({'component': 'ocr_v5_v6_v1.reader.RUNTIME', 'vendored': R.RUNTIME['cpu_threads'],
                                  'used': threads})
    R.RUNTIME['cpu_threads'] = threads
    from paddleocr import PaddleOCR
    original = PaddleOCR.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        file, line = _caller(root)
        LEDGER.paddle_threads.append({'component': 'paddleocr.PaddleOCR', 'callsite': f'{file}:{line}',
                                      'vendored': kwargs.get('cpu_threads'), 'used': threads})
        kwargs['cpu_threads'] = threads
        original(self, *args, **kwargs)
    PaddleOCR.__init__ = init


def _install_vision(root, shared, startup):
    from rshb_vine.io import read_json, sha256, verify
    import rshb_vine.vision_ocr as V
    from rshb_vine.ocr_v5_v6_v1 import pipeline as P
    plan = verify(read_json(root / VISION_PLAN))

    def init(self, executable):
        file, line = _caller(root, skip=('rshb_vine/vision_ocr.py',))
        slot = SLOT_CONSTRUCTORS.get(file)
        if slot is None:
            raise RuntimeError(f'VisionOCR constructed at an unmapped site {file}:{line}')
        self.executable = Path(executable).resolve()
        self.executable_sha256 = sha256(self.executable)
        if self.executable_sha256 != plan['binary_sha256']:
            raise ValueError('Vision worker binary differs from the pinned plan')
        self.info = deepcopy(plan['reader_info'])
        self.lock, self.worker = threading.Lock(), None
        LEDGER.shells[id(self)] = (self, slot, P.SlotReader(slot, ARM, shared(), startup, 'paddle'))
        LEDGER.vision_shells.append({'slot': slot, 'callsite': f'{file}:{line}',
                                     'executable_sha256': self.executable_sha256})

    def forbidden(self, *args, **kwargs):
        file, line = _caller(root, skip=('rshb_vine/vision_ocr.py',))
        LEDGER.vision_leaks.append({'callsite': f'{file}:{line}', 'at': time.time()})
        raise RuntimeError('Apple Vision worker is forbidden in the Linux CPU runtime')

    class ShellRead:
        """Resolves to the shell's own SlotReader.read, so no adapter frame precedes the vendored OCR callsite."""

        def __get__(self, obj, cls=None):
            if obj is None:
                return self
            shell = LEDGER.shells.get(id(obj))
            if shell is None or shell[0] is not obj:
                return types.MethodType(forbidden, obj)
            return shell[2].read
    V.VisionOCR.__init__ = init
    V.VisionOCR._start_worker = V.VisionOCR._response_line = forbidden
    V.VisionOCR.read = ShellRead()
    V.VisionOCR.close = lambda self: None
    return V.VisionOCR


def _install_version_mapping(root):
    """The pinned guards name the macOS wheel versions; the Linux CPU wheels of the same release carry +cpu."""
    original = importlib.metadata.version

    def version(name):
        actual = original(name)
        label = LOCAL_BUILDS.get(name)
        file, line = _caller(root)
        if label and actual.endswith(label) and file in VERSION_GUARDS:
            public = actual[:-len(label)]
            LEDGER.versions.setdefault(f'{file}:{line}', {})[name] = {'installed': actual, 'reported_to_guard': public}
            return public
        return actual
    importlib.metadata.version = version


def _install_exec_guard(root):
    original = subprocess.Popen.__init__

    @functools.wraps(original)
    def init(self, args, *rest, **kwargs):
        argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
        if argv and Path(os.fsdecode(argv[0]).split()[0]).name == 'vision-ocr':
            file, line = _caller(root)
            LEDGER.blocked.append({'exec': os.fsdecode(argv[0]), 'callsite': f'{file}:{line}'})
            raise PermissionError('Apple Vision worker execution is forbidden in the Linux CPU runtime')
        original(self, args, *rest, **kwargs)
    subprocess.Popen.__init__ = init


def torch_census():
    import torch
    modules = [o for o in gc.get_objects() if isinstance(o, torch.nn.Module)]
    devices, storages = {}, {}
    for module in modules:
        for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            devices[tensor.device.type] = devices.get(tensor.device.type, 0) + 1
            storage = tensor.untyped_storage()
            storages[(tensor.device.type, storage.data_ptr())] = storage.nbytes()
    return {'tensor_devices': devices, 'unique_storage_bytes': sum(storages.values()), 'modules': len(modules),
            'threads': torch.get_num_threads()}


def health(calls, leaks):
    lines = [line for call in calls for line in call.get('lines') or []]
    return {'reader_errors': sum(1 for c in calls if c.get('error')),
            'line_reader_errors': sum(1 for line in lines if line.get('drop_reason') == 'reader_error'),
            'unexpected_callsites': sum(1 for c in calls if c['crop_context'] == 'unexpected_callsite'),
            'vision_leaks': len(leaks), 'blocked_exec': len(LEDGER.blocked)}


class LinuxV6Runtime:
    """The vendored parent release with its OCR readers swapped; ``recognize`` returns the parent response."""

    def __init__(self, root, threads, provenance):
        started = time.perf_counter()
        root = Path(root).resolve()
        self.memory = {'start': memory()}
        _install_exec_guard(root)
        _install_version_mapping(root)
        provenance.install()
        import torch
        torch.set_num_threads(threads)
        from rshb_vine.io import sha256
        from rshb_vine.ocr_v5_v6_v1 import pipeline as P
        from rshb_vine.ocr_v5_v6_v1.reader import PaddleLineReader
        self.descriptor = P.adapter_descriptor(ARM, DETECTOR)
        if self.descriptor['checksum'] != provenance.expected['v6_descriptor']:
            raise ValueError('V6 adapter descriptor differs from the exported one')
        if sha256(root / P.PROFILE) != P.PROFILE_FILE_SHA256:
            raise ValueError('Active profile file changed')
        _install_devices(root)
        _install_paddle_threads(root, threads)
        holder = {}

        def shared():
            if 'reader' not in holder:
                holder['reader'] = PaddleLineReader(ARM, root, detector=DETECTOR)
                if holder['reader'].info()['models']['detector']['sha256'] != self.descriptor['detector']['sha256']:
                    raise RuntimeError('Loaded detector differs from the adapter descriptor')
            return holder['reader']
        self.startup = P.RequestTrace()
        self.startup.begin()
        VisionOCR = _install_vision(root, shared, self.startup)
        from rshb_vine.models import OCR
        from rshb_vine.recognition_factory import load_runtime
        parent = load_runtime(root, root / P.PROFILE)
        self.startup.end()
        if parent.profile['checksum'] != P.PROFILE_CHECKSUM or parent.manifest['checksum'] != P.RUNTIME_CHECKSUM:
            raise ValueError('Loaded runtime is not the pinned parent profile/runtime')
        provenance.check_used()
        self.parent = parent
        self.ledger = P.RequestTrace()
        guarded = [VisionOCR, OCR]
        rotated = sys.modules.get('rshb_vine.reading_recovery_v1.rotated_ocr')
        if rotated is not None:
            guarded.append(rotated.RotatedSparseReread)
        found, _ = P.census(parent, tuple(guarded))
        shells = [e for e in found.values() if isinstance(e['object'], VisionOCR)]
        sparse = [e for e in found.values() if type(e['object']) is OCR]
        if len(shells) + len(sparse) != len(found):
            raise RuntimeError('An uninstalled reader stage is reachable: %s' % sorted(e['type'] for e in found.values()))
        slots = {}
        for entry in shells:
            names = {P._slot_of(r) for r in entry['refs']} - {None}
            if len(names) != 1 or names & set(slots):
                raise RuntimeError('Reader shell does not map to exactly one slot')
            slot = names.pop()
            if LEDGER.shells[id(entry['object'])][1] != slot:
                raise RuntimeError('Constructor slot differs from graph slot ' + slot)
            slots[slot] = entry
        if set(slots) != set(P.SLOTS) or len(sparse) != 1:
            raise RuntimeError('Unexpected reader census: shells=%d sparse=%d' % (len(shells), len(sparse)))
        for slot, entry in slots.items():
            reader = P.SlotReader(slot, ARM, shared(), self.ledger, 'paddle')
            for ref in entry['refs']:
                P._rebind(ref, reader)
        for ref in sparse[0]['refs']:
            P._rebind(ref, P.SlotReader('sparse', ARM, sparse[0]['object'], self.ledger, 'sparse_existing'))
        for entry in shells:
            entry['object'].read = types.MethodType(VisionOCR._start_worker, entry['object'])
        if P.census(parent, tuple(guarded))[0]:
            raise RuntimeError('Reader references remain after the swap')
        if LEDGER.vision_leaks or LEDGER.blocked:
            raise RuntimeError('Apple Vision was reached: %s' % (LEDGER.vision_leaks or LEDGER.blocked))
        self.startup_health = health(self.startup.calls, LEDGER.vision_leaks)
        if any(self.startup_health.values()):
            raise RuntimeError('Startup OCR is not clean: %s' % self.startup_health)
        census = torch_census()
        if set(census['tensor_devices']) != {'cpu'}:
            raise RuntimeError('Non-CPU tensors after load: %s' % census['tensor_devices'])
        gc.collect()
        self.memory['loaded'] = memory()
        self.load_seconds = time.perf_counter() - started
        self.execution = {
            'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version.split()[0],
            'cpu_count': os.cpu_count(), 'device': 'cpu', 'torch_threads': census['threads'], 'paddle_threads': threads,
            'ocr': {'A': 'PP-OCRv6 medium det+rec (v6_medium_native)', 'B': 'PP-OCRv6 medium det+rec (v6_medium_native)',
                    'sparse': 'PP-OCRv5 mobile det + eslav_PP-OCRv5_mobile_rec (vendored models.OCR)'},
            'apple_vision_executed': False, 'packages': {p: importlib.metadata.version(p) for p in PACKAGES},
            'deviations': {'device': LEDGER.devices, 'paddle_threads': LEDGER.paddle_threads,
                           'vision_shells': LEDGER.vision_shells, 'package_version_mapping': LEDGER.versions,
                           'vision_reader_info_source': VISION_PLAN + ' (pinned, not observed)'},
            'reader_slots': {s: sorted(r['path'] for r in e['refs']) for s, e in slots.items()},
            'startup_ocr_calls': len(self.startup.calls), 'torch': census}

    def recognize(self, data, roi=None, bottles='addressed'):
        self.ledger.begin()
        try:
            started = time.perf_counter()
            result = self.parent.recognize(data, roi, bottles)
            seconds = time.perf_counter() - started
        finally:
            calls, _ = self.ledger.end()
        return result, {'ocr_calls': len(calls), 'health': health(calls, LEDGER.vision_leaks), 'seconds': seconds}
