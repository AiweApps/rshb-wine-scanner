"""Front-label repair candidate over the loaded B3-only release object (profile 83e97cb3, never the current pointer).

Installed per instance, frozen sources untouched:
  RepairSplitStage   directly before SplitWineObjectStage in the object-gate instance MRO; records the image
                     the split verifier crops from
  RepairIndex        wraps base.index; in split-verifier searches ([detected_bottle, automatic_main_label]) a
                     recovered label replaces the served label view, so downstream selection and OCR stages that
                     read the target views see the full label; the served box is kept in the trace only
  orphan retry       wrapped; inside a retry whose PNG pixels equal the crop of an orphan-stage SSDlite box, the
                     crop frame counts as that SSDlite detection
Rotated (orientation_normalized_*) and all other searches pass through unchanged.
"""
from contextvars import ContextVar
from copy import deepcopy
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image

from rshb_vine.b3_only_v1.split import SplitWineObjectStage
from rshb_vine.front_label_repair_v1 import geometry as G
from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.preprocessing import checked_box
from rshb_vine.visual_core import View
from rshb_vine.wine_object_profile import WineObjectStage

KIND = 'front-label-repair-v1-candidate-profile'
BASE_PROFILE = 'config/recognition-b3-only-v1-release.json'
BASE_CHECKSUM = '83e97cb3f6e8d1a89451b1ffcdd8de4e20f6417706b488f1e4d426c538cafbbb'
PROFILE = 'runs/recognition-repair-20260926/crop/candidate/profile.json'
SOURCES = ('rshb_vine/front_label_repair_v1/__init__.py', 'rshb_vine/front_label_repair_v1/geometry.py',
           'rshb_vine/front_label_repair_v1/runtime.py', 'scripts/front_label_repair_v1.py')
SPLIT_SOURCES = ('detected_bottle', 'automatic_main_label')
VIEW_SOURCE = 'two_detector_label_boundary'

IMAGE = ContextVar('front_label_repair_image', default=None)
ORPHAN_FRAME = ContextVar('front_label_repair_orphan_frame', default=None)
ORPHAN_IMAGE = ContextVar('front_label_repair_orphan_image', default=None)
TRACES = ContextVar('front_label_repair_traces', default=None)


class RepairSplitStage(SplitWineObjectStage):
    def recognize_image(self, image, roi=None):
        token = IMAGE.set(image)
        try:
            return super().recognize_image(image, roi)
        finally:
            IMAGE.reset(token)


class RepairIndex:
    def __init__(self, index, encoder_holder, ssd):
        self._index, self._holder, self._ssd = index, encoder_holder, ssd

    def __getattr__(self, name):
        return getattr(self._index, name)

    def reuse(self, images, views, encoder_id):
        return self._index.reuse(images, views, encoder_id)

    def search(self, query, views, encoder_id):
        image, traces = IMAGE.get(), TRACES.get()
        if (image is None or traces is None or len(views) != 2
                or tuple(v.source for v in views) != SPLIT_SOURCES
                or tuple(v.kind for v in views) != ('context', 'detected_label')
                or any(list(v.original_size) != list(image.size) for v in views)):
            return self._index.search(query, views, encoder_id)
        context, label = views
        ssd = [{'bbox': r['bbox'], 'score': r['detector_score'], 'source': 'ssdlite_same_pixels'}
               for r in self._ssd.detect(image)]
        orphan = ORPHAN_FRAME.get()
        if orphan is not None and tuple(orphan['size']) == tuple(image.size):
            ssd.append({'bbox': [0, 0, *image.size], 'score': orphan['score'], 'source': 'orphan_stage_ssd_crop'})
        box, trace = G.recovered_label(label.bbox, context.bbox, ssd)
        trace.update(image_size=list(image.size), orphan_retry=orphan is not None,
                     orphan_ssd_bbox=orphan['bbox'] if orphan else None)
        traces.append(trace)
        if box is None:
            return self._index.search(query, views, encoder_id)
        box = checked_box(box, image.size)
        primary = View('detected_label', box, VIEW_SOURCE, None, list(image.size), G.POLICY)
        vector = np.asarray(self._holder.encoder.encode([image.crop(box)]), dtype='float32')
        query = np.asarray(query, dtype='float32')
        return self._index.search(np.concatenate([query[:1], vector]), [context, primary], encoder_id)


def _ssd_provenance(image, detections, data):
    """The orphan-stage SSDlite detection whose exact crop pixels were sent to this retry, else None."""
    if image is None:
        return None
    with Image.open(io.BytesIO(data)) as im:
        sent = np.asarray(im.convert('RGB'))
    for d in detections:
        box = checked_box(d['bbox'], image.size)
        if d['detector_score'] >= G.SSD_SCORE and sent.shape[1::-1] == (box[2] - box[0], box[3] - box[1]) \
                and np.array_equal(sent, np.asarray(image.crop(box).convert('RGB'))):
            return {'size': list(sent.shape[1::-1]), 'bbox': list(box), 'score': float(d['detector_score'])}
    return None


def install(release):
    instance = release.runtime.control.expanded.core.parent.parent.parent
    cls = type(instance)
    if not isinstance(instance, SplitWineObjectStage):
        raise ValueError('Object-gate instance is not the B3-only split stage')
    repaired = type('FrontLabelRepair' + cls.__name__, (cls, RepairSplitStage), {'__module__': __name__})
    mro = repaired.__mro__
    if not (mro.index(RepairSplitStage) + 1 == mro.index(SplitWineObjectStage)
            and mro.index(SplitWineObjectStage) + 1 == mro.index(WineObjectStage) and mro[1] is cls):
        raise ValueError('Repair stage is not directly before SplitWineObjectStage')
    orphan = release.geometry.inner.inner.parent.route.runtime.control.orphans
    if type(orphan).__name__ != 'GeometryOrphanRecovery':
        raise ValueError('Loaded orphan-label stage is not GeometryOrphanRecovery')
    ssd = getattr(orphan.labels, 'detector', None)
    if type(orphan.labels).__name__ != 'CapturingLabels' or type(ssd).__name__ != 'LabelDetector':
        raise ValueError('Orphan stage label detector is not the SSDlite main-label detector')
    retry, apply = orphan.retry, orphan.apply

    def orphan_apply(data, baseline, roi=None):
        from rshb_vine.preprocessing import decode
        token = ORPHAN_IMAGE.set(decode(data)[0] if roi is None and baseline.get('decision') != 'invalid_image' else None)
        try:
            return apply(data, baseline, roi)
        finally:
            ORPHAN_IMAGE.reset(token)

    def orphan_retry(data):
        token = ORPHAN_FRAME.set(_ssd_provenance(ORPHAN_IMAGE.get(), orphan.labels.last, data))
        try:
            return retry(data)
        finally:
            ORPHAN_FRAME.reset(token)
    orphan.apply = orphan_apply
    orphan.retry = orphan_retry
    base = instance.base
    base.index = RepairIndex(base.index, base, ssd)
    instance.__class__ = repaired
    return {'instance_mro': [c.__name__ for c in mro], 'orphan_stage': type(orphan).__module__,
            'ssd_model_id': ssd.model_id, 'base_encoder_id': base.encoder_id,
            'index': 'base.index wrapped; split-verifier searches only'}


def load_profile(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile['base_profile'] != BASE_PROFILE or profile['base_checksum'] != BASE_CHECKSUM:
        raise ValueError('Unsupported front-label repair profile')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('Front-label repair pinned file changed: ' + path)
    return profile


def freeze_body(root):
    root = Path(root).resolve()
    base = verify(read_json(root / BASE_PROFILE))
    if base['checksum'] != BASE_CHECKSUM:
        raise ValueError('Base B3-only release profile changed')
    pins = {p: sha256(root / p) for p in SOURCES}
    pins[BASE_PROFILE] = sha256(root / BASE_PROFILE)
    pins['runs/label-detector-real-v2/manifest.json'] = sha256(root / 'runs/label-detector-real-v2/manifest.json')
    return {'kind': KIND, 'name': 'front-label-repair-v1-candidate', 'policy': G.POLICY,
            'base_profile': BASE_PROFILE, 'base_checksum': BASE_CHECKSUM, 'pins_sha256': pins,
            'constants': {'ssd_score': G.SSD_SCORE, 'same_label_iou': G.SAME_LABEL_IOU,
                          'min_growth': G.MIN_GROWTH},
            'port': 8188, 'release_status': 'experimental_candidate_not_admitted', 'activated': False,
            'calibrated': False, 'probability': None, 'fit_run': False,
            'rollback': 'stop 8188; 8175 and config/recognition-current.json are never modified by this candidate'}


class FrontLabelRepairRecognition:
    def __init__(self, root, profile_path=PROFILE):
        from rshb_vine.b3_only_v1.release import B3OnlyRelease
        root = Path(root).resolve()
        self.profile = load_profile(root, profile_path)
        self.release = B3OnlyRelease(root, BASE_PROFILE)
        if self.release.profile['checksum'] != BASE_CHECKSUM:
            raise ValueError('Loaded B3-only release differs from the pinned base')
        self.installation = install(self.release)
        self.manifest = seal({'kind': 'front-label-repair-v1-runtime', 'policy': G.POLICY,
                              'profile_checksum': self.profile['checksum'],
                              'runtime_descriptor_checksum': self.profile['checksum'],
                              'parent_runtime': self.release.manifest['checksum'],
                              'base_profile_checksum': BASE_CHECKSUM, 'installation': self.installation,
                              'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        started = time.perf_counter()
        token = TRACES.set([])
        try:
            result = self.release.recognize(data, roi, bottles)
            traces = deepcopy(TRACES.get())
        finally:
            TRACES.reset(token)
        if result.get('decision') == 'invalid_image':
            return result
        added = [t['recovered'] for t in traces if t['added']]
        served = [{'instance_id': str(t['instance_id']),
                   'recovered_views': [v['bbox'] for v in t['retrieval'].get('views', []) if v.get('source') == VIEW_SOURCE]}
                  for t in result.get('targets', [])]
        result['front_label_repair_v1'] = {
            'policy': G.POLICY, 'profile_checksum': self.profile['checksum'],
            'runtime_checksum': self.manifest['checksum'], 'base_profile_checksum': BASE_CHECKSUM,
            'split_calls': traces, 'views_added': len(added), 'targets': served,
            'calibrated': False, 'probability': None, 'release_admitted': False,
            'seconds': time.perf_counter() - started}
        return result
