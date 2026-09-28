"""Build the pinned target-contract-v2 runtime once, then substitute only the B3 encoder/index in place.

Upper layers (GuardedConsensus replay, selector arms) read holder attributes at call time, so S0 and the
selector both see the substituted B3. B0, the object gate, detector, OCR, ranker and resolver objects stay.
"""
import numpy as np

from rshb_vine.io import read_json, seal, sha256, verify
from rshb_vine.catalog_training_evaluation_v2 import pins, reindex


def load_pinned():
    from rshb_vine.recognition_factory import load_runtime
    profile = pins.ROOT / pins.PROFILE
    if sha256(profile) != pins.PROFILE_SHA256 or verify(read_json(profile))['checksum'] != pins.PROFILE_CHECKSUM:
        raise ValueError('Pinned target-contract-v2 profile changed')
    pipeline = load_runtime(pins.ROOT, profile)
    if pipeline.profile['checksum'] != pins.PROFILE_CHECKSUM:
        raise ValueError('Loaded runtime is not the pinned profile')
    return pipeline


def recognition_runtime(pipeline):
    runtime = pipeline.inner.parent.runtime
    if type(runtime).__name__ != 'RecognitionRuntime' or type(runtime.holders['B3']).__name__ != 'GuardedConsensus':
        raise ValueError('Runtime chain ownership changed')
    return runtime


def _unwrap(obj, name):
    return getattr(obj, name, obj)


def arm_components(arm, model_path, gallery_dir, device):
    """Fresh encoder model and LabelFirstIndex for one arm; gallery must be bound to that encoder."""
    from rshb_vine.visual_core import LabelFirstIndex
    if gallery_dir == 'current':
        gallery, vectors = pins.read('current_gallery', sealed=True), np.load(pins.checked('current_vectors'))
        if gallery['encoder_id'] != arm['encoder_id']:
            raise ValueError('Current B3 gallery is only valid for parent A')
    elif gallery_dir == 'white':
        gallery, vectors = pins.read('white_gallery', sealed=True), np.load(pins.checked('white_vectors'))
        if gallery['encoder_id'] != arm['encoder_id']:
            raise ValueError('White B3 gallery is only valid for parent A')
    else:
        gallery, vectors = reindex.load_gallery(gallery_dir, arm)
    refs, _ = reindex.references()
    if gallery['references'] != refs:
        raise ValueError('Arm gallery reference order differs from serving order')
    encoder = reindex.load_encoder_model(model_path, device)
    return encoder, LabelFirstIndex(vectors, gallery['references'], arm['encoder_id']), gallery


def _retire(obj, names):
    """Any later call through a stale reference to a replaced parent object fails instead of silently scoring A."""
    def tripwire(*_, **__):
        raise RuntimeError('Stale reference to the replaced parent B3 object')
    for name in names:
        setattr(obj, name, tripwire)


def substitute(pipeline, arm, encoder, index, gallery):
    """Replace B3 holder eid/encoder/index; new request cache rewraps both arms so memo keys follow the new encoder."""
    from rshb_vine.request_visual_cache import RequestVisualCache
    runtime = recognition_runtime(pipeline)
    base, core = runtime.holders['B0'], runtime.holders['B3']
    b0_eid = runtime.arms['B0'][0]
    old_b3 = runtime.arms['B3'][0]
    if base.encoder_id != b0_eid or core.eid != old_b3 or old_b3 != pins.PARENT_EID:
        raise ValueError('Runtime arms differ from the frozen control binding before substitution')
    b0_encoder, b0_index = _unwrap(base.encoder, '_encoder'), _unwrap(base.index, '_index')
    old_encoder, old_index = _unwrap(core.encoder, '_encoder'), _unwrap(core.index, '_index')
    if old_encoder is b0_encoder or old_encoder is encoder or old_index is index:
        raise ValueError('Substitution must use fresh B3 objects distinct from B0 and the parent')
    cache = RequestVisualCache(capture_retrieval=runtime.selection is not None)
    base.encoder = cache.wrap(b0_encoder, b0_eid)
    core.eid, core.encoder = arm['encoder_id'], cache.wrap(encoder, arm['encoder_id'])
    if runtime.selection is not None:
        base.index = cache.wrap_index(b0_index, b0_eid)
        core.index = cache.wrap_index(index, arm['encoder_id'])
    else:
        base.index, core.index = b0_index, index
    runtime.visual_cache = cache
    runtime.arms = {'B0': (b0_eid, base.encoder, base.index), 'B3': (core.eid, core.encoder, core.index)}
    _retire(old_encoder, ('encode', 'features'))
    _retire(old_index, ('search', 'reuse'))
    return seal({'kind': 'catalog-training-v2-b3-substitution-v1', 'profile_checksum': pins.PROFILE_CHECKSUM,
                 'parent_runtime_manifest': runtime.manifest['checksum'], 'arm': arm,
                 'B0_encoder_id_unchanged': b0_eid, 'B3_parent_encoder_id': old_b3,
                 'B3_gallery_checksum': gallery['checksum'], 'B3_vectors_sha256': gallery['vectors_sha'],
                 'request_cache': 'new instance, both arms rewrapped, memo keyed by substituted encoder_id',
                 'replaced_parent_B3_objects': 'tripwired: encode/features/search/reuse raise on stale use',
                 'frozen_sources_changed': False, 'release_admitted': False, 'calibration_status': 'unknown'})
