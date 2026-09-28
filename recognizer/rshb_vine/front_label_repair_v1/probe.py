"""Fixed-bounds crop counterfactual on the frozen B3-4343 encoder and release gallery (no pipeline, no fit)."""
from pathlib import Path
import time

import numpy as np

from rshb_vine.io import sha256
from rshb_vine.preprocessing import checked_box, decode
from rshb_vine.visual_core import View

RELEASE = 'config/recognition-b3-only-v1-release.json'


def load_arm(root, device='mps'):
    """B3-4343 encoder and gallery through the release's own profile and gallery-migration checks."""
    from rshb_vine.b3_only_v1 import release
    from rshb_vine.b3_only_v1.runtime import ARM_INPUTS
    from rshb_vine.catalog_training_evaluation_v2 import reindex
    from rshb_vine.visual_core import LabelFirstIndex
    root = Path(root)
    profile = release.load_profile(root, RELEASE)
    a = {k: str(root / v) for k, v in ARM_INPUTS.items()}
    arm, model_path = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                          a['fit_admission'])
    if arm['encoder_id'] != profile['b3_encoder_id']:
        raise ValueError('Probe arm differs from the release B3 encoder')
    gallery, vectors = release._gallery(root, profile, arm)
    encoder = reindex.load_encoder_model(model_path, device)
    return arm, encoder, LabelFirstIndex(vectors, gallery['references'], arm['encoder_id']), gallery


def channel_ranks(index, vectors, kinds, size, gt, related):
    """Per-variant GT slug rank inside one channel using LabelFirstIndex.search with a single query view."""
    out = []
    for vec, kind in zip(vectors, kinds):
        view = View('detected_label' if kind == 'label' else 'context', [0, 0, 1, 1], 'probe', None, size)
        found = index.search(vec[None], [view], index.encoder_id)
        channel = 'front_label' if kind == 'label' else 'context'
        scores = found['channel_top20'][channel]
        full = sorted(_channel_scores(index, vec, channel).items(), key=lambda kv: -kv[1])
        order = [s for s, _ in full]
        rank = {s: order.index(s) + 1 for s in gt + related if s in order}
        out.append({'channel': channel, 'top3': [[r['slug'], round(r['score'], 4)] for r in scores[:3]],
                    'gt_rank': min((rank[s] for s in gt if s in rank), default=None),
                    'gt_score': round(max(v for s, v in full if s in gt), 4),
                    'related_ranks': {s: rank.get(s) for s in related}})
    return out


def _channel_scores(index, vec, channel):
    ids = index.channel_ids[channel]
    values = index.vectors[ids] @ vec
    best = {}
    for i, v in zip(ids, values):
        slug = index.references[i]['slug']
        best[slug] = max(best.get(slug, -9), float(v))
    return best


def run(root, cases, out_dir, device='mps'):
    root = Path(root).resolve()
    started = time.perf_counter()
    arm, encoder, index, gallery = load_arm(root, device)
    loaded = time.perf_counter()
    rows, crops_encoded = [], 0
    for case in cases:
        data = (root / case['path']).read_bytes()
        if sha256(root / case['path']) != case['sha256']:
            raise ValueError('Query SHA changed: ' + case['path'])
        image, _ = decode(data)
        boxes = [checked_box(v['bbox'], image.size) for v in case['variants']]
        pics = [image.crop(b) for b in boxes]
        for v, pic in zip(case['variants'], pics):
            pic.save(Path(out_dir) / ('%s__%s.jpg' % (case['id'], v['name'])), quality=88)
        vecs = np.asarray(encoder.encode(pics), dtype='float32')
        crops_encoded += len(pics)
        kinds = [v['channel'] for v in case['variants']]
        ranks = channel_ranks(index, vecs, kinds, list(image.size), case['gt'], case.get('related', []))
        for v, b, r in zip(case['variants'], boxes, ranks):
            rows.append({'id': case['id'], 'variant': v['name'], 'bbox': b, 'basis': v['basis'], **r})
    return {'arm_encoder_id': arm['encoder_id'], 'gallery_checksum': gallery['checksum'],
            'vectors_sha256': gallery['vectors_sha'], 'device': device, 'images': len(cases),
            'crops_encoded': crops_encoded, 'load_seconds': loaded - started,
            'inference_seconds': time.perf_counter() - loaded, 'rows': rows}
