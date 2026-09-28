"""Encoder identity for parent A / exported B and full 4206-view B3 gallery reindex (production white decode)."""
from hashlib import sha256 as _sha256
import io
from pathlib import Path

import numpy as np

from rshb_vine.io import digest, read_json, seal, sha256, verify, write_json
from rshb_vine.catalog_training_evaluation_v2 import pins

PREFIX, ADDED, SOURCES = 3998, 208, 2103


def encoder_arm(encoder_dir=None, export_receipt=None, recipe=None, checkpoint=None, fit_admission=None):
    """Return (arm descriptor, model path). Parent A uses its historical receipt; B its own stage5 export chain."""
    from rshb_vine.encoder_artifact import encoder_spec
    if encoder_dir is None:
        if any(x is not None for x in (export_receipt, recipe, checkpoint, fit_admission)):
            raise ValueError('Parent A takes no export/recipe/checkpoint inputs')
        spec = encoder_spec(pins.ROOT, pins.PARENT_ENCODER)
        if spec['encoder_id'] != pins.PARENT_EID:
            raise ValueError('Parent B3 encoder identity changed')
        return {'arm': 'A', 'encoder_id': spec['encoder_id'], 'encoder_dir': pins.PARENT_ENCODER,
                'files': spec['files'], 'export': None}, spec['path']
    if any(x is None for x in (export_receipt, recipe, checkpoint, fit_admission)):
        raise ValueError('B arm needs export receipt, exact recipe, checkpoint and fit admission')
    from rshb_vine.catalog_training_v2 import prepare, trainer
    pins.checked('trainer')
    pins.checked('trainer_contract')
    folder, receipt_path = Path(encoder_dir).resolve(), Path(export_receipt).resolve()
    recipe_path, checkpoint_path = Path(recipe).resolve(), Path(checkpoint).resolve()
    receipt, admission = read_json(receipt_path), read_json(Path(fit_admission).resolve())
    recipe_sha = sha256(recipe_path)
    trainer.verify_recipe(recipe_path, prepare.validate_runtime_seal())
    steps = pins.checkpoint_steps()
    if (folder.name != 'encoder' or receipt.get('status') != 'exported_unreleased'
            or receipt.get('gallery_reindex_required') is not True
            or receipt.get('global_step') not in steps['selectable'] + steps['diagnostic_only']
            or receipt.get('diagnostic_only') != (receipt['global_step'] in steps['diagnostic_only'])):
        raise ValueError('Stage5 export receipt for a frozen checkpoint step is required')
    if receipt.get('recipe_sha256') != recipe_sha or admission.get('exact_recipe_sha256') != recipe_sha:
        raise ValueError('Export, recipe and fit admission do not bind the same exact recipe')
    if receipt.get('checkpoint_sha256') != sha256(checkpoint_path):
        raise ValueError('Checkpoint bytes differ from the export receipt')
    files = {'encoder/' + name: value for name, value in receipt['files'].items()}
    if not any(k.endswith('.safetensors') for k in files) or sorted(receipt['files']) != sorted(
            p.name for p in folder.iterdir() if p.is_file()):
        raise ValueError('Export directory differs from its receipt file list')
    for name, expected in receipt['files'].items():
        if sha256(folder / name) != expected:
            raise ValueError('Exported encoder file changed: ' + name)
    identity = digest({'files': files, 'source': sha256(pins.ROOT / 'rshb_vine/models.py'),
                       'preprocessing': pins.PREPROCESSING})
    if identity == pins.PARENT_EID:
        raise ValueError('Exported encoder is byte-identical to parent A')
    return {'arm': 'B%d' % receipt['global_step'], 'encoder_id': identity, 'encoder_dir': str(folder),
            'files': files, 'export': {'receipt_sha256': sha256(receipt_path), 'recipe_sha256': recipe_sha,
                                       'checkpoint_sha256': receipt['checkpoint_sha256'],
                                       'fit_admission_sha256': sha256(Path(fit_admission).resolve()),
                                       'global_step': receipt['global_step'],
                                       'diagnostic_only': receipt['diagnostic_only']}}, folder


def load_encoder_model(model_path, device):
    from rshb_vine.encoder_artifact import ArtifactEncoder
    return ArtifactEncoder(model_path, device)


def references():
    """Current B3 gallery references in serving order, joined to their byte-verified source paths."""
    gallery = pins.read('current_gallery', sealed=True)
    frozen = pins.read('references')
    refs = gallery['references']
    if (len(refs) != PREFIX + ADDED or len(frozen) != len(refs)
            or not all(r['index'] == i and all(r[k] == g[k] for k in ('slug', 'kind', 'image_sha256', 'bbox'))
                       for i, (r, g) in enumerate(zip(frozen, refs)))):
        raise ValueError('Frozen source/reference order differs from the serving B3 gallery')
    return refs, [r['source_path'] for r in frozen]


def reindex(encoder, refs, sources, batch):
    from rshb_vine.preprocessing import checked_box, decode
    from rshb_vine.visual_core import validate_vectors
    vectors, pics, verified = [], [], set()
    previous, image = None, None
    for ref, source in zip(refs, sources):
        if ref['image_sha256'] != previous:
            data = (pins.ROOT / source).read_bytes()
            if _sha256(data).hexdigest() != ref['image_sha256']:
                raise ValueError('Canonical reference source SHA changed: ' + source)
            image, _ = decode(data, max_bytes=1 << 30, max_pixels=36_000_000)
            previous = ref['image_sha256']
            verified.add(previous)
        if list(image.size) != ref['original_size'] or ref['coordinate_space'] != 'EXIF-oriented original pixels':
            raise ValueError('Canonical reference EXIF geometry changed')
        pics.append(image.crop(checked_box(ref['bbox'], image.size)))
        if len(pics) >= batch:
            vectors.extend(encoder.encode(pics))
            pics.clear()
    if pics:
        vectors.extend(encoder.encode(pics))
    if len(verified) != SOURCES:
        raise ValueError('Canonical reference source coverage incomplete')
    array = np.asarray(vectors, dtype=np.float32)
    validate_vectors(array, len(refs))
    return array


def write_gallery(out, arm, refs, array, extra):
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    out.mkdir(parents=True)
    (out / 'vectors.npy').write_bytes(buf.getvalue())
    gallery = seal({'encoder_id': arm['encoder_id'], 'references': refs, 'vectors_sha': sha256(out / 'vectors.npy'),
                    'weights_fit': arm['arm'] != 'A', 'reindex_of': pins.FILES['current_gallery'][1],
                    'decode': 'production white RGBA->RGB, EXIF, 36MP offline limit', 'release_admitted': False})
    write_json(out / 'gallery.json', gallery)
    receipt = seal({'kind': 'catalog-training-v2-b3-gallery-reindex-v1', 'arm': arm, 'references': len(refs),
                    'gallery_checksum': gallery['checksum'], 'gallery_sha256': sha256(out / 'gallery.json'),
                    'vectors_sha256': gallery['vectors_sha'], 'code_sha256': pins.code_sha(), **extra,
                    'current_or_release_changed': False})
    write_json(out / 'receipt.json', receipt)
    return receipt


def parent_parity(array):
    """A reindex vs signed white parent vectors and the serving current prefix."""
    signed = np.load(pins.checked('signed_white_parent_vectors'), allow_pickle=False)
    current = np.load(pins.checked('current_vectors'), allow_pickle=False)
    white = np.load(pins.checked('white_vectors'), allow_pickle=False)
    return {'max_abs_vs_signed_white_parent': float(np.max(np.abs(array - signed))),
            'max_abs_vs_current_prefix3998': float(np.max(np.abs(array[:PREFIX] - current[:PREFIX]))),
            'max_abs_vs_white_candidate': float(np.max(np.abs(array - white))),
            'max_abs_vs_current_additions208': float(np.max(np.abs(array[PREFIX:] - current[PREFIX:])))}


def load_gallery(folder, arm):
    """A write_gallery output: receipt, frozen code, arm/export provenance and bytes must all bind."""
    folder = Path(folder).resolve()
    receipt = verify(read_json(folder / 'receipt.json'))
    gallery = verify(read_json(folder / 'gallery.json'))
    if (receipt.get('kind') != 'catalog-training-v2-b3-gallery-reindex-v1' or receipt['arm'] != arm
            or receipt['code_sha256'] != pins.code_sha()
            or receipt['gallery_checksum'] != gallery['checksum'] or receipt['gallery_sha256'] != sha256(folder / 'gallery.json')
            or gallery['encoder_id'] != arm['encoder_id'] or receipt['references'] != PREFIX + ADDED
            or sha256(folder / 'vectors.npy') != gallery['vectors_sha'] or receipt['vectors_sha256'] != gallery['vectors_sha']):
        raise ValueError('Reindexed gallery provenance does not bind this arm: ' + str(folder))
    # Same 3e-6 bound prepare_catalog_additions_alpha_repair.py uses for this parent re-encode parity.
    if arm['arm'] == 'A' and not receipt['parent_parity']['max_abs_vs_signed_white_parent'] <= 3e-6:
        raise ValueError('A reindex does not reproduce the signed white parent vectors')
    refs, _ = references()
    if gallery['references'] != refs:
        raise ValueError('Gallery reference order differs from serving B3 order')
    return gallery, np.load(folder / 'vectors.npy', allow_pickle=False)
