"""The appended B3 gallery: the frozen 4206 rows and vectors byte-for-byte, then 4 rows encoded by frozen B4343.

``build`` is the only function that loads a model; it encodes exactly the admitted views with the parent's encoder,
decode and crop rule (production white RGBA->RGB, EXIF, ``checked_box``). The output folder is written once and is
immutable; ``load`` re-verifies every byte and the prefix against the parent before any consumer receives it.
"""
import io
import shutil
from pathlib import Path

import numpy as np

from rshb_vine.io import digest, local_path, read_json, seal, sha256, verify, write_json
from rshb_vine.reference_additions_v1 import admission as A

OUT = 'runs/release-next-v1/references/build-v1'
RECEIPT_KIND = 'reference-additions-release-next-v1-gallery'
PREPROCESSING = 'letterbox384-existing-Encoder-v1'
DECODE = 'production white RGBA->RGB, EXIF, 36MP offline limit'


def parent(root):
    folder = local_path(root, A.PARENT_GALLERY)
    gallery = verify(read_json(folder / 'gallery.json'))
    receipt = verify(read_json(folder / 'receipt.json'))
    if (sha256(folder / 'gallery.json') != A.PARENT_GALLERY_SHA or gallery['vectors_sha'] != A.PARENT_VECTORS_SHA
            or sha256(folder / 'vectors.npy') != A.PARENT_VECTORS_SHA or receipt['gallery_sha256'] != A.PARENT_GALLERY_SHA):
        raise ValueError('Parent B3 gallery bytes changed')
    return gallery, np.load(folder / 'vectors.npy', allow_pickle=False), receipt


def encoder_identity(root, arm):
    """B4343 encoder files as the parent receipt pins them, and the identity the parent gallery was encoded with."""
    folder = Path(arm['encoder_dir'])
    if folder.resolve() != local_path(root, 'runs/catalog-training-data-v2/stage5/evaluator/actual-fit-v1/B4343/encoder'):
        raise ValueError('Parent encoder directory is not the frozen B4343 export')
    for name, expected in arm['files'].items():
        if sha256(folder.parent / name) != expected:
            raise ValueError('Frozen B4343 encoder file changed: ' + name)
    identity = digest({'files': arm['files'], 'source': sha256(local_path(root, 'rshb_vine/models.py')),
                       'preprocessing': PREPROCESSING})
    if identity != arm['encoder_id']:
        raise ValueError('B4343 encoder identity differs from the parent gallery')
    return folder, identity


def new_rows(root, config, out):
    rows = []
    for record in config['records']:
        sha = record['image']['sha256']
        image_path = f'{out}/images/{sha}.webp'
        for kind in A.KINDS:
            rows.append({'bbox': record['views'][kind], 'complete': True,
                         'coordinate_space': 'EXIF-oriented original pixels', 'cropper_version': config['cropper_version'],
                         'image_path': image_path, 'image_sha256': sha, 'kind': kind,
                         'original_size': record['views']['size'], 'slug': record['slug'],
                         'source': config['row_source'][kind]})
    return rows


def crops(root, config):
    from rshb_vine.preprocessing import checked_box, decode
    pics = []
    for record in config['records']:
        data = local_path(root, record['image']['path']).read_bytes()
        image, _ = decode(data, max_bytes=1 << 30, max_pixels=36_000_000)
        if list(image.size) != record['views']['size']:
            raise ValueError('Decoded admitted image size changed: ' + record['slug'])
        pics.extend(image.crop(checked_box(record['views'][kind], image.size)) for kind in A.KINDS)
    return pics


def build(root, device='mps', out=OUT, batch=4):
    """Encode the admitted views once and publish the appended gallery; refuses an existing output folder."""
    from rshb_vine.encoder_artifact import ArtifactEncoder
    from rshb_vine.visual_core import validate_vectors
    root = Path(root).resolve()
    config, report = A.load(root)
    gallery, vectors, receipt = parent(root)
    folder, identity = encoder_identity(root, receipt['arm'])
    target = local_path(root, out)
    if target.exists():
        raise FileExistsError('Build output already exists: ' + out)
    rows = new_rows(root, config, out)
    pics = crops(root, config)
    encoder = ArtifactEncoder(folder, device)
    added = np.concatenate([encoder.encode(pics[i:i + batch]) for i in range(0, len(pics), batch)]).astype(np.float32)
    validate_vectors(added, len(rows))
    array = np.concatenate([vectors, added])
    target.mkdir(parents=True)
    (target / 'images').mkdir()
    for record in config['records']:
        shutil.copyfile(local_path(root, record['image']['path']), target / 'images' / (record['image']['sha256'] + '.webp'))
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    (target / 'vectors.npy').write_bytes(buf.getvalue())
    doc = seal({'encoder_id': identity, 'references': [*gallery['references'], *rows],
                'vectors_sha': sha256(target / 'vectors.npy'), 'weights_fit': gallery['weights_fit'],
                'reindex_of': gallery['reindex_of'], 'appended_to': A.PARENT_GALLERY_SHA,
                'parent_rows': A.PARENT_ROWS, 'decode': DECODE, 'release_admitted': False})
    write_json(target / 'gallery.json', doc)
    out_receipt = seal({'kind': RECEIPT_KIND, 'admission': report, 'arm': receipt['arm'], 'device': device,
                        'parent_gallery_sha256': A.PARENT_GALLERY_SHA, 'parent_vectors_sha256': A.PARENT_VECTORS_SHA,
                        'gallery_checksum': doc['checksum'], 'gallery_sha256': sha256(target / 'gallery.json'),
                        'vectors_sha256': doc['vectors_sha'], 'rows': len(doc['references']),
                        'encoded_views': len(rows), 'fit_run': False,
                        'images_sha256': {p.name: sha256(p) for p in sorted((target / 'images').iterdir())},
                        'code_sha256': {p: sha256(local_path(root, p)) for p in SOURCES}})
    write_json(target / 'receipt.json', out_receipt)
    return out_receipt


def load(root, out=OUT):
    """(gallery, vectors, receipt) of a finished build, with the parent prefix proven byte-identical."""
    root = Path(root).resolve()
    target = local_path(root, out)
    receipt = verify(read_json(target / 'receipt.json'))
    doc = verify(read_json(target / 'gallery.json'))
    if (receipt.get('kind') != RECEIPT_KIND or receipt['gallery_sha256'] != sha256(target / 'gallery.json')
            or receipt['gallery_checksum'] != doc['checksum'] or sha256(target / 'vectors.npy') != doc['vectors_sha']
            or receipt['vectors_sha256'] != doc['vectors_sha']):
        raise ValueError('Reference additions gallery bytes do not bind its receipt')
    for name, expected in receipt['images_sha256'].items():
        if sha256(target / 'images' / name) != expected or name != expected + '.webp':
            raise ValueError('Copied reference image changed: ' + name)
    config, report = A.load(root)
    if receipt['admission'] != report:
        raise ValueError('Admission config changed since the build')
    gallery, vectors, parent_receipt = parent(root)
    array = np.load(target / 'vectors.npy', allow_pickle=False)
    added = new_rows(root, config, out)
    if (doc['encoder_id'] != gallery['encoder_id'] or doc['references'][:A.PARENT_ROWS] != gallery['references']
            or doc['references'][A.PARENT_ROWS:] != added or array.shape != (A.PARENT_ROWS + len(added), vectors.shape[1])
            or array[:A.PARENT_ROWS].tobytes() != vectors.tobytes() or receipt['arm'] != parent_receipt['arm']):
        raise ValueError('Appended gallery differs from parent prefix or admitted rows')
    return doc, array, receipt


SOURCES = ('rshb_vine/reference_additions_v1/__init__.py', 'rshb_vine/reference_additions_v1/admission.py',
           'rshb_vine/reference_additions_v1/gallery.py')
