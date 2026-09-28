"""Verify and load one development runtime bundle, shared by HTTP and replay."""
from pathlib import Path
import importlib.metadata

import numpy as np

from rshb_vine.io import read_json, read_jsonl, verify, sha256, local_path, digest, seal, write_json
from rshb_vine.encoder_artifact import encoder_spec, load_encoder

CODE_FILES = ('rshb_vine/models.py', 'rshb_vine/encoder_artifact.py', 'rshb_vine/visual_core.py',
              'rshb_vine/label_detector.py', 'rshb_vine/label_first_pipeline.py',
              'rshb_vine/bottle_label_localizer.py',
              'rshb_vine/conditional_ocr.py', 'rshb_vine/resolution/identity.py',
              'rshb_vine/preprocessing/__init__.py', 'rshb_vine/api.py',
              'rshb_vine/label_first_release.py', 'rshb_vine/io.py')


def inspect_bundle(root, descriptor):
    root = Path(root).resolve()
    release = verify(read_json(descriptor))
    if release.get('schema') != 'label-first-runtime-v1' or release.get('purpose') != 'development_candidate':
        raise ValueError('Explicit development runtime descriptor required')
    if type(release.get('max_targets')) is not int or not 1 <= release['max_targets'] <= 16:
        raise ValueError('Invalid target limit')
    localization = release.get('localization_mode', 'direct_label')
    if localization not in ('direct_label', 'bottle_first', 'bottle_parent'):
        raise ValueError('Unsupported localization mode')
    if localization in ('bottle_first', 'bottle_parent'):
        models = verify(read_json(root / 'models/manifest.json'))
        files = {k: v for k, v in models['files'].items() if k.startswith('detector/')}
        if not files or release.get('bottle_detector_files') != files:
            raise ValueError('Explicit bottle detector fingerprint required')
        for name, expected in files.items():
            if sha256(local_path(root / 'models', name)) != expected:
                raise ValueError('Bottle detector changed')
    elif release.get('bottle_detector_files'):
        raise ValueError('Unused bottle detector configuration')
    if set(release.get('code_files', {})) != set(CODE_FILES):
        raise ValueError('Complete runtime code fingerprint required')
    for name, expected in release['code_files'].items():
        if sha256(local_path(root, name)) != expected:
            raise ValueError('Runtime code changed: ' + name)
    required_packages = {'torch', 'torchvision', 'transformers', 'numpy', 'Pillow'}
    if release.get('ocr') is not None:
        required_packages.update(('paddleocr', 'paddlepaddle'))
    if not required_packages <= set(release.get('packages', {})):
        raise ValueError('Runtime package versions required')
    for name, version in release['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Runtime package changed: ' + name)
    components = {}
    for name in ('gallery', 'detector'):
        spec = release[name]
        path = local_path(root, spec['manifest'])
        manifest = verify(read_json(path))
        if manifest['checksum'] != spec['checksum']:
            raise ValueError(name + ' manifest changed')
        components[name] = (path.parent, manifest)
    folder, gallery = components['gallery']
    for name in ('vectors.npy', 'references.json'):
        if sha256(folder / name) != gallery['files'][name]:
            raise ValueError('Gallery data changed')
    from rshb_vine.visual_core import LabelFirstIndex
    references = read_json(folder / 'references.json')
    if not references:
        raise ValueError('Empty reference gallery')
    LabelFirstIndex(np.load(folder / 'vectors.npy', allow_pickle=False), references, gallery['encoder_id'])
    directory = release.get('encoder_directory')
    encoder = encoder_spec(root, directory)
    if encoder['encoder_id'] != gallery['encoder_id'] or encoder['encoder_id'] != release['encoder_id']:
        raise ValueError('Encoder/gallery mismatch; rebuild both gallery channels')
    folder, detector = components['detector']
    from rshb_vine.label_detector import ARCHITECTURE
    if (detector.get('architecture') != ARCHITECTURE or detector.get('label_semantics') != 'visible_main_label'
            or 'selected_epoch' not in detector):
        raise ValueError('Trained main-label detector required, not COCO initializer')
    trial = verify(read_json(folder / 'trial.json'))
    if trial['checksum'] != detector['trial_checksum'] or trial['dataset_checksum'] != detector['dataset_checksum']:
        raise ValueError('Detector trial/data mismatch')
    epochs = trial['recipe']['epochs']
    if not isinstance(epochs, int) or epochs < 1 or not 0 <= detector['selected_epoch'] < epochs:
        raise ValueError('Invalid detector training completion')
    for epoch in range(epochs):
        receipt = verify(read_json(folder / f'epoch-{epoch:02d}.json'))
        if receipt['epoch'] != epoch or receipt.get('mean_train_loss') is None:
            raise ValueError('Incomplete detector trial')
    if sha256(local_path(folder, detector['weights_file'])) != detector['weights_sha256']:
        raise ValueError('Detector weights changed')
    ocr = release.get('ocr')
    if ocr is not None:
        if type(ocr.get('apply_reranking')) is not bool:
            raise ValueError('Explicit boolean OCR ranking policy required')
        catalog = local_path(root, ocr['catalog'])
        if sha256(catalog) != ocr['catalog_sha256']:
            raise ValueError('OCR catalog changed')
        known = {r['slug'] for r in read_jsonl(catalog) if not r.get('excluded_from_retrieval')}
        if {r['slug'] for r in references} - known:
            raise ValueError('OCR catalog does not cover gallery')
        files = ocr['files']
        if not files or not all(k.startswith(('ocr_detector/', 'ocr_recognizer/')) for k in files):
            raise ValueError('Explicit local OCR component hashes required')
        for name, expected in files.items():
            if sha256(local_path(root / 'models', name)) != expected:
                raise ValueError('OCR model changed')
    return release, components


def assemble_bundle(root, gallery_directory, detector_directory, output, *, encoder_directory=None,
                    catalog_path=None, ocr_mode='none', max_targets=16, localization_mode='direct_label'):
    """Publish only after all supplied artifacts pass read-only checks."""
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Use a new runtime descriptor')
    if ocr_mode not in ('none', 'evidence', 'contradictions') or not 1 <= max_targets <= 16:
        raise ValueError('Invalid runtime policy')
    payload = {'schema': 'label-first-runtime-v1', 'purpose': 'development_candidate',
               'encoder_directory': str(encoder_directory) if encoder_directory is not None else None,
               'encoder_id': encoder_spec(root, encoder_directory)['encoder_id'], 'max_targets': max_targets,
               'code_files': {p: sha256(root / p) for p in CODE_FILES},
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'transformers', 'numpy', 'Pillow')},
               'ocr': None, 'calibrated': False, 'independent_acceptance_passed': False}
    payload['localization_mode'] = localization_mode
    if localization_mode in ('bottle_first', 'bottle_parent'):
        models = verify(read_json(root / 'models/manifest.json'))
        payload['bottle_detector_files'] = {k: v for k, v in models['files'].items() if k.startswith('detector/')}
    for name, directory in [('gallery', gallery_directory), ('detector', detector_directory)]:
        path = local_path(root, directory) / 'manifest.json'
        manifest = verify(read_json(path))
        payload[name] = {'manifest': path.relative_to(root).as_posix(), 'checksum': manifest['checksum']}
    if ocr_mode != 'none':
        if catalog_path is None:
            raise ValueError('OCR requires an explicit catalog snapshot')
        catalog = local_path(root, catalog_path)
        models = verify(read_json(root / 'models/manifest.json'))
        payload['ocr'] = {'catalog': catalog.relative_to(root).as_posix(), 'catalog_sha256': sha256(catalog),
                          'apply_reranking': ocr_mode == 'contradictions',
                          'files': {k: v for k, v in models['files'].items() if k.startswith(('ocr_detector/', 'ocr_recognizer/'))}}
        payload['packages'].update({p: importlib.metadata.version(p) for p in ('paddleocr', 'paddlepaddle')})
    release = seal(payload)
    # The verifier reads documents through the normal guarded reader. Staging is
    # never the final descriptor and is removed even if verification fails.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        staging = Path(folder) / 'runtime.json'
        write_json(staging, release)
        inspect_bundle(root, staging)
    write_json(output, release)
    return release


def load_bundle(root, descriptor, device='mps', detector_device='cpu'):
    from rshb_vine.visual_core import LabelFirstIndex
    from rshb_vine.label_detector import LabelDetector
    from rshb_vine.label_first_pipeline import LabelFirstPipeline
    from rshb_vine.conditional_ocr import ConditionalOCR
    from rshb_vine.models import OCR
    root = Path(root)
    release, components = inspect_bundle(root, descriptor)
    encoder_id, encoder = load_encoder(root, release.get('encoder_directory'), device)
    folder, _ = components['gallery']
    references = read_json(folder / 'references.json')
    index = LabelFirstIndex(np.load(folder / 'vectors.npy', allow_pickle=False), references, encoder_id)
    if release.get('localization_mode') in ('bottle_first', 'bottle_parent'):
        from rshb_vine.bottle_label_localizer import BottleLabelLocalizer
        detector = BottleLabelLocalizer.from_artifacts(root, components['detector'][0], detector_device,
                                                            merge_parent_labels=release['localization_mode']=='bottle_parent')
    else:
        detector = LabelDetector(components['detector'][0], detector_device)
    text = None
    if release.get('ocr') is not None:
        spec = release['ocr']
        catalog = read_jsonl(local_path(root, spec['catalog']))
        known = {r['slug'] for r in catalog if not r.get('excluded_from_retrieval')}
        if {r['slug'] for r in references} - known:
            raise ValueError('OCR catalog does not cover this gallery')
        text = ConditionalOCR(catalog, OCR(root), digest(spec['files']), apply_reranking=spec['apply_reranking'])
    pipeline = LabelFirstPipeline(encoder, index, detector, encoder_id,
                                 max_targets=release['max_targets'], conditional_ocr=text)
    pipeline.manifest = {**pipeline.manifest, 'runtime_descriptor_checksum': release['checksum']}
    # Preserve a sealed health manifest after attaching the bundle identity.
    pipeline.manifest = seal({k: v for k, v in pipeline.manifest.items() if k != 'checksum'})
    return pipeline
