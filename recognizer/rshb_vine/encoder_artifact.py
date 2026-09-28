"""Shared offline encoder loading/identity for gallery builds and serving."""
from pathlib import Path

from rshb_vine.io import read_json, verify, digest, sha256, local_path
from rshb_vine.models import Encoder, device_checked


def encoder_spec(root, directory=None):
    root = Path(root)
    selected = 'B1'
    if directory is None:
        manifest = verify(read_json(root / 'models/manifest.json'))
        files = {k: v for k, v in manifest['files'].items() if k.startswith('encoder/')}
        base = root / 'models'
        path = base / 'encoder'
    else:
        path = local_path(root, directory)
        receipt = verify(read_json(path.parent / 'result.json'))
        if receipt.get('selected') not in ('B1', 'B2') or 'encoder_files' not in receipt:
            raise ValueError('Completed metric-training receipt with encoder hashes required')
        selected = receipt['selected']
        files = {'encoder/' + k: v for k, v in receipt['encoder_files'].items()}
        base = path.parent
        if path.name != 'encoder':
            raise ValueError('Expected canonical trained encoder directory')
    if not files or not any(k.endswith('.safetensors') for k in files):
        raise ValueError('Encoder weights missing')
    for name, expected in files.items():
        if sha256(local_path(base, name)) != expected:
            raise ValueError('Changed encoder artifact: ' + name)
    identity = digest({'files': files, 'source': sha256(root / 'rshb_vine/models.py'),
                       'preprocessing': 'letterbox384-existing-Encoder-v1'})
    return {'encoder_id': identity, 'path': path, 'selected': selected, 'files': files}


class ArtifactEncoder(Encoder):
    """Keep Encoder.features/encode unchanged for merged adapted weights."""
    def __init__(self, path, device):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device_checked(device)
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
        self.model.requires_grad_(False)


def load_encoder(root, directory=None, device='mps'):
    spec = encoder_spec(root, directory)
    encoder = Encoder(root, device) if directory is None else ArtifactEncoder(spec['path'], device)
    return spec['encoder_id'], encoder
