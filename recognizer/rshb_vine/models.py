"""Frozen local model adapters. Network is allowed only in fetch-models."""
import importlib.metadata
import os
from pathlib import Path

import numpy as np
from PIL import ImageOps

from rshb_vine.io import read_json, seal, sha256, verify, write_json


def device_checked(name):
    import torch
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable implicit PYTORCH_ENABLE_MPS_FALLBACK; choose devices explicitly')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS is unavailable; configure a measured CPU profile explicitly')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    return name


def synchronize(device):
    import torch
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def fetch_models(root):
    from huggingface_hub import snapshot_download
    root = Path(root)
    specs = read_json(root / 'config/models.json')
    for name, spec in specs.items():
        snapshot_download(spec['repo'], revision=spec['revision'],
                          local_dir=root / 'models' / name,
                          allow_patterns=['*.json', '*.safetensors', '*.pdiparams', '*.yml', '*.yaml'])
    # Official LightGlue downloads happen explicitly in this setup operation only.
    os.environ['TORCH_HOME'] = str(root / 'models/torch')
    from lightglue import ALIKED, LightGlue
    ALIKED(max_num_keypoints=1024).eval()
    LightGlue(features='aliked', flash=False).eval()
    files = {p.relative_to(root / 'models').as_posix(): sha256(p)
             for p in (root / 'models').rglob('*') if p.is_file() and '.cache' not in p.parts}
    versions = {n: importlib.metadata.version(n) for n in
                ('torch','torchvision','transformers','paddleocr','paddlepaddle','lightglue','numpy')}
    manifest = seal({'specs': specs, 'files': files, 'versions': versions, 'weights_fit': False})
    write_json(root / 'models/manifest.json', manifest)
    return {'model_checksum': manifest['checksum'], 'files': len(files)}


def verify_models(root):
    root = Path(root)
    manifest = verify(read_json(root / 'models/manifest.json'))
    for package, expected in manifest['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Model runtime package mismatch: ' + package)
    for name, expected in manifest['files'].items():
        if sha256(root / 'models' / name) != expected:
            raise ValueError('Model weights checksum mismatch: ' + name)
    return manifest


class Encoder:
    def __init__(self, root, device):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device_checked(device)
        path = Path(root) / 'models/encoder'
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
        self.model.requires_grad_(False)

    def features(self, images):
        """Shared differentiable input/feature path for inference and metric fitting."""
        import torch
        # Square letterbox preserves the narrow bottle aspect ratio, shared by query/gallery.
        squares = [ImageOps.pad(im, (384, 384), color='white') for im in images]
        inputs = self.processor(images=squares, return_tensors='pt').to(self.device)
        vectors = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(vectors, dim=-1)

    def encode(self, images):
        import torch
        with torch.inference_mode():
            vectors = self.features(images)
        synchronize(self.device)
        return vectors.cpu().numpy().astype('float32')


class Detector:
    def __init__(self, root, device):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.device = device_checked(device)
        path = Path(root) / 'models/detector'
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.model = AutoModelForObjectDetection.from_pretrained(path, local_files_only=True).eval().to(device)
        self.model.requires_grad_(False)

    def detect(self, image):
        import torch
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(outputs, threshold=.4, target_sizes=[image.size[::-1]])[0]
        synchronize(self.device)
        rows = []
        for label, score, box in zip(results['labels'], results['scores'], results['boxes']):
            if self.model.config.id2label[int(label)] == 'bottle':
                b = box.cpu().tolist()
                b = [max(0, b[0]), max(0, b[1]), min(image.width, b[2]), min(image.height, b[3])]
                if b[2] > b[0] and b[3] > b[1]:
                    rows.append({'bbox': b, 'score': float(score)})
        # Duplicate predictions of one object do not count as multiple targets.
        kept = []
        for row in sorted(rows, key=lambda r: -r['score']):
            a = row['bbox']
            duplicate = False
            for other in kept:
                b = other['bbox']
                overlap = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
                union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
                duplicate |= overlap/max(union,1) > .6
            if not duplicate:
                kept.append(row)
        return kept


class OCR:
    def __init__(self, root):
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        from paddleocr import PaddleOCR
        self.model = PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',
                              text_detection_model_dir=str(Path(root) / 'models/ocr_detector'),
                              text_recognition_model_name='eslav_PP-OCRv5_mobile_rec',
                              text_recognition_model_dir=str(Path(root) / 'models/ocr_recognizer'),
                              use_doc_orientation_classify=False, use_doc_unwarping=False,
                              use_textline_orientation=False, device='cpu', enable_mkldnn=False,
                              cpu_threads=4)

    def read(self, image):
        # Input is original-resolution RGB; Paddle expects BGR ndarray.
        rows = []
        for result in self.model.predict(np.array(image)[:, :, ::-1].copy()):
            for text, score, polygon in zip(result['rec_texts'], result['rec_scores'], result['rec_polys']):
                rows.append({'raw_text': text, 'score': float(score), 'polygon': np.asarray(polygon).tolist()})
        return rows
