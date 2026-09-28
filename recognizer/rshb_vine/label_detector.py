"""Compact visible-main-label detector; no identity or visibility classification."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from rshb_vine.io import read_json, verify, sha256, local_path
from rshb_vine.models import device_checked, synchronize
from rshb_vine.preprocessing import checked_box


INPUT_SIZE = 320
ARCHITECTURE = 'torchvision-ssdlite320-mobilenet-v3-large-main-label-v1'


def letterbox(image, boxes=None):
    """Preserve aspect ratio; record actual rounded resize scales for inversion."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError('Empty image')
    ratio = INPUT_SIZE / max(width, height)
    resized = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    left, top = (INPUT_SIZE - resized[0]) // 2, (INPUT_SIZE - resized[1]) // 2
    canvas = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), 'white')
    canvas.paste(image.convert('RGB').resize(resized, Image.Resampling.BILINEAR), (left, top))
    sx, sy = resized[0] / width, resized[1] / height
    transform = {'original_size': [width, height], 'resized_size': list(resized),
                 'offset': [left, top], 'scale': [sx, sy]}
    tensor = torch.from_numpy(np.array(canvas, copy=True)).permute(2, 0, 1).float() / 255
    transformed = None
    if boxes is not None:
        transformed = []
        for box in boxes:
            x1, y1, x2, y2 = checked_box(box, image.size)
            transformed.append([x1 * sx + left, y1 * sy + top, x2 * sx + left, y2 * sy + top])
        transformed = torch.tensor(transformed, dtype=torch.float32).reshape(-1, 4)
    return tensor, transformed, transform


def original_boxes(boxes, transform):
    """Invert letterbox and clip padding predictions to the original pixel plane."""
    boxes = boxes.clone()
    width, height = transform['original_size']
    left, top = transform['offset']
    sx, sy = transform['scale']
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / sx).clamp(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / sy).clamp(0, height)
    return boxes


def build_model():
    # No implicit downloads: pretrained state is loaded by the explicit setup step.
    return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2)


def initialize_from_coco(model, path, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ValueError('Initializer checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    own = model.state_dict()
    discarded = []
    for key in list(state):
        if key not in own:
            raise ValueError('Unexpected initializer key: ' + key)
        if state[key].shape != own[key].shape:
            if not key.startswith('head.classification_head.'):
                raise ValueError('Backbone/regression initializer shape mismatch')
            discarded.append(key)
            del state[key]
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(discarded) or len(discarded) != 12:
        raise ValueError('Expected six class-output weights/biases replaced')
    return {'reinitialized_parameters': discarded,
            'parameter_count': sum(p.numel() for p in model.parameters())}


class LabelDetector:
    def __init__(self, artifact, device='cpu'):
        artifact = Path(artifact)
        manifest = verify(read_json(artifact / 'manifest.json'))
        if manifest['architecture'] != ARCHITECTURE or manifest.get('label_semantics') != 'visible_main_label':
            raise ValueError('Unsupported detector artifact')
        weights = local_path(artifact, manifest['weights_file'])
        if sha256(weights) != manifest['weights_sha256']:
            raise ValueError('Detector weights changed')
        self.device = device_checked(device)
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.model_id = manifest['checksum']

    def detect(self, image, score_threshold=0.3):
        """Accept an already decoded RGB image, returning original-pixel regions."""
        if not 0 <= score_threshold <= 1:
            raise ValueError('Invalid detector score threshold')
        tensor, _, transform = letterbox(image)
        with torch.inference_mode():
            output = self.model([tensor.to(self.device)])[0]
        synchronize(self.device)
        boxes = original_boxes(output['boxes'].cpu(), transform)
        rows = []
        for box, score, label in zip(boxes, output['scores'].cpu(), output['labels'].cpu()):
            if int(label) != 1 or not torch.isfinite(box).all() or not torch.isfinite(score):
                continue
            if float(score) < score_threshold or box[2] <= box[0] or box[3] <= box[1]:
                continue
            rows.append({'bbox': box.tolist(), 'detector_score': float(score),
                         'source': 'automatic_main_label', 'model_id': self.model_id,
                         'visibility_assessed': False})
        return rows
