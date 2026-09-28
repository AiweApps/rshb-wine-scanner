"""Fixed SSDLite label training, separate from identity adaptation and acceptance."""
from pathlib import Path
import random

import torch
from torch import nn
from torchvision.ops import box_iou
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

from rshb_vine.io import read_json, verify, seal, write_json, local_path, sha256, digest
from rshb_vine.label_detector import ARCHITECTURE, build_model, initialize_from_coco, letterbox
from rshb_vine.models import device_checked, synchronize
from rshb_vine.preprocessing import decode


RECIPE = {'architecture': ARCHITECTURE, 'seed': 20260917, 'epochs': 30,
          'batch_size': 4, 'learning_rate': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005,
          'augmentation': 'none; aspect-preserving 320px white letterbox',
          'batch_norm': 'pretrained running statistics frozen, affine parameters trainable',
          'selection': 'one-to-one main-label recall@IoU0.5 at score0.3, then precision; earliest tie'}


def fetch_initializer(directory):
    """Explicit network setup; training and inference are otherwise offline."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    path = directory / Path(weights.url).name
    if (directory / 'manifest.json').exists():
        receipt = verify(read_json(directory / 'manifest.json'))
        if receipt['url'] != weights.url or sha256(path) != receipt['sha256']:
            raise ValueError('Initializer changed')
        return receipt
    torch.hub.load_state_dict_from_url(weights.url, model_dir=str(directory),
                                      map_location='cpu', check_hash=True, weights_only=True)
    receipt = seal({'url': weights.url, 'file': path.name, 'sha256': sha256(path),
                    'weights': 'SSDLite320_MobileNet_V3_Large_Weights.COCO_V1',
                    'purpose': 'Initialization only; not a trained label detector'})
    write_json(directory / 'manifest.json', receipt)
    return receipt


def validate_manifest(root, manifest):
    verify(manifest)
    if manifest.get('status') != 'admitted' or not manifest.get('admission_evidence'):
        raise ValueError('Detector data admission required; proposals cannot fit')
    if not manifest.get('protocol') or not manifest.get('provenance_limits'):
        raise ValueError('Explicit protocol and provenance limits required')
    if sha256(root / 'config/curation.json') != manifest['curation_sha256']:
        raise ValueError('Curation changed after admission')
    curation = read_json(root / 'config/curation.json')
    admission = curation.get('__label_fit_admissions_v1__', {}).get(manifest.get('admission_id'))
    fingerprint = digest({k: manifest[k] for k in ('protocol', 'records', 'selection_candidates', 'allocation_checksum')})
    if (not admission or admission.get('purpose') != 'label_detector_fit'
            or admission.get('dataset_fingerprint') != fingerprint
            or not admission.get('protocol_decision_evidence') or admission.get('status') != 'admitted'):
        raise ValueError('Source-bound detector admission receipt required')
    allocation = verify(read_json(local_path(root, manifest['allocation_path'])))
    if allocation['checksum'] != manifest['allocation_checksum']:
        raise ValueError('Allocation checksum mismatch')
    by_id = {r['annotation_id']: r for r in allocation['records']}
    approved = {}
    scope_checksums = [allocation['scope_checksum'], *allocation.get('supplement_scope_checksums', [])]
    for scope_checksum in scope_checksums:
        registry = curation['__front_label_annotation_v1__']['scopes'][scope_checksum]
        for batch in registry['batches']:
            data = verify(read_json(local_path(root, batch['path'])))
            if data['checksum'] != batch['checksum'] or data['scope_checksum'] != scope_checksum:
                raise ValueError('Published label batch mismatch')
            approved.update({r['annotation_id']: r for r in data['records']})
    frame_reviews = {}
    forbidden_train_sha = set()
    forbidden_train_ids = set()
    family_links = {}
    for batch in curation.get('__detector_frame_reviews_v1__', {}).get('reviews', []):
        frame_reviews.update({r['image_sha256']: r for r in batch['records']})
        family_links.update({r['image_sha256']: r for r in batch.get('family_links', [])})
    forbidden_train_sha.update(sha for sha, r in family_links.items() if r.get('fit_exclusion'))
    for batch in curation.get('__cross_split_visual_reviews_v1__', {}).get('reviews', []):
        forbidden_train_sha.update(r['train_sha256'] for r in batch['records']
                                   if r['decision'] == 'suspected_shared_photo_quarantine')
    for batch in curation.get('__detector_source_group_holds_v1__', {}).get('reviews', []):
        forbidden_train_ids.update(i for r in batch['records'] for i in r['annotation_ids'])
    reserved = verify(read_json(root / 'data/evaluation/reserved-kultovo.json'))
    reserved_sha = {r['image_sha256'] for r in reserved['entries']}
    protected_hashes, protected_groups = set(), set()
    for row in allocation['records']:
        if row['allocation'] in ('validation_candidate', 'calibration_candidate', 'final_test_candidate', 'development_no_fit'):
            protected_hashes.update(row['all_source_sha256'])
            protected_groups.add(row['split_group_id'])
    role_hashes, role_groups = {}, {}
    for name, expected in [('records', 'train_candidate'), ('selection_candidates', 'validation_candidate')]:
        frames = manifest[name]
        if not frames or len({r['image_sha256'] for r in frames}) != len(frames):
            raise ValueError('Empty or repeated frames')
        hashes, groups = set(), set()
        for frame in frames:
            review = frame_reviews.get(frame['image_sha256'], {})
            if review.get('decision') != 'foreground_labels_accounted_for' or review.get('labels_checksum') != digest(frame['labels']):
                raise ValueError('Missing or stale frame completeness review')
            if not frame['labels']:
                raise ValueError('No published targets')
            if len({r['annotation_id'] for r in frame['labels']}) != len(frame['labels']):
                raise ValueError('Repeated targets in one frame')
            for label in frame['labels']:
                row = by_id[label['annotation_id']]
                owner = approved[label['annotation_id']]
                if row['allocation'] != expected or row.get('owner_pool', 'active') != 'active':
                    raise ValueError('Wrong allocation/owner pool')
                if row['image_sha256'] != frame['image_sha256'] or row['image_path'] != frame['image_path']:
                    raise ValueError('Annotation/source mismatch')
                if owner['image_sha256'] != frame['image_sha256'] or owner['front_label']['bbox'] != label['bbox'] or owner['front_label']['visibility'] != label['visibility']:
                    raise ValueError('Owner annotation changed')
                if label['visibility'] not in ('complete', 'partial'):
                    raise ValueError('Unusable label')
                hashes.update(row['all_source_sha256'])
                groups.add(row['split_group_id'])
                if name == 'records' and (label['annotation_id'] in forbidden_train_ids or frame['image_sha256'] in forbidden_train_sha or row.get('reserved_no_fit')):
                    raise ValueError('Quarantined source cannot fit')
            if sha256(local_path(root, frame['image_path'])) != frame['image_sha256']:
                raise ValueError('Source changed')
        role_hashes[name], role_groups[name] = hashes, groups
    if role_hashes['records'] & reserved_sha:
        raise ValueError('Reserved source cannot fit')
    if role_hashes['records'] & protected_hashes or role_groups['records'] & protected_groups:
        raise ValueError('Protected source/group cannot fit')
    if role_hashes['records'] & role_hashes['selection_candidates'] or role_groups['records'] & role_groups['selection_candidates']:
        raise ValueError('Train/selection source group overlap')


def batch_inputs(root, frames, device):
    images, targets = [], []
    for frame in frames:
        image, _ = decode(local_path(root, frame['image_path']).read_bytes(), max_pixels=48_000_000)
        tensor, boxes, _ = letterbox(image, [r['bbox'] for r in frame['labels']])
        images.append(tensor.to(device))
        targets.append({'boxes': boxes.to(device), 'labels': torch.ones(len(boxes), dtype=torch.int64, device=device)})
    return images, targets


def match_counts(prediction, truth, score_threshold=0.3):
    """Score-ordered, one-to-one IoU0.5 matching; duplicates count as false positives."""
    keep = (prediction['labels'].cpu() == 1) & (prediction['scores'].cpu() >= score_threshold)
    boxes = prediction['boxes'].cpu()[keep]
    scores = prediction['scores'].cpu()[keep]
    truth = truth.cpu()
    used, tp = set(), 0
    for i in scores.argsort(descending=True):
        ious = box_iou(boxes[i].reshape(1, 4), truth)[0]
        available = [(float(value), j) for j, value in enumerate(ious) if j not in used and value >= 0.5]
        if available:
            _, j = max(available)
            used.add(j)
            tp += 1
    return tp, len(boxes) - tp, len(truth) - tp


def evaluate(root, model, frames, device):
    model.eval()
    total = [0, 0, 0]
    with torch.inference_mode():
        for start in range(0, len(frames), RECIPE['batch_size']):
            images, targets = batch_inputs(root, frames[start:start + RECIPE['batch_size']], device)
            for pred, target in zip(model(images), targets):
                total = [a + b for a, b in zip(total, match_counts(pred, target['boxes']))]
    tp, fp, fn = total
    return {'tp': tp, 'fp': fp, 'fn': fn, 'recall_iou50': tp / (tp + fn),
            'precision_iou50': tp / (tp + fp) if tp + fp else 0., 'score_threshold': 0.3,
            'metric': 'fixed-threshold one-to-one detection counts, not COCO mAP or recognition accuracy'}


def train(root, dataset, initializer, output, device='cpu'):
    root, output, initializer = Path(root), Path(output), Path(initializer)
    manifest = verify(read_json(dataset))
    validate_manifest(root, manifest)
    device_checked(device)
    if output.exists():
        raise FileExistsError('New immutable trial directory required')
    init = verify(read_json(initializer / 'manifest.json'))
    torch.manual_seed(RECIPE['seed'])
    model = build_model()
    transfer = initialize_from_coco(model, local_path(initializer, init['file']), init['sha256'])
    model.to(device)
    output.mkdir(parents=True)
    trial = seal({'recipe': RECIPE, 'dataset_checksum': manifest['checksum'],
                  'initializer_checksum': init['checksum'], 'transfer': transfer, 'device': device,
                  'code_sha256': {n: sha256(root / 'rshb_vine' / n) for n in ('label_detector.py', 'label_detector_training.py')}})
    write_json(output / 'trial.json', trial)
    optimizer = torch.optim.SGD(model.parameters(), lr=RECIPE['learning_rate'],
                                momentum=RECIPE['momentum'], weight_decay=RECIPE['weight_decay'])
    best, best_epoch = None, None
    for epoch in range(RECIPE['epochs']):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        frames = list(manifest['records'])
        random.Random(RECIPE['seed'] + epoch).shuffle(frames)
        losses = []
        for start in range(0, len(frames), RECIPE['batch_size']):
            images, targets = batch_inputs(root, frames[start:start + RECIPE['batch_size']], device)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite detector loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            if float(norm) == 0:
                raise RuntimeError('Zero detector gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        score = evaluate(root, model, manifest['selection_candidates'], device)
        write_json(output / f'epoch-{epoch:02d}.json', seal({'epoch': epoch, 'mean_train_loss': sum(losses) / len(losses), 'selection': score}))
        key = (score['recall_iou50'], score['precision_iou50'])
        if best is None or key > best:
            best, best_epoch = key, epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / 'best.pt')
        print({'epoch': epoch, 'selection': score, 'best_epoch': best_epoch}, flush=True)
    synchronize(device)
    write_json(output / 'manifest.json', seal({'architecture': ARCHITECTURE, 'label_semantics': 'visible_main_label',
               'weights_file': 'best.pt', 'weights_sha256': sha256(output / 'best.pt'),
               'trial_checksum': trial['checksum'], 'dataset_checksum': manifest['checksum'],
               'selected_epoch': best_epoch, 'provenance_limits': manifest['provenance_limits'],
               'recognition_comparison_required': True, 'release_ready': False}))
