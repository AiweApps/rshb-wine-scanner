"""Stage5 B3 evaluator: plan, full gallery reindex, isolated B3-only runtime, validation46 scorer.

Inference commands (reindex/evaluate) require an exact root run admission; parity-probe is limited to
three web232 train photos. Nothing here changes current, frozen profiles, splits or GT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import read_json, seal, sha256, verify, write_json
from rshb_vine.catalog_training_evaluation_v2 import adapter, pins, reindex, score

PROBE_LIMIT, PROBE_SECONDS = 3, 240
DEVICES = ('mps',)


def _deadline(seconds):
    def expire(*_):
        raise TimeoutError('Hard evaluator deadline reached')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)


def _admission(path, command, out, extra):
    if extra.get('device') not in DEVICES:
        raise ValueError('Pinned runtime and encode conditions are validated on MPS only')
    admission = verify(read_json(path))
    expected = {'kind': 'catalog-training-v2-evaluator-run-admission-v1', 'command': command,
                'root_admitted': True, 'output_directory': str(out.relative_to(ROOT)),
                'profile_checksum': pins.PROFILE_CHECKSUM, 'code_sha256': pins.code_sha(), **extra}
    wrong = sorted(k for k, v in expected.items() if admission.get(k) != v)
    if wrong:
        raise ValueError('Root run admission does not match this run: ' + ', '.join(wrong))
    if not isinstance(admission.get('max_seconds'), int) or admission['max_seconds'] <= 0:
        raise ValueError('Root run admission needs max_seconds')
    return admission


def _arm(args):
    return reindex.encoder_arm(args.encoder_dir, args.export_receipt, args.recipe, args.checkpoint, args.fit_admission)


def _plan(path):
    document = verify(read_json(path))
    if (document.get('kind') != 'catalog-training-v2-evaluator-plan-v1' or document['profile_checksum'] != pins.PROFILE_CHECKSUM
            or document['code_sha256'] != pins.code_sha()
            or document['inputs_sha256'] != {v[0]: v[1] for v in pins.FILES.values()}):
        raise ValueError('Evaluator plan differs from the frozen code/input pins')
    return document


def plan(args):
    out = pins.output_dir(args.out)
    for name in pins.FILES:
        pins.checked(name)
    refs, sources = reindex.references()
    targets = score.validation_targets()
    arm, _ = reindex.encoder_arm()
    document = seal({'kind': 'catalog-training-v2-evaluator-plan-v1', 'profile': pins.PROFILE,
        'profile_checksum': pins.PROFILE_CHECKSUM, 'inputs_sha256': {v[0]: v[1] for v in pins.FILES.values()},
        'code_sha256': pins.code_sha(), 'parent_arm': arm, 'checkpoint_steps': pins.checkpoint_steps(),
        'gallery': {'references': len(refs), 'base_rows': reindex.PREFIX, 'addition_rows': reindex.ADDED,
                    'source_images': reindex.SOURCES, 'order': 'serving data/catalog-additions-20260921/B3',
                    'decode': 'rshb_vine.preprocessing.decode white RGBA composite, EXIF, max 36MP'},
        'arms': {'A_current': 'parent encoder + pinned current B3 gallery; integration parity only, never the selection control',
                 'A_white': 'parent encoder + white alpha gallery d5949dda; selection control (same white decode/order as B)',
                 'A_reindex': 'parent encoder + this evaluator reindex; admissible control only with receipt parity vs signed white parent',
                 'B<step>': 'stage5 export + its own full reindex, same order/boxes/white decode'},
        'preprocessing_delta': 'A_current -> A_white reported separately, never attributed to training',
        'device': 'mps only (pinned runtime loads MPS); Linux/CPU full-runtime portability not claimed',
        'validation': {'photos': len(targets), 'gt': 'validation.json acceptable_slugs -> registry03 product_id',
                       'gt_geometry': 'none (no bbox/ROI annotation)', 'request_mode': 'full_frame, roi=null',
                       'focal_instance_id_used': False, 'unmapped_gt_photos': sum(bool(t['unmapped_gt_slugs']) for t in targets),
                       'multi_gt_photos': sum(len(t['acceptable_products03']) > 1 for t in targets),
                       'targets': targets},
        'primary_metric': 'image-level published best_candidate ProductID03 correct / 46; null/ambiguous/unmapped counted non-correct',
        'secondary': ['confirmed slug separately', 'any-target GT hit diagnostic only',
                      'B3 pool Top1/5/20 only if response exposes B3 arm retrieval'],
        'selection': 'all selectable 2895/4343/5790 scored: most correct, then fewest new wrong Product, then earliest; must exceed A_white (or admitted A_reindex)',
        'closed_roles_opened': False, 'fit_admitted': False, 'release_admitted': False})
    out.mkdir(parents=True)
    write_json(out / 'plan.json', document)
    print(json.dumps({'plan': str(out / 'plan.json'), 'checksum': document['checksum']}))


def run_reindex(args):
    out = pins.output_dir(args.out)
    arm, model_path = _arm(args)
    admission = _admission(args.admission, 'reindex', out, {'arm': arm, 'device': args.device, 'batch': args.batch})
    _deadline(admission['max_seconds'])
    started = time.perf_counter()
    refs, sources = reindex.references()
    encoder = reindex.load_encoder_model(model_path, args.device)
    array = reindex.reindex(encoder, refs, sources, args.batch)
    extra = {'admission_sha256': sha256(args.admission), 'device': args.device, 'batch': args.batch,
             'seconds': time.perf_counter() - started}
    if arm['arm'] == 'A':
        extra['parent_parity'] = reindex.parent_parity(array)
    receipt = reindex.write_gallery(out, arm, refs, array, extra)
    print(json.dumps({'receipt': str(out / 'receipt.json'), 'checksum': receipt['checksum'],
                      **extra.get('parent_parity', {})}))


def _train_probe_images(ids):
    admission = pins.read('probe_admission')
    train = {r['id']: r for r in pins.read('train')['records']}
    admitted = {r['id']: r for r in admission['records']}
    if (admission['profile_checksum'] != pins.PROFILE_CHECKSUM or admission['profile_file_sha256'] != pins.PROFILE_SHA256
            or admission['train_manifest_sha256'] != pins.FILES['train'][1]
            or admission['max_total_seconds_including_model_load'] != PROBE_SECONDS
            or admission['max_photos'] != PROBE_LIMIT or admission['passes_per_photo'] != 2
            or admission['validation_inference'] or admission['gallery_reindex']):
        raise ValueError('Root probe admission does not bind this probe')
    if not 1 <= len(ids) <= PROBE_LIMIT or len(set(ids)) != len(ids) or not set(ids) <= set(admitted):
        raise ValueError('Parity probe accepts only root-admitted train IDs')
    rows = []
    for i in ids:
        record = train.get(i)
        if record is None or record['split'] != 'train' or record['sha256'] != admitted[i]['sha256']:
            raise ValueError('Parity probe is limited to admitted web232 train records: ' + i)
        path = pins.recovered_image(record['sha256'])
        if str(path.relative_to(ROOT)) != admitted[i]['path']:
            raise ValueError('Probe image path differs from root admission: ' + i)
        rows.append((i, record['sha256'], path))
    return rows


def parity_probe(args):
    out = pins.output_dir(args.out)
    if args.device not in DEVICES:
        raise ValueError('Pinned runtime is validated on MPS only')
    images = _train_probe_images(args.ids.split(','))
    _deadline(PROBE_SECONDS)
    started = time.perf_counter()
    pipeline = adapter.load_pinned()
    loaded = time.perf_counter()
    plain = [pipeline.recognize(path.read_bytes()) for _, _, path in images]
    plain_done = time.perf_counter()
    arm, model_path = reindex.encoder_arm()
    encoder, index, gallery = adapter.arm_components(arm, model_path, 'current', args.device)
    substitution = adapter.substitute(pipeline, arm, encoder, index, gallery)
    swapped = time.perf_counter()
    adapted = [pipeline.recognize(path.read_bytes()) for _, _, path in images]
    finished = time.perf_counter()
    signal.alarm(0)
    rows = []
    for (i, sha, path), left, right in zip(images, plain, adapted):
        split = score.classify(score.diff(left, right))
        rows.append({'id': i, 'image_sha256': sha, 'image_path': str(path.relative_to(ROOT)),
                     'plain': {'decision': left.get('decision'), 'best_candidate': left.get('best_candidate'),
                               'slug': left.get('slug'), 'targets': len(left.get('targets') or [])},
                     'adapted': {'decision': right.get('decision'), 'best_candidate': right.get('best_candidate'),
                                 'slug': right.get('slug'), 'targets': len(right.get('targets') or [])},
                     'timing_only_diff_paths': split['timing_only'], 'behavior_diff_paths': split['behavior']})
    out.mkdir(parents=True)
    write_json(out / 'responses.json', {'plain': plain, 'adapted': adapted})
    receipt = seal({'kind': 'catalog-training-v2-evaluator-parity-probe-v1', 'scope': 'web232 train, root-approved',
        'profile_checksum': pins.PROFILE_CHECKSUM, 'probe_admission_sha256': pins.FILES['probe_admission'][1],
        'code_sha256': pins.code_sha(), 'substitution': substitution,
        'device': args.device, 'records': rows, 'responses_sha256': sha256(out / 'responses.json'),
        'parity': all(not r['behavior_diff_paths'] for r in rows),
        'seconds': {'load': loaded - started, 'plain_pass': plain_done - loaded, 'substitute': swapped - plain_done,
                    'adapted_pass': finished - swapped, 'total': finished - started},
        'claim': 'integration parity of the B3 substitution path on parent A only; no quality claim',
        'validation_or_closed_opened': False})
    write_json(out / 'receipt.json', receipt)
    print(json.dumps({'receipt': str(out / 'receipt.json'), 'parity': receipt['parity'],
                      'seconds': receipt['seconds']}))


def evaluate(args):
    out = pins.output_dir(args.out)
    arm, model_path = _arm(args)
    if args.gallery in ('current', 'white') and arm['arm'] != 'A':
        raise ValueError('Pinned A galleries are only valid for parent A')
    plan_doc = _plan(args.plan)
    admission = _admission(args.admission, 'evaluate', out, {'arm': arm, 'gallery': args.gallery,
                                                             'scope': 'web232-validation46', 'device': args.device,
                                                             'plan_checksum': plan_doc['checksum']})
    targets = score.validation_targets()
    mapping = pins.product03()
    _deadline(admission['max_seconds'])
    started = time.perf_counter()
    pipeline = adapter.load_pinned()
    encoder, index, gallery = adapter.arm_components(arm, model_path, args.gallery, args.device)
    substitution = adapter.substitute(pipeline, arm, encoder, index, gallery)
    out.mkdir(parents=True)
    rows = []
    with (out / 'responses.jsonl').open('x') as sink:
        for target in targets:
            data = pins.recovered_image(target['sha256']).read_bytes()
            result = pipeline.recognize(data)
            sink.write(json.dumps({'id': target['id'], 'result': result}, ensure_ascii=False) + '\n')
            rows.append(score.score_record(target, result, mapping, hashlib.sha256(data).hexdigest()))
    signal.alarm(0)
    report = seal({'kind': 'catalog-training-v2-validation46-arm-report-v1', 'arm': arm, 'gallery': args.gallery,
        'plan_checksum': plan_doc['checksum'], 'substitution': substitution, 'admission_sha256': sha256(args.admission), 'rows': rows,
        'summary': score.summarize(list(score.complete(rows).values())), 'responses_sha256': sha256(out / 'responses.jsonl'),
        'seconds': time.perf_counter() - started, 'device': args.device})
    write_json(out / 'report.json', report)
    print(json.dumps({'report': str(out / 'report.json'), 'summary': report['summary']}, ensure_ascii=False))


def compare(args):
    out = pins.output_dir(args.out)
    plan_doc = _plan(args.plan)
    a = verify(read_json(args.a))
    if a['arm']['arm'] != 'A' or a['gallery'] == 'current' or a['plan_checksum'] != plan_doc['checksum']:
        raise ValueError('Selection control must be parent A on the white gallery or an admitted A reindex')
    delta = None
    if args.a_current is not None:
        current = verify(read_json(args.a_current))
        if current['arm']['arm'] != 'A' or current['gallery'] != 'current' or current['plan_checksum'] != plan_doc['checksum']:
            raise ValueError('--a-current must be parent A on the pinned current gallery')
        delta = score.paired(current['rows'], a['rows'])
    arms = {}
    for raw in args.b:
        report = verify(read_json(raw))
        step = (report['arm'].get('export') or {}).get('global_step')
        if step is None or step in arms or report['plan_checksum'] != plan_doc['checksum']:
            raise ValueError('Each B report must be one distinct exported checkpoint')
        arms[step] = report['rows']
    document = seal({'kind': 'catalog-training-v2-validation46-selection-v1',
        'A_report_checksum': a['checksum'], 'B_reports': {str(k): v for k, v in sorted(arms.items())},
        'pairs': {str(k): score.paired(a['rows'], v) for k, v in sorted(arms.items())},
        'selection': score.select(a['rows'], arms), 'control_gallery': a['gallery'],
        'A_current_to_control_preprocessing_delta_not_training': delta, 'release_admitted': False,
        'next_required': 'full HTTP/ROI pass for the selected checkpoint and causal review of regressions'})
    out.mkdir(parents=True)
    write_json(out / 'selection.json', document)
    print(json.dumps(document['selection'], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--out', required=True)
    for name in ('reindex', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--admission', type=Path, required=True)
        p.add_argument('--encoder-dir')
        p.add_argument('--export-receipt')
        p.add_argument('--recipe')
        p.add_argument('--checkpoint')
        p.add_argument('--fit-admission')
        p.add_argument('--device', default='mps')
        if name == 'reindex':
            p.add_argument('--batch', type=int, default=32)
        else:
            p.add_argument('--gallery', required=True, help="'current', 'white' or a reindex output directory")
            p.add_argument('--plan', type=Path, required=True)
    p = sub.add_parser('parity-probe')
    p.add_argument('--ids', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    p = sub.add_parser('compare')
    p.add_argument('--a', type=Path, required=True)
    p.add_argument('--b', type=Path, nargs='+', required=True)
    p.add_argument('--a-current', type=Path)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', required=True)
    args = parser.parse_args()
    {'plan': plan, 'reindex': run_reindex, 'parity-probe': parity_probe, 'evaluate': evaluate,
     'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
