"""Existing-target parent N (N1 physical parent, N1b unlocalized label): bounded pre-probe, component freeze/describe.

preprobe-plan  CPU only: seal the fixed beside-target label_only class of census 57c886c8 (30 labels): E1 inputs not
               measured by residual probe f1b4eb54 (<=18) and every label crop for the frozen B0 gate (<=30).
preprobe-run   once, after runs/release-next-v1/geometry/root-admission.json: loads the bde4fa52 release in-process
               (no HTTP, no OCR, no retrieval) and records E1 proposals and frozen B0 gate scores of label crops.
               Executed once by the v1 source (component/v1-archive); results 49be6fef are immutable.
freeze         component descriptor (own source SHA + immutable bde4fa52 profile); no active pointer is written.
describe       descriptor + source checks, no models loaded.
The candidate is served by rshb_vine.release_next_v1 (recipe N or D,N) through component.py.
This script never touches 8175/8187, the factory, the gallery or the current pointer.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import read_json, seal, sha256, verify, write_json  # noqa: E402

OUT = 'runs/release-next-v1/geometry'
PREPROBE = OUT + '/preprobe'
DESCRIPTOR = OUT + '/component/descriptor-v3.json'
CENSUS = 'runs/atlas-repair-v1/geometry/census-b0d8688c.json'
CENSUS_CHECKSUM = '57c886c8c9b80615e09775588d7c599a47cb5358603615402eb744f8131875c4'
MEASURED = 'runs/atlas-repair-v1/geometry/residual-probe/summary.json'
MEASURED_CHECKSUM = 'f1b4eb542e415718a666dd9643977c00a36ba087be0a87e4173942e2b1ea3edf'
ADMISSION = OUT + '/root-admission.json'
ADMISSION_CHECKSUM = '18bd8f4ab3f214cf2ccfe64c5ac6aaecf03f696fd4245c9cffed1c9a443cc731'
E1_MAX, B0_MAX = 18, 30
AGENT_VISUAL = {'atlas:0294': 'foreground_wine_bottle', 'atlas:0296': 'foreground_wine_bottle',
                'atlas:0508': 'foreground_wine_bottle',
                'open:real_organizer_unlabeled:organizer-007': 'second_wine_bottle_at_frame_edge',
                'open:real_organizer_unlabeled:organizer-058': 'uncertain_shelf_background'}


def _image_size(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return list(ImageOps.exif_transpose(im).size)


def preprobe_plan(root):
    from rshb_vine.existing_target_parent_v1 import plan as P
    census = verify(read_json(root / CENSUS))
    measured = verify(read_json(root / MEASURED))
    if census['checksum'] != CENSUS_CHECKSUM or measured['checksum'] != MEASURED_CHECKSUM:
        raise SystemExit('Census or residual-probe summary is not the reviewed version')
    done = {r['key'] for r in measured['rows']}
    items = []
    for row in census['rows']:
        if not row['targets']:
            continue
        for trial in row.get('trials') or []:
            if trial.get('kind') != 'label_only' or trial.get('skip') != 'request_has_targets':
                continue
            if sha256(Path(row['input_path'])) != row['input_sha256']:
                raise SystemExit('Input bytes changed: ' + row['key'])
            size = _image_size(row['input_path'])
            label = list(trial['label_bbox'])
            items.append({'key': row['key'], 'source': row['source'], 'input_path': row['input_path'],
                          'input_sha256': row['input_sha256'], 'image_size_exif': size, 'label_bbox': label,
                          'label_score': trial.get('label_score'), 'column': P.stripe(label, size),
                          'agent_visual_class': AGENT_VISUAL.get(row['key'], 'non_wine_region'),
                          'e1': row['key'] not in done, 'e1_measured_by': MEASURED if row['key'] in done else None})
    e1, b0 = sum(i['e1'] for i in items), len(items)
    if e1 > E1_MAX or b0 > B0_MAX:
        raise SystemExit('Pre-probe exceeds the admitted budget: E1 %d/%d, B0 %d/%d' % (e1, E1_MAX, b0, B0_MAX))
    doc = seal({'kind': 'existing-target-parent-v1-preprobe-plan', 'census': CENSUS_CHECKSUM, 'measured': MEASURED_CHECKSUM,
                'admission': ADMISSION_CHECKSUM, 'constants': P.CONSTANTS, 'e1_calls': e1, 'b0_encodes': b0,
                'b0': 'frozen object gate of the loaded bde4fa52 graph: sigmoid(B0(label crop) . coef + bias) >= spec threshold;'
                      ' context = checked label box (G label_only convention beside another bottle)',
                'visual_classes': 'agent review of overlays, not owner GT; organizer-058 never counts as positive',
                'repeat_rule': 'one run; a failed call is recorded, never repeated', 'items': items})
    write_json(root / PREPROBE / 'plan.json', doc)
    return {'plan': PREPROBE + '/plan.json', 'checksum': doc['checksum'], 'e1_calls': e1, 'b0_encodes': b0}


def preprobe_run(root):
    import numpy as np
    from rshb_vine.atlas_repair_release_v1.release import AtlasRepairRelease
    from rshb_vine.b3_only_v1 import split as S
    from rshb_vine.existing_target_parent_v1 import runtime as N
    from rshb_vine.preprocessing import checked_box, decode
    doc = verify(read_json(root / PREPROBE / 'plan.json'))
    admission = verify(read_json(root / ADMISSION))
    if admission['checksum'] != ADMISSION_CHECKSUM or admission['decision'] != 'admit_N1_implementation_and_bounded_preprobe':
        raise SystemExit('Root admission is not the reviewed N1 pre-probe admission')
    if (root / PREPROBE / 'started.json').exists():
        raise SystemExit('Pre-probe already started; results are immutable and never repeated')
    write_json(root / PREPROBE / 'started.json', seal({'plan': doc['checksum'], 'admission': admission['checksum'],
                                                       'started': time.time()}))
    release = AtlasRepairRelease(root, N.PARENT_PROFILE)
    if release.profile['checksum'] != N.PARENT_CHECKSUM or release.manifest['checksum'] != N.PARENT_RUNTIME:
        raise SystemExit('Loaded release differs from bde4fa52/54f09a20')
    orphan, _, detector, gate, _ = N.locate(release)
    stage = N.ExistingTargetParent(orphan, orphan.apply, orphan.retry, detector, gate)
    spec = gate.spec
    rows = []
    for item in doc['items']:
        data = Path(item['input_path']).read_bytes()
        row = {'key': item['key'], 'label_bbox': item['label_bbox']}
        rows.append(row)
        if sha256(Path(item['input_path'])) != item['input_sha256']:
            row['error'] = 'input bytes changed'
            continue
        image, _ = decode(data)
        if list(image.size) != item['image_size_exif']:
            row['error'] = 'decoded size differs from plan'
            continue
        try:
            if item['e1']:
                row['e1'] = stage._e1(image, item['label_bbox'])
            started = time.perf_counter()
            with S.GATE:
                vector = gate.gate_encoder.encode([image.crop(checked_box(item['label_bbox'], image.size))])[0]
            score = float(1 / (1 + np.exp(-np.clip(np.dot(vector, gate.coef) + gate.bias, -40, 40))))
            row['b0'] = {'score': score, 'threshold': spec['threshold'], 'passes': score >= spec['threshold'],
                         'ms': 1000 * (time.perf_counter() - started)}
        except Exception as error:
            row['error'] = type(error).__name__ + ': ' + str(error)[:300]
        print(json.dumps({'key': item['key'], 'e1': len((row.get('e1') or {}).get('proposals') or []) if item['e1'] else None,
                          'b0': (row.get('b0') or {}).get('score'), 'error': row.get('error')}), flush=True)
    results = seal({'kind': 'existing-target-parent-v1-preprobe-results', 'plan': doc['checksum'],
                    'admission': admission['checksum'], 'gate_spec_checksum': spec['checksum'],
                    'gate_encoder_id': gate.gate_encoder_id, 'release': N.PARENT_CHECKSUM, 'runtime': N.PARENT_RUNTIME,
                    'rows': rows})
    write_json(root / PREPROBE / 'results.json', results)
    return summarize(root)


def summarize(root):
    from rshb_vine.existing_target_parent_v1 import plan as P
    doc = verify(read_json(root / PREPROBE / 'plan.json'))
    results = verify(read_json(root / PREPROBE / 'results.json'))
    measured = {r['key']: r for r in verify(read_json(root / MEASURED))['rows']}
    by_key = {r['key']: r for r in results['rows']}
    out = []
    for item in doc['items']:
        row = by_key[item['key']]
        entry = {'key': item['key'], 'class': item['agent_visual_class'], 'error': row.get('error'),
                 'b0_score': (row.get('b0') or {}).get('score'), 'b0_passes': (row.get('b0') or {}).get('passes')}
        if item['e1'] and 'e1' in row:
            holding = [p for p in row['e1']['proposals'] if p['inside_column'] and p['label_containment'] >= P.CONTAINED]
            entry.update(e1_source='this_preprobe', e1_proposals=len(row['e1']['proposals']), e1_unique_parent=len(holding) == 1,
                         e1_containment=[round(p['label_containment'], 4) for p in row['e1']['proposals']])
        elif not item['e1']:
            m = measured[item['key']]
            entry.update(e1_source='residual_probe_f1b4eb54', e1_proposals=m.get('proposals'),
                         e1_unique_parent=bool(m.get('e1') and m['e1'][0]), e1_containment=(m.get('containment') or [[]])[0])
        out.append(entry)
    positive = {'foreground_wine_bottle', 'second_wine_bottle_at_frame_edge'}
    negatives = [e for e in out if e['class'] == 'non_wine_region']
    summary = seal({'kind': 'existing-target-parent-v1-preprobe-summary', 'plan': doc['checksum'], 'results': results['checksum'],
                    'rows': out,
                    'e1_positive_wine': [e['key'] for e in out if e['class'] in positive and e.get('e1_unique_parent')],
                    'e1_non_wine': [e['key'] for e in negatives if e.get('e1_unique_parent')],
                    'b0_positive_wine': [e['key'] for e in out if e['class'] in positive and e.get('b0_passes')],
                    'b0_non_wine_passing': [e['key'] for e in negatives if e.get('b0_passes')],
                    'uncertain': [e['key'] for e in out if e['class'].startswith('uncertain')],
                    'n1b_condition': 'every visually non-wine label crop rejected by the frozen gate',
                    'n1b_condition_met': not any(e.get('b0_passes') for e in negatives) and not any(e['error'] for e in out),
                    'note': 'agent visual classes are not owner GT; N1b source only after root review'})
    write_json(root / PREPROBE / 'summary.json', summary)
    return {k: summary[k] for k in ('checksum', 'e1_positive_wine', 'e1_non_wine', 'b0_positive_wine',
                                    'b0_non_wine_passing', 'n1b_condition_met')}


def freeze(root, path):
    from rshb_vine.existing_target_parent_v1 import runtime as N
    if (root / path).exists():
        raise SystemExit(path + ' already exists; immutable descriptor preserved')
    doc = N.freeze(root)
    write_json(root / path, doc)
    return {'descriptor': path, 'checksum': doc['checksum'], 'sources_sha256': doc['sources_sha256'],
            'parent_release': doc['parent_release']['checksum']}


def describe(root, path):
    from rshb_vine.existing_target_parent_v1 import runtime as N
    doc = N.load_descriptor(root, path)
    return {'descriptor': path, 'checksum': doc['checksum'], 'policy': doc['policy'], 'constants': doc['constants'],
            'parent_release': doc['parent_release'], 'activated': doc['activated'],
            'release_admitted': doc['release_admitted'], 'probability': None, 'models_loaded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preprobe-plan', 'preprobe-run', 'preprobe-summarize'):
        sub.add_parser(name)
    for name in ('freeze', 'describe'):
        sub.add_parser(name).add_argument('--descriptor', default=DESCRIPTOR)
    args = parser.parse_args()
    if args.command == 'preprobe-plan':
        result = preprobe_plan(ROOT)
    elif args.command == 'preprobe-run':
        result = preprobe_run(ROOT)
    elif args.command == 'preprobe-summarize':
        result = summarize(ROOT)
    elif args.command == 'freeze':
        result = freeze(ROOT, args.descriptor)
    else:
        result = describe(ROOT, args.descriptor)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
