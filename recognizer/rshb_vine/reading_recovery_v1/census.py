"""Saved-body eligibility census of R and O over the admitted open593 roster; response metadata only, no model."""
import collections
from pathlib import Path

from rshb_vine.io import read_json, read_jsonl, seal, sha256
from rshb_vine.reading_recovery_v1 import rotated_ocr as O
from rshb_vine.reading_recovery_v1 import route as R

ROOT = Path(__file__).resolve().parents[2]
GATE = 'runs/recognition-repair-20260926/open-http-gate'
ADMISSION = 'runs/evidence-consistency-v1/open593-admission.json'
CATALOG = 'data/catalog-additions-20260921'
CONTROL_LABELLED = 10
CONTROL_NEGATIVE = 15
ZERO_TARGET_RETENTION = 'main183:W0353'


def census_lexicon(root=ROOT):
    """The lexicon of the frozen alternative engine as ExpandedCatalogPipeline extends it (CPU, no reader)."""
    from rshb_vine.catalog_additions import extend_profile
    from rshb_vine.sparse_alternative_ocr_v2 import AlternateEvidence
    from rshb_vine.io import verify
    directory = root / CATALOG
    manifest = verify(read_json(directory / 'manifest.json'))
    engine = AlternateEvidence(root)
    extend_profile(engine, read_jsonl(directory / 'catalog.jsonl'), manifest['added_canonical_slugs'],
                   manifest['checksum'], root)
    return engine.lexicon


def run(root=ROOT):
    import sys
    sys.path.insert(0, str(root / 'scripts'))
    import recognition_repair_v2 as RR
    protocol = read_json(root / GATE / 'protocol.json')
    ledger = {r['key']: r for r in read_jsonl(root / GATE / 'receipts/ledger.jsonl')}
    forms = O.producer_forms(census_lexicon(root), root)
    rows, saved = [], set()
    for row, body, _ in RR.roster():
        saved.add(row['key'])
        r, o = R.plan(body), O.plan(body, forms)
        rows.append({'key': row['key'], 'image_sha256': row['sha256'], 'path': row['path'], 'set': row['set'],
                     'cohort': row['cohort'], 'roi': row['roi'], 'gt_level': row.get('gt_level'),
                     'targets': len(body.get('targets') or []), 'response_sha256': ledger[row['key']]['response_sha256'],
                     'R': {**r, 'eligible': r['eligible'] and row['roi'] is None,
                           'reason': r['reason'] if row['roi'] is None or not r['eligible'] else 'roi_request'},
                     'O': o})
    invalid = [{'key': r['key'], 'image_sha256': r['sha256'], 'cohort': r['cohort'], 'http_status': ledger[r['key']]['http_status']}
               for r in protocol['rows'] if r['key'] not in saved]
    by_key = {r['key']: r for r in rows}

    def eligible(arm):
        return sorted((r for r in rows if r[arm]['eligible']), key=lambda r: r['key'])

    def entry(r, why):
        return {'key': r['key'], 'image_sha256': r['image_sha256'], 'cohort': r['cohort'], 'roi': r['roi'], 'why': why}

    real = [r for r in rows if r['cohort'] not in ('negative_nonwine',) and not r['cohort'].startswith('synthetic')]
    labelled = sorted((r for r in real if r['gt_level'] and r['gt_level'] != 'none' and r['roi'] is None and r['targets']),
                      key=lambda r: r['key'])
    negatives = sorted((r for r in rows if r['cohort'] == 'negative_nonwine'), key=lambda r: r['key'])
    roi = sorted((r for r in rows if r['roi'] is not None), key=lambda r: r['key'])
    arms = {}
    for arm in ('R', 'O'):
        chosen = eligible(arm)
        keys = {r['key'] for r in chosen}
        controls = [r for r in labelled if r['key'] not in keys][:CONTROL_LABELLED]
        controls = [entry(r, 'unchanged_labelled_real_first_by_key') for r in controls]
        controls += [entry(r, 'negative_nonwine_first_by_key') for r in negatives if r['key'] not in keys][:CONTROL_NEGATIVE]
        controls += [entry(r, 'real_supplied_roi') for r in roi if r['key'] not in keys]
        if ZERO_TARGET_RETENTION in by_key and ZERO_TARGET_RETENTION not in keys:
            controls.append(entry(by_key[ZERO_TARGET_RETENTION], 'zero_target_route_retention'))
        arms[arm] = {'eligible': [entry(r, r[arm]['reason']) for r in chosen], 'controls': controls,
                     'paired_images': len(chosen) + len(controls),
                     'reasons': dict(collections.Counter(r[arm]['reason'] for r in rows)),
                     'eligible_by_cohort': dict(collections.Counter(r['cohort'] for r in chosen))}
    return seal({'kind': 'reading-recovery-v1-census', 'roster': GATE + '/protocol.json',
                 'roster_sha256': sha256(root / GATE / 'protocol.json'),
                 'admission': ADMISSION, 'admission_sha256': sha256(root / ADMISSION) if (root / ADMISSION).exists() else None,
                 'saved_bodies': len(rows), 'invalid_uploads': invalid,
                 'negative_nonwine_all': [entry(r, {'R': r['R']['reason'], 'O': r['O']['reason']}) for r in negatives],
                 'sources_sha256': {p: sha256(root / p) for p in ('rshb_vine/reading_recovery_v1/census.py',
                                                                  'rshb_vine/reading_recovery_v1/route.py',
                                                                  'rshb_vine/reading_recovery_v1/rotated_ocr.py')},
                 'producer_forms': len(forms), 'arms': arms,
                 'rows': [{k: r[k] for k in ('key', 'image_sha256', 'cohort', 'roi', 'targets', 'R', 'O')} for r in rows],
                 'limits': 'saved 013c845a-candidate bodies: route/OCR upstream equals fa317ef2 except zero-target (a), '
                           'which acts only on instance-free responses (disjoint from R); no pixel, OCR or model run'})
