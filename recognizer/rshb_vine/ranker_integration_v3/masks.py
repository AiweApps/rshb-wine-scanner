"""Loss masks from the global confusability graph applied to every actual real train pool.

A candidate is masked (neither positive nor negative) when any of its card slugs forms a
never_negative / loss_ignore / negative_unverified pair with any acceptable GT slug.
Positive sets are never changed: a positive product whose other card slugs hit a pair stays positive.
"""
from collections import Counter
from pathlib import Path

from rshb_vine.io import read_jsonl, seal, sha256, write_json

REAL_TABLE = 'runs/ranker-dataset-v1-root-admission01/admitted-table.jsonl'
REAL_TABLE_SHA256 = '5b4e698636ec0087e94b38eb3c23e416f62ee3db40f221c943b310a7f90a2517'
EXTRA_ROWS = 'runs/ranker-completion-v2/admitted-extra-rows.jsonl'
EXTRA_ROWS_SHA256 = '40f1bef2209296220e6d868c4d0ca4abeb9f8213a1c6ef2b752d50cafc81f5ed'
IGNORED = 'data/global-wine-confusability-v1/curriculum/ignored-pairs.jsonl'
IGNORED_SHA256 = '6aa23b563cef14fe14d451804272747a4e556ff4b6f79b70f26b59be3d20b096'
GRADES = {'never_negative', 'loss_ignore', 'negative_unverified'}
OUT = 'runs/ranker-integration-v3/masks'


def pair_key(a, b):
    return '||'.join(sorted((a, b)))


def ignored_pairs(root):
    return {r['pair']: r for r in read_jsonl(Path(root) / IGNORED) if r['grade'] in GRADES}


def mask_row(row, pairs):
    """Return (added_ids, occurrences) for one real pool; existing ids are kept separately."""
    positives = set(row['ground_truth']['current_products'])
    gt = set(row['ground_truth']['acceptable_slugs'])
    existing = set(row.get('loss_ignore_candidate_ids', []))
    added, occurrences = [], []
    for c in row['candidate_pool']:
        slugs = set(c['card_slugs']) | {c['representative_slug']}
        hits = sorted({pair_key(g, s) for g in gt for s in slugs - gt if pair_key(g, s) in pairs})
        if not hits:
            continue
        occurrences.append({'candidate_id': c['candidate_id'], 'product_id': c['product_id'],
                            'pairs': hits, 'grades': sorted({pairs[h]['grade'] for h in hits}),
                            'grade_reasons': sorted({x for h in hits for x in pairs[h].get('grade_reasons', [])}),
                            'already_ignored': c['candidate_id'] in existing,
                            'positive_kept': c['product_id'] in positives})
        if c['product_id'] not in positives and c['candidate_id'] not in existing:
            added.append(c['candidate_id'])
    return added, occurrences


def build(root):
    root = Path(root)
    for path, expected in ((REAL_TABLE, REAL_TABLE_SHA256), (EXTRA_ROWS, EXTRA_ROWS_SHA256), (IGNORED, IGNORED_SHA256)):
        if sha256(root / path) != expected:
            raise ValueError('Input changed: ' + path)
    pairs = ignored_pairs(root)
    rows = [(REAL_TABLE, r) for r in read_jsonl(root / REAL_TABLE) if r['ranker_split'] == 'train']
    rows += [(EXTRA_ROWS, r) for r in read_jsonl(root / EXTRA_ROWS)]
    records, counts = [], Counter()
    for source, row in rows:
        added, occurrences = mask_row(row, pairs)
        positives = set(row['ground_truth']['current_products'])
        ignore = set(row.get('loss_ignore_candidate_ids', [])) | set(added)
        keep = [c for c in row['candidate_pool'] if c['candidate_id'] not in ignore]
        n_pos = sum(c['product_id'] in positives for c in keep)
        counts['rows'] += 1
        counts['occurrences'] += len(occurrences)
        counts['occurrences_already_ignored'] += sum(o['already_ignored'] for o in occurrences)
        counts['occurrences_on_positive_kept'] += sum(o['positive_kept'] for o in occurrences)
        counts['added_candidate_ids'] += len(added)
        counts['rows_with_added'] += bool(added)
        counts['rows_losing_all_negatives'] += n_pos > 0 and n_pos == len(keep)
        for o in occurrences:
            for g in o['grades']:
                counts['occurrences.' + g] += 1
        records.append({'query_id': row['query_id'], 'source_rows': source, 'image_sha256': row['image_sha256'],
                        'positive_products': sorted(positives),
                        'positive_candidate_ids_after_mask': sorted(c['candidate_id'] for c in keep
                                                                    if c['product_id'] in positives),
                        'previous_loss_ignore_candidate_ids': sorted(row.get('loss_ignore_candidate_ids', [])),
                        'added_loss_ignore_candidate_ids': sorted(added),
                        'occurrences': occurrences, 'pool_size': len(row['candidate_pool']),
                        'kept_after_mask': len(keep)})
    counts['unique_pairs'] = len({p for r in records for o in r['occurrences'] for p in o['pairs']})
    document = seal({'kind': 'ranker-integration-v3-loss-masks',
                     'rule': 'mask candidate if any card slug pairs with any acceptable GT slug in ignored-pairs '
                             'grades ' + ','.join(sorted(GRADES)) + '; positives unchanged; union with previous ids',
                     'inputs_sha256': {REAL_TABLE: REAL_TABLE_SHA256, EXTRA_ROWS: EXTRA_ROWS_SHA256,
                                       IGNORED: IGNORED_SHA256},
                     'counts': dict(counts), 'records': records})
    write_json(root / OUT / 'masks.json', document, replace=True)
    return document['counts']
