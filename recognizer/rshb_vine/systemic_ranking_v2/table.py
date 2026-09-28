"""Fit table from saved receipts over the frozen pool; labels attached after features in snapshot-03 semantics.

Roster, groups, lineage and CV folds are the immutable interaction-ranker-v1 fit roster (471 admitted rows,
46 components); nothing is re-drawn. A candidate is positive when any of its cards belongs to a snapshot-03
product of the frozen GT (multiple positives per accepted product). Loss masks are the combined ones.
"""
from rshb_vine.data_protection import load_protection
from rshb_vine.interaction_ranker_v1.source import check_trace, row_input
from rshb_vine.io import digest
from rshb_vine.ranker_integration_v3 import masks as v3_masks
from rshb_vine.systemic_ranking_v2 import evidence as E


def build_row(root, selection, projection, row, meta, pairs, forbidden):
    from rshb_vine.combined_ranker_v1.identity import ignored_cards, loss_ignore
    if row['image_sha256'] in forbidden:
        raise ValueError('Protected image reached the systemic table: ' + row['query_id'])
    payload, provenance, report, kind = row_input(root, row)
    problems = check_trace(provenance, report, payload['observations'])
    out = selection.evaluate(**payload, provenance=provenance)
    frozen_positives = set(row['ground_truth']['current_products'])
    positives03 = projection.gt_from_frozen_products(frozen_positives)
    frozen_cards = ignored_cards(row, pairs)
    candidates = []
    for c, f, ev in zip(out['base']['candidates'], out['rows'], out['evidence']):
        products03 = sorted({projection.product(s) for s in c['card_slugs']})
        positive = bool(set(products03) & positives03)
        learned = next(r for r in out['learned']['ranked_candidates'] if r['candidate_id'] == c['candidate_id'])
        candidates.append({'candidate_id': c['candidate_id'], 'product_id': c['product_id'], 'products03': products03,
                           'card_slugs': list(c['card_slugs']), 'positive': positive,
                           'positive_frozen': c['product_id'] in frozen_positives,
                           'loss_ignore': False if positive else loss_ignore(c, frozen_positives, row, pairs, frozen_cards),
                           'old103_rank': learned['rank'], 'old103_score': learned.get('score'),
                           'features': {k: v for k, v in f.items() if v}, 'evidence': ev})
    checks = {'old103_feature_digest': digest(out['enhanced']) == row['feature_digest'],
              'old103_selected': out['legacy']['representative_slug'] == row['selected_slug'],
              'provenance_trace': not problems}
    return {'query_id': row['query_id'], 'group': meta['group'], 'lineage_group': meta['lineage_group'],
            'cv_fold': meta['cv_fold'], 'fit_eligible': meta['fit_eligible'], 'source_kind': kind,
            'instance_id': str(row['instance_id']), 'image_sha256': row['image_sha256'],
            'control_slug': payload['control_slug'], 'old103_selected': out['legacy']['representative_slug'],
            'positive_products_frozen': sorted(frozen_positives), 'positive_products03': sorted(positives03),
            'feature_schema': E.SCHEMA_VERSION, 'feature_digest': out['feature_digest'],
            'provenance_digest': digest(provenance), 'trace_problems': problems, 'parity': checks,
            'query': out['query'], 'seconds': out['seconds'], 'candidates': candidates}


def build(root, selection, projection, rows, metas):
    forbidden = set(load_protection(root)['forbidden_sha256'])
    pairs = v3_masks.ignored_pairs(root)
    return [build_row(root, selection, projection, row, metas[row['query_id']], pairs, forbidden) for row in rows]
