"""Runtime filter: a contradicted photo binding casts no visual vote for its card.

The card stays in the catalog and remains reachable through independent text evidence; the sibling keeps its
references. Scores use the parent's full-channel matmul (bitwise parity for every other reference) and only skip
excluded rows. Install on the innermost LabelFirstIndex so request caches above it record filtered results only.
"""
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from rshb_vine.visual_core import LabelFirstIndex, validate_vectors


class ConflictFilteredIndex:
    def __init__(self, index, guard):
        if type(index) is not LabelFirstIndex:
            raise TypeError('ConflictFilteredIndex wraps a LabelFirstIndex')
        if not getattr(guard, 'validated', False):
            raise ValueError('Reference-conflict guard is not validated')
        self._index = index
        self.excluded = guard.excluded_references(index.references)
        self.drop = frozenset(e['reference_index'] for e in self.excluded)
        self.provenance = {'policy': 'reference-conflicts-v1', 'config_checksum': guard.checksum,
                           'excluded_references': self.excluded, 'product_records_preserved': True,
                           'text_evidence_unaffected': True}

    def __getattr__(self, name):
        return getattr(self._index, name)

    def search(self, query, views, encoder_id):
        """LabelFirstIndex.search with excluded rows skipped after the unchanged full-channel matmul."""
        parent = self._index
        if encoder_id != parent.encoder_id:
            raise ValueError('Query/gallery encoder mismatch; rebuild both channels')
        query = np.asarray(query, dtype='float32'); validate_vectors(query, len(views))
        scores = {}; lists = {}
        for kind in ('context', 'front_label'):
            qids = [i for i, v in enumerate(views) if v.kind == kind or (kind == 'front_label' and v.kind in ('partial_label', 'detected_label'))]
            ids = parent.channel_ids[kind]
            per_slug = {}
            if qids and len(ids):
                values = parent.vectors[ids] @ query[qids].T
                for offset, i in enumerate(ids):
                    if int(i) in self.drop:
                        continue
                    j = int(np.argmax(values[offset])); value = float(values[offset, j]); slug = parent.references[i]['slug']
                    if slug not in per_slug or value > per_slug[slug]['score']:
                        per_slug[slug] = {'slug': slug, 'score': value, 'reference_index': int(i), 'query_view_index': qids[j]}
            scores[kind] = per_slug
            lists[kind] = sorted(per_slug.values(), key=lambda r: (-r['score'], r['slug']))[:20]
        primary = 'front_label' if lists['front_label'] else 'context'
        pool = {r['slug'] for branch in lists.values() for r in branch}
        ranked = []
        for slug in pool:
            evidence = {kind: scores[kind].get(slug) for kind in scores}
            primary_score = evidence[primary]['score'] if evidence[primary] else None
            ranked.append({'slug': slug, 'score': primary_score, 'channels': evidence})
        ranked.sort(key=lambda r: (r['score'] is None, -r['score'] if r['score'] is not None else 0, r['slug']))
        disagreement = bool(lists['context'] and lists['front_label'] and lists['context'][0]['slug'] != lists['front_label'][0]['slug'])
        return {'ranked_candidates': ranked, 'best_candidate': ranked[0]['slug'] if ranked else None,
                'channel_top20': lists, 'candidate_union': sorted(pool), 'primary_channel': primary,
                'channel_disagreement': disagreement, 'decision': 'uncertain' if ranked else 'unknown',
                'slug': None, 'probability_correct': None,
                'reasons': ['confidence_not_calibrated'] + (['partial_label_evidence'] if any(v.kind == 'partial_label' for v in views) else []) + (['channel_disagreement'] if disagreement else []) +
                           (['label_visibility_unassessed'] if any(v.kind == 'detected_label' for v in views) else []) +
                           (['complete_label_unavailable_context_fallback'] if primary == 'context' else []),
                'views': [asdict(v) for v in views], 'reference_conflicts': deepcopy(self.provenance)}


def filter_index(index, guard):
    """Composable entry point for a versioned candidate (e.g. right after adapter.arm_components)."""
    return ConflictFilteredIndex(index, guard)
