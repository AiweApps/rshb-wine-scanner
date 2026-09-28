"""Opt-in selector shared by receipt replay and the live runtime (same feature path for both).

old103 runs unchanged on the same frozen pool (baseline, rollback, parity). The systemic evidence is built
over that pool with OCR provenance; a loaded ranker orders every pool candidate with one linear function and
its proposal passes the unchanged existing-evidence resolver. Without a model the public answer is old103.
The candidate generator (4 visual Top20 + supplied 8175 cards) is never extended.
"""
from pathlib import Path
import time

import numpy as np

from rshb_vine.candidate_discriminators import enrich_candidate_features
from rshb_vine.candidate_listwise import _representative
from rshb_vine.interaction_ranker_v1.source import trace_result
from rshb_vine.io import digest, read_json, verify
from rshb_vine.learned_selection_features import build_candidate_features
from rshb_vine.runtime_product_selection import RuntimeProductSelection, _ReadOnlyEvidence
from rshb_vine.system_selection_v4.context import SelectionContext
from rshb_vine.system_selection_v4.features import build_features
from rshb_vine.systemic_ranking_v2 import evidence as E
from rshb_vine.systemic_ranking_v2 import model as M

BASELINE_PROFILE = 'config/recognition-old103-v1.json'
ADAPTER_VERSION = 'systemic-ranking-v2-selection'


def matrix(rows):
    return np.asarray([[r[n] for n in E.FEATURE_NAMES] for r in rows], dtype=np.float64).reshape(len(rows), len(E.FEATURE_NAMES))


def feature_digest(rows, candidate_ids):
    return digest([{'candidate_id': c, 'features': r} for c, r in zip(candidate_ids, rows)])


class SystemicSelection:
    def __init__(self, root, legacy, *, model_path=None, public='legacy', context=None):
        if public not in ('legacy', 'systemic') or (public == 'systemic' and model_path is None):
            raise ValueError('Publishing the systemic ranker requires a sealed model')
        self.root = Path(root)
        self.legacy = legacy
        self.registry = legacy.registry
        self.model = legacy.model
        self.context = context or SelectionContext(root)
        self.stats = E.CatalogStats(self.registry, self.context)
        from rshb_vine.combined_ranker_v1.identity import load_candidate
        self.identity_handoff, self.identity = load_candidate(self.root)
        if set(self.identity.cards) != set(self.registry.cards):
            raise ValueError('snapshot-03 identity must keep the frozen card roster')
        self.ranker = None
        if model_path is not None:
            self.use_ranker(verify(read_json(self.root / model_path)))
        self.public = public
        self.current_result = None
        self.records = []

    def use_ranker(self, model):
        if model['features'] != list(E.FEATURE_NAMES) or model['signs'] != {n: E.SIGNS[n] for n in E.FEATURE_NAMES}:
            raise ValueError('systemic-ranking-v2 model/schema mismatch')
        M.check_loadable(model)
        self.ranker = model

    @classmethod
    def from_profile(cls, root, profile_path=BASELINE_PROFILE, **kwargs):
        from rshb_vine.recognition_runtime import load_profile
        profile, _, _ = load_profile(Path(root), Path(root) / profile_path)
        selector = profile['selector']
        legacy = RuntimeProductSelection(root, profile['product_bundle'], selector['protocol'], selector['model'])
        return cls(root, legacy, **kwargs)

    def evaluate(self, *, control_slug, raw_visual, observations, control_candidates, provenance):
        started = time.perf_counter()
        base = build_candidate_features(control_slug=control_slug, raw_visual=raw_visual, observations=observations,
                                        cards=self.registry.cards, claims=self.registry.claims,
                                        control_candidates=control_candidates)
        for candidate in base['candidates']:
            for key in ('provenance', 'text_evidence'):
                candidate[key] = _ReadOnlyEvidence(candidate[key])
        enhanced = enrich_candidate_features(base)
        learned = self.legacy.head.predict(enhanced, self.legacy.model)
        legacy = self.legacy.resolver.resolve(base, learned)
        legacy_seconds = time.perf_counter() - started
        v4 = build_features(base, raw_visual, self.context, self.registry.cards, self.registry.claims)
        rows, raw, query = E.build(base, v4, provenance, self.context, self.stats, control_slug, self.identity.product_id)
        ids = [c['candidate_id'] for c in v4['candidates']]
        if ids != [c['candidate_id'] for c in enhanced['candidates']]:
            raise ValueError('v4 and old103 pools differ')
        proposal_raw = proposal = None
        if self.ranker is not None:
            proposal_raw = self._propose(base, v4, rows, control_slug)
            proposal = self.legacy.resolver.resolve(base, proposal_raw)
        return {'base': base, 'enhanced': enhanced, 'learned': learned, 'legacy': legacy, 'v4': v4,
                'rows': rows, 'evidence': raw, 'query': query, 'candidate_ids': ids,
                'feature_digest': feature_digest(rows, ids), 'raw_proposal': proposal_raw, 'proposal': proposal,
                'seconds': {'legacy': legacy_seconds, 'total': time.perf_counter() - started}}

    def _propose(self, base, v4, rows, control_slug):
        values = M.score(self.ranker, matrix(rows))
        if not np.isfinite(values).all():
            raise ValueError('Non-finite systemic score')
        candidates = base['candidates']
        order = sorted(range(len(candidates)), key=lambda i: (-values[i], -v4['candidates'][i]['features']['vis.support'],
                                                              candidates[i]['candidate_id']))
        ranked = []
        for position, i in enumerate(order, 1):
            c = candidates[i]
            representative, source = _representative(c)
            ranked.append({'rank': position, 'input_pool_index': i, 'candidate_id': c['candidate_id'],
                           'product_id': c['product_id'], 'identity_status': c['identity_status'],
                           'card_slugs': list(c['card_slugs']), 'representative_slug': representative,
                           'representative_source': source, 'score': float(values[i]),
                           'is_control_proposal': control_slug in c['card_slugs']})
        winner = ranked[0] if ranked else None
        return {'schema_version': 'systemic-ranking-v2-proposal', 'model_checksum': self.ranker['checksum'],
                'ranked_candidates': ranked, 'candidate_id': winner and winner['candidate_id'],
                'product_id': winner and winner['product_id'], 'representative_slug': winner and winner['representative_slug'],
                'best_candidate': winner and winner['representative_slug'],
                'representative_source': winner and winner['representative_source'], 'score': winner and winner['score'],
                'score_margin': ranked[0]['score'] - ranked[1]['score'] if len(ranked) > 1 else None,
                'exact_slug': None, 'vintage': {'value': None, 'status': 'resolved_after_selection'},
                'probability': None, 'confidence': {'status': 'uncalibrated_proposal', 'probability': None},
                'reason': 'systemic_ranking_v2' if winner else 'empty_candidate_pool', 'tie_policy': M.TIE_POLICY,
                'release_admitted': False}

    def select(self, *, control_slug, raw_visual, observations, control_candidates):
        """Live entry: provenance comes from the in-flight control result set by ``attach``."""
        if self.current_result is None:
            raise ValueError('Live systemic selection needs the in-flight control result')
        provenance, report = trace_result(self.current_result, raw_visual['instance_id'], observations)
        out = self.evaluate(control_slug=control_slug, raw_visual=raw_visual, observations=observations,
                            control_candidates=control_candidates, provenance=provenance)
        public = out['proposal'] if self.public == 'systemic' else out['legacy']
        for key in ('identity_registry_checksum', 'claims_checksum'):
            public[key] = getattr(self.registry, key.replace('identity_registry_', ''))
        public['product_resolution'] = self.legacy.describe(public['representative_slug'], observations,
                                                            {'kind': 'same_instance_ocr', 'instance_id': raw_visual['instance_id']})
        record = {'adapter': ADAPTER_VERSION, 'instance_id': str(raw_visual['instance_id']), 'public': self.public,
                  'legacy_selected': out['legacy']['representative_slug'], 'legacy_feature_digest': digest(out['enhanced']),
                  'systemic_feature_digest': out['feature_digest'], 'trace_report': report,
                  'ranker': self.ranker['checksum'] if self.ranker else None,
                  'systemic_selected': out['proposal']['representative_slug'] if out['proposal'] else None,
                  'seconds': out['seconds']}
        self.records.append(record)
        return {'features': out['enhanced'], 'proposal': public, 'systemic_ranking_v2': record}


def attach(runtime, selection):
    """Swap a RecognitionRuntime's selector for ``selection`` and expose the control result during selection."""
    original = runtime._attach_products

    def hooked(result, data, control_seconds):
        selection.current_result = result
        try:
            return original(result, data, control_seconds)
        finally:
            selection.current_result = None

    runtime.selection = selection
    runtime._attach_products = hooked
    return runtime
