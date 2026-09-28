"""Composable adapter: the frozen SystemicSelection with OCR catalogue-phrase candidates added to its pool.

Without an injection the call is the parent ``evaluate`` itself (byte-identical features and proposal).
With one, the same frozen builders run on the pool plus the injected cards; those cards carry an explicit
``ocr_catalog_injection`` provenance, a zero supplied-8175 feature and no visual rank (their rows ride the
supplied-candidate list only so the frozen representative logic can name them; the source is relabelled). The frozen ranker orders the
pool and the unchanged existing-evidence resolver decides; probability and exact slug stay None.
"""
from pathlib import Path
import time

from rshb_vine.candidate_discriminators import enrich_candidate_features
from rshb_vine.interaction_ranker_v1.source import trace_result
from rshb_vine.io import digest
from rshb_vine.learned_selection_features import _candidate_key, build_candidate_features
from rshb_vine.ocr_candidate_repair_v1 import injection as I
from rshb_vine.runtime_product_selection import _ReadOnlyEvidence
from rshb_vine.system_selection_v4.features import build_features
from rshb_vine.systemic_ranking_v2 import evidence as E
from rshb_vine.systemic_ranking_v2.selection import feature_digest

ADAPTER_VERSION = 'ocr-candidate-repair-v1-selection'
SOURCE = 'ocr_catalog_phrase_injection'


def _relabel(proposal, injected_keys):
    """The frozen representative logic reads injected rows from the supplied-candidate list; name their real source."""
    for row in proposal.get('ranked_candidates', []):
        row['ocr_catalog_injected'] = row['candidate_id'] in injected_keys
        if row['ocr_catalog_injected']:
            row['representative_source'] = SOURCE
    if proposal.get('candidate_id') in injected_keys and proposal.get('representative_source') == 'actual8175_supplied_candidate':
        proposal['representative_source'] = SOURCE


class OcrCandidateSelection:
    def __init__(self, root, inner, conflicts=None):
        self.root, self.inner = Path(root), inner
        self.registry, self.legacy, self.model = inner.registry, inner.legacy, inner.model
        self.index = I.CatalogPhraseIndex(inner, I.eligible_slugs(self.root, inner.registry))
        self.conflicts = conflicts
        self.current_result = None
        self.records = []

    @property
    def ranker(self):
        return self.inner.ranker

    @property
    def public(self):
        return self.inner.public

    def use_ranker(self, model):
        self.inner.use_ranker(model)

    @staticmethod
    def pool_slugs(raw_visual, control_slug, control_candidates):
        slugs = {r['slug'] for arm in raw_visual.get('arms', {}).values()
                 for rows in arm.get('retrieval', {}).get('channel_top20', {}).values() for r in rows}
        slugs |= {r if isinstance(r, str) else r['slug'] for r in control_candidates or []}
        return slugs | ({control_slug} if control_slug else set())

    def evaluate(self, *, control_slug, raw_visual, observations, control_candidates, provenance, other_line_keys=()):
        decision = self.index.propose(observations, provenance,
                                      self.pool_slugs(raw_visual, control_slug, control_candidates),
                                      other_line_keys, self.conflicts)
        if not decision['injected']:
            out = self.inner.evaluate(control_slug=control_slug, raw_visual=raw_visual, observations=observations,
                                      control_candidates=control_candidates, provenance=provenance)
            out['ocr_candidate_injection'] = decision
            return out
        return self._evaluate_injected(control_slug, raw_visual, observations, control_candidates, provenance, decision)

    def _evaluate_injected(self, control_slug, raw_visual, observations, control_candidates, provenance, decision):
        inner, started = self.inner, time.perf_counter()
        supplied = list(control_candidates or [])
        extra = [{'slug': s, 'source': SOURCE} for s in decision['injected']]
        base = build_candidate_features(control_slug=control_slug, raw_visual=raw_visual, observations=observations,
                                        cards=self.registry.cards, claims=self.registry.claims,
                                        control_candidates=supplied + extra)
        injected_keys = {_candidate_key(s, self.registry.cards) for s in decision['injected']}
        by_slug = {m['slug']: m for m in decision['matched']}
        for candidate in base['candidates']:
            if candidate['candidate_id'] in injected_keys:
                if candidate['provenance']['visual'] or candidate['features']['control.is_proposal']:
                    raise ValueError('Injected candidate already carried visual/control evidence')
                candidate['features']['control.in_supplied_candidates'] = 0.
                candidate['provenance']['ocr_catalog_injection'] = [by_slug[s] for s in candidate['card_slugs']]
            for key in ('provenance', 'text_evidence'):
                candidate[key] = _ReadOnlyEvidence(candidate[key])
        enhanced = enrich_candidate_features(base)
        learned = inner.legacy.head.predict(enhanced, inner.legacy.model)
        legacy = inner.legacy.resolver.resolve(base, learned)
        legacy_seconds = time.perf_counter() - started
        v4 = build_features(base, raw_visual, inner.context, self.registry.cards, self.registry.claims)
        rows, raw, query = E.build(base, v4, provenance, inner.context, inner.stats, control_slug,
                                   inner.identity.product_id)
        ids = [c['candidate_id'] for c in v4['candidates']]
        if ids != [c['candidate_id'] for c in enhanced['candidates']]:
            raise ValueError('v4 and old103 pools differ')
        proposal_raw = proposal = None
        if inner.ranker is not None:
            proposal_raw = inner._propose(base, v4, rows, control_slug)
            _relabel(proposal_raw, injected_keys)
            proposal = inner.legacy.resolver.resolve(base, proposal_raw)
        _relabel(learned, injected_keys)
        _relabel(legacy, injected_keys)
        return {'base': base, 'enhanced': enhanced, 'learned': learned, 'legacy': legacy, 'v4': v4,
                'rows': rows, 'evidence': raw, 'query': query, 'candidate_ids': ids,
                'feature_digest': feature_digest(rows, ids), 'raw_proposal': proposal_raw, 'proposal': proposal,
                'ocr_candidate_injection': decision,
                'seconds': {'legacy': legacy_seconds, 'total': time.perf_counter() - started}}

    def select(self, *, control_slug, raw_visual, observations, control_candidates):
        """Live entry mirroring the parent ``select``; the in-flight result is set by the parent ``attach``."""
        if self.current_result is None:
            raise ValueError('Live selection needs the in-flight control result')
        iid = raw_visual['instance_id']
        provenance, report = trace_result(self.current_result, iid, observations)
        out = self.evaluate(control_slug=control_slug, raw_visual=raw_visual, observations=observations,
                            control_candidates=control_candidates, provenance=provenance,
                            other_line_keys=I.scene_line_keys(self.current_result, iid))
        public = out['proposal'] if self.public == 'systemic' else out['legacy']
        for key in ('identity_registry_checksum', 'claims_checksum'):
            public[key] = getattr(self.registry, key.replace('identity_registry_', ''))
        public['product_resolution'] = self.legacy.describe(public['representative_slug'], observations,
                                                            {'kind': 'same_instance_ocr', 'instance_id': iid})
        decision = out['ocr_candidate_injection']
        record = {'adapter': ADAPTER_VERSION, 'instance_id': str(iid), 'public': self.public,
                  'legacy_selected': out['legacy']['representative_slug'], 'legacy_feature_digest': digest(out['enhanced']),
                  'systemic_feature_digest': out['feature_digest'], 'trace_report': report,
                  'ranker': self.ranker['checksum'] if self.ranker else None,
                  'systemic_selected': out['proposal']['representative_slug'] if out['proposal'] else None,
                  'ocr_candidate_injection': {k: decision[k] for k in ('rule', 'index_checksum', 'status', 'injected',
                                                                      'skipped', 'matched_products')},
                  'discriminator_guard': out.get('discriminator_guard'),
                  'seconds': out['seconds']}
        self.records.append(record)
        return {'features': out['enhanced'], 'proposal': public, 'systemic_ranking_v2': record}


def install(recognition, conflicts=None):
    """Wrap the selector of a constructed B3OnlyRecognition in place of its frozen SystemicSelection."""
    from rshb_vine.systemic_ranking_v2.selection import attach
    runtime, inner = recognition.runtime, recognition.target.inner.selection
    if runtime.selection is not inner or type(inner).__name__ != 'SystemicSelection':
        raise ValueError('Systemic selection ownership changed')
    wrapper = OcrCandidateSelection(inner.root, inner, conflicts)
    attach(runtime, wrapper)
    recognition.target.inner.selection = wrapper
    return wrapper
