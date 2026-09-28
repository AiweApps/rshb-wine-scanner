"""Identical frozen selection with request-owned provenance shared internally.

The frozen builder already copies every external input. Enrichment only writes
numeric features and query annotations; resolution only writes its comparison
query and proposal. Their large, read-only provenance/text evidence can therefore
share the builder-owned objects until this one select call returns. No registry,
request input, or cross-request object is borrowed by this optimization.
"""
from rshb_vine.product_selection import ProductSelection
from rshb_vine.learned_selection_features import build_candidate_features
from rshb_vine.candidate_discriminators import enrich_candidate_features


class _ReadOnlyEvidence(dict):
    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def _readonly(self, *_args, **_kwargs):
        raise TypeError('Selection evidence is read-only during this call')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly


class RuntimeProductSelection(ProductSelection):
    def select(self, *, control_slug, raw_visual, observations, control_candidates):
        base = build_candidate_features(control_slug=control_slug, raw_visual=raw_visual,
            observations=observations, cards=self.registry.cards, claims=self.registry.claims,
            control_candidates=control_candidates)
        for candidate in base['candidates']:
            for key in ('provenance', 'text_evidence'):
                candidate[key] = _ReadOnlyEvidence(candidate[key])
        enhanced = enrich_candidate_features(base)
        learned = self.head.predict(enhanced, self.model)
        resolved = self.resolver.resolve(base, learned)
        resolved['identity_registry_checksum'] = self.registry.checksum
        resolved['claims_checksum'] = self.registry.claims_checksum
        resolved['product_resolution'] = self.describe(resolved['representative_slug'], observations,
            {'kind':'same_instance_ocr','instance_id':raw_visual['instance_id']})
        return {'features':enhanced,'proposal':resolved}
