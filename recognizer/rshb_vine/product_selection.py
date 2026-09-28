"""Version-bound product selection shared by replay and the diagnostic HTTP API."""
from copy import deepcopy
from pathlib import Path

from rshb_vine.catalog.product_registry import ProductRegistry
from rshb_vine.candidate_discriminators import enrich_candidate_features
from rshb_vine.candidate_listwise import CandidateListwise
from rshb_vine.io import read_json, verify
from rshb_vine.learned_selection_features import build_candidate_features
from rshb_vine.product_evidence_resolution import ProductEvidenceResolution
from rshb_vine.product_first.year_evidence import describe_year


class ProductSelection:
    def __init__(self, root, bundle, selector_protocol, model_path):
        root = Path(root)
        self.registry = ProductRegistry.from_bundle(root, bundle)
        protocol = verify(read_json(root / selector_protocol))
        self.model = verify(read_json(root / model_path))
        self.head = CandidateListwise(**protocol['feature_schema'],
                                     feature_source_paths=protocol['feature_source_paths'])
        self.resolver = ProductEvidenceResolution(root)

    def describe(self, slug, observations, source):
        result = describe_year({'representative_slug': slug,
                                'product_id': self.registry.product_id(slug)},
                               observations, self.registry.claims, source)
        result.update(identity_registry_checksum=self.registry.checksum,
                      claims_checksum=self.registry.claims_checksum,
                      identity_binding_status=self.registry.cards[slug]['binding_status'] if slug else 'no_product',
                      catalog_card_slugs=self.registry.members[self.registry.product_id(slug)] if slug else [])
        return result

    def select(self, *, control_slug, raw_visual, observations, control_candidates):
        features = build_candidate_features(control_slug=control_slug, raw_visual=raw_visual,
            observations=observations, cards=self.registry.cards, claims=self.registry.claims,
            control_candidates=control_candidates)
        enhanced = enrich_candidate_features(features)
        learned = self.head.predict(enhanced, self.model)
        resolved = self.resolver.resolve(features, learned)
        resolved['identity_registry_checksum'] = self.registry.checksum
        resolved['claims_checksum'] = self.registry.claims_checksum
        resolved['product_resolution'] = self.describe(resolved['representative_slug'], observations,
            {'kind': 'same_instance_ocr', 'instance_id': raw_visual['instance_id']})
        return {'features': enhanced, 'proposal': resolved}


def project_training_table(table, previous, current):
    """Rebind a saved table only when each query retains its exact candidate roster.

    A merge within a query requires rebuilding from raw model evidence; do not
    guess feature aggregation or turn an old negative into a positive here.
    This projection itself never authorizes another fit.
    """
    verify(table)
    result = deepcopy(table)
    result.pop('checksum')
    changed, reordered = [], []
    for query in result['queries']:
        candidates = query['feature_output']['candidates']
        ids = [current.product_id(c['card_slugs'][0]) for c in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError('Training query needs a raw-evidence rebuild after product merge: ' + query['id'])
        if len(query['positive_mask']) != len(candidates):
            raise ValueError('Training positive mask shape mismatch')
        for candidate in candidates:
            slugs = candidate['card_slugs']
            if len({current.product_id(s) for s in slugs}) != 1:
                raise ValueError('Previously grouped candidate was split')
            if any(candidate['provenance']['cards'][s] != previous.cards[s] for s in slugs):
                raise ValueError('Training card provenance differs from source registry')
            pid = current.product_id(slugs[0])
            if candidate['product_id'] != pid:
                changed.append({'query_id': query['id'], 'card_slugs': slugs,
                                'before': candidate['product_id'], 'after': pid})
            candidate.update(candidate_id=current.candidate_id(slugs[0]), product_id=pid,
                             identity_status=current.cards[slugs[0]]['binding_status'])
            candidate['features']['source.product_admitted'] = float(
                candidate['identity_status'] == 'source_admitted_product')
            for slug in slugs:
                candidate['provenance']['cards'][slug] = deepcopy(current.cards[slug])
                old_claim = candidate['provenance']['claims'][slug]
                new_claim = current.claims.get(slug)
                if old_claim is not None and new_claim is not None:
                    if any(old_claim[k] != new_claim[k] for k in old_claim if k not in ('product_id', 'binding_status')):
                        raise ValueError('Training attribute evidence changed')
                elif old_claim != new_claim:
                    raise ValueError('Training claim availability changed')
                candidate['provenance']['claims'][slug] = deepcopy(new_claim)
                match = candidate.get('text_evidence', {}).get('card_matches', {}).get(slug)
                if match is not None:
                    if match['product_id'] != previous.product_id(slug):
                        raise ValueError('Text match belongs to a different source product')
                    match['product_id'] = current.product_id(slug)
        before_order = [tuple(c['card_slugs']) for c in candidates]
        paired = sorted(zip(candidates, query['positive_mask'], strict=True), key=lambda row: row[0]['candidate_id'])
        query['feature_output']['candidates'] = [c for c, _ in paired]
        query['positive_mask'] = [positive for _, positive in paired]
        if before_order != [tuple(c['card_slugs']) for c, _ in paired]:
            reordered.append(query['id'])
    result.update(source_table_checksum=table['checksum'], identity_registry_checksum=current.checksum,
                  claims_checksum=current.claims_checksum, identity_projection_changes=changed,
                  positive_labels_changed=False, candidate_order_reindexed_queries=reordered,
                  groups_changed=False, fit_admitted=False,
                  fit_admission_reason='Identity migration only; a new fitting protocol and source admission remain separate.')
    return result
