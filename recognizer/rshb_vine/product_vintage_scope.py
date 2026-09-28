"""Experimental product ranking separated from vintage evidence.

Year-bearing source signatures remain intact. Only the product-ranking view
omits year; audited within-product vintage switches retain their guard.
"""
import copy
from rshb_vine.catalog_constrained_title import CatalogConstrainedTitle
from rshb_vine.product_name_selector import ProductNameSelector
from rshb_vine.within_producer_title import WithinProducerTitle
from rshb_vine.label_specialist_consensus import VariantGuard, combine
from rshb_vine.gallery_variant_text import signature


class ProductVintageScope:
    def __init__(self, catalog, source_signatures, product_groups):
        self.source_signatures = copy.deepcopy(source_signatures)
        self.groups = copy.deepcopy(product_groups)
        product_signatures = {
            slug: {key: value for key, value in data.items() if key != 'year'}
            for slug, data in source_signatures.items()
        }
        self.single = CatalogConstrainedTitle(catalog, product_signatures)
        self.multi = ProductNameSelector(catalog, product_signatures)
        self.title = WithinProducerTitle(self.single)
        self.guard = VariantGuard(catalog, product_signatures, product_groups)

    @staticmethod
    def observations(result, sid):
        if len(result['targets']) == 1:
            vt = result.get('variant_text', {})
            return vt.get('observations', []) if vt.get('performed') else []
        return next((t.get('observations', []) for t in result.get('instance_text', {}).get('targets', [])
                     if t.get('performed') and str(t['instance_id']) == str(sid)), [])

    def product_rank(self, source):
        result = copy.deepcopy(source)
        for target in result['targets']:
            ret = target['retrieval']
            obs = self.observations(result, target['instance_id'])
            visual = ret.get('visual_ranked_candidates', ret['ranked_candidates'])
            policy = self.single if len(result['targets']) == 1 else self.multi
            ranked, trace = policy.rerank(copy.deepcopy(visual), obs)
            ret.update(ranked_candidates=ranked, best_candidate=ranked[0]['slug'] if ranked else None)
            ret['product_scope_rerank'] = trace
        if len(result['targets']) == 1:
            ret = result['targets'][0]['retrieval']
            result.update(best_candidate=ret['best_candidate'], ranked_candidates=ret['ranked_candidates'])
        return result

    def apply(self, b0, b3):
        result = combine(self.product_rank(b0), self.product_rank(b3), self.guard, self.title)
        for target in result['targets']:
            slug = target['retrieval']['best_candidate']
            observed = signature(self.observations(result, target['instance_id']))['year']
            reference = self.groups.get(slug, {}).get('year') or self.source_signatures.get(slug, {}).get('year')
            reference = str(reference) if reference is not None else None
            status = 'unknown' if not observed or not reference else 'same_year' if observed == reference else 'different_year'
            target['vintage_evidence'] = {
                'observed_year': observed, 'reference_year': reference,
                'status': status, 'exact_card_verified': False,
                'meaning': 'Year evidence only; product candidate is not exact-card acceptance',
            }
        result['product_vintage_scope'] = 'product-first-vintage-evidence-v1'
        return result
