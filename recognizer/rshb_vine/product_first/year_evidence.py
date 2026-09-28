"""Post-ranking year evidence. Reference year is never a catalog-card year."""
from copy import deepcopy

from rshb_vine.gallery_variant_text import signature
from rshb_vine.product_vintage_scope import ProductVintageScope


def describe_year(proposal, observations, claims, source):
    slug = proposal['representative_slug']
    vintage = claims.get(slug, {}).get('vintage', {})
    reference = deepcopy(vintage.get('reference_year', {'status': 'unknown', 'claims': []}))
    catalog = deepcopy(vintage.get('catalog_vintage', {'status': 'unknown', 'claims': []}))
    observed = signature(observations)['year']
    reference_value = reference.get('comparison_value')
    reference_verified = reference.get('status') == 'source_verified' and not reference.get('review_required')
    relation = 'unknown'
    if observed is not None and reference_verified and reference_value is not None:
        relation = 'same_as_reference' if str(reference_value) == observed else 'different_from_reference'
    return {
        'product_id': proposal['product_id'], 'representative_slug': slug,
        'product_status': 'uncalibrated_proposal' if proposal['product_id'] else 'unknown',
        'exact_slug': None,
        'exact_reason': 'product_and_catalog_card_not_confirmed',
        'vintage': {
            'value': int(observed) if observed is not None else None,
            'status': 'ocr_year_candidate' if observed else 'unknown',
            'relation_to_reference': relation,
            'role_confirmed': False,
            'source': deepcopy(source),
            'observations': deepcopy(observations),
            'extractor': 'gallery_variant_text.signature:standalone_or_style_year',
        },
        'reference_year': reference, 'catalog_vintage': catalog,
        'ranking_changed': False,
    }


def attach_year_evidence(shadow, claims, source):
    result = deepcopy(shadow)
    outputs = []
    for target in result.get('targets', []):
        sid = str(target['instance_id'])
        observations = ProductVintageScope.observations(result, target['instance_id'])
        resolved = describe_year(target['visual_product_shadow'], observations, claims,
                                 {**source, 'instance_id': sid})
        target['product_resolution'] = resolved
        outputs.append({'instance_id': sid, 'resolution': resolved})
    result['product_resolution'] = {'mode': 'shadow_only', 'targets': outputs,
                                    'accepted': False, 'year_used_for_ranking': False}
    return result
