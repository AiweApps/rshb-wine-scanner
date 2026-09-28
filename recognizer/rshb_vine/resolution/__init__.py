"""Explicit variant conflicts outrank cosmetic similarity; no uncalibrated acceptance."""
import re


def year_proposals(observations):
    values = []
    for row in observations:
        text = row['raw_text'].casefold()
        role = ('founding' if re.search(r'основан|основания|since|est\b',text) else
                'bottling' if re.search(r'розлив|bottl',text) else
                'harvest' if re.search(r'урожа|винтаж|vintage|harvest',text) else 'unknown')
        for value in re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',text):
            values.append({'value':value,'role':role,'score':row['score'],'raw_text':row['raw_text'],
                           'polygon':row['polygon']})
    return values


def resolve(candidates, catalog, relationships, observations, ambiguous_target=False, incomplete=False):
    years = year_proposals(observations)
    # Unqualified four digits remain a proposal, never a hard veto.
    trusted = {y['value'] for y in years if y['role']=='harvest' and y['score']>=.85}
    rows = []
    for candidate in candidates:
        row = dict(candidate)
        item = catalog[row['slug']]
        if item.get('excluded_from_retrieval'):
            continue
        expected = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',item['fields'].get('Название вина','')))
        conflicts = []
        if len(trusted)==1 and len(expected)==1 and trusted.isdisjoint(expected):
            conflicts.append('vintage_conflict')
        color_words = {'красное','белое','розовое'}
        seen_colors = {word for obs in observations if obs['score'] >= .9
                       for word in re.findall(r'\w+',obs['raw_text'].casefold()) if word in color_words}
        expected_colors = set(re.findall(r'\w+',item['fields'].get('Категория','').casefold())) & color_words
        if len(seen_colors)==1 and len(expected_colors)==1 and seen_colors.isdisjoint(expected_colors):
            conflicts.append('color_conflict')
        row['attribute_evidence'] = {'catalog_year_proposals':sorted(expected), 'observed_years':years,
                                     'conflicts':conflicts}
        row['contradicted'] = bool(conflicts)
        rows.append(row)
    rows.sort(key=lambda r:(r['contradicted'], -int((r.get('local') or {}).get('inliers') or 0), -r['rrf'], r['slug']))
    return decision_from_ranked(rows, relationships, observations, ambiguous_target, incomplete)


def decision_from_ranked(rows, relationships, observations, ambiguous_target=False, incomplete=False):
    """Apply response/uncertainty semantics without replacing an accepted ordering."""
    years = year_proposals(observations)
    best = rows[0]['slug'] if rows else None
    reasons = ['confidence_not_calibrated']
    if rows and rows[0]['contradicted']:
        reasons.append('all_candidates_contradicted')
    if best:
        rel = relationships.get(best,{})
        same_family = [r for r in rows if r['slug'] != best and rel.get('family_id') and relationships.get(r['slug'],{}).get('family_id')==rel['family_id'] and not r['contradicted']]
        if same_family:
            reasons.append('unresolved_variant')
        if rel.get('ambiguity_group_id'):
            reasons.append('catalog_ambiguity')
    if incomplete:
        reasons.append('incomplete_evidence')
    return {'ranked_candidates':rows,'best_candidate':best,'slug':None,
            'decision':'ambiguous_target' if ambiguous_target else 'uncertain' if rows else 'unknown',
            'probability_correct':None,'reasons':reasons,'year_evidence':years}
