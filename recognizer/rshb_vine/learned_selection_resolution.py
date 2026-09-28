"""Resolve a learned proposal through the existing8175 product evidence rules.

No fitting, additional OCR, novel wine-specific aliases, year product veto,
new candidates or target changes. The ranker proposes; admitted evidence remains
owned by the existing guard/positive-variant/identity-preservation components.
"""
from copy import deepcopy
from pathlib import Path
from rshb_vine.io import read_json, read_jsonl, verify
from rshb_vine.catalog_additions import SourceBoundGuard
from rshb_vine.positive_variant_text import PositiveVariantText
from rshb_vine.bottle_isolation_evidence import BottleIsolationEvidence
from rshb_vine.resolution.identity import SUGAR, values


SOURCE_PATHS = (
    'rshb_vine/learned_selection_resolution.py', 'rshb_vine/catalog_additions.py',
    'rshb_vine/label_specialist_consensus.py', 'rshb_vine/positive_variant_text.py',
    'rshb_vine/bottle_isolation_evidence.py', 'rshb_vine/resolution/identity.py',
    'rshb_vine/gallery_variant_text.py',
    'data/catalog-additions-20260921/catalog.jsonl',
    'data/catalog-additions-20260921/manifest.json',
    'runs/gallery-variant-source-review-v1/gallery/signatures.json',
    'runs/sugar-evidence-v1/reference-ledger.json',
)


class ExistingEvidenceResolution:
    def __init__(self, root):
        root = Path(root)
        catalog = read_jsonl(root / 'data/catalog-additions-20260921/catalog.jsonl')
        addition = verify(read_json(root / 'data/catalog-additions-20260921/manifest.json'))
        added = addition['added_canonical_slugs']
        signatures = verify(read_json(root / 'runs/gallery-variant-source-review-v1/gallery/signatures.json'))['signatures']
        signatures = {s:v for s,v in signatures.items() if s not in added}
        # Empty year groups: this decision is product-only. Existing year reasons
        # are retained in trace but do not keep a different product in place.
        self.guard = SourceBoundGuard(catalog, signatures, {}, added)
        allowed = {r['slug'] for r in catalog if not r.get('excluded_from_retrieval')}
        self.positive = PositiveVariantText(catalog, allowed)
        self.identity = BottleIsolationEvidence(catalog, retry=None)
        self.sugar = verify(read_json(root / 'runs/sugar-evidence-v1/reference-ledger.json'))['records']

    def resolve(self, features, proposal):
        output = deepcopy(proposal)
        control = next((c for c in features['candidates'] if c['features']['control.is_proposal']), None)
        before = control['provenance']['control_proposal_slug'] if control else None
        after = proposal.get('representative_slug')
        trace = {'policy': 'learned-proposal-existing-product-evidence-v1',
                 'control': before, 'proposal': after, 'selected': after, 'blocked': False,
                 'reasons': [], 'new_wine_specific_rules': False, 'year_product_veto': False}
        output['existing_evidence_resolution'] = trace
        if not before or not after or before == after:
            trace['reason'] = 'no_changed_product_proposal'
            return output
        observations = features['query_evidence']['observations']
        if before not in self.guard.fields or after not in self.guard.fields:
            trace['reasons'].append('missing_existing_catalog_binding')
        else:
            _, guard_reasons = self.guard.check(before, after, observations)
            trace['legacy_guard_reasons'] = guard_reasons
            trace['reasons'].extend(r for r in guard_reasons
                                    if r.startswith(('color:', 'grape:', 'sugar:', 'style:')))
            if before in self.positive.items and after in self.positive.items:
                pair, positive = self.positive.rerank([{'slug': after}, {'slug': before}], observations)
                trace['positive_variant_pair'] = positive
                if pair[0]['slug'] == before:
                    trace['reasons'].append('existing_positive_variant_prefers_control')
            lost = self.identity.lost_evidence(observations, before, after)
            trace['lost_identity_phrases'] = lost
            if lost is None or lost:
                trace['reasons'].append('existing_identity_phrase_preservation')
            seen = set().union(*(values(o['raw_text'], SUGAR) for o in observations if o['score'] >= .85))
            fact = self.sugar.get(after, {})
            trace['observed_sweetness'] = sorted(seen)
            trace['source_sugar_fact'] = fact
            if len(seen) == 1 and fact.get('verified') and fact.get('value') not in seen:
                trace['reasons'].append('existing_verified_sugar_contradiction')
        if not trace['reasons']:
            trace['reason'] = 'proposal_passes_existing_product_evidence'
            return output
        # Keep the original proposal and scores for explanation; choosing the
        # control does not erase any candidate or manufacture a probability.
        selected = next(r for r in proposal['ranked_candidates'] if before in r['card_slugs'])
        trace.update(blocked=True, selected=before, reason='retained_existing_product_evidence')
        output.update(candidate_id=selected['candidate_id'], product_id=selected['product_id'],
                      representative_slug=before, best_candidate=before,
                      representative_source='actual8175_proposal_preserved_by_existing_evidence',
                      score=selected['score'], score_margin=None,
                      reason='existing_product_evidence_preserved', exact_slug=None,
                      probability=None, release_admitted=False)
        for row in output['ranked_candidates']:
            row['learned_rank'] = row['rank']
        output['ranked_candidates'].sort(key=lambda r: before not in r['card_slugs'])
        for rank, row in enumerate(output['ranked_candidates'], 1): row['rank'] = rank
        return output
