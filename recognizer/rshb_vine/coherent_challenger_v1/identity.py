"""Explicit candidate-selection adapter: old103 chooses on its frozen pool, snapshot-03 names the product.

old103 was fitted and admitted on the frozen grouping, where unverified cards stay card-local candidates.
snapshot-03 merges 64 of those cards into source-admitted products and splits Aristov rosé, so running
old103 natively on snapshot-03 would aggregate its features over a different pool. The adapter therefore
keeps the frozen pool and selected card unchanged and only re-describes the chosen card, its members and
the ranked card groups under snapshot-03. Both identities stay in the output; nothing is presented as
equal when the pool grouping differs.
"""
from copy import deepcopy
from pathlib import Path

from rshb_vine.catalog.product_registry import ProductRegistry
from rshb_vine.combined_ranker_v1.identity import FROZEN_BUNDLE, load_candidate
from rshb_vine.io import digest
from rshb_vine.learned_selection_features import build_candidate_features
from rshb_vine.product_first.year_evidence import describe_year

POLICY = 'frozen-old103-pool-then-snapshot03-identity-v1'
MARKER = 'coherent_challenger_v1'


class IdentityAdapter:
    def __init__(self, root, frozen_registry=None):
        root = Path(root).resolve()
        self.handoff, self.candidate = load_candidate(root)
        self.frozen = frozen_registry or ProductRegistry.from_bundle(root, FROZEN_BUNDLE)
        self.changed_cards = sorted(s for s in self.frozen.cards if
                                    (self.frozen.cards[s]['product_id'], self.frozen.cards[s]['binding_status']) !=
                                    (self.candidate.cards[s]['product_id'], self.candidate.cards[s]['binding_status']))
        if set(self.frozen.cards) != set(self.candidate.cards):
            raise ValueError('Candidate identity must keep the frozen card roster')
        self.descriptor = {'policy': POLICY, 'selection_pool_registry': self.frozen.checksum,
                           'selection_pool_bundle': self.frozen.bundle_checksum,
                           'identity_registry': self.candidate.checksum,
                           'identity_bundle': self.candidate.bundle_checksum,
                           'identity_claims': self.candidate.claims_checksum,
                           'identity_handoff': self.handoff['checksum'],
                           'cards_changed_product': len(self.changed_cards)}

    def _describe(self, slug, observations, sid):
        registry = self.candidate
        result = describe_year({'representative_slug': slug, 'product_id': registry.product_id(slug)},
                               observations, registry.claims, {'kind': 'same_instance_ocr', 'instance_id': sid})
        result.update(identity_registry_checksum=registry.checksum, claims_checksum=registry.claims_checksum,
                      identity_binding_status=registry.cards[slug]['binding_status'] if slug else 'no_product',
                      catalog_card_slugs=registry.members[registry.product_id(slug)] if slug else [])
        return result

    def project(self, result, copy=True):
        """Product identity becomes snapshot-03; selected cards, ranks and geometry unchanged."""
        if result.get('decision') == 'invalid_image':
            return result
        out = deepcopy(result) if copy else result
        frozen_resolution = out.get('product_resolution')
        if frozen_resolution is not None and frozen_resolution.get('identity_registry_checksum') != self.frozen.checksum:
            raise ValueError('Result was not produced on the frozen old103 identity')
        evidence = {str(e['instance_id']): e for e in (out.get('product_identity_evidence') or {}).get('targets', [])}
        rows = []
        for target in out.get('targets', []):
            sid = str(target['instance_id'])
            slug = target['retrieval']['best_candidate']
            if sid not in evidence:
                raise ValueError('Target has no same-request selection evidence: ' + sid)
            frozen = target.get('product_resolution') or {}
            target['frozen_product_resolution'] = frozen
            target['frozen_product_candidates'] = target.get('product_candidates')
            target['product_resolution'] = self._describe(slug, evidence[sid]['ocr_observations'], sid)
            target['product_candidates'] = self.candidate.group_ranked_cards(target['retrieval']['ranked_candidates'])
            before, after = self.frozen.product_id(slug), self.candidate.product_id(slug)
            row = {'instance_id': sid, 'selected_card': slug, 'frozen_product_id': before, 'product_id': after,
                   'identity_changed': before != after,
                   'frozen_members': self.frozen.members[before] if slug else [],
                   'members': self.candidate.members[after] if slug else [],
                   'ranked_groups_frozen': len(target['frozen_product_candidates'] or []),
                   'ranked_groups': len(target['product_candidates'])}
            target['candidate_identity'] = row
            rows.append(row)
        out['frozen_product_resolution'] = frozen_resolution
        out['product_resolution'] = dict(frozen_resolution or {}, identity_registry_checksum=self.candidate.checksum,
            bundle_checksum=self.candidate.bundle_checksum, claims_checksum=self.candidate.claims_checksum,
            targets=[{'instance_id': str(t['instance_id']), 'resolution': t['product_resolution']}
                     for t in out.get('targets', [])])
        out[MARKER] = dict(self.descriptor, targets=rows, public_card_unchanged=True, calibrated=False, probability=None)
        return out

    def pool_review(self, payload):
        """Compare old103's frozen pool with the pool the same evidence would form under snapshot-03."""
        def pool(registry):
            features = build_candidate_features(control_slug=payload['control_slug'], raw_visual=payload['raw_visual'],
                observations=payload['ocr_observations'], cards=registry.cards, claims=registry.claims,
                control_candidates=payload['control_candidates'])
            return {c['candidate_id']: sorted(c['card_slugs']) for c in features['candidates']}
        frozen, candidate = pool(self.frozen), pool(self.candidate)
        frozen_groups = {tuple(v) for v in frozen.values()}
        candidate_groups = {tuple(v) for v in candidate.values()}
        merged = sorted(g for g in candidate_groups - frozen_groups if len(g) > 1)
        return {'frozen_candidates': len(frozen), 'candidate_candidates': len(candidate),
                'pool_changed': frozen_groups != candidate_groups, 'merged_groups': [list(g) for g in merged],
                'cards_in_changed_groups': sorted({s for g in frozen_groups ^ candidate_groups for s in g}),
                'frozen_digest': digest(sorted(frozen_groups)), 'candidate_digest': digest(sorted(candidate_groups))}
