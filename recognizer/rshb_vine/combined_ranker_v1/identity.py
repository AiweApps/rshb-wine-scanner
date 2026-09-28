"""Challenger identity (snapshot-03-candidate) for v4 features, labels and masks; old103 stays on the frozen bundle.

Only the product grouping/claims change: cards, raw metadata and gallery references are identical to
current (checked by the identity handoff). Labels migrate through preserved card membership
(``rebind_ids``); a frozen product split by the candidate needs label review and is held.
"""
from pathlib import Path
import time

from rshb_vine.catalog.product_registry import ProductRegistry
from rshb_vine.io import read_json, sha256, verify
from rshb_vine.learned_selection_features import build_candidate_features
from rshb_vine.ranker_integration_v3.masks import pair_key
from rshb_vine.system_selection_v4 import scorer
from rshb_vine.system_selection_v4.context import SelectionContext
from rshb_vine.system_selection_v4.features import build_features
from rshb_vine.text_runtime_integration_v1.contract import TextCandidateContract

FROZEN_BUNDLE = 'config/product-identity-current.json'
CANDIDATE_HANDOFF = 'runs/identity-link-review-v1/candidate-bundle-handoff.json'
CANDIDATE_BUNDLE = 'data/product-identity-v1/snapshot-03-candidate/manifest.json'


def load_candidate(root):
    root = Path(root)
    handoff = verify(read_json(root / CANDIDATE_HANDOFF))
    bundle = handoff['bundle']
    if bundle['manifest'] != CANDIDATE_BUNDLE or sha256(root / CANDIDATE_BUNDLE) != bundle['manifest_sha256']:
        raise ValueError('Candidate identity bundle differs from its handoff')
    registry = ProductRegistry.from_bundle(root, CANDIDATE_BUNDLE)
    if registry.checksum != bundle['registry']['checksum'] or registry.bundle_checksum != bundle['manifest_checksum']:
        raise ValueError('Candidate identity registry checksum mismatch')
    if handoff['use'] != 'challenger only; current pointer not switched':
        raise ValueError('Unexpected candidate identity use')
    return handoff, registry


def candidate_labels(row, frozen, candidate):
    """(candidate product ids, status); split products are not migrated silently."""
    try:
        products = candidate.rebind_ids(row['ground_truth']['current_products'], frozen)
    except ValueError:
        return [], 'split_requires_label_review'
    slugs = {candidate.product_id(s) for s in row['ground_truth']['acceptable_slugs'] if s in candidate.cards}
    if not slugs <= set(products):
        return products, 'acceptable_slug_outside_migrated_product'
    changed = sorted(products) != sorted(row['ground_truth']['current_products'])
    return sorted(products), 'migrated_changed_id' if changed else 'unchanged'


def ignored_cards(row, pairs):
    """Card slugs the frozen row masks: row loss_ignore candidates plus global pair masks on its old pool."""
    ignore = set(row.get('loss_ignore_candidate_ids', []))
    cards = set()
    gt = set(row['ground_truth']['acceptable_slugs'])
    for c in row['candidate_pool']:
        slugs = set(c['card_slugs']) | {c['representative_slug']}
        if c['candidate_id'] in ignore or any(pair_key(g, s) in pairs for g in gt for s in slugs - gt):
            cards |= slugs - gt
    return cards


def loss_ignore(candidate, positives, row, pairs, frozen_cards):
    if candidate['product_id'] in positives:
        return False
    slugs = set(candidate['card_slugs'])
    gt = set(row['ground_truth']['acceptable_slugs'])
    return bool(slugs & frozen_cards) or any(pair_key(g, s) in pairs for g in gt for s in slugs - gt)


class CandidateSelector:
    """v4 features over the candidate identity with the same text contract; no old103 here."""

    def __init__(self, root, registry, model_path=None):
        self.root = Path(root)
        self.registry = registry
        self.context = SelectionContext(root)
        self.contract = TextCandidateContract(registry, root)
        self.model = None
        if model_path is not None:
            self.model = verify(read_json(self.root / model_path))
            scorer.check_model(self.model, self.root)

    def features(self, payload):
        started = time.perf_counter()
        extras = self.contract.extra_candidates(payload['observations'])
        base = build_candidate_features(control_slug=payload['control_slug'], raw_visual=payload['raw_visual'],
                                        observations=payload['observations'], cards=self.registry.cards,
                                        claims=self.registry.claims, control_candidates=payload['control_candidates'])
        output = build_features(base, payload['raw_visual'], self.context, self.registry.cards, self.registry.claims,
                                extras)
        return output, extras, time.perf_counter() - started

    def propose(self, output, control_slug):
        proposal = scorer.propose(output, self.model, control_slug=control_slug)
        proposal['identity_registry_checksum'] = self.registry.checksum
        return proposal
