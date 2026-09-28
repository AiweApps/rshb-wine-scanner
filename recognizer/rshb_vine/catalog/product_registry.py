"""One version-bound product/card identity reader for inference, fit and scoring."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from rshb_vine.io import read_json, verify, sha256


class ProductRegistry:
    def __init__(self, document, claims=None):
        self.document = verify(document)
        self.checksum = document['checksum']
        self.cards = {c['slug']: c for c in document['cards']}
        self.products = {p['product_id']: p for p in document['products']}
        if len(self.cards) != len(document['cards']) or len(self.products) != len(document['products']):
            raise ValueError('Duplicate product/card IDs')
        members = defaultdict(list)
        for slug, card in self.cards.items():
            pid = card['product_id']
            if pid not in self.products:
                raise ValueError('Card has no product: ' + slug)
            product = self.products[pid]
            if card['binding_status'] == 'source_admitted_product':
                if product['identity_status'] != 'source_admitted' or not product.get('evidence'):
                    raise ValueError('Product membership has no admitted source: ' + slug)
            elif (card['binding_status'] != 'bootstrap_not_verified_product' or
                  pid != 'provisional-card:' + slug or product['identity_status'] != 'provisional_card_local'):
                raise ValueError('Unresolved identity must stay card-local: ' + slug)
            members[pid].append(slug)
        self.members = {pid: sorted(slugs) for pid, slugs in members.items()}
        if set(self.members) != set(self.products):
            raise ValueError('Orphan product in registry')
        for pid, product in self.products.items():
            if product['identity_status'] == 'source_admitted' and set(product['card_slugs']) != set(self.members[pid]):
                raise ValueError('Product membership differs between cards and products')
        for ref in document['reference_images']:
            if set(ref['product_ids']) != {self.product_id(s) for s in ref['card_slugs']}:
                raise ValueError('Reference/product binding mismatch')
        self.claims = {}
        self.claims_checksum = None
        if claims is not None:
            verify(claims)
            if claims['identity_registry_checksum'] != self.checksum:
                raise ValueError('Claims belong to a different identity registry')
            self.claims = {c['slug']: c for c in claims['records']}
            self.claims_checksum = claims['checksum']
            if len(self.claims) != len(claims['records']):
                raise ValueError('Duplicate claim profiles')
            for slug, profile in self.claims.items():
                if (profile['product_id'] != self.product_id(slug) or
                        profile['binding_status'] != self.cards[slug]['binding_status']):
                    raise ValueError('Claim/card identity mismatch: ' + slug)
        self.bundle_checksum = None

    @classmethod
    def from_bundle(cls, root, path):
        root = Path(root).resolve()
        path = Path(path)
        manifest = verify(read_json(path if path.is_absolute() else root / path))
        documents = {}
        for kind in ('registry', 'claims'):
            ref = manifest[kind]
            source = (root / ref['path']).resolve()
            if not source.is_relative_to(root) or sha256(source) != ref['sha256']:
                raise ValueError('Product bundle source changed: ' + kind)
            documents[kind] = verify(read_json(source))
            if documents[kind]['checksum'] != ref['checksum']:
                raise ValueError('Product bundle document changed: ' + kind)
        result = cls(documents['registry'], documents['claims'])
        result.bundle_checksum = manifest['checksum']
        return result

    def product_id(self, slug):
        return self.cards[slug]['product_id'] if slug is not None else None

    def candidate_id(self, slug):
        card = self.cards[slug]
        return card['product_id'] if card['binding_status'] == 'source_admitted_product' else 'unresolved-card:' + slug

    def same_product(self, left, right):
        return left is not None and right is not None and self.product_id(left) == self.product_id(right)

    def rebind_ids(self, product_ids, previous):
        """Migrate labels through preserved card membership, never model answers."""
        output = set()
        for old_id in product_ids:
            mapped = {self.product_id(s) for s in previous.members[old_id]}
            if len(mapped) != 1:
                raise ValueError('Splitting a previously admitted product needs separate label review')
            output.update(mapped)
        return sorted(output)

    def group_ranked_cards(self, rows):
        """Collapse admitted duplicates without inventing scores or reordering products."""
        grouped = {}
        for rank, row in enumerate(rows, 1):
            slug = row['slug']
            pid = self.product_id(slug)
            group = grouped.setdefault(pid, {
                'product_id': pid, 'identity_status': self.cards[slug]['binding_status'],
                'representative_slug': slug, 'first_card_rank': rank,
                'card_slugs': [], 'card_evidence': []})
            if slug not in group['card_slugs']:
                group['card_slugs'].append(slug)
            group['card_evidence'].append({'original_rank': rank, 'row': deepcopy(row)})
        return [{'rank': i, **row} for i, row in enumerate(grouped.values(), 1)]


def bind_claim_profiles(claims, previous, current):
    """Project only identity ownership; preserve every attribute and its source."""
    verify(claims)
    result = deepcopy(claims)
    result.pop('checksum')
    changed = []
    for row in result['records']:
        slug = row['slug']
        if (row['product_id'] != previous.product_id(slug) or
                row['binding_status'] != previous.cards[slug]['binding_status']):
            raise ValueError('Source claims/registry mismatch: ' + slug)
        if row['product_id'] != current.product_id(slug):
            changed.append({'slug': slug, 'before': row['product_id'], 'after': current.product_id(slug)})
        row['product_id'] = current.product_id(slug)
        row['binding_status'] = current.cards[slug]['binding_status']
    result.update(identity_registry_checksum=current.checksum,
                  source_claims_checksum=claims['checksum'], source_registry_checksum=previous.checksum,
                  identity_changes=changed, attribute_claims_changed=False)
    return result
