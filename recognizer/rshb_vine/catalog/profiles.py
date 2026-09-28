"""Verified, offline profile reader. No ranking or runtime admission implied."""
from copy import deepcopy
from pathlib import Path

from rshb_vine.io import local_path, read_json, read_jsonl, sha256, verify


class CatalogProfiles:
    def __init__(self, root: Path):
        pointer = verify(read_json(root / 'data/catalog-profiles-v1/current.json'))
        snapshot = local_path(root, pointer['snapshot'])
        manifest = verify(read_json(snapshot / 'manifest.json'))
        proof = verify(read_json(snapshot / 'verification.json'))
        if (manifest['checksum'] != pointer['manifest_checksum']
                or proof['checksum'] != pointer['verification_checksum']
                or proof['manifest_checksum'] != manifest['checksum']
                or proof['profiles_sha'] != manifest['profiles_sha']
                or sha256(snapshot / 'profiles.jsonl') != manifest['profiles_sha']):
            raise ValueError('Profile snapshot is not bound to its verification')
        for path, expected in {**manifest['sources'], **manifest['ocr_sources']}.items():
            if sha256(local_path(root, path)) != expected:
                raise ValueError(f'Profile source changed: {path}')
        rows = read_jsonl(snapshot / 'profiles.jsonl')
        self._rows = {row['slug']: row for row in rows}
        if len(self._rows) != len(rows) or len(rows) != manifest['n']:
            raise ValueError('Profile roster mismatch')
        self.snapshot_checksum = manifest['checksum']

    def evidence(self, slug: str) -> dict:
        """Keep product, vintage and appearance separate; never infer hard vetoes."""
        row = self._rows[slug]
        groups = {'product': {}, 'vintage': {}, 'appearance': {}}
        for name, prop in row['properties'].items():
            group = ('vintage' if name in {'reference_year', 'catalog_vintage'}
                     else 'appearance' if name == 'label_color' else 'product')
            usable = (prop['status'] in {'catalog_asserted', 'source_verified'}
                      and not prop['review_required'])
            groups[group][name] = {
                'status': prop['status'],
                'comparison_value': deepcopy(prop['value']) if usable else None,
                'review_required': prop['review_required'],
                'claims': deepcopy(prop['claims']),
                'hard_veto_allowed': False,
            }
        return {
            'slug': slug, 'snapshot_checksum': self.snapshot_checksum,
            'active_in_gallery': row['active_in_gallery'],
            'retrieval_eligible': row['active_in_gallery'] and not row['excluded_from_retrieval'],
            **groups, 'references': deepcopy(row['references']),
            'source_review': deepcopy(row.get('source_review')),
            'external_product_candidates': deepcopy(row['external_product_candidates']),
            'exact_card_verified': False, 'runtime_admitted': False,
        }
