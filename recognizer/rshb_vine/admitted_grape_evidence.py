"""Source-verified grape supplements for positive text matching only.

The adapter constructs a transient catalogue copy. It never makes a grape
list exhaustive and must not feed negative-evidence guards or SKU identity.
"""
from copy import deepcopy
from pathlib import Path

from rshb_vine.catalog.profiles import CatalogProfiles
from rshb_vine.io import read_json, verify, sha256, digest, seal
from rshb_vine.positive_variant_text import ALIASES, canonical, contains


class AdmittedGrapeEvidence:
    def __init__(self, root, protocol_path):
        self.root = Path(root)
        self.protocol = verify(read_json(protocol_path))
        for path, checksum in self.protocol['sources'].items():
            if sha256(self.root / path) != checksum:
                raise ValueError('Admitted grape input changed: ' + path)
        pointer = verify(read_json(self.root / 'data/catalog-profiles-v1/current.json'))
        if pointer['snapshot'] != self.protocol['snapshot']:
            raise ValueError('Unexpected profile snapshot')
        self.profiles = CatalogProfiles(self.root)
        if self.profiles.snapshot_checksum != self.protocol['snapshot_checksum']:
            raise ValueError('Profile manifest changed')

    def adapt(self, catalog):
        result = deepcopy(catalog)
        records = []
        for row in result:
            slug = row['slug']
            try:
                evidence = self.profiles.evidence(slug)['product']['grapes']
            except KeyError:
                continue
            if evidence['status'] != 'source_verified' or evidence['review_required']:
                continue
            values = evidence['comparison_value']
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ValueError('Invalid admitted grape comparison value: ' + slug)
            surfaces_by_value = {}
            for value in values:
                if value in ALIASES:
                    surfaces_by_value[value] = ALIASES[value][0]
                elif value.startswith('unmapped:') and value.removeprefix('unmapped:').strip():
                    # The profile builder preserves the source's literal unrecognized
                    # component after this prefix; match it as a phrase, not an alias.
                    surfaces_by_value[value] = value.removeprefix('unmapped:').strip()
                else:
                    raise ValueError('Unsupported admitted grape value: ' + value)
            original = row['fields'].get('Сорт винограда', '')
            if not isinstance(original, str):
                raise ValueError('Non-text raw grape field: ' + slug)
            existing = canonical(original)
            added = [value for value in values
                     if not contains(existing, canonical(surfaces_by_value[value]))]
            surfaces = [surfaces_by_value[value] for value in added]
            if surfaces:
                row['fields']['Сорт винограда'] = ', '.join(
                    value for value in [original, *surfaces] if value)
            records.append({
                'slug': slug, 'raw_grapes': original, 'admitted_values': values,
                'added_values': added, 'added_surfaces': surfaces,
                'effective_grapes': row['fields'].get('Сорт винограда', ''),
                'changed': bool(surfaces), 'supports_negative_inference_from_absence': False,
                'claims': evidence['claims'],
            })
        inventory = seal({
            'kind': 'admitted-grape-positive-evidence-inventory-v1',
            'protocol': self.protocol['checksum'], 'catalog_records': len(catalog),
            'catalog_digest': digest(catalog), 'profile_snapshot': self.profiles.snapshot_checksum,
            'admitted_records': len(records), 'changed_records': sum(r['changed'] for r in records),
            'records': records, 'raw_mutated': False, 'negative_evidence_granted': False,
        })
        return result, inventory

    def install(self, unified, catalog):
        """Rebuild positive consumers on a separately constructed candidate.

        The caller owns runtime admission and must include this returned receipt
        in its new manifest. Never call this on the running frozen service.
        """
        from rshb_vine.product_name_selector import ProductNameSelector
        from rshb_vine.catalog_constrained_title import CatalogConstrainedTitle
        from rshb_vine.within_producer_title import WithinProducerTitle

        adapted, inventory = self.adapt(catalog)
        expanded = unified.expanded
        core = expanded.core
        title = core.parent.parent
        instance = title.parent
        guard_before = digest(core.guard.fields)
        sugar_before = digest(unified.sugar.catalog)
        policies = [
            (instance, 'variant_policy', ProductNameSelector),
            (title, 'policy', CatalogConstrainedTitle),
            (core, 'single', CatalogConstrainedTitle),
            (core, 'multi', ProductNameSelector),
            (unified.series, 'names', ProductNameSelector),
        ]
        replacements = []
        for owner, attr, cls in policies:
            old = getattr(owner, attr)
            allowed = set(old.items)
            selected = [row for row in adapted if row['slug'] in allowed]
            policy = cls(selected, old.gallery)
            if set(policy.items) != allowed:
                raise ValueError('Positive policy roster changed: ' + attr)
            replacements.append((owner, attr, policy))
        # Construct all replacements successfully before changing candidate state.
        for owner, attr, policy in replacements:
            setattr(owner, attr, policy)
        core.title = WithinProducerTitle(core.single)
        if digest(core.guard.fields) != guard_before or digest(unified.sugar.catalog) != sugar_before:
            raise ValueError('Positive supplement altered a non-positive consumer')
        return seal({
            'kind': 'admitted-grape-positive-evidence-installed-v1',
            'protocol': self.protocol['checksum'], 'inventory': inventory['checksum'],
            'changed_records': inventory['changed_records'],
            'positive_consumers': ['instance.variant_policy', 'title.policy',
                                   'core.single', 'core.multi', 'series.names'],
            'within_producer_title_rebuilt': True,
            'guard_fields_unchanged': guard_before, 'sugar_catalog_unchanged': sugar_before,
            'raw_catalog_changed': False, 'gallery_changed': False,
            'admission': 'pending_replay_and_http',
        })
