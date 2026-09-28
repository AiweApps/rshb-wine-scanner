"""Product-only adapter for the existing proposal resolution components.

Standalone1800..2099 years are removed only from comparison copies. Original
catalogue and OCR inputs remain intact; visible vintages stay separate evidence.
"""
from copy import deepcopy
from pathlib import Path
import re

from rshb_vine.io import read_jsonl
from rshb_vine.learned_selection_resolution import ExistingEvidenceResolution, SOURCE_PATHS as PARENT_SOURCES
from rshb_vine.positive_variant_text import PositiveVariantText
from rshb_vine.bottle_isolation_evidence import BottleIsolationEvidence

SOURCE_PATHS = (*PARENT_SOURCES, 'rshb_vine/product_evidence_resolution.py')
YEAR = re.compile(r'(?<!\w)(?:18|19|20)\d{2}(?!\w)')


def product_text(text):
    return ' '.join(YEAR.sub(' ', text).split())


class ProductEvidenceResolution(ExistingEvidenceResolution):
    def __init__(self, root):
        super().__init__(root)
        catalog = read_jsonl(Path(root) / 'data/catalog-additions-20260921/catalog.jsonl')
        comparison = deepcopy(catalog)
        for row in comparison:
            for key in ('Винодельня', 'Название вина', 'Сорт винограда'):
                row['fields'][key] = product_text(row['fields'].get(key, ''))
        allowed = {r['slug'] for r in comparison if not r.get('excluded_from_retrieval')}
        self.positive = PositiveVariantText(comparison, allowed)
        self.identity = BottleIsolationEvidence(comparison, retry=None)

    def resolve(self, features, proposal):
        comparison = deepcopy(features)
        changed = []
        for i, line in enumerate(comparison['query_evidence']['observations']):
            original = line.get('raw_text', '')
            projected = product_text(original)
            if projected != original:
                changed.append({'observation_index': i, 'original_text': original,
                                'product_comparison_text': projected})
            line['raw_text'] = projected
        result = super().resolve(comparison, proposal)
        trace = result['existing_evidence_resolution']
        trace['policy'] = 'learned-proposal-existing-product-evidence-v2'
        trace['product_comparison_projection'] = {
            'kind': 'standalone_year_removal_1800_2099', 'original_input_mutated': False,
            'source_catalog_mutated': False, 'changed_observation_copies': changed,
            'other_numeric_name_tokens_preserved': True,
            'vintage_result': 'not_resolved_by_product_ranking'}
        return result
