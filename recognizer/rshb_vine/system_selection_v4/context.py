"""Catalogue-side evidence shared by every request; built once, read-only afterwards."""
from pathlib import Path

from rshb_vine.io import read_json, read_jsonl, sha256, verify
from rshb_vine.frozen_candidate_selector import GENERIC
from rshb_vine.system_selection_v4 import text

CATALOG = 'data/catalog-additions-20260921/catalog.jsonl'
PRODUCER_ALIASES = 'config/producer-brand-aliases.json'
SUGAR_LEDGER = 'runs/sugar-evidence-v1/reference-ledger.json'
SOURCE_PATHS = (
    CATALOG, PRODUCER_ALIASES, SUGAR_LEDGER,
    'rshb_vine/system_selection_v4/context.py', 'rshb_vine/system_selection_v4/text.py',
    'rshb_vine/system_selection_v4/features.py', 'rshb_vine/typed_catalog_lexicon.py',
    'rshb_vine/catalog_text_retrieval.py', 'rshb_vine/resolution/identity.py',
    'rshb_vine/gallery_variant_text.py', 'rshb_vine/positive_variant_text.py',
    'rshb_vine/frozen_candidate_selector.py', 'rshb_vine/learned_selection_features.py',
    'rshb_vine/product_first/visual_selector.py', 'rshb_vine/product_first/partial_evidence.py',
)
# Card fields that may fill a typed claim the normalized profile left unknown.
RAW_FIELDS = {'grape_blend': ('Сорт винограда',), 'color': ('Категория',),
              'sugar': ('Категория', 'Название вина'), 'style': ('Название вина',)}


class SelectionContext:
    def __init__(self, root):
        root = Path(root)
        self.root = root
        catalog = read_jsonl(root / CATALOG)
        self.lexicon = text.TypedLexicon(catalog, root)
        self.generic = frozenset(t.forms[0] for word in GENERIC for t in text.tokenize(word))
        self.producer_aliases = {}
        for row in read_json(root / PRODUCER_ALIASES)['aliases']:
            if row.get('status') == 'verified':
                forms = self.producer_aliases.setdefault(text.base_norm(row['producer']), set())
                forms.update(row['aliases'])
        self.sugar_facts = {slug: fact for slug, fact in
                            verify(read_json(root / SUGAR_LEDGER))['records'].items() if fact.get('verified')}
        self.source_checksums = {path: sha256(root / path) for path in SOURCE_PATHS}

    def describe(self):
        return {'lexicon': self.lexicon.describe(), 'verified_producer_alias_groups': len(self.producer_aliases),
                'verified_sugar_facts': len(self.sugar_facts), 'sources_sha256': self.source_checksums}
