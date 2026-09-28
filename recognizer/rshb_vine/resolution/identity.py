"""Experimental, uncalibrated identity reranking; preserves every candidate."""
from collections import Counter
import re
import unicodedata


def norm(value):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', value).casefold().replace('ё', 'е')))


GRAPES = {
    'riesling': ['рислинг', 'riesling'], 'chardonnay': ['шардоне', 'chardonnay'],
    'pinot_noir': ['пино нуар', 'pinot noir'], 'pinot_blanc': ['пино блан', 'pinot blanc'],
    'sauvignon_blanc': ['совиньон блан', 'sauvignon blanc'],
    'cabernet_sauvignon': ['каберне совиньон', 'cabernet sauvignon'],
    'cabernet_franc': ['каберне фран', 'cabernet franc'], 'merlot': ['мерло', 'merlot'],
    'marselan': ['марселан', 'marselan'], 'saperavi': ['саперави', 'saperavi'],
    'kokur': ['кокур', 'kokur'], 'rkatsiteli': ['ркацители', 'rkatsiteli'],
    'syrah': ['сира', 'шираз', 'syrah', 'shiraz'], 'aligote': ['алиготе', 'aligote'],
}
SUGAR = {'extra_brut': ['экстра брют', 'extra brut'], 'brut': ['брют', 'brut'],
         'dry': ['сухое', 'сухой', 'dry'], 'semidry': ['полусухое', 'полусухой', 'semi dry'],
         'semisweet': ['полусладкое', 'полусладкий', 'semi sweet'], 'sweet': ['сладкое', 'сладкий', 'sweet']}
COLOR = {'white': ['белое', 'белый', 'white wine', 'blanc de blancs', 'blanc de noirs'],
         'red': ['красное', 'красный', 'red wine'], 'rose': ['розовое', 'розовый', 'rose', 'rosé']}
STYLE = {'blanc_de_blancs': ['blanc de blancs', 'блан де блан'],
         'blanc_de_noirs': ['blanc de noirs', 'блан де нуар']}


def values(text, aliases):
    # Consume longer phrases first: extra brut must not also become brut.
    remaining = ' '+norm(text)+' '
    found = set()
    phrases = sorted(((norm(a), key) for key, group in aliases.items() for a in group), key=lambda x: -len(x[0]))
    for phrase, key in phrases:
        pattern = r'(?<!\w)'+re.escape(phrase)+r'(?!\w)'
        if re.search(pattern, remaining):
            found.add(key)
            remaining = re.sub(pattern, ' ', remaining)
    return found


def compare(observed, expected, *, subset=False):
    if not observed or not expected:
        return 'unknown'
    if observed <= expected if subset else observed == expected:
        return 'matched'
    # Multiple inconsistent readings are not a trustworthy veto.
    if len(observed) > 1:
        return 'unknown'
    return 'contradicted' if observed.isdisjoint(expected) else 'unknown'


class IdentityReranker:
    version = 'identity-evidence-experiment-v1'

    def __init__(self, catalog, minimum_ocr_score=.85, policy='evidence_count'):
        if policy not in ('evidence_count', 'contradictions_only'):
            raise ValueError('Unknown identity policy')
        self.policy = policy
        self.version = 'identity-evidence-experiment-v1' if policy == 'evidence_count' else 'identity-contradictions-experiment-v2'
        self.catalog = {r['slug']: r for r in catalog if not r.get('excluded_from_retrieval')}
        self.minimum_ocr_score = minimum_ocr_score
        counts = Counter(t for r in self.catalog.values() for t in set(norm(r['fields'].get('Название вина', '')).split()))
        self.rare = {t for t, n in counts.items() if len(t) >= 4 and n <= 10}
        self.producers = {norm(r['fields'].get('Винодельня', '')) for r in self.catalog.values()} - {''}

    def rerank(self, candidates, observations):
        usable = [o for o in observations if o['score'] >= self.minimum_ocr_score]
        text = ' '.join(o['raw_text'] for o in usable)
        observed = {k: set().union(*(values(o['raw_text'], aliases) for o in usable))
                    for k, aliases in [('grape', GRAPES), ('sugar', SUGAR), ('color', COLOR), ('style', STYLE)]}
        # Producer must occur as a complete known phrase, not a generic word fragment.
        read_producers = {p for p in self.producers if ' '+p+' ' in ' '+norm(text)+' '}
        seen_name_tokens = set(norm(text).split()) & self.rare
        rows = []
        for baseline_rank, candidate in enumerate(candidates, 1):
            if candidate['slug'] not in self.catalog:
                continue
            item = self.catalog[candidate['slug']]
            f = item['fields']
            name = f.get('Название вина', '')
            expected = {'grape': values(f.get('Сорт винограда', ''), GRAPES),
                        'sugar': values(name, SUGAR),
                        'color': values(f.get('Категория', ''), COLOR), 'style': values(name, STYLE)}
            evidence = {k: {'observed': sorted(observed[k]), 'catalog': sorted(v),
                            'state': compare(observed[k], v, subset=k == 'grape'),
                            'trust': 'provisional_ocr_and_catalog_not_calibrated'} for k, v in expected.items()}
            producer = norm(f.get('Винодельня', ''))
            evidence['producer'] = {'observed': sorted(read_producers), 'catalog': [producer] if producer else [],
                                    'state': compare(read_producers, {producer} if producer else set()),
                                    'trust': 'literal_catalog_phrase_only'}
            matched_name = sorted(seen_name_tokens & set(norm(name).split()))
            conflicts = sum(e['state'] == 'contradicted' for e in evidence.values())
            matches = sum(e['state'] == 'matched' for e in evidence.values())
            row = dict(candidate, identity_evidence=evidence, matched_name_tokens=matched_name,
                       identity_conflicts=conflicts, identity_matches=matches,
                       baseline_rank=baseline_rank, identity_version=self.version)
            rows.append(row)
        # Explicit legacy color/vintage conflicts still dominate; no new hard deletion.
        if self.policy == 'contradictions_only':
            rows.sort(key=lambda r: (bool(r.get('contradicted')), bool(r['identity_conflicts']), r['baseline_rank']))
        else:
            rows.sort(key=lambda r: (bool(r.get('contradicted')), r['identity_conflicts'],
                                     -r['identity_matches'], -min(2, len(r['matched_name_tokens'])), r['baseline_rank']))
        return rows
