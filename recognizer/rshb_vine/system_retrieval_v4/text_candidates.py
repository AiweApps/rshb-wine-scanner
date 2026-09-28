"""OCR-driven catalogue candidate channel T_catalog: positive multi-token matches only, uncalibrated."""
import math
from collections import Counter

from rshb_vine.frozen_candidate_selector import GENERIC
from rshb_vine.gallery_variant_text import STYLE_ALIASES
from rshb_vine.io import read_json
from rshb_vine.positive_variant_text import ALIASES
from rshb_vine.resolution.identity import COLOR, SUGAR
from rshb_vine.system_retrieval_v4.normalize import catalogue_skeletons, fuzzy_equal, ocr_skeletons

CHANNEL = 'T_catalog'
LINE_MIN_SCORE = 0.5
NAME_DF_MAX = 20
PRODUCER_DF_MAX = 60
LIMIT = 5
FIELDS = ('Винодельня', 'Название вина', 'Сорт винограда')


class _FuzzyVocabulary:
    def __init__(self, terms):
        self.terms = set(terms)
        self.buckets = {}
        for t in self.terms:
            self.buckets.setdefault((len(t), t[0]), set()).add(t)
            self.buckets.setdefault((len(t), '$' + t[-1]), set()).add(t)
        self.cache = {}

    def match(self, variants):
        key = frozenset(variants)
        if key not in self.cache:
            found = {v for v in variants if v in self.terms}
            for v in variants:
                for length in range(len(v) - 2, len(v) + 3):
                    for term in self.buckets.get((length, v[0]), set()) | self.buckets.get((length, '$' + v[-1]), set()):
                        if term not in found and fuzzy_equal(term, {v}):
                            found.add(term)
            self.cache[key] = found
        return self.cache[key]


class CatalogueTextCandidates:
    def __init__(self, registry, root):
        aliases = {}
        for a in read_json(root / 'config/producer-brand-aliases.json')['aliases']:
            if a['status'] == 'verified':
                aliases.setdefault(a['producer'], set()).update([a['alias'], *a.get('aliases', [])])
        generic = {s for phrase in [*GENERIC, *(a for m in (SUGAR, COLOR, STYLE_ALIASES) for v in m.values() for a in v)]
                   for s in catalogue_skeletons(phrase)}
        self.registry = registry
        self.cards = {}
        grape_forms = set(s for v in ALIASES.values() for a in v for s in catalogue_skeletons(a))
        for slug, card in registry.cards.items():
            if not card.get('active_in_gallery'):
                continue
            meta = card.get('raw_metadata') or {}
            producer = meta.get('Винодельня', '')
            brand = (registry.claims.get(slug, {}).get('roles', {}).get('visible_brand') or {}).get('values') or []
            producer_tokens = set(catalogue_skeletons(producer))
            for form in [*aliases.get(producer, ()), *brand]:
                producer_tokens.update(catalogue_skeletons(form))
            grape_tokens = [t for t in dict.fromkeys(catalogue_skeletons(meta.get('Сорт винограда', ''))) if len(t) >= 3]
            grape_forms.update(grape_tokens)
            name_tokens = [t for t in dict.fromkeys(catalogue_skeletons(meta.get('Название вина', '')))
                           if t not in producer_tokens and len(t) >= 3 and not t.isdigit()]
            self.cards[slug] = {'producer': producer, 'producer_tokens': sorted(t for t in producer_tokens if len(t) >= 3),
                                'name_tokens': name_tokens, 'grape_tokens': grape_tokens,
                                'all_tokens': set(s for f in FIELDS for s in catalogue_skeletons(meta.get(f, '')))}
        vocabulary = _FuzzyVocabulary(t for c in self.cards.values() for t in c['all_tokens'])
        owners = {}
        for slug, c in self.cards.items():
            for t in c['all_tokens']:
                owners.setdefault(t, set()).add(slug)
        role = _FuzzyVocabulary(generic | grape_forms)
        generic_vocabulary = _FuzzyVocabulary(generic)
        terms = {t for c in self.cards.values() for t in [*c['name_tokens'], *c['producer_tokens']]}
        self.df = {t: len(set().union(*(owners.get(u, set()) for u in vocabulary.match({t})))) or 1 for t in terms}
        self.generic = {t for t in terms if generic_vocabulary.match({t})}
        self.role = {t for t in terms if role.match({t})}
        n = len(self.cards)
        self.idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in self.df.items()}
        postings = {}
        for slug, c in self.cards.items():
            c['within_name'] = [t for t in c['name_tokens'] if t not in self.generic]
            c['global_name'] = [t for t in c['within_name'] if t not in self.role and self.df[t] <= NAME_DF_MAX]
            c['producer_key'] = [t for t in c['producer_tokens'] if t not in self.generic and self.df[t] <= PRODUCER_DF_MAX]
            for t in c['global_name'] + c['producer_key']:
                postings.setdefault(t, set()).add(slug)
        self.postings = postings
        self.vocabulary = _FuzzyVocabulary(t for c in self.cards.values()
                                           for t in c['within_name'] + c['producer_key'])

    def generate(self, observations, limit=LIMIT):
        lines = [(i, o) for i, o in enumerate(observations or []) if float(o.get('score') or 0) >= LINE_MIN_SCORE]
        read = []
        term_lines = {}
        for i, o in lines:
            for variants in ocr_skeletons(o.get('raw_text', '')):
                read.append(variants)
                for term in self.vocabulary.match(variants):
                    term_lines.setdefault(term, set()).add(i)
        seeds = set().union(*(self.postings.get(t, set()) for t in term_lines)) if term_lines else set()
        rows = [row for row in (self._score(slug, term_lines, read) for slug in seeds) if row]
        best = {}
        for row in sorted(rows, key=lambda r: (-r['score'], r['slug'])):
            best.setdefault(row['candidate_id'], row)
        ranked = sorted(best.values(), key=lambda r: (-r['score'], r['slug']))[:limit]
        for rank, row in enumerate(ranked, 1):
            row['rank'] = rank
            row['support_lines'] = [{'line_index': i, 'raw_text': observations[i].get('raw_text'),
                                     'score': observations[i].get('score')} for i in row.pop('_lines')]
        return ranked

    def _score(self, slug, term_lines, read):
        card = self.cards[slug]
        producer_hit = [t for t in card['producer_key'] if t in term_lines]
        if producer_hit:
            names = card['within_name']
            matched = [t for t in names if t in term_lines]
            if not matched:
                return None
            coverage = sum(self.idf[t] for t in matched) / sum(self.idf[t] for t in names)
            if coverage < 0.5:
                return None
            rule = 'A_producer_and_name'
        else:
            names = card['global_name']
            matched = [t for t in names if t in term_lines]
            if not names or len(matched) < len(names):
                return None
            if not (len(names) >= 2 or (self.df[names[0]] <= 3 and len(names[0]) >= 6)):
                return None
            coverage = 1.0
            rule = 'B_full_distinctive_name'
        grapes = [t for t in card['grape_tokens'] if any(fuzzy_equal(t, v) for v in read)]
        grape_coverage = len(grapes) / len(card['grape_tokens']) if card['grape_tokens'] else 0.0
        support = sorted(set().union(*(term_lines[t] for t in matched + producer_hit)))
        return {'channel': CHANNEL, 'slug': slug, 'candidate_id': self.registry.candidate_id(slug),
                'product_id': self.registry.product_id(slug), 'rule': rule,
                'score': round(2 * coverage + (1 if producer_hit else 0) + 0.5 * grape_coverage, 6),
                'name_coverage': round(coverage, 6), 'matched_name_tokens': matched,
                'unmatched_name_tokens': [t for t in names if t not in term_lines],
                'matched_producer_tokens': producer_hit, 'matched_grape_tokens': grapes,
                'calibrated': False, '_lines': support}
