"""Experimental positive variant evidence; no accepted identity or missing-word veto."""
import math
import re
from rshb_vine.catalog_text_retrieval import normalize
from rshb_vine.resolution.identity import GRAPES


ALIASES = {**GRAPES,
    'nebbiolo': ['неббиоло', 'nebbiolo'],
    'tempranillo': ['темпранильо', 'tempranillo'],
    'pinot_gris': ['пино гри', 'pinot gris', 'pinot grigio'],
    'viognier': ['вионье', 'viognier'],
    'gewurztraminer': ['гевюрцтраминер', 'gewurztraminer'],
    'reserve': ['резерв', 'reserve', 'reserva'],
}
PHRASES = sorted(((normalize(a), key) for key, group in ALIASES.items()
                  for a in group), key=lambda p: -len(p[0]))


def canonical(text):
    text = normalize(text)
    for phrase, key in PHRASES:
        text = re.sub(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)',
                      'zz' + key.replace('_', '') + 'zz', text)
    return text


def contains(text, phrase):
    return bool(phrase) and ' ' + phrase + ' ' in ' ' + text + ' '


class PositiveVariantText:
    """Promote only a unique stronger match within the visual leader's producer.

    Evidence is a set, not a weighted sum. A promotion must retain all evidence
    supporting the leader. OCR line boundaries remain intact for name matching.
    This is a reranker only; unseen candidates and multi-object choice are separate.
    """
    def __init__(self, catalog, allowed):
        self.items = {}
        for row in catalog:
            if row['slug'] not in allowed or row.get('excluded_from_retrieval'):
                continue
            f = row['fields']
            self.items[row['slug']] = {
                'producer': normalize(f.get('Винодельня', '')),
                'name': canonical(f.get('Название вина', '')),
                'grapes': canonical(f.get('Сорт винограда', '')),
            }
        if set(self.items) != set(allowed):
            raise ValueError('Catalog must cover gallery exactly')
        self.distinct_names = {
            s for s, item in self.items.items()
            if item['name'] and not any(
                t != s and other['producer'] == item['producer']
                and contains(other['name'], item['name'])
                for t, other in self.items.items())
        }

    def rerank(self, candidates, observations):
        rows = [dict(r) for r in candidates]
        lines = []
        for o in observations:
            score = float(o['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid OCR confidence')
            if score >= .85:
                lines.append(canonical(o['raw_text']))
        trace = {'policy': 'positive-variant-v2', 'evidence': {}, 'promoted': None}
        if not rows:
            return rows, trace
        leader = self.items[rows[0]['slug']]
        if not leader['producer']:
            return rows, trace
        pool = [r for r in rows[:20]
                if self.items[r['slug']]['producer'] == leader['producer']]
        evidence = {}
        for row in pool:
            item = self.items[row['slug']]
            matches = set()
            for key in ALIASES:
                token = 'zz' + key.replace('_', '') + 'zz'
                field = item['name'] if key == 'reserve' else item['grapes']
                if contains(field, token) and any(contains(line, token) for line in lines):
                    matches.add(key)
            name = item['name']
            # Full phrase must occur on one OCR line, never assembled from fragments.
            # Shared names, including shorter sibling names anywhere in the gallery,
            # cannot identify a variant even when that sibling is absent from Top20.
            if len(name) >= 5 and row['slug'] in self.distinct_names:
                if any(contains(line, name) for line in lines):
                    matches.add('name:' + name)
            evidence[row['slug']] = matches
        trace['evidence'] = {s: sorted(e) for s, e in evidence.items()}
        initial = evidence[rows[0]['slug']]
        stronger = [s for s, e in evidence.items() if e > initial]
        maximal = [s for s in stronger
                   if not any(evidence[s] < evidence[t] for t in stronger)]
        if len(maximal) == 1:
            slug = maximal[0]
            trace['promoted'] = slug
            rows = [r for r in rows if r['slug'] == slug] + [r for r in rows if r['slug'] != slug]
        return rows, trace
