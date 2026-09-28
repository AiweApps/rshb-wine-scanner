"""Development challenger: positive complete title without mandatory brand OCR.

Operates after the existing selector. No pool expansion, weight changes,
negative inference from missing OCR, or special handling of individual wines.
"""
from collections import defaultdict

from rshb_vine.catalog_phrase_text import phrases
from rshb_vine.frozen_candidate_selector import spelling
from rshb_vine.gallery_variant_text import signature
from rshb_vine.positive_variant_text import contains


class WithinProducerTitle:
    def __init__(self, existing_policy):
        self.policy = existing_policy
        self.producers_by_name = defaultdict(set)
        for slug, name in self.policy.product_names.items():
            if name:
                self.producers_by_name[name].add(self.policy.items[slug]['producer'])

    def rerank(self, candidates, observations):
        rows = list(candidates)
        trace = {'policy': 'within-producer-complete-title-v1', 'matches': {},
                 'promoted': None, 'pool_unchanged': True}
        if not rows:
            return rows, trace
        first = rows[0]['slug']
        producer = self.policy.items[first]['producer']
        lines = [spelling(line) for line in phrases(observations)]
        year = signature(observations)['year']
        for row in rows:
            slug = row['slug']
            name = self.policy.product_names[slug]
            if (self.policy.items[slug]['producer'] != producer
                    or len(name.split()) < 2
                    or self.producers_by_name[name] != {producer}):
                continue
            reference_year = self.policy.gallery.get(slug, {}).get('year')
            if year and reference_year and year != reference_year:
                continue
            if any(contains(line, name) for line in lines):
                trace['matches'][slug] = name
        # Different full titles are ambiguous. Same-title vintage cards retain
        # their existing order; a missing year does not establish exact identity.
        if (trace['matches'] and first not in trace['matches']
                and len(set(trace['matches'].values())) == 1):
            chosen = next(iter(trace['matches']))
            rows = [r for r in rows if r['slug'] == chosen] + [
                r for r in rows if r['slug'] != chosen]
            trace['promoted'] = chosen
        return rows, trace
