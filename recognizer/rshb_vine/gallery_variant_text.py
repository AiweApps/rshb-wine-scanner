"""Bounded positive evidence from frozen gallery labels for same-name siblings."""
import math
import re
from rshb_vine.catalog_text_retrieval import normalize


STYLE_ALIASES = {
    'brut_nature': ['brut nature', 'брют натюр', 'брют натур', 'zero dosage', 'зеро дозаж'],
    'extra_brut': ['extra brut', 'экстра брют'],
    'brut': ['brut', 'брют'],
    'extra_dry': ['extra dry', 'экстра драй'],
    'demi_sec': ['demi sec', 'demi-sec', 'деми сек'],
    'doux': ['doux'],
}


def signature(observations):
    styles, years = set(), set()
    previous = ''
    phrases = sorted({(normalize(alias), style) for style, aliases in STYLE_ALIASES.items()
                      for alias in aliases}, key=lambda pair: -len(pair[0]))
    for observation in observations:
        score = float(observation['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid OCR confidence')
        # Catalog name normalization intentionally drops digits. Preserve them
        # here because reference vintage evidence needs the printed year.
        text = ' '.join(part if part.isdigit() else normalize(part)
                        for part in re.split(r'(\d+)', observation['raw_text']))
        text = re.sub(r'\s+', ' ', text).strip()
        if score < .85:
            previous = text
            continue
        remaining = text
        for phrase, style in phrases:
            pattern = r'(?<!\w)' + re.escape(phrase) + r'(?!\w)'
            if re.search(pattern, remaining):
                styles.add(style)
                remaining = re.sub(pattern, ' ', remaining)
        # Only a standalone year or style+year line is eligible; SINCE/EST and
        # producer history prose must not become a vintage contradiction.
        year = remaining.strip()
        establishment = {'since', 'est', 'established', 'founded', 'osnovano', 'osnovan'}
        if re.fullmatch(r'(?:19|20)\d{2}', year) and previous not in establishment:
            years.add(year)
        previous = text
    return {'style': next(iter(styles)) if len(styles) == 1 else None,
            'year': next(iter(years)) if len(years) == 1 else None}


class GalleryVariantText:
    def __init__(self, base_policy, gallery_signatures):
        self.base_policy = base_policy
        self.gallery = gallery_signatures

    def rerank(self, candidates, observations):
        rows, base_trace = self.base_policy.rerank(candidates, observations)
        query = signature(observations)
        trace = {'policy': 'gallery-variant-v1', 'base': base_trace,
                 'query': query, 'evidence': {}, 'promoted': None}
        if not rows or query['style'] is None:
            return rows, trace
        items = self.base_policy.items
        first = rows[0]['slug']
        leader = items[first]
        evidence = {}
        base_evidence = base_trace['evidence']
        for row in rows[:20]:
            slug = row['slug']
            item = items[slug]
            if (item['producer'], item['name']) != (leader['producer'], leader['name']):
                continue
            ref = self.gallery.get(slug, {})
            matches = set()
            if ref.get('style') == query['style']:
                matches.add('style:' + query['style'])
                if query['year'] and ref.get('year') == query['year']:
                    matches.add('year:' + query['year'])
            evidence[slug] = matches
        trace['evidence'] = {s: sorted(e) for s, e in evidence.items()}
        initial = evidence.get(first, set())
        stronger = []
        for slug, matches in evidence.items():
            ref = self.gallery.get(slug, {})
            if query['year'] and ref.get('year') and query['year'] != ref['year']:
                continue
            if matches > initial and set(base_evidence.get(slug, [])) >= set(base_evidence.get(first, [])):
                stronger.append(slug)
        maximal = [s for s in stronger if not any(evidence[s] < evidence[t] for t in stronger)]
        if len(maximal) == 1:
            chosen = maximal[0]
            trace['promoted'] = chosen
            rows = [r for r in rows if r['slug'] == chosen] + [r for r in rows if r['slug'] != chosen]
        return rows, trace
