"""Frozen catalogue-derived identity evidence across an unchanged visual pool.

Preserve the released positive-variant behaviour when no stronger identity is
visible. No new candidates, query-specific aliases, learned weights or refusals.
"""
import json
from pathlib import Path
import math
import re
from rshb_vine.catalog_text_retrieval import normalize

def spelling(text):
    """Unify common transliterated ia/iya endings without wine-specific aliases."""
    return re.sub(r"iya\b", "ia", text)
from rshb_vine.positive_variant_text import PositiveVariantText, canonical, contains, ALIASES
from rshb_vine.gallery_variant_text import GalleryVariantText, signature
from rshb_vine.catalog_phrase_text import CatalogPhrase, phrases

# Generic catalogue vocabulary is not distinctive brand/name evidence by itself.
GENERIC = set(normalize('wine wines winery estate chateau domaine reserve collection '
    'вино вина винодельня винодельческое хозяйство усадьба поместье завод дом '
    'белое красное розовое сухое полусухое сладкое полусладкое игристое '
    'white red rose orange orangevoe oranzh oranzhevoe dry sweet brut sparkling premium classic selection').split())
GENERIC |= {canonical(a) for aliases in ALIASES.values() for a in aliases}


class FrozenCandidateSelector:
    def __init__(self, catalog, gallery_signatures):
        catalog = [r for r in catalog if not r.get('excluded_from_retrieval')]
        allowed = {r['slug'] for r in catalog}
        base = PositiveVariantText(catalog, allowed)
        self.parent = CatalogPhrase(GalleryVariantText(base, gallery_signatures), catalog)
        self.items = base.items
        self.gallery = gallery_signatures
        self.distinct_names = base.distinct_names
        self.grape_words = {w for item in self.items.values() for w in item['grapes'].split()}
        alias_path = Path(__file__).resolve().parents[1] / 'config/producer-brand-aliases.json'
        aliases = json.loads(alias_path.read_text())['aliases']
        self.producer_aliases = {}
        for entry in aliases:
            if entry['status'] != 'verified':
                raise ValueError('Unverified producer alias')
            producer = normalize(entry['producer'])
            self.producer_aliases.setdefault(producer, set()).update(spelling(canonical(a)) for a in entry['aliases'])
        self.product_names = {}
        self.producer_names = {}
        for slug, item in self.items.items():
            name = spelling(item['name'])
            producer = spelling(canonical(item['producer']))
            self.producer_names[slug] = ' '.join(w for w in producer.split() if w not in GENERIC)
            # Remove an exact producer phrase from a name before judging whether
            # the remaining text identifies a product rather than just a brand.
            remainder = re.sub(r'(?<!\w)' + re.escape(producer) + r'(?!\w)', ' ', name) if producer else name
            remainder = ' '.join(remainder.split())
            informative = {w for w in remainder.split() if len(w) >= 5
                           and w not in GENERIC and w not in self.grape_words and not w.startswith('zz')}
            self.product_names[slug] = remainder if informative else ''

    def rerank(self, candidates, observations):
        for o in observations:
            score = float(o['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid OCR confidence')
        original = [dict(r) for r in candidates]
        if len({r['slug'] for r in original}) != len(original):
            raise ValueError('Duplicate candidate slug')
        if any(r['slug'] not in self.items for r in original):
            raise ValueError('Candidate outside frozen catalogue')
        rows, parent_trace = self.parent.rerank(original, observations)
        trace = {'policy': 'frozen-global-identity-v6', 'parent': parent_trace,
                 'evidence': {}, 'promoted': None, 'pool_unchanged': True}
        if not rows:
            return rows, trace
        lines = [spelling(line) for line in phrases(observations)]
        raw_lines = [normalize(o['raw_text']) for o in observations if float(o['score']) >= .85]
        query = signature(observations)
        evidence = {}
        for row in rows:
            slug = row['slug']; item = self.items[slug]
            producer = self.producer_names[slug]
            brand_forms = self.producer_aliases.get(item['producer'], set()) | ({producer} if producer else set())
            brand_match = any(contains(line,form) for form in brand_forms for line in lines)
            product_name = self.product_names[slug]
            name_terms = {product_name} if product_name and any(contains(line,product_name) for line in lines) else set()
            qualifiers = set()
            for key in ALIASES:
                token = 'zz'+key.replace('_','')+'zz'
                field = item['name'] if key == 'reserve' else item['grapes']
                if contains(field,token) and any(contains(line,token) for line in lines):
                    qualifiers.add(key)
            ref = self.gallery.get(slug,{})
            if query['style'] and ref.get('style') == query['style']:
                qualifiers.add('style:'+query['style'])
            year_conflict = bool(query['year'] and ref.get('year') and query['year'] != ref['year'])
            if query['year'] and ref.get('year') == query['year']:
                qualifiers.add('year:'+query['year'])
            # Evidence keys refer to observed text, not candidate IDs.
            anchors = {'brand:'+item['producer']} if brand_match else set()
            anchors |= {'name:'+word for word in name_terms}
            evidence[slug] = (anchors,qualifiers,year_conflict)
            trace['evidence'][slug] = {'identity':sorted(anchors),'qualifiers':sorted(qualifiers),
                                       'year_conflict':year_conflict}
        first = rows[0]['slug']
        def comparable_qualifiers(slug, other):
            # A year/style match cannot retain a wrong producer, or distinguish
            # two different products when source signatures cover only one.
            q = evidence[slug][1]
            if self.items[slug]['name'] == self.items[other]['name']:
                return q
            return {v for v in q if not v.startswith(('year:', 'style:'))}
        def stronger(slug, other):
            a,q,conflict = evidence[slug]; b,r,_ = evidence[other]
            if conflict or not a:
                return False
            same_producer = self.items[slug]['producer'] == self.items[other]['producer']
            if a > b:
                return not same_producer or comparable_qualifiers(slug,other) >= comparable_qualifiers(other,slug)
            return a == b and same_producer and comparable_qualifiers(slug,other) > comparable_qualifiers(other,slug)
        leader_has_brand = any(v.startswith('brand:') for v in evidence[first][0])
        # This challenger changes producer selection only with positive brand
        # evidence. Existing within-producer selection remains the fallback.
        eligible = [s for s in evidence if not leader_has_brand
                    and self.items[s]['producer'] != self.items[first]['producer']
                    and any(v.startswith('brand:') for v in evidence[s][0])
                    and not evidence[s][2]]
        maximal = [s for s in eligible if not any(stronger(t,s) for t in eligible if t != s)]
        # Equal positive evidence defines a group. Retain visual order inside it,
        # rather than retaining an unsupported leader of another producer.
        identities = {self.items[s]['producer'] for s in maximal}
        if maximal and len(identities) == 1:
            chosen = maximal[0];trace['promoted'] = chosen
            rows = [r for r in rows if r['slug']==chosen]+[r for r in rows if r['slug']!=chosen]
        trace['unresolved_maxima'] = maximal if len(maximal)>1 else []
        assert {r['slug'] for r in rows} == {r['slug'] for r in original}
        return rows,trace
