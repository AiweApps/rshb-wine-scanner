"""Independent all-catalog text search and distinct-slug visual fusion."""
from collections import defaultdict
import math
import re
import unicodedata

import numpy as np

STOP = {'вино','россия','wine','russia','мл','ml','alc','vol','алк','об'}


def tokens(text):
    return [t for t in re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())
            if len(t) > 1 and t not in STOP and not t.isdigit()]


def grams(text):
    value = ' '.join(tokens(text))
    return {value[i:i+3] for i in range(max(0,len(value)-2))}


class TextIndex:
    def __init__(self, catalog, reference_text=None):
        reference_text = reference_text or {}
        self.fields = {}
        frequencies = defaultdict(int)
        for row in catalog:
            if row.get('excluded_from_retrieval'):
                continue
            f = row['fields']
            # Description intentionally excluded: marketing text is not identity evidence.
            fields = [f.get(k,'') for k in ('Название вина','Винодельня','Сорт винограда','Категория')]
            fields.extend(reference_text.get(row['slug'], []))
            self.fields[row['slug']] = fields
            for t in set(tokens(' '.join(fields))):
                frequencies[t] += 1
        self.idf = {t: math.log(1+len(catalog)/(1+n)) for t,n in frequencies.items()}
        self.features = {s:(set(tokens(' '.join(f))),grams(' '.join(f))) for s,f in self.fields.items()}

    def search(self, observations, k=20):
        text = ' '.join(r['raw_text'] for r in observations if r['score'] >= .4)
        ts, gs = set(tokens(text)), grams(text)
        if not ts:
            return []
        denominator = sum(self.idf.get(t,1.) for t in ts)
        rows = []
        for slug,(ct,cg) in self.features.items():
            intersection = ts & ct
            weighted = sum(self.idf[t] for t in intersection)/max(denominator,1e-8)
            character = len(gs & cg)/max(len(gs | cg),1)
            score = .75*weighted + .25*character
            if intersection or character >= .12:
                rows.append({'slug':slug,'score':score,'tokens':sorted(intersection)})
        return sorted(rows,key=lambda r:(-r['score'],r['slug']))[:k]


def visual_search(vectors, references, query, k=20):
    scores = vectors @ query.T
    per_slug = {}
    for i, row in enumerate(references):
        score = float(scores[i].max())
        if row['slug'] not in per_slug or score > per_slug[row['slug']]['score']:
            per_slug[row['slug']] = {'slug': row['slug'], 'score': score, 'reference_index': i}
    return sorted(per_slug.values(), key=lambda r:(-r['score'],r['slug']))[:k]


def fuse(visual, text, relationships, k=60, limit=40):
    candidates = {}
    for branch, rows in [('visual',visual),('text',text)]:
        for rank, row in enumerate(rows,1):
            if relationships.get(row['slug'], {}).get('excluded_from_retrieval'):
                continue
            candidate = candidates.setdefault(row['slug'], {'slug':row['slug'],'rrf':0.,'sources':{},'expanded_from':[]})
            candidate['sources'][branch] = dict(row, rank=rank)
            candidate['rrf'] += 1/(k+rank)
    for slug in list(candidates):
        family = relationships.get(slug,{}).get('family_id')
        if family:
            for other, rel in relationships.items():
                if rel.get('excluded_from_retrieval'):
                    continue
                if rel.get('family_id') == family and other != slug:
                    candidate = candidates.setdefault(other,{'slug':other,'rrf':0.,'sources':{},'expanded_from':[]})
                    candidate['expanded_from'].append(slug)
    ranked = sorted(candidates.values(),key=lambda r:(-r['rrf'],r['slug']))
    # Expanded variants must survive even if they exceed the normal union budget.
    kept = ranked[:limit]
    families = {relationships.get(r['slug'],{}).get('family_id') for r in kept} - {None,''}
    kept.extend(r for r in ranked[limit:] if relationships.get(r['slug'],{}).get('family_id') in families)
    return kept, {'total_union_expanded':len(ranked),'returned':len(kept),'truncated':len(kept)<len(ranked)}
