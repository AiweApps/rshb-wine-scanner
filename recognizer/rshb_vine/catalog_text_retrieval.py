"""Local positive lexical evidence over catalog identity fields, no confidence fit.

Character TF-IDF is fitted to the reference catalog only. OCR never generates a
slug and missing words never become contradictions. This is a fixed development
challenger, not calibrated exact-identity acceptance.
"""
import math
import re
import unicodedata
from collections import Counter

from rshb_vine.io import digest

_CYR = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                ['a','b','v','g','d','e','e','zh','z','i','y','k','l','m','n','o','p','r','s','t','u','f','kh','ts','ch','sh','shch','','y','','e','yu','ya']))
FIELDS = ('Название вина', 'Винодельня', 'Сорт винограда')


def normalize(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(text).lower()) if not unicodedata.combining(c))
    text = ''.join(_CYR.get(c,c) for c in text)
    return ' '.join(re.findall('[a-z]+', text))


def features(text):
    result = Counter()
    for token in normalize(text).split():
        if len(token) < 3:
            continue
        padded = ' '+token+' '
        for n in (3,4,5):
            result.update(padded[i:i+n] for i in range(len(padded)-n+1))
    return result


class CatalogTextIndex:
    def __init__(self, catalog, allowed_slugs):
        allowed = set(allowed_slugs)
        records = {r['slug']:r for r in catalog if r['slug'] in allowed and not r.get('excluded_from_retrieval')}
        if set(records) != allowed:
            raise ValueError('Text catalog must cover the fixed gallery exactly')
        self.slugs = sorted(records)
        self.documents = {s:' '.join(str(records[s]['fields'].get(f,'')) for f in FIELDS) for s in self.slugs}
        counts = {s:features(self.documents[s]) for s in self.slugs}
        df=Counter(k for c in counts.values() for k in c)
        self.idf={k:math.log((len(counts)+1)/(v+1))+1 for k,v in df.items()}
        self.vectors={s:self._vector(c) for s,c in counts.items()}
        self.policy_id=digest({'policy':'catalog-positive-char-tfidf-v1','fields':FIELDS,
                               'ngrams':[3,4,5],'minimum_ocr_score':.85,'documents':self.documents})

    def _vector(self, counts):
        values={k:(1+math.log(v))*self.idf[k] for k,v in counts.items() if k in self.idf}
        norm=math.sqrt(sum(v*v for v in values.values()))
        return {k:v/norm for k,v in values.items()} if norm else {}

    def search(self, observations, limit=20):
        lines=[]
        for r in observations:
            score=float(r['score'])
            if not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('Invalid OCR score')
            if score>=.85:
                lines.append(str(r['raw_text']))
        query=self._vector(features(' '.join(lines)))
        if not query:
            return []
        scores=[{'slug':s,'text_score':sum(v*query.get(k,0.) for k,v in vector.items()),
                 'matched_terms': sorted(set(normalize(self.documents[s]).split()) & set(normalize(' '.join(lines)).split())),
                 'evidence':'positive_similarity_only'} for s,vector in self.vectors.items()]
        return sorted((r for r in scores if r['text_score']>0),key=lambda r:(-r['text_score'],r['slug']))[:limit]


def combine_visual_text(visual, text):
    """Fixed equal RRF60; preserve visual tie order, then lexical slug."""
    if not text:
        return [dict(r) for r in visual]
    scores={};original={r['slug']:r for r in visual};positions={r['slug']:i for i,r in enumerate(visual)}
    for channel in (visual[:20],text[:20]):
        for rank,r in enumerate(channel,1):
            scores[r['slug']]=scores.get(r['slug'],0.)+1/(60+rank)
    text_by_slug={r['slug']:r for r in text}
    return [{**original.get(s,{'slug':s,'score':None,'channels':{}}),
             'fusion_score':scores[s],'text_evidence':text_by_slug.get(s)}
            for s in sorted(scores,key=lambda s:(-scores[s],positions.get(s,10**6),s))]
