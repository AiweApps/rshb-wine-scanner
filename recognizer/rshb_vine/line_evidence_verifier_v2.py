"""Candidate support across separately typed OCR lines, fixed pool and threshold."""
from copy import deepcopy
from collections import defaultdict
from rshb_vine.profile_verifier_v5 import ProfileVerifier
from rshb_vine.typed_catalog_lexicon import normalize
from rshb_vine.positive_variant_text import contains

class LineEvidenceVerifier(ProfileVerifier):
    def __init__(self,root):
        super().__init__(root)
        self.name_producers=defaultdict(set)
        for s,item in self.lexicon.items.items():
            phrase=item['product_phrase']
            if phrase:self.name_producers[phrase].add(normalize(item['raw_producer']))

    def eligible(self,target):
        ret=target['retrieval'];rows=ret.get('ranked_candidates',[])
        if len(rows)<2 or not ret.get('channel_disagreement'):return False
        a,b=[self.profiles[r['slug']]['product']['producer']['comparison_value'] for r in rows[:2]]
        return bool(a and a==b)

    def apply(self,baseline):
        result=super().apply(baseline)
        # Re-evaluate from original ordering; parent serves extraction only.
        for original,target,trace in zip(baseline.get('targets',[]),result.get('targets',[]),result['profile_verifier']['targets']):
            target['retrieval']=deepcopy(original['retrieval']);ret=target['retrieval'];rows=ret.get('ranked_candidates',[]);before=ret.get('best_candidate');sid=str(target['instance_id'])
            obs=baseline.get('variant_text',{}).get('observations',[]) if baseline.get('variant_text',{}).get('performed') else []
            if len(result['targets'])>1:obs=next((x.get('observations',[]) for x in baseline.get('instance_text',{}).get('targets',[]) if str(x['instance_id'])==sid and x.get('performed')),[])
            lines=[normalize(o['raw_text']) for o in obs if o['score']>=.85];supports={};conflicts={}
            for row in rows:
                slug=row['slug'];item=self.lexicon.items[slug];e=trace['candidates'][slug];support=set(e['matched'])
                producer=normalize(item['raw_producer']);name=item['product_phrase']
                if e['brand']:support.add('brand:'+producer)
                # Name is family evidence, never exact SKU. Requires an intact line,
                # >=2 words, exclusive producer across the full catalog, and is
                # always combined with preservation of observed variant fields.
                if name and len(name.split())>=2 and self.name_producers[name]=={producer} and any(contains(line,name) for line in lines):support.add('name:'+name)
                supports[slug]=support;conflicts[slug]=e['conflicts']
                e['line_support']=sorted(support)
            chosen=before
            if rows and self.eligible(original):
                old=supports.get(before,set());stronger=[s for s,v in supports.items() if v>old and any(x.startswith(('brand:','name:')) for x in v) and any(not x.startswith(('brand:','name:')) for x in v) and not conflicts[s]]
                maximal=[s for s in stronger if not any(supports[s]<supports[t] for t in stronger)]
                if maximal:
                    # Retain visual ordering among equally supported candidates.
                    chosen=next(r['slug'] for r in rows if r['slug'] in maximal)
            if chosen!=before:ret['ranked_candidates']=[r for r in rows if r['slug']==chosen]+[r for r in rows if r['slug']!=chosen];ret['best_candidate']=chosen
            trace.update(before=before,after=chosen)
        if len(result.get('targets',[]))==1:
            ret=result['targets'][0]['retrieval'];result.update(best_candidate=ret['best_candidate'],ranked_candidates=ret['ranked_candidates'])
        result['profile_verifier']['policy']='line-evidence-v2'
        return result
