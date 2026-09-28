"""Experimental positive profile evidence on a fixed candidate pool."""
from copy import deepcopy
from collections import defaultdict
from rshb_vine.catalog.profiles import CatalogProfiles
from rshb_vine.resolution.identity import values,GRAPES,COLOR,SUGAR
from rshb_vine.gallery_variant_text import signature
from rshb_vine.frozen_candidate_selector import FrozenCandidateSelector,spelling,GENERIC
from rshb_vine.positive_variant_text import canonical,contains
from rshb_vine.io import read_jsonl
from rshb_vine.typed_catalog_lexicon import TypedCatalogLexicon

class ProfileVerifier:
    def __init__(self, root):
        profiles=CatalogProfiles(root)
        self.checksum=profiles.snapshot_checksum
        self.profiles={r['slug']:profiles.evidence(r['slug']) for r in read_jsonl(root/'data/normalized/catalog.jsonl')}
        self.lexicon=TypedCatalogLexicon(read_jsonl(root/'data/normalized/catalog.jsonl'),root)

    def apply(self, baseline):
        result=deepcopy(baseline);traces=[]
        for target in result.get('targets',[]):
            sid=str(target['instance_id']);ret=target['retrieval'];rows=ret.get('ranked_candidates',[])
            obs=baseline.get('variant_text',{}).get('observations',[]) if baseline.get('variant_text',{}).get('performed') else []
            if len(result['targets'])>1:
                obs=next((x.get('observations',[]) for x in baseline.get('instance_text',{}).get('targets',[]) if str(x['instance_id'])==sid and x.get('performed')),[])
            usable=[o for o in obs if o['score']>=.85]
            lines=[spelling(canonical(o['raw_text'])) for o in usable]
            seen={k:set().union(*(values(o['raw_text'],a) for o in usable)) for k,a in [('grapes',GRAPES),('wine_color',COLOR),('sweetness',SUGAR)]}
            seen['grapes']=self.lexicon.observe_grapes(usable)
            sig=signature(obs);seen['sparkling_style']={sig['style']} if sig['style'] else set()
            before=ret.get('best_candidate');evidence={};scores={}
            for row in rows:
                slug=row['slug'];profile=self.profiles[slug];product=profile['product'];matched=[];conflicts=[]
                for key,observed in seen.items():
                    expected=product[key]['comparison_value'];expected=set(expected if isinstance(expected,list) else [expected]) if expected is not None else set()
                    if key=='grapes':expected=self.lexicon.expected_grapes(expected)
                    if observed and expected and observed<=expected:matched.append(key)
                    elif len(observed)==1 and expected and observed.isdisjoint(expected) and key!='grapes':conflicts.append(key)
                brand,title=self.lexicon.identity(slug,usable)
                # Positive identity is mandatory; no grape-absence veto, no year in ranking.
                identity=int(brand)+int(title);score=(identity,len(matched)) if identity and (title or matched) and not conflicts else (0,0)
                scores[slug]=score
                evidence[slug]={'matched':matched,'conflicts':conflicts,'brand':brand,'title':title,'score':list(score)}
            chosen=before
            if scores:
                best=max(scores.values());winners=[s for s,v in scores.items() if v==best]
                if len(winners)==1 and best>scores.get(before,(0,0)) and best[0]>0:chosen=winners[0]
            if chosen!=before:
                ret['ranked_candidates']=[r for r in rows if r['slug']==chosen]+[r for r in rows if r['slug']!=chosen];ret['best_candidate']=chosen
            trace={'instance_id':sid,'before':before,'after':chosen,'observations':len(obs),'usable_observations':len(usable),'observed':{k:sorted(v) for k,v in seen.items()},'year':sig['year'],'candidates':evidence}
            traces.append(trace)
        if len(result.get('targets',[]))==1:
            ret=result['targets'][0]['retrieval'];result.update(best_candidate=ret['best_candidate'],ranked_candidates=ret['ranked_candidates'])
        result['profile_verifier']={'policy':'positive-profile-v5','profile_checksum':self.checksum,'targets':traces,'calibrated':False}
        return result
