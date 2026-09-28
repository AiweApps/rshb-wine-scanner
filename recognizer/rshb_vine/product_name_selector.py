"""Experimental complete product-name evidence with intervening brand removed."""
import re
from rshb_vine.frozen_candidate_selector import FrozenCandidateSelector,spelling,GENERIC
from rshb_vine.positive_variant_text import canonical,contains
from rshb_vine.catalog_phrase_text import phrases

class ProductNameSelector(FrozenCandidateSelector):
    def rerank(self,candidates,observations):
        rows,parent=super().rerank(candidates,observations)
        trace={'policy':'complete-product-name-v1','parent':parent,'matched':{},'promoted':None}
        if not rows:return rows,trace
        first=rows[0]['slug'];producer=self.items[first]['producer']
        evidence=parent.get('evidence',{})
        if not any(x.startswith('brand:') for x in evidence.get(first,{}).get('identity',[])):
            return rows,trace
        forms=self.producer_aliases.get(producer,set())|{spelling(canonical(producer)),self.producer_names[first]}
        def strip_brand(text):
            for form in sorted(forms,key=len,reverse=True):
                if form:text=re.sub(r'(?<!\w)'+re.escape(form)+r'(?!\w)',' ',text)
            return ' '.join(text.split())
        lines=[strip_brand(spelling(x)) for x in phrases(observations)]
        matches=[]
        for row in rows:
            slug=row['slug']
            if self.items[slug]['producer']!=producer:continue
            name=strip_brand(spelling(self.items[slug]['name']));tokens=name.split()
            if len(tokens)<2 or not any(len(t)>=5 and t not in GENERIC and not t.startswith('zz') for t in tokens):continue
            if evidence.get(slug,{}).get('year_conflict'):continue
            if any(contains(line,name) for line in lines):
                matches.append(slug);trace['matched'][slug]=name
        # Different matched names are ambiguous; identical vintage-card names
        # retain their previous order. Missing words do not veto other products.
        if matches and len({trace['matched'][s] for s in matches})==1 and first not in matches:
            chosen=matches[0];rows=[r for r in rows if r['slug']==chosen]+[r for r in rows if r['slug']!=chosen];trace['promoted']=chosen
        return rows,trace
