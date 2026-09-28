"""Conditional whole-title evidence; does not lower other OCR acceptance gates."""
import re
from rshb_vine.product_name_selector import ProductNameSelector
from rshb_vine.frozen_candidate_selector import spelling,GENERIC
from rshb_vine.positive_variant_text import canonical,contains

def title_phrases(observations):
    # Keep original scores. Only this complete-title channel admits >= .5.
    result=[canonical(o['raw_text']) for o in observations if o['score']>=.5]
    lines=[]
    for o in observations:
        if not o.get('polygon'):continue
        xs,ys=zip(*o['polygon']);lines.append((min(xs),min(ys),max(xs),max(ys),canonical(o['raw_text']),o['score']))
    lines.sort(key=lambda x:(x[1],x[0]))
    for i,line in enumerate(lines):
        if line[5]<.5:continue
        text=line[4];previous=line
        for following in lines[i+1:i+4]:
            x1,y1,x2,y2=previous[:4];a,b,c,d=following[:4]
            if following[5]<.5 or min(x2,c)-max(x1,a)<.25*min(x2-x1,c-a) or b-y2>2*max(y2-y1,d-b) or b<y1:break
            text+=' '+following[4];result.append(text);previous=following
    return result

class CatalogConstrainedTitle(ProductNameSelector):
    def rerank(self,candidates,observations):
        rows,old=super().rerank(candidates,observations);trace={'policy':'catalog-constrained-whole-title-v1','parent':old,'matched':{},'promoted':None}
        if not rows:return rows,trace
        first=rows[0]['slug'];producer=self.items[first]['producer'];evidence=old.get('parent',{}).get('evidence',{})
        if not any(x.startswith('brand:') for x in evidence.get(first,{}).get('identity',[])):return rows,trace
        forms=self.producer_aliases.get(producer,set())|{spelling(canonical(producer)),self.producer_names[first]}
        def strip(text):
            for form in sorted(forms,key=len,reverse=True):
                if form:text=re.sub(r'(?<!\w)'+re.escape(form)+r'(?!\w)',' ',text)
            return ' '.join(text.split())
        lines=[strip(spelling(t)) for t in title_phrases(observations)];matched=[]
        for row in rows:
            slug=row['slug'];item=self.items[slug]
            if item['producer']!=producer or evidence.get(slug,{}).get('year_conflict'):continue
            name=strip(spelling(item['name']));tokens=name.split()
            if len(tokens)<2 or not any(len(t)>=5 and t not in GENERIC and not t.startswith('zz') for t in tokens):continue
            if any(contains(line,name) for line in lines):matched.append(slug);trace['matched'][slug]=name
        if matched and len({trace['matched'][s] for s in matched})==1 and first not in matched:
            chosen=matched[0];rows=[r for r in rows if r['slug']==chosen]+[r for r in rows if r['slug']!=chosen];trace['promoted']=chosen
        return rows,trace
