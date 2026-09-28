import re
from rshb_vine.positive_variant_text import canonical, contains


def phrases(observations):
    lines = []
    for o in observations:
        p = o.get('polygon')
        if not p:
            continue
        xs, ys = zip(*p)
        lines.append((min(xs), min(ys), max(xs), max(ys), canonical(o['raw_text']),float(o['score'])))
    lines.sort(key=lambda x: (x[1], x[0]))
    result = [canonical(o['raw_text']) for o in observations if float(o['score']) >= .85]
    for i, line in enumerate(lines):
        if line[5] < .85:
            continue
        text = line[4]
        previous = line
        for following in lines[i+1:i+4]:
            x1,y1,x2,y2,_,_ = previous
            a,b,c,d,t,score = following
            if score < .85:
                break
            overlap = min(x2,c)-max(x1,a)
            if overlap < .25*min(x2-x1,c-a) or b-y2 > 2*max(y2-y1,d-b) or b < y1:
                break
            text += ' '+t
            result.append(text)
            previous = following
    return result


class CatalogPhrase:
    def __init__(self, parent, catalog):
        self.parent = parent
        self.base = parent.base_policy
        self.grapes = {r['slug']: {canonical(g) for g in re.split('[,;]',r['fields'].get('Сорт винограда','')) if len(canonical(g))>=5 and not re.search(r'zz[a-z]+zz',canonical(g))} for r in catalog}

    def rerank(self, candidates, observations):
        rows, old = self.parent.rerank(candidates, observations)
        trace = {'policy':'catalog-phrase-v2','parent':old,'evidence':{},'promoted':None}
        if not rows:
            return rows,trace
        first=rows[0]['slug']; leader=self.base.items[first]
        if not leader['producer']:
            return rows,trace
        lines=phrases(observations)
        single=[canonical(o['raw_text']) for o in observations if float(o['score'])>=.85]
        ev={}
        for row in rows[:20]:
            slug=row['slug'];item=self.base.items[slug]
            if item['producer'] != leader['producer']:
                continue
            e=set(old['base']['evidence'].get(slug,[])) | set(old['evidence'].get(slug,[]))
            for grape in self.grapes[slug]:
                if any(contains(line,grape) for line in single):
                    e.add('catalog-grape:'+grape)
            if slug in self.base.distinct_names and len(item['name'])>=5 and any(contains(line,item['name']) for line in lines):
                e.add('name:'+item['name'])
            ev[slug]=e
        stronger=[s for s in ev if ev[s]>ev[first] and not (old['query']['year'] and self.parent.gallery.get(s,{}).get('year') and old['query']['year'] != self.parent.gallery[s]['year'])]
        maximal=[s for s in stronger if not any(ev[s]<ev[t] for t in stronger)]
        if len(maximal)==1:
            chosen=maximal[0];trace['promoted']=chosen
            rows=[r for r in rows if r['slug']==chosen]+[r for r in rows if r['slug']!=chosen]
        trace['evidence']={s:sorted(e) for s,e in ev.items()}
        return rows,trace
