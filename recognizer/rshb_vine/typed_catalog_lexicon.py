"""Catalog-derived role separation; number-preserving normalization, no query fit."""
import re
import unicodedata
from rshb_vine.catalog_text_retrieval import _CYR
from rshb_vine.positive_variant_text import ALIASES,contains
from rshb_vine.frozen_candidate_selector import GENERIC
from rshb_vine.io import read_json
from rshb_vine.resolution.identity import SUGAR,COLOR
from rshb_vine.gallery_variant_text import STYLE_ALIASES


def normalize(text):
    text=''.join(c for c in unicodedata.normalize('NFKD',str(text).lower()) if not unicodedata.combining(c))
    text=''.join(_CYR.get(c,c) for c in text)
    return ' '.join(re.findall('[a-z0-9]+',text))


class TypedCatalogLexicon:
    def __init__(self,catalog,root):
        self.grapes={k:list(v) for k,v in ALIASES.items() if k!='reserve'}
        self.grapes.update({'sangiovese':['санджовезе','санджове́зе','sangiovese'],
                            'muscat_family':['мускат','muscat','moscato'],
                            'chenin_blanc':['шенен блан','chenin blanc']})
        self.grape_forms={normalize(a):k for k,v in self.grapes.items() for a in v}
        for row in catalog:
            for raw in re.split('[,;]',row['fields'].get('Сорт винограда','')):
                form=normalize(raw)
                if form:self.grape_forms.setdefault(form,'unmapped:'+raw.strip())
        aliases={}
        for a in read_json(root/'config/producer-brand-aliases.json')['aliases']:
            if a['status']=='verified':aliases.setdefault(normalize(a['producer']),set()).update(normalize(x) for x in a['aliases'])
        generic=set(GENERIC)|set(normalize('виноград сорт сорта винограда резерв коллекция серия premium reserve classic blend cuvee').split())
        role_forms=set(self.grape_forms)|{normalize(a) for mapping in [SUGAR,COLOR,STYLE_ALIASES] for aliases in mapping.values() for a in aliases}
        self.items={}
        for row in catalog:
            fields=row['fields'];producer=normalize(fields['Винодельня']);forms=aliases.get(producer,set())|{producer}
            forms={f for f in forms if any(t not in generic for t in f.split())}
            name=normalize(fields['Название вина'])
            full_name=name
            for form in sorted(forms,key=len,reverse=True):
                full_name=re.sub(r'(?<!\w)'+re.escape(form)+r'(?!\w)',' ',full_name)
            full_name=' '.join(full_name.split())
            for form in sorted(forms|role_forms,key=len,reverse=True):
                if form:name=re.sub(r'(?<!\w)'+re.escape(form)+r'(?!\w)',' ',name)
            # Years/strength in product titles are qualifiers, not name evidence.
            words=[w for w in name.split() if not w.isdigit() and w not in generic]
            self.items[row['slug']]={'brand_forms':sorted(forms),'product_phrase':' '.join(words),'full_product_phrase':full_name,
                                     'raw_producer':fields['Винодельня'],'raw_name':fields['Название вина']}

    def observe_grapes(self,observations):
        found=set()
        for o in observations:
            text=' '+normalize(o['raw_text'])+' '
            for form,key in sorted(self.grape_forms.items(),key=lambda x:-len(x[0])):
                if contains(text.strip(),form):
                    found.add(key);text=text.replace(' '+form+' ',' ')
        return found

    def identity(self,slug,observations):
        item=self.items[slug];lines=[normalize(o['raw_text']) for o in observations]
        brand=any(contains(line,form) for form in item['brand_forms'] for line in lines)
        # Role extraction cannot shorten the phrase used as identity proof.
        name=item['full_product_phrase']
        distinctive=item['product_phrase']
        other_names={x['full_product_phrase'] for x in self.items.values()
                     if x['product_phrase']==distinctive}
        title=bool(distinctive and len(other_names)==1 and name
                   and any(contains(line,name) for line in lines))
        return brand,title

    def expected_grapes(self,values):
        return {self.grape_forms.get(normalize(v[len('unmapped:'):]),v)
                if v.startswith('unmapped:') else v for v in values}
