"""Typed sweetness evidence within an identical producer/product family."""
from copy import deepcopy
from rshb_vine.io import read_json, read_jsonl, verify
from rshb_vine.resolution.identity import values,SUGAR,norm
class SugarEvidence:
 def __init__(self,root):
  self.catalog={r['slug']:r for r in read_jsonl(root/'data/normalized/catalog.jsonl')}
  self.ledger=verify(read_json(root/'runs/sugar-evidence-v1/reference-ledger.json'))['records']
 def family(self,slug):
  f=self.catalog[slug]['fields'];return (norm(f.get('Винодельня','')),norm(f.get('Название вина','')),norm(f.get('Сорт винограда','')))
 def apply(self,baseline):
  d=deepcopy(baseline);traces=[]
  for t in d.get('targets',[]):
   ret=t['retrieval'];before=ret['best_candidate'];sid=str(t['instance_id']);obs=d.get('variant_text',{}).get('observations',[])
   if len(d['targets'])>1:obs=next((p.get('observations',[]) for p in d.get('instance_text',{}).get('targets',[]) if str(p['instance_id'])==sid),[])
   seen=set().union(*(values(o['raw_text'],SUGAR) for o in obs if o['score']>=.85));old=self.ledger.get(before,{});chosen=before
   # Only verified reference facts can establish contradiction; OCR candidates alone cannot.
   if len(seen)==1 and old.get('verified') and old['value'] not in seen:
    candidates=[r['slug'] for r in ret['ranked_candidates'] if self.family(r['slug'])==self.family(before) and self.ledger.get(r['slug'],{}).get('verified') and self.ledger[r['slug']]['value'] in seen]
    if len(candidates)==1:chosen=candidates[0]
   if chosen!=before:
    ret['best_candidate']=chosen;ret['ranked_candidates']=sorted(ret['ranked_candidates'],key=lambda r:r['slug']!=chosen)
   traces.append({'instance_id':sid,'before':before,'after':chosen,'observed_sweetness':sorted(seen),'reference_before':old})
  if len(d.get('targets',[]))==1:d.update(best_candidate=d['targets'][0]['retrieval']['best_candidate'],ranked_candidates=d['targets'][0]['retrieval']['ranked_candidates'])
  d['sugar_evidence']={'policy':'verified-sugar-family-v1','targets':traces};return d
