"""One local alternate OCR only for sparse single-target native transcripts."""
import re,io,time
from collections import defaultdict
from copy import deepcopy
from PIL import Image,ImageOps
from rshb_vine.line_evidence_verifier_v2 import LineEvidenceVerifier
from rshb_vine.typed_catalog_lexicon import normalize
class AlternateEvidence(LineEvidenceVerifier):
 def __init__(self,root):
  super().__init__(root);owners=defaultdict(set);forms={}
  for s,item in self.lexicon.items.items():
   producer=normalize(item['raw_producer']);short=re.sub(r'^(?:zmv|zavod marochnykh vin|ooo|pao|ao)\s+','',producer)
   if short and short!=producer:forms[s]=short;owners[short].add(producer)
  self.derived_aliases=[]
  for s,f in forms.items():
   if len(owners[f])==1:
    self.lexicon.items[s]['brand_forms'].append(f);self.derived_aliases.append({'slug':s,'form':f,'source':'catalog producer minus explicit organizational prefix, unique producer ownership'})
 def eligible(self,target):return bool(target.get('alternative_ocr_performed'))
 def apply(self,baseline):
  result=super().apply(baseline)
  for original,target,trace in zip(baseline.get('targets',[]),result.get('targets',[]),result['profile_verifier']['targets']):
   before=original['retrieval'].get('best_candidate')
   # Missing catalog fields must not act as a penalty. An alternate reader
   # may replace the answer only when it establishes an explicit contradiction.
   if not trace['candidates'].get(before,{}).get('conflicts'):
    target['retrieval']=deepcopy(original['retrieval']);trace['after']=before
  if len(result.get('targets',[]))==1:
   ret=result['targets'][0]['retrieval'];result.update(best_candidate=ret['best_candidate'],ranked_candidates=ret['ranked_candidates'])
  result['profile_verifier']['policy']='sparse-alternative-contradiction-v2'
  return result
class SparseAlternativeOCR:
 def __init__(self,root,reader=None):
  self.root=root;self.engine=AlternateEvidence(root);self.reader=reader
 def eligible(self,d):
  if len(d.get('targets',[]))!=1:return False
  t=d['targets'][0];rows=t['retrieval'].get('ranked_candidates',[])
  if len(rows)<2 or not d.get('variant_text',{}).get('performed'):return False
  count=sum(o['score']>=.85 and len(re.sub(r'[^a-zа-яё]','',o['raw_text'].lower()))>=3 for o in d['variant_text'].get('observations',[]))
  a,b=[self.engine.profiles[r['slug']]['product']['producer']['comparison_value'] for r in rows[:2]]
  return count<=1 and bool(a and a==b)
 def apply(self,data,baseline):
  if not self.eligible(baseline):return deepcopy(baseline)
  if self.reader is None:
   from rshb_vine.models import OCR
   self.reader=OCR(self.root)
  x=deepcopy(baseline);t=x['targets'][0];views=[v for v in t['retrieval']['views'] if v['kind'] in ('detected_label','front_label','partial_label')]
  if not views:return x
  im=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB');crop=im.crop(tuple(views[0]['bbox']));angle=t.get('retrieval_pixel_rotation_ccw',0)
  if angle:crop=crop.rotate(angle,expand=True)
  start=time.perf_counter();obs=self.reader.read(crop);ms=(time.perf_counter()-start)*1000
  x['variant_text']['observations']+=[{**o,'polygon':[]} for o in obs];t['alternative_ocr_performed']=True
  result=self.engine.apply(x);result['alternative_ocr']={'reader':'PaddlePP-OCRv5CPU','observations':obs,'bbox':views[0]['bbox'],'ms':ms,'confidence':'model-specific uncalibrated; .85 experimental admission, not probability equality with Vision'};return result
