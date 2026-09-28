"""One conditional whole-label OCR pass, shared by live recognition and replay."""
import io,time
from copy import deepcopy
from PIL import Image,ImageOps
from rshb_vine.line_evidence_verifier_v2 import LineEvidenceVerifier
from rshb_vine.vision_ocr import VisionOCR
class ConditionalLineProfile:
 def __init__(self,root):
  self.engine=LineEvidenceVerifier(root);self.reader=VisionOCR(root/'runs/vision-text-v1/vision-ocr')
 def apply(self,data,baseline):
  x=deepcopy(baseline);targets=[t for t in x.get('targets',[]) if self.engine.eligible(t)];reads=[]
  if targets:
   image=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
   for t in targets:
    views=[v for v in t['retrieval']['views'] if v['kind'] in ('detected_label','front_label','partial_label')]
    if not views:continue
    crop=image.crop(tuple(views[0]['bbox']));angle=t.get('retrieval_pixel_rotation_ccw',0)
    if angle:crop=crop.rotate(angle,expand=True)
    start=time.perf_counter();obs=self.reader.read(crop);reads.append({'instance_id':str(t['instance_id']),'bbox':views[0]['bbox'],'rotation':angle,'observations':obs,'ms':(time.perf_counter()-start)*1000})
    incoming=[{**o,'polygon':[]} for o in obs]
    if len(x['targets'])==1:x['variant_text']={**x.get('variant_text',{}),'performed':True,'observations':x.get('variant_text',{}).get('observations',[])+incoming}
    else:
     it=x.setdefault('instance_text',{}).setdefault('targets',[]);part=next((z for z in it if str(z['instance_id'])==str(t['instance_id'])),None)
     if part is None:part={'instance_id':str(t['instance_id']),'observations':[]};it.append(part)
     part['observations']=part.get('observations',[])+incoming;part['performed']=True
  result=self.engine.apply(x);result['conditional_label_ocr']={'performed':bool(reads),'reads':reads,'reader':'Vision revision3','polygons':'new observations in individual rotated label crop pixels; omitted from merged line-only evidence'};return result
