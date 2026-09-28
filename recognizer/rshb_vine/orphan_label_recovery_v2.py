"""Orphan recovery with detached instance history and complete OCR packets.

Geometry, acceptance and candidate selection are identical to the v1 policy.
"""
from copy import deepcopy
import io,time
from rshb_vine.preprocessing import decode,checked_box
from rshb_vine.bottle_instances import intersection,area,iou

def containing(parents,label):
 out=[]
 for p in parents:
  b=p.get('bottle_bbox') or p.get('bbox')
  if b and intersection(b,label)/max(area(label),1)>=.8 and not any(iou(b,c)>.65 for c in out):out.append(b)
 return out


def detach_instance_roster(result):
 """Do not mutate shared historical baseline/retry instance lists."""
 result['instances'] = deepcopy(result.get('instances', []))


def publish_recovered_text(result, baseline, retry, sid, source_box):
 """Preserve full OCR packets and the frame in which their polygons live.

 Polygons remain in the OCR crop's rotated pixels. Only the enclosing unrotated
 bbox is translated to original-image coordinates; rotation_ccw is retained.
 """
 packets = result.setdefault('instance_text', {}).setdefault('targets', [])
 old_targets = baseline.get('targets', [])
 if len(old_targets) == 1:
  old_id = str(old_targets[0]['instance_id'])
  if not any(str(p['instance_id']) == old_id for p in packets):
   previous = deepcopy(baseline.get('variant_text', {}))
   previous['instance_id'] = old_id
   if previous.get('bbox_original') is not None:
    previous['bbox'] = list(previous['bbox_original'])
   packets.append(previous)
 packet = deepcopy(retry.get('variant_text', {}))
 packet.update(instance_id=sid, source_label_bbox=list(source_box))
 packet['retry_text_source'] = deepcopy(retry.get('variant_text', {}))
 local_box = packet.get('bbox_original')
 if local_box is not None:
  packet['retry_bbox_original'] = list(local_box)
  packet['bbox_original'] = [local_box[0]+source_box[0], local_box[1]+source_box[1],
                             local_box[2]+source_box[0], local_box[3]+source_box[1]]
  packet['bbox'] = list(packet['bbox_original'])
  packet['bbox_coordinate_space'] = 'EXIF-oriented original pixels'
 else:
  # Keep unavailable transform metadata explicitly unavailable, not guessed.
  packet['bbox_coordinate_space'] = 'unavailable; see retry_text_source'
 packets.append(packet)
 result['instance_text']['performed'] = any(p.get('performed', False) for p in packets)


def finalize_text_contract(result):
 packets = result.get('instance_text', {}).get('targets', [])
 targets = result.get('targets', [])
 if len(targets) == 1:
  sid = str(targets[0]['instance_id'])
  result['variant_text'] = deepcopy(next(p for p in packets if str(p['instance_id']) == sid))
 else:
  result['variant_text'] = {'performed': False, 'reason': 'not_single_target',
                            'evidence_location': 'instance_text.targets'}

class OrphanLabelRecovery:
 def __init__(self,label_detector,bottle_detector,retry):self.labels=label_detector;self.bottles=bottle_detector;self.retry=retry
 def apply(self,data,baseline,roi=None):
  result=deepcopy(baseline);trace={'attempted':False,'trials':[],'added':[]};result['orphan_label_recovery']=trace
  if roi is not None or baseline.get('decision')=='invalid_image':return result
  detach_instance_roster(result)
  image,_=decode(data);trace['attempted']=True;start=time.perf_counter();labels=self.labels.detect(image);old=baseline.get('targets',[])
  labels=[l for l in labels if max([intersection(l['bbox'],t['bbox'])/max(area(l['bbox']),1) for t in old] or [0])<.2][:2]
  parents=baseline.get('instances',[])+baseline.get('bottle_rescue',{}).get('retry_instances',[])
  for j,label in enumerate(labels):
   box=checked_box(label['bbox'],image.size);matches=containing(parents,box);trial={'label_bbox':box,'stripe_attempted':False};trace['trials'].append(trial)
   if not matches:
    pad=.1*(box[2]-box[0]);stripe=[max(0,int(box[0]-pad)),0,min(image.width,int(box[2]+pad)),image.height];ps=self.bottles.detect(image.crop(stripe))
    for p in ps:p['bbox']=[p['bbox'][0]+stripe[0],p['bbox'][1],p['bbox'][2]+stripe[0],p['bbox'][3]]
    matches=containing(ps,box);trial.update(stripe_attempted=True,stripe_bbox=stripe,proposals=ps)
   trial['parents']=matches
   if len(matches)!=1:trial['reason']='no_unique_physical_parent';continue
   parent=matches[0]
   if any(iou(parent,t.get('bottle_bbox') or t.get('context_bbox') or t['bbox'])>.65 for t in result.get('targets',[])):trial['reason']='existing_physical_target';continue
   crop=image.crop(box);buf=io.BytesIO();crop.save(buf,format='PNG',compress_level=1)
   if len(buf.getvalue())>20*1024*1024:trial['reason']='payload_limit';continue
   retry=self.retry(buf.getvalue());ts=retry.get('targets',[])
   if len(ts)!=1:trial['reason']='not_one_valid_wine_target';continue
   t=deepcopy(ts[0]);sid=f'orphan-{j}';t.update(instance_id=sid,parent_id=sid,physical_bottle_localized=True,bottle_bbox=parent,context_bbox=parent)
   def restore(b):return [max(0,min(image.width,b[0]+box[0])),max(0,min(image.height,b[1]+box[1])),max(0,min(image.width,b[2]+box[0])),max(0,min(image.height,b[3]+box[1]))]
   t['bbox']=restore(t['bbox'])
   for view in t['retrieval'].get('views',[]):
    if view.get('bbox'):view['bbox']=restore(view['bbox'])
    view['original_size']=[image.width,image.height]
   result.setdefault('targets',[]).append(t)
   # Keep the physical-instance roster consistent, including an existing failed parent.
   instance=next((p for p in result.get('instances',[]) if p.get('bottle_bbox') and iou(parent,p['bottle_bbox'])>.65),None)
   if instance is None:
    instance={'bottle_id':sid};result.setdefault('instances',[]).append(instance)
   instance.update(bottle_bbox=parent,label_bbox=t['bbox'],label_id=sid+'-label',status='label_available',wine_score=t.get('wine_score'))
   t['parent_id']=instance['bottle_id']
   publish_recovered_text(result,baseline,retry,sid,box)
   trial.update(reason='recovered',instance_id=sid,answer=t['retrieval']['best_candidate']);trace['added'].append(sid)
  if trace['added']:
   finalize_text_contract(result)
   if len(result['targets'])==1:
    t=result['targets'][0];result.update(best_candidate=t['retrieval']['best_candidate'],ranked_candidates=t['retrieval']['ranked_candidates'],slug=None,decision='uncertain',requires_target_selection=False)
   else:result.update(best_candidate=None,slug=None,ranked_candidates=[],decision='ambiguous_target',requires_target_selection=True)
  trace['ms']=1000*(time.perf_counter()-start);return result
