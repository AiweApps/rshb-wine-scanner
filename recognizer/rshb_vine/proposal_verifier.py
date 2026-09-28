"""Frozen RT-DETR and label localization with a learned crop eligibility gate."""
import time
import numpy as np
from rshb_vine.io import verify,read_json,seal
from rshb_vine.preprocessing import decode,checked_box,InvalidImage
from rshb_vine.visual_core import View
from rshb_vine.label_first_pipeline import consolidate_regions
from rshb_vine.bottle_instances import area,intersection,iou

class ProposalVerifierPipeline:
    def __init__(self,baseline,artifact):
        self.base=baseline;self.spec=verify(read_json(artifact))
        assert self.spec['encoder_id']==baseline.encoder_id
        assert self.spec['baseline_runtime']==baseline.manifest['runtime_descriptor_checksum']
        self.coef=np.array(self.spec['coef']);self.bias=self.spec['intercept'];self.detector=baseline.detector
        self.manifest=seal(dict(kind='frozen-cascade-proposal-verifier-v1',baseline=baseline.manifest['checksum'],verifier=self.spec['checksum'],policy='v2: retain explicit label closeup route without inventing bottle geometry; parent envelope; label/parent duplicate gate; no context-only SKU'))
    def recognize(self,data,roi=None):
        started=time.perf_counter()
        try:
            image,_=decode(data);roi=checked_box(roi,image.size) if roi is not None else None
        except (InvalidImage,ValueError,TypeError) as exc:
            return dict(decision='invalid_image',slug=None,best_candidate=None,targets=[],reasons=[str(exc)])
        result=self.recognize_image(image,roi);result['timing_ms']['total']=(time.perf_counter()-started)*1000;return result
    def recognize_image(self,image,roi=None):
        t=time.perf_counter();regions=self.detector.detect(image);parents={}
        regions,_=consolidate_regions(regions)
        for j,r in enumerate(regions):
            # Retain the existing close-up label input route; no bottle geometry claimed.
            if 'context_bbox' not in r:
                r={**r,'context_bbox':[0,0,*image.size],'parent_id':f'closeup-{j}','kind':'detected_label','unlocalized':True}
            box=checked_box(r['context_bbox'],image.size)
            if roi and not (roi[0]<=(box[0]+box[2])/2<=roi[2] and roi[1]<=(box[1]+box[3])/2<=roi[3]):continue
            p=parents.setdefault(r['parent_id'],dict(parent_id=r['parent_id'],context_bbox=box,labels=[],unlocalized=r.get('unlocalized',False),detector_score=r['detector_score']))
            if r['kind']=='detected_label':p['labels'].append(r['bbox'])
        detector_ms=(time.perf_counter()-t)*1000
        parents=list(parents.values());views=[]
        for p in parents:
            p['context_vector']=len(views);views.append(View('context',p['context_bbox'],'detected_bottle',True,list(image.size),self.detector.model_id))
            if p['labels']:
                bs=p['labels'];p['bbox']=[min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)];p['label_vector']=len(views)
                views.append(View('detected_label',checked_box(p['bbox'],image.size),'automatic_main_label',None,list(image.size),self.detector.model_id))
        t=time.perf_counter();vectors=np.concatenate([self.base.encoder.encode([image.crop(v.bbox) for v in views[j:j+8]]) for j in range(0,len(views),8)]) if views else []
        encode_ms=(time.perf_counter()-t)*1000;t=time.perf_counter();candidates=[];instances=[]
        for p in parents:
            score=float(1/(1+np.exp(-np.clip(np.dot(vectors[p['context_vector']],self.coef)+self.bias,-40,40))))
            ins=dict(bottle_id=str(p['parent_id']),bottle_bbox=None if p['unlocalized'] else p['context_bbox'],label_bbox=p.get('bbox'),wine_score=score,status='label_available' if p['labels'] else 'insufficient_evidence')
            instances.append(ins)
            if not p['labels']:continue
            ids=[p['context_vector'],p['label_vector']];retrieval=self.base.index.search(vectors[ids],[views[i] for i in ids],self.base.encoder_id)
            candidates.append(dict(instance_id=str(p['parent_id']),parent_id=p['parent_id'],bbox=p['bbox'],bottle_bbox=None if p['unlocalized'] else p['context_bbox'],context_bbox=p['context_bbox'],physical_bottle_localized=not p['unlocalized'],kind='detected_label',detector_score=p['detector_score'],wine_score=score,retrieval=retrieval))
        kept=[];duplicates=[]
        for c in sorted(candidates,key=lambda r:-r['detector_score']):
            duplicate=next((k for k in kept if iou(c['bbox'],k['bbox'])>.5 and intersection(c['context_bbox'],k['context_bbox'])/max(min(area(c['context_bbox']),area(k['context_bbox'])),1)>.9),None)
            if duplicate:duplicates.append(dict(instance_id=c['instance_id'],kept_instance_id=duplicate['instance_id']))
            else:kept.append(c)
        def resolve(targets,ins):
            r=dict(decision='insufficient_evidence',slug=None,best_candidate=None,probability_correct=None,ranked_candidates=[],targets=targets,instances=ins,requires_target_selection=len(targets)>1,reasons=['confidence_not_calibrated'],suppressed_instances=duplicates)
            if len(targets)==1:
                found=targets[0]['retrieval'];r.update(decision=found['decision'],best_candidate=found['best_candidate'],ranked_candidates=found['ranked_candidates'])
            elif len(targets)>1:r['decision']='ambiguous_target'
            else:r['reasons'].append('no_verified_bottle_with_label')
            return r
        structure=resolve(kept,instances)
        threshold=self.spec['threshold'];result=resolve([c for c in kept if c['wine_score']>=threshold],[i for i in instances if i['wine_score']>=threshold])
        result['rejected_instances']=[i for i in instances if i['wine_score']<threshold];result['structure_only']=structure
        result['timing_ms']=dict(detector=detector_ms,encode=encode_ms,retrieval=(time.perf_counter()-t)*1000,ocr=0.)
        return result
