"""Bounded experimental label ownership and per-instance text profile."""
import time,copy
from rshb_vine.bottle_rescue import BottleRescueProfile
from rshb_vine.label_region_ownership import resolve_label_ownership
from rshb_vine.preprocessing import checked_box

class OwnershipDetector:
    def __init__(self,inner):
        self.inner=inner;self.ownership_events=[]
    def __getattr__(self,name):return getattr(self.inner,name)
    def detect(self,image):
        regions=self.inner.detect(image)
        result,self.ownership_events=resolve_label_ownership(regions)
        return result

class InstanceTextProfile(BottleRescueProfile):
    def recognize_image(self,image,roi=None):
        result=super().recognize_image(image,roi)
        result['label_ownership']=copy.deepcopy(self.detector.detector.ownership_events)
        targets=result.get('targets',[])
        trace={'performed':False,'targets':[],'limit':16}
        result['instance_text']=trace
        if not 1<len(targets)<=16:return result
        started=time.perf_counter()
        for target in targets:
            ret=target['retrieval'];box=checked_box(target['bbox'],image.size)
            crop=image.crop(box);angle=target.get('retrieval_pixel_rotation_ccw',0)
            if angle in (90,270):crop=crop.rotate(angle,expand=True)
            entry={'instance_id':str(target['instance_id']),'bbox':list(box)}
            try:
                obs=self.variant_reader.read(crop)
                pool=ret.get('visual_ranked_candidates',ret['ranked_candidates'])
                ranked,evidence=self.variant_policy.rerank(copy.deepcopy(pool),obs)
            except Exception as exc:
                entry.update(performed=False,error=type(exc).__name__)
            else:
                ret.update(visual_ranked_candidates=pool,ranked_candidates=ranked,
                           best_candidate=ranked[0]['slug'] if ranked else None)
                entry.update(performed=True,observations=obs,evidence=evidence)
            trace['targets'].append(entry)
        trace['performed']=any(x['performed'] for x in trace['targets'])
        result['timing_ms']['instance_text']=(time.perf_counter()-started)*1000
        # Multiple physical targets remain multiple: no arbitrary global SKU.
        return result
