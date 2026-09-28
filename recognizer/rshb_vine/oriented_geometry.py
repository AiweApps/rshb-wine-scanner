"""Carry detector-chosen bottle orientation into visual crops without fitting weights."""
import time
from PIL import Image
from rshb_vine.geometry_loop import GeometryPipeline
from rshb_vine.visual_core import View
from rshb_vine.preprocessing import checked_box

class OrientedGeometryPipeline(GeometryPipeline):
    def recognize_image(self,image,roi=None):
        result=super().recognize_image(image,roi)
        trace={str(t['parent_id']):t for t in result['localization_trace']}
        for target in result['targets']:
            t=trace.get(str(target['parent_id']),{})
            angle=t.get('selected_rotation') if t.get('rotation_attempted') else None
            if angle not in (90,270):continue
            method=Image.Transpose.ROTATE_90 if angle==90 else Image.Transpose.ROTATE_270
            views=[View('context',checked_box(target['context_bbox'],image.size),'orientation_normalized_bottle',True,list(image.size),self.detector.model_id),View('detected_label',checked_box(target['bbox'],image.size),'orientation_normalized_label',None,list(image.size),self.detector.model_id)]
            started=time.perf_counter();vectors=self.base.encoder.encode([image.crop(v.bbox).transpose(method) for v in views]);result['timing_ms']['encode']+=(time.perf_counter()-started)*1000
            started=time.perf_counter();target['retrieval']=self.base.index.search(vectors,views,self.base.encoder_id);result['timing_ms']['retrieval']+=(time.perf_counter()-started)*1000
            target['retrieval_pixel_rotation_ccw']=angle
        if len(result['targets'])==1:
            r=result['targets'][0]['retrieval'];result.update(decision=r['decision'],best_candidate=r['best_candidate'],ranked_candidates=r['ranked_candidates'])
        return result
