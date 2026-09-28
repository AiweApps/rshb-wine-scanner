"""Bounded geometry challengers; all learned weights and retrieval unchanged."""
from PIL import Image
from rshb_vine.bottle_label_localizer import BottleLabelLocalizer
from rshb_vine.crop_stage_pipeline import CropStagePipeline
from rshb_vine.io import digest,seal
from rshb_vine.preprocessing import checked_box

class GeometryLocalizer(BottleLabelLocalizer):
    def __init__(self,previous,mode):
        self.__dict__.update(previous.__dict__)
        self.mode=mode;self.rotate=mode in ('rotate','combined')
        size=960 if mode in ('resolution960','combined') else 640
        self.bottles.processor.size={'height':size,'width':size}
        self.model_id=digest(dict(parent=previous.model_id,mode=mode,size=size,rotation='only label-empty bottle crop with width/height>1.25; choose90/270 by maximum label detection confidence'))
        self.last_trace=[]

    def detect(self,image):
        self.last_trace=[];bottles=self.bottles.detect(image)
        if not bottles:return self.labels.detect(image)
        result=[]
        for i,bottle in enumerate(bottles):
            parent=checked_box(bottle['bbox'],image.size);common=dict(parent_id=i,context_bbox=parent,bottle_score=float(bottle['score']));labels=[];direction=0;trial=False
            if len(bottles)<=self.max_bottles:
                crop=image.crop(parent);regions=self.labels.detect(crop)
                if self.rotate and not regions and crop.width>1.25*crop.height:
                    trial=True;candidates=[]
                    for angle,method in [(90,Image.Transpose.ROTATE_90),(270,Image.Transpose.ROTATE_270)]:
                        rotated=crop.transpose(method);found=self.labels.detect(rotated)
                        candidates.append((max((r['detector_score'] for r in found),default=-1),angle,found))
                    _,direction,regions=max(candidates,key=lambda x:(x[0],-x[1]))
                    restored=[]
                    for region in regions:
                        x1,y1,x2,y2=region['bbox']
                        box=[crop.width-y2,x1,crop.width-y1,x2] if direction==90 else [y1,crop.height-x2,y2,crop.height-x1]
                        restored.append({**region,'bbox':checked_box(box,crop.size)})
                    regions=restored
                for region in regions:
                    x1,y1,x2,y2=checked_box(region['bbox'],crop.size)
                    labels.append(dict(bbox=[x1+parent[0],y1+parent[1],x2+parent[0],y2+parent[1]],detector_score=region['detector_score']))
            self.last_trace.append(dict(parent_id=i,bottle_bbox=parent,rotation_attempted=trial,selected_rotation=direction if labels else None,label_count=len(labels)))
            if not labels:
                result.append({**common,'bbox':parent,'kind':'bottle_context','detector_score':float(bottle['score']),'raw_label_regions':[],'source':'bottle_context_no_label'})
            groups=[labels] if self.merge_parent_labels and labels else self.components(labels)
            for group in groups:
                box=[min(r['bbox'][0] for r in group),min(r['bbox'][1] for r in group),max(r['bbox'][2] for r in group),max(r['bbox'][3] for r in group)]
                result.append({**common,'bbox':box,'kind':'detected_label','detector_score':max(r['detector_score'] for r in group),'raw_label_regions':group,'source':'label_inside_detected_bottle'})
        return result

class GeometryPipeline(CropStagePipeline):
    def recognize_image(self,image,roi=None):
        result=super().recognize_image(image,roi)
        result['localization_trace']=self.detector.last_trace
        return result
