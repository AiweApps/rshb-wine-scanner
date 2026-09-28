"""Bounded bottle retry for a single unlocalized, frame-sized label proposal."""
from copy import deepcopy
import io,time
from rshb_vine.preprocessing import decode,checked_box
from rshb_vine.bottle_instances import area
from rshb_vine.square_rescue import SquareRescue


def eligible(result,roi=None):
    if roi is not None or len(result.get('targets',[]))!=1:return False
    t=result['targets'][0]
    if t.get('physical_bottle_localized') or t.get('bottle_bbox') is not None:return False
    views=t['retrieval'].get('views',[])
    if not views:return False
    w,h=views[0]['original_size']
    return area(t['bbox'])/(w*h)>=.6


class OversizedLabelRescue:
    def __init__(self,parent,low_bottle_detector,png_compression=6):
        self.parent=parent;self.detector=low_bottle_detector;self.png_compression=png_compression

    def apply(self,data,baseline,roi=None):
        result=deepcopy(baseline);trace={'attempted':False,'recovered':False}
        result['oversized_label_rescue']=trace
        if not eligible(baseline,roi):return result
        start=time.perf_counter();trace['attempted']=True
        t=time.perf_counter();image,_=decode(data);trace['decode_ms']=(time.perf_counter()-t)*1000
        t=time.perf_counter();parents=self.detector.detect(image)[:2];trace['detector_ms']=(time.perf_counter()-t)*1000;trace['parents']=parents;trace['trials']=[];valid=[];trace['costs']=[]
        for parent in parents:
            box=checked_box(parent['bbox'],image.size)
            if area(box)>.5*image.width*image.height:
                trace['trials'].append({'box':box,'reason':'parent_not_smaller'});continue
            t0=time.perf_counter();crop=image.crop(box);buf=io.BytesIO();crop.save(buf,format='PNG',compress_level=self.png_compression);payload=buf.getvalue();cost={'box':box,'encode_ms':(time.perf_counter()-t0)*1000,'payload_bytes':len(payload)};trace['costs'].append(cost)
            if len(payload)>20*1024*1024:
                trace['trials'].append({'box':box,'reason':'payload_limit'});continue
            t0=time.perf_counter();retry=self.parent(payload);cost['parent_ms']=(time.perf_counter()-t0)*1000;cost['parent_timing']=retry.get('timing_ms',{})
            if len(retry.get('targets',[]))!=1:
                trace['trials'].append({'box':box,'reason':'not_one_target'});continue
            t=retry['targets'][0];lb=t['bbox']
            good=t.get('physical_bottle_localized') and t.get('bottle_bbox') is not None and area(lb)<=.5*crop.width*crop.height
            trace['trials'].append({'box':box,'reason':'valid' if good else 'unlocalized_or_oversized','best':retry.get('best_candidate'),'label_bbox_crop':lb})
            if not good:continue
            recovered=deepcopy(retry);SquareRescue._restore(recovered,-box[0],-box[1],image.width,image.height)
            selection=recovered.get('roi_selection',{})
            if selection.get('roi') is not None:
                x1,y1,x2,y2=selection['roi']
                selection['roi']=[x1+box[0],y1+box[1],x2+box[0],y2+box[1]]
                selection['coordinate_space']='EXIF-oriented original pixels'
                selection['source']='automatic_bottle_retry_not_user_roi'
            valid.append(recovered)
        # No similarity-based choice between competing physical proposals.
        if len(valid)==1:
            result=valid[0];oldid=baseline['targets'][0]['instance_id'];newid=result['targets'][0]['instance_id']
            def ids(x):
                if isinstance(x,dict):
                    for k,v in x.items():
                        if k in ('instance_id','bottle_id','parent_id','selected_parent_id') and str(v)==str(newid):x[k]=oldid
                        elif isinstance(v,(dict,list)):ids(v)
                elif isinstance(x,list):
                    for v in x:ids(v)
            ids(result);trace['recovered']=True;trace['before']=baseline['targets'][0]['retrieval']['best_candidate'];trace['old_label_bbox']=baseline['targets'][0]['bbox']
        trace['valid_proposals']=len(valid);trace['ms']=(time.perf_counter()-start)*1000;result['oversized_label_rescue']=trace
        return result
