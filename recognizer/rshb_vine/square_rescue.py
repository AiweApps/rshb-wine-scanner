"""One bounded square-canvas retry after no targets; original-coordinate output."""
import copy,io,time
from PIL import Image
from rshb_vine.preprocessing import decode
from rshb_vine.io import seal,sha256

class SquareRescue:
    def __init__(self,parent):
        self.parent=parent
        self.manifest=seal({'kind':'zero-target-square-rescue-v1','parent':parent.manifest,'source_sha':sha256(__file__),'max_square_pixels':24000000,'calibrated':False})
    def recognize(self,data,roi=None):
        started=time.perf_counter();baseline=self.parent.recognize(data,roi)
        if roi is not None or baseline.get('targets') or baseline.get('decision')=='invalid_image':return baseline
        image,_=decode(data);w,h=image.size;side=max(w,h)
        if w==h or side*side>24000000:return baseline
        dx,dy=(side-w)//2,(side-h)//2
        square=Image.new('RGB',(side,side),'white');square.paste(image,(dx,dy));buf=io.BytesIO();square.save(buf,format='PNG');payload=buf.getvalue()
        if len(payload)>20*1024*1024:return baseline
        retry=self.parent.recognize(payload);chosen=baseline;reason='no_recovered_target'
        if retry.get('targets'):
            recovered=copy.deepcopy(retry)
            try:
                self._restore(recovered,dx,dy,w,h)
                chosen=recovered;reason='recovered'
            except ValueError:
                reason='invalid_recovered_geometry'
        result=copy.deepcopy(chosen)
        result['square_rescue']={'attempted':True,'recovered':reason=='recovered','reason':reason,'padding_left_top':[dx,dy],'square_size':[side,side],'original_size':[w,h],'bbox_coordinate_space':'EXIF-oriented original pixels','local_ocr_coordinates_unchanged':True}
        elapsed=(time.perf_counter()-started)*1000
        result.setdefault('timing_ms',{})['square_rescue_total']=elapsed
        result['timing_ms']['total']=elapsed
        return result
    @staticmethod
    def _restore(result,dx,dy,w,h):
        boxkeys={'bbox','bottle_bbox','label_bbox','context_bbox','bbox_original'}
        visited=set()
        def visit(x):
            if isinstance(x,(dict,list)):
                if id(x) in visited:return
                visited.add(id(x))
            if isinstance(x,dict):
                for k,v in list(x.items()):
                    if k in boxkeys and v is not None:
                        if not isinstance(v,(list,tuple)) or len(v)!=4:raise ValueError('Unknown bbox')
                        b=[max(0,min(w,v[0]-dx)),max(0,min(h,v[1]-dy)),max(0,min(w,v[2]-dx)),max(0,min(h,v[3]-dy))]
                        if b[0]>=b[2] or b[1]>=b[3]:raise ValueError('Box outside original')
                        x[k]=b
                    elif k=='original_size' and isinstance(v,(list,tuple)) and len(v)==2:x[k]=[w,h]
                    else:visit(v)
            elif isinstance(x,list):
                for v in x:visit(v)
        # OCR polygons are crop-local, as stated by polygon_coordinate_space.
        # Translate global bbox origins only, never these local polygons.
        visit(result)
