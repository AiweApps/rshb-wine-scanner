"""Experimental cross-parent label deduplication before parent-envelope use."""
import copy
from rshb_vine.bottle_instances import iou

def resolve_label_ownership(regions, overlap=.65):
    """Retain highest-confidence observation of a shared label across parents.

    Only cross-parent duplicates are removed; within-parent multipart labels
    remain untouched. This does not resolve occlusion or establish SKU identity.
    """
    result=copy.deepcopy(regions)
    observations=[(float(label['detector_score']),i,j,label) for i,r in enumerate(result)
                  for j,label in enumerate(r.get('raw_label_regions',[]))]
    observations.sort(key=lambda x:(-x[0],x[1],x[2]))
    retained=[];removed={};events=[]
    for score,i,j,label in observations:
        owner=next((x for x in retained if result[x[1]].get('parent_id')!=result[i].get('parent_id')
                    and iou(label['bbox'],x[3]['bbox'])>=overlap),None)
        if owner is None:retained.append((score,i,j,label));continue
        removed.setdefault(i,set()).add(j)
        events.append({'from_parent':result[i].get('parent_id'),'to_parent':result[owner[1]].get('parent_id'),
                       'removed_bbox':label['bbox'],'owner_bbox':owner[3]['bbox'],
                       'iou':iou(label['bbox'],owner[3]['bbox'])})
    for i,indices in removed.items():
        r=result[i];labels=[label for j,label in enumerate(r['raw_label_regions']) if j not in indices]
        r['raw_label_regions']=labels
        if labels:
            boxes=[l['bbox'] for l in labels]
            r['bbox']=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            r['detector_score']=max(l['detector_score'] for l in labels)
        else:
            r.update(bbox=r['context_bbox'],kind='bottle_context',source='bottle_context_no_label',
                     detector_score=r.get('bottle_score',r['detector_score']))
    return result,events
