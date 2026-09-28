"""Physical bottle instances and label association; no retrieval or identity changes."""
from rshb_vine.preprocessing import checked_box

SCHEMA = 'wine-bottle-instances-v1'


def area(b):
    return max(0, b[2]-b[0])*max(0, b[3]-b[1])


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def iou(a,b):
    overlap=intersection(a,b)
    return overlap/max(area(a)+area(b)-overlap,1e-9)


def validate_frame(frame):
    """Source annotation validation; empty, reviewed negative frames are valid."""
    if frame.get('schema') != SCHEMA:
        raise ValueError('Wrong instance schema')
    size=frame['oriented_size']; seen=set(); labels=set()
    for instance in frame['instances']:
        bid=instance['bottle_id']
        if bid in seen:raise ValueError('Duplicate bottle identity')
        seen.add(bid);box=checked_box(instance['bottle_bbox'],size)
        label=instance.get('label_bbox');lid=instance.get('label_id')
        if label is None:
            if lid is not None or not instance.get('label_not_visible',False):
                raise ValueError('Missing label must be explicit')
        else:
            if not lid or lid in labels or instance.get('label_not_visible',False):
                raise ValueError('Label identity/visibility conflict')
            labels.add(lid);label=checked_box(label,size)
            if intersection(box,label)/area(label)<.65:
                raise ValueError('Label not contained by its bottle')
    if not frame['instances'] and frame.get('negative_review_status')!='visually_verified':
        raise ValueError('Empty detections do not prove a negative image')
    return frame


def associate(bottles,labels):
    """Each label belongs to at most one bottle; labels never create wine targets.

    High-IoU duplicates are merged before association. Ambiguous cross-bottle
    assignments abstain. Bottle proposals without labels remain visible but unscored.
    """
    kept=[]
    for b in sorted(bottles,key=lambda r:-r['score']):
        if not any(iou(b['bbox'],k['bbox'])>.65 for k in kept):kept.append(b)
    instances=[dict(bottle_id=f'bottle-{i}',bottle_bbox=b['bbox'],bottle_score=b['score'],labels=[]) for i,b in enumerate(kept)]
    orphan=[]
    for label in labels:
        box=label['bbox']; candidates=[]
        for i,b in enumerate(kept):
            fraction=intersection(box,b['bbox'])/max(area(box),1)
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            if fraction>=.8 and b['bbox'][0]<=cx<=b['bbox'][2] and b['bbox'][1]<=cy<=b['bbox'][3]:
                candidates.append((fraction,i))
        if not candidates:
            orphan.append(dict(label,reason='no_containing_bottle'));continue
        # Nested duplicate proposals should have been removed. Do not guess which
        # adjacent bottle owns a label substantially contained by both.
        if len(candidates)>1:
            orphan.append(dict(label,reason='ambiguous_bottle_association'));continue
        instances[candidates[0][1]]['labels'].append(label)
    for instance in instances:
        unique=[]
        for label in sorted(instance.pop('labels'),key=lambda r:-r['score']):
            if not any(iou(label['bbox'],x['bbox'])>.5 for x in unique):unique.append(label)
        if unique:
            boxes=[r['bbox'] for r in unique]
            instance.update(label_bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)],label_id=instance['bottle_id']+'-label',label_regions=unique,status='label_available')
        else:
            instance.update(label_bbox=None,label_id=None,label_regions=[],status='insufficient_evidence')
    return instances,orphan
