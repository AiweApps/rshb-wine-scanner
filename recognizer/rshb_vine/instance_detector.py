"""One full-frame model for bottle+label, or same-data label-only control."""
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD
from rshb_vine.label_detector import letterbox,original_boxes
from rshb_vine.io import read_json,verify,sha256


def empty_aware_loss(self,targets,head_outputs,anchors,matched_idxs):
    losses=SSD.compute_loss(self,targets,head_outputs,anchors,matched_idxs)
    empty=[i for i,t in enumerate(targets) if len(t['boxes'])==0]
    if empty:
        logits=head_outputs['cls_logits'][empty]
        loss=F.cross_entropy(logits.flatten(0,1),torch.zeros(logits.shape[0]*logits.shape[1],dtype=torch.long,device=logits.device),reduction='none').reshape(logits.shape[:2])
        losses['empty_background']=.25*loss.topk(min(32,loss.shape[1]),dim=1).values.mean()
    return losses


def build(mode):
    if mode not in ('joint','label_only'):raise ValueError('Unknown detector mode')
    m=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3 if mode=='joint' else 2)
    m.compute_loss=MethodType(empty_aware_loss,m)
    return m


class InstanceDetector:
    def __init__(self,artifact,device='mps'):
        p=Path(artifact);self.manifest=verify(read_json(p/'manifest.json'));self.mode=self.manifest['mode'];self.device=device
        if sha256(p/'best.pt')!=self.manifest['weights_sha256']:raise ValueError('Changed joint/control weights')
        self.model=build(self.mode);self.model.load_state_dict(torch.load(p/'best.pt',map_location='cpu',weights_only=True));self.model.eval().requires_grad_(False).to(device)
        self.model_id=self.manifest['checksum']
    def detect(self,image):
        tensor,_,tr=letterbox(image)
        with torch.inference_mode():r=self.model([tensor.to(self.device)])[0]
        boxes=original_boxes(r['boxes'].cpu(),tr);out={'bottles':[],'labels':[]}
        for box,score,cls in zip(boxes,r['scores'].cpu(),r['labels'].cpu()):
            if float(score)<.3 or box[2]<=box[0] or box[3]<=box[1]:continue
            kind='bottles' if self.mode=='joint' and int(cls)==1 else 'labels'
            out[kind].append({'bbox':box.tolist(),'score':float(score)})
        return out
