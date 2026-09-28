"""Fixed crop-stage comparison on open real queries; optional isolated HTTP."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import *
from rshb_vine.label_first_release import load_bundle
from rshb_vine.bottle_label_localizer import BottleLabelLocalizer
from rshb_vine.proposal_verifier import ProposalVerifierPipeline
from scripts.run_label_first import query_manifest

def load(mode):
 base=load_bundle(ROOT,ROOT/'runs/real-detector-v2/candidate-runtime.json','mps','mps')
 if mode in ('crop_label','crop_structure'):base.detector=BottleLabelLocalizer.from_artifacts(ROOT,ROOT/'runs/crop-label-v1/label-model','mps',merge_parent_labels=base.detector.merge_parent_labels)
 from rshb_vine.crop_stage_pipeline import CropStagePipeline
 cls=CropStagePipeline if mode=='crop_structure' else ProposalVerifierPipeline
 pipe=cls(base,ROOT/'runs/crop-label-v1/verifier/verifier.json');pipe.manifest=seal({**{k:v for k,v in pipe.manifest.items() if k!='checksum'},'stage_mode':mode,'actual_localizer_id':base.detector.model_id,'source_sha256':{p:sha256(ROOT/p) for p in ['scripts/run_crop_stage.py','rshb_vine/proposal_verifier.py','rshb_vine/crop_stage_pipeline.py']}});return pipe

def replay(mode):
 pipe=load(mode);allocation=verify(read_json(ROOT/'data/label-real-v2/allocation.json'));qs,excluded=query_manifest(allocation);root=ROOT/'runs/crop-label-v1'/mode
 prot=seal(dict(runtime=pipe.manifest,queries=qs,excluded=excluded,purpose='Fixed open426 comparison; no threshold or weight changes after replay; closed20 unused'))
 if (root/'protocol.json').exists():assert verify(read_json(root/'protocol.json'))==prot
 else:write_json(root/'protocol.json',prot)
 cached={}
 for n,q in enumerate(qs):
  name=digest([q['dataset'],q['query_id']])+'.json';dest=root/'records'/name
  if dest.exists():continue
  p=ROOT/q['image_path'];assert sha256(p)==q['image_sha256']
  if q['image_sha256'] not in cached:cached[q['image_sha256']]=pipe.recognize(p.read_bytes())
  auto={k:v for k,v in cached[q['image_sha256']].items() if k!='structure_only'};rank=next((i for i,r in enumerate(auto['ranked_candidates'],1) if r['slug']==q['ground_truth_slug']),None)
  write_json(dest,seal(dict(query_id=q['query_id'],dataset=q['dataset'],ground_truth_slug=q['ground_truth_slug'],automatic_rank=rank,automatic=auto,protocol_checksum=prot['checksum'])))
  if (n+1)%50==0:print(mode,n+1,'/',len(qs),flush=True)
 cases={r['query_id']:r['visible_bottle_case'] for r in verify(read_json(ROOT/'runs/label-fullframe-audit-v1/bottle-cases.json'))['records']};rs=[verify(read_json(p)) for p in (root/'records').glob('*.json')];summary={}
 for cohort in ['kultovo_open','single_main_bottle','multiple_visible_bottles']:
  selected=[r for r in rs if (r['dataset'] if r['dataset']=='kultovo_open' else cases[r['query_id']])==cohort]
  summary[cohort]=dict(n=len(selected),hits={str(k):sum(r['automatic_rank'] is not None and r['automatic_rank']<=k for r in selected) for k in [1,5,20]},ambiguous=sum(r['automatic']['decision']=='ambiguous_target' for r in selected),insufficient=sum(r['automatic']['decision']=='insufficient_evidence' for r in selected),accepted=sum(r['automatic'].get('slug') is not None for r in selected))
 write_json(root/'report.json',seal(dict(summary=summary)));print(mode,summary,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['replay','serve']);p.add_argument('--mode',choices=['gate_only','crop_label','crop_structure'],required=True);p.add_argument('--port',type=int,default=8089);a=p.parse_args()
 if a.action=='replay':replay(a.mode)
 else:
  import uvicorn
  from rshb_vine.api import create_app
  uvicorn.run(create_app(load(a.mode)),host='127.0.0.1',port=a.port)
