"""Selected candidate with startup warm-up before the HTTP server becomes ready."""
from pathlib import Path
import sys,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import read_json,verify,seal,write_json,sha256

def load(receipt):
 from scripts.run_instance_text_candidate import load as candidate
 decision=verify(read_json(ROOT/'runs/joint-instance-text-v1/decision.json'))
 for path,expected in decision['sources'].items():
  if sha256(ROOT/path)!=expected:raise ValueError('Selected source changed: '+path)
 started=time.perf_counter();app=candidate();loaded=time.perf_counter()
 ref=read_json(ROOT/'config/reference-supplements.json')['references'][0]
 if sha256(ROOT/ref['image_path'])!=ref['image_sha256']:raise ValueError('Warmup reference changed')
 # An existing gallery reference is used only for initialization, never scoring.
 for _ in range(2):
  result=app.recognize((ROOT/ref['image_path']).read_bytes())
  if result.get('decision')=='invalid_image':raise ValueError('Warmup image failed decode')
 elapsed=time.perf_counter()-started
 app.manifest=seal({'kind':'selected-instance-recognition-warmed-v1','parent':app.manifest,'selected_decision':decision['checksum'],'runner_sha':sha256(__file__),'startup_policy':'two gallery-reference queries before binding HTTP','calibrated':False})
 write_json(receipt,seal({'snapshot':app.manifest['checksum'],'load_seconds':loaded-started,'startup_seconds':elapsed,'warmup_source_sha':ref['image_sha256'],'warmup_runs':2,'quality_evaluation':False}))
 return app

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8144);p.add_argument('--startup-receipt',type=Path,required=True);a=p.parse_args()
 if a.port in (8112,8138,8141,8142,8143):raise ValueError('Retain existing profiles')
 import uvicorn
 from rshb_vine.api import create_app
 uvicorn.run(create_app(load(a.startup_receipt)),host='127.0.0.1',port=a.port)
