"""Run separate joint ownership/per-instance text development candidate."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import read_json,read_jsonl,verify,seal,sha256

def load():
 from scripts.run_unified_candidate import load as parent
 from rshb_vine.instance_text_profile import InstanceTextProfile,OwnershipDetector
 from rshb_vine.product_name_selector import ProductNameSelector
 app=parent();allowed={r['slug'] for r in app.base.index.references};cat=[r for r in read_jsonl(ROOT/'data/normalized/catalog.jsonl') if r['slug'] in allowed]
 app.variant_policy=ProductNameSelector(cat,verify(read_json(ROOT/'runs/gallery-variant-source-review-v1/gallery/signatures.json'))['signatures'])
 app.detector.detector=OwnershipDetector(app.detector.detector);app.__class__=InstanceTextProfile
 app.manifest=seal({'kind':'joint-instance-text-v1','parent':app.manifest,'sources':{p:sha256(ROOT/p) for p in ['rshb_vine/instance_text_profile.py','rshb_vine/product_name_selector.py','rshb_vine/label_region_ownership.py','scripts/run_instance_text_candidate.py']},'production_selected':False,'calibrated':False});return app
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8143);a=p.parse_args()
 if a.port in (8112,8138,8141,8142):raise ValueError('Preserve previous profiles')
 import uvicorn
 from rshb_vine.api import create_app
 uvicorn.run(create_app(load()),host='127.0.0.1',port=a.port)
