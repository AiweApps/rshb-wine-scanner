"""B0 + verified supplemental gallery + generic selector v6, separate HTTP candidate."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import read_json,read_jsonl,verify,seal,sha256

def load():
 from scripts.run_gallery_supplement import load as parent
 from rshb_vine.frozen_candidate_selector import FrozenCandidateSelector
 run=ROOT/'runs/unified-text-v1';protocol=verify(read_json(run/'protocol.json'))
 for path in ['rshb_vine/frozen_candidate_selector.py','config/producer-brand-aliases.json','data/normalized/catalog.jsonl']:
  if sha256(ROOT/path)!=protocol['sources'][path]:raise ValueError('Frozen candidate input changed: '+path)
 report=verify(read_json(run/'combined-gallery/report.json'));validation=verify(read_json(run/'combined-gallery/validation.json'))
 if report['summary']['combined']<=report['summary']['baseline'] or report['summary']['regressions'] or validation['summary']['regressions']:raise ValueError('Candidate quality gate failed')
 app=parent();slugs={r['slug'] for r in app.base.index.references};catalog=[r for r in read_jsonl(ROOT/'data/normalized/catalog.jsonl') if r['slug'] in slugs];signatures=verify(read_json(ROOT/'runs/gallery-variant-source-review-v1/gallery/signatures.json'))
 app.variant_policy=FrozenCandidateSelector(catalog,signatures['signatures']);app.manifest=seal({'kind':'unified-B0-gallery-selector-v1','parent':app.manifest,'development_report':report['checksum'],'validation_report':validation['checksum'],'selector_sha':sha256(ROOT/'rshb_vine/frozen_candidate_selector.py'),'signatures':signatures['checksum'],'source_sha':sha256(__file__),'calibrated':False,'production_selected':False});return app

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8142);args=p.parse_args()
 if args.port in (8085,8112,8138,8141):raise ValueError('Keep previous profiles')
 import uvicorn
 from rshb_vine.api import create_app
 uvicorn.run(create_app(load()),host='127.0.0.1',port=args.port)
