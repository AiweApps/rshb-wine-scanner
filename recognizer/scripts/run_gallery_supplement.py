"""Separate B0 profile with verified additional gallery views, unchanged weights."""
from pathlib import Path
import sys,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import read_json,verify,sha256,seal

def load():
 from scripts.run_bottle_rescue import load as load_parent
 from rshb_vine.visual_core import LabelFirstIndex
 run=ROOT/'runs/winecraft-gallery-v1';g=verify(read_json(run/'gallery.json'));r=verify(read_json(run/'corpus/report.json'))
 if r['summary']['response_regressions'] or r['summary']['mapped_regressions']:raise ValueError('Gallery regression gate')
 if sha256(ROOT/'config/reference-supplements.json')!=g['source_supplement_sha']:raise ValueError('Supplement changed')
 if sha256(run/'gallery-vectors.npy')!=g['vectors_sha']:raise ValueError('Gallery checksum')
 base=load_parent();assert base.manifest['checksum']==g['baseline_runtime'];old=base.base.index;v=np.load(run/'gallery-vectors.npy');assert g['encoder_id']==old.encoder_id;assert g['references'][:len(old.references)]==old.references;assert np.array_equal(v[:len(old.vectors)],old.vectors)
 base.base.index=LabelFirstIndex(v,g['references'],g['encoder_id']);base.manifest=seal({'kind':'B0-gallery-supplement-v1','parent':base.manifest,'gallery_checksum':g['checksum'],'corpus_check':r['checksum'],'source_sha':sha256(__file__),'production_selected':False,'calibrated':False});return base
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8141);args=p.parse_args()
 if args.port in (8085,8112,8138):raise ValueError('Keep frozen profiles')
 import uvicorn
 from rshb_vine.api import create_app
 uvicorn.run(create_app(load()),host='127.0.0.1',port=args.port)
