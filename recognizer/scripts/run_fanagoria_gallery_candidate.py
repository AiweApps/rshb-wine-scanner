"""Extra source-confirmed Fanagoria view over the existing title candidate."""
from pathlib import Path
import sys,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import read_json,verify,seal,sha256

def load(receipt):
 from scripts.run_selected_recognition import load as load_parent
 from scripts.run_catalog_title_candidate import TitleCandidate
 from rshb_vine.visual_core import LabelFirstIndex
 run=ROOT/'runs/fanagoria-gallery-v1';g=verify(read_json(run/'gallery.json'));check=verify(read_json(run/'common/report.json'));admission=verify(read_json(run/'admission.json'))
 if check['summary']['regressions'] or check['summary']['after']<=check['summary']['before']:raise ValueError('Gallery gain gate failed')
 if sha256(run/'vectors.npy')!=g['vectors_sha']:raise ValueError('Gallery vectors changed')
 parent=load_parent(receipt);old=parent.base.index;vectors=np.load(run/'vectors.npy');assert g['encoder_id']==old.encoder_id;assert g['references'][:len(old.references)]==old.references;assert np.array_equal(vectors[:len(old.vectors)],old.vectors)
 parent.base.index=LabelFirstIndex(vectors,g['references'],g['encoder_id']);parent.manifest=seal({'kind':'fanagoria-extra-gallery-v1','parent':parent.manifest,'gallery':g['checksum'],'admission':admission['checksum'],'common_check':check['checksum'],'runner_sha':sha256(__file__),'calibrated':False})
 return TitleCandidate(parent)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8146);p.add_argument('--startup-receipt',type=Path,required=True);a=p.parse_args()
 if a.port in (8112,8138,8141,8142,8143,8144,8145):raise ValueError('Retain earlier profiles')
 import uvicorn
 from rshb_vine.api import create_app
 uvicorn.run(create_app(load(a.startup_receipt)),host='127.0.0.1',port=a.port)
