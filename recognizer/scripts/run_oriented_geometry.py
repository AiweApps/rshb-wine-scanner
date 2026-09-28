"""Second bounded geometry hypothesis: consistent detector/encoder orientation."""
import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_geometry_loop import load as load_geometry,replay
from rshb_vine.oriented_geometry import OrientedGeometryPipeline
from rshb_vine.io import seal,sha256

def load(mode='upright'):
    parent=load_geometry('rotate')
    pipe=OrientedGeometryPipeline(parent.base,ROOT/'runs/crop-label-v1/verifier/verifier.json')
    pipe.manifest=seal({**{k:v for k,v in parent.manifest.items() if k!='checksum'},'mode':'upright','orientation_policy':'Same detector-chosen90/270 used for label detection and encoder crops; never choose by catalog score','source_sha256':{p:sha256(ROOT/p) for p in ['rshb_vine/geometry_loop.py','rshb_vine/oriented_geometry.py','scripts/run_geometry_loop.py','scripts/run_oriented_geometry.py']}})
    return pipe
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['replay','serve']);p.add_argument('--cohort',choices=['scenes','packshots','all'],default='scenes');p.add_argument('--port',type=int,default=8090);a=p.parse_args()
    if a.action=='replay':replay('upright',a.cohort,'geometry-upright-v1',load)
    else:
        import uvicorn
        from rshb_vine.api import create_app
        uvicorn.run(create_app(load()),host='127.0.0.1',port=a.port)
