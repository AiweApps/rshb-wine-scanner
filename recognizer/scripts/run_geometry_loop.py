"""Geometry-only scene screen then packshot/HTTP verification of selected policy."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import *
from rshb_vine.geometry_loop import GeometryLocalizer,GeometryPipeline
from scripts.run_crop_stage import load as load_parent

def load(mode):
    parent=load_parent('crop_structure');base=parent.base
    base.detector=GeometryLocalizer(base.detector,mode)
    pipe=GeometryPipeline(base,ROOT/'runs/crop-label-v1/verifier/verifier.json')
    pipe.manifest=seal({**{k:v for k,v in pipe.manifest.items() if k!='checksum'},'mode':mode,'localizer_id':base.detector.model_id,'source_sha256':{p:sha256(ROOT/p) for p in ['scripts/run_geometry_loop.py','rshb_vine/geometry_loop.py']}})
    return pipe

def replay(mode,cohort,experiment="geometry-loop-v1",loader=load):
    plan=verify(read_json(ROOT/'runs'/experiment/'plan.json'));pipe=loader(mode);parent=verify(read_json(ROOT/'runs/crop-label-v1/crop_structure/protocol.json'));qs=parent['queries'];queries=[q for q in qs if cohort=='all' or (q['dataset']=='rshb_v2')==(cohort=='scenes')];out=ROOT/'runs'/experiment/mode
    proto=seal(dict(plan_checksum=plan['checksum'],runtime=pipe.manifest,queries=qs,purpose='Open development only. All weights/gallery/ranking frozen. No closed synthetic20 access.'))
    if (out/'protocol.json').exists():assert verify(read_json(out/'protocol.json'))==proto
    else:write_json(out/'protocol.json',proto)
    cache={}
    for n,q in enumerate(queries):
        dest=out/'records'/(digest([q['dataset'],q['query_id']])+'.json')
        if dest.exists():continue
        p=ROOT/q['image_path'];assert sha256(p)==q['image_sha256']
        if q['image_sha256'] not in cache:cache[q['image_sha256']]=pipe.recognize(p.read_bytes())
        auto=cache[q['image_sha256']];rank=next((i for i,c in enumerate(auto.get('ranked_candidates',[]),1) if c['slug']==q['ground_truth_slug']),None)
        write_json(dest,seal(dict(query_id=q['query_id'],dataset=q['dataset'],ground_truth_slug=q['ground_truth_slug'],automatic_rank=rank,automatic=auto,protocol_checksum=proto['checksum'])))
        if (n+1)%10==0:print(mode,cohort,n+1,'/',len(queries),flush=True)
    cases={r['query_id']:r['visible_bottle_case'] for r in verify(read_json(ROOT/'runs/label-fullframe-audit-v1/bottle-cases.json'))['records']};rs=[verify(read_json(p)) for p in (out/'records').glob('*.json')];summary={}
    for name in ['kultovo_open','single_main_bottle','multiple_visible_bottles']:
        rows=[r for r in rs if (r['dataset'] if r['dataset']=='kultovo_open' else cases[r['query_id']])==name]
        summary[name]=dict(n=len(rows),top1=sum(r['automatic_rank']==1 for r in rows),top20=sum(r['automatic_rank'] is not None and r['automatic_rank']<=20 for r in rows),abstain=sum(r['automatic']['decision']=='insufficient_evidence' for r in rows),ambiguous=sum(r['automatic']['decision']=='ambiguous_target' for r in rows))
    write_json(out/f'report-{cohort}.json',seal(dict(summary=summary)));print(summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['replay','serve']);p.add_argument('--mode',required=True,choices=['rotate','resolution960','combined']);p.add_argument('--cohort',choices=['scenes','packshots','all'],default='scenes');p.add_argument('--port',type=int,default=8090);a=p.parse_args()
    if a.action=='replay':replay(a.mode,a.cohort)
    else:
        import uvicorn
        from rshb_vine.api import create_app
        uvicorn.run(create_app(load(a.mode)),host='127.0.0.1',port=a.port)
