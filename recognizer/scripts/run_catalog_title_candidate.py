"""Single-target catalogue-title challenger over the frozen selected runtime."""
from pathlib import Path
import sys,argparse,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rshb_vine.io import read_json,read_jsonl,verify,seal,sha256

class TitleCandidate:
    def __init__(self,parent):
        from rshb_vine.catalog_constrained_title import CatalogConstrainedTitle
        self.parent=parent;allowed={r['slug'] for r in parent.base.index.references};cat=[r for r in read_jsonl(ROOT/'data/normalized/catalog.jsonl') if r['slug'] in allowed];self.policy=CatalogConstrainedTitle(cat,verify(read_json(ROOT/'runs/gallery-variant-source-review-v1/gallery/signatures.json'))['signatures'])
        val=verify(read_json(ROOT/'runs/catalog-constrained-title-v1/validation/report.json'));dev=verify(read_json(ROOT/'runs/catalog-constrained-title-v1/report.json'))
        if any(x['regressions'] for x in val['summary'].values()) or dev['summary']['comparison_to575']['regressions']:raise ValueError('Regression gate failed')
        self.manifest=seal({'kind':'single-catalog-title-v1','parent':parent.manifest,'validation':val['checksum'],'development':dev['checksum'],'source_sha':sha256(__file__),'selector_sha':sha256(ROOT/'rshb_vine/catalog_constrained_title.py'),'calibrated':False,'production_selected':False})
    def recognize(self,data,roi=None):
        result=self.parent.recognize(data,roi)
        if len(result.get('targets',[]))!=1:return result
        started=time.perf_counter();target=result['targets'][0];ret=target['retrieval'];vt=result.get('variant_text',{});obs=vt.get('observations',[]) if vt.get('performed') else [];pool=ret.get('visual_ranked_candidates',ret['ranked_candidates']);ranked,evidence=self.policy.rerank(copy.deepcopy(pool),copy.deepcopy(obs));ret.update(ranked_candidates=ranked,best_candidate=ranked[0]['slug'] if ranked else None);result.update(ranked_candidates=ranked,best_candidate=ret['best_candidate'],catalog_title=evidence);ms=(time.perf_counter()-started)*1000;result.setdefault('timing_ms',{})['catalog_title']=ms
        if 'total' in result['timing_ms']:result['timing_ms']['total']+=ms
        return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8145);p.add_argument('--startup-receipt',type=Path,required=True);a=p.parse_args()
 if a.port in (8112,8138,8141,8142,8143,8144):raise ValueError('Retain controls')
 from scripts.run_selected_recognition import load
 from rshb_vine.api import create_app
 import uvicorn
 uvicorn.run(create_app(TitleCandidate(load(a.startup_receipt))),host='127.0.0.1',port=a.port)
