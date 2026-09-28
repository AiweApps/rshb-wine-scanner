"""Frozen label-first gallery and manual-label development replay. Never fit/locked."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rshb_vine.io import digest, local_path, read_json, seal, sha256, verify, write_json
from rshb_vine.encoder_artifact import load_encoder, encoder_spec
from rshb_vine.preprocessing import decode, checked_box
from rshb_vine.visual_core import LabelFirstIndex, prepare_views, validate_vectors

ROOT=Path(__file__).resolve().parents[1]


def encoder_setup():
    return load_encoder(ROOT)


def build(data, output, encoder_directory=None):
    allocation=verify(read_json(data/'allocation.json'))
    if output.exists():raise FileExistsError('Immutable gallery already exists')
    enc_id,encoder=load_encoder(ROOT, encoder_directory)
    refs=[r for r in allocation['records'] if r['source']=='rshb_gallery_reference' and r.get('owner_pool','active')=='active']
    signature=seal({'allocation_checksum':allocation['checksum'],'encoder_id':enc_id,
                    'runner_sha256':sha256(__file__),'core_sha256':sha256(ROOT/'rshb_vine/visual_core.py'),
                    'gallery_policy':'canonical_context_and_owner_complete_label_no_augmentation','references':len(refs)})
    staging=output.with_name(output.name+'.building')
    if staging.exists():
        if verify(read_json(staging/'build.json'))!=signature:raise ValueError('Stale partial gallery')
    else:write_json(staging/'build.json',signature)
    vectors=[];metadata=[];batch_images=[];batch_jobs=[]
    def flush():
        if not batch_jobs:return
        encoded=encoder.encode(batch_images);validate_vectors(encoded,len(batch_images));pos=0
        for r,views in batch_jobs:
            n=len(views);record=seal({'signature':signature['checksum'],'annotation_id':r['annotation_id'],
                                    'views':[dict(asdict(v),slug=r['known_slug'],image_sha256=r['image_sha256']) for v in views],
                                    'vectors':encoded[pos:pos+n].tolist()})
            write_json(staging/'records'/(digest(r['annotation_id'])+'.json'),record);pos+=n
        batch_images.clear();batch_jobs.clear()
    for i,r in enumerate(refs):
        path=local_path(ROOT,r['image_path'])
        if sha256(path)!=r['image_sha256']:raise ValueError('Gallery source changed')
        dest=staging/'records'/(digest(r['annotation_id'])+'.json')
        if not dest.exists():
            image,_=decode(path.read_bytes(),max_pixels=36_000_000)
            views,images=prepare_views(image,approved_label=r['approved_front_label'])
            batch_images.extend(images);batch_jobs.append((r,views))
            if len(batch_images)>=16:flush()
        if (i+1)%100==0:print('gallery',i+1,'/',len(refs),flush=True)
    flush()
    for r in refs:
        record=verify(read_json(staging/'records'/(digest(r['annotation_id'])+'.json')))
        if record['signature']!=signature['checksum']:raise ValueError('Wrong gallery checkpoint')
        vectors.extend(record['vectors']);metadata.extend(record['views'])
    arr=np.asarray(vectors,dtype='float32');validate_vectors(arr,len(metadata));np.save(staging/'vectors.npy',arr)
    write_json(staging/'references.json',metadata)
    release=seal({'build':signature,'encoder_id':enc_id,'references':len(refs),'view_counts':dict(Counter(r['kind'] for r in metadata)),
                  'files':{p:sha256(staging/p) for p in ('vectors.npy','references.json')},
                  'weights_fit':encoder_spec(ROOT, encoder_directory)['selected']=='B2'})
    write_json(staging/'manifest.json',release);staging.rename(output);print(release['view_counts'],flush=True)


def query_manifest(allocation):
    hold_path = ROOT / 'data/evaluation/label-holds.json'
    holds = verify(read_json(hold_path)) if hold_path.exists() else None
    held_sha = {r['image_sha256'] for r in holds['entries'] if r['status']=='needs_review'} if holds else set()
    by_sha={}
    for r in allocation['records']:
        if r['source']!='rshb_gallery_reference':by_sha.setdefault(r['image_sha256'],[]).append(r)
    records=[];excluded=[]
    # Only already-open development, never proposal validation/calibration/final.
    for split in ('dev','validation'):
        source=f'data/evaluation/reset-v1/{split}/manifest.json';m=verify(read_json(ROOT/source))
        for q in m['queries']:
            if q['image_sha256'] in held_sha:
                excluded.append({'query_id':q['query_id'],'dataset':'kultovo_open','reason':'current_source_identity_hold'});continue
            matches=by_sha.get(q['image_sha256'],[])
            r=next((r for r in matches if r.get('known_slug')==q['ground_truth_slug']),None)
            if not r or r.get('owner_pool','active')!='active' or r.get('identity_hold'):
                excluded.append({'query_id':q['query_id'],'dataset':'kultovo_open','reason':'missing_active_exact_binding_or_hold'});continue
            if r['allocation']!='development_no_fit':raise ValueError('Refusing non-development query')
            old=verify(read_json(ROOT/f'data/evaluation/reset-v1/{split}/records'/(q['query_id']+'.json')))['results']['FAST_V0']['result']
            baseline=verify(read_json(ROOT/'runs/multiview-v1/queries'/(q['query_id']+'.json')))
            records.append({'query_id':q['query_id'],'dataset':'kultovo_open','image_path':q['image_path'],
                            'image_sha256':q['image_sha256'],'ground_truth_slug':q['ground_truth_slug'],
                            'target_bbox':old['trace']['target_bbox'],'approved_label':r['approved_front_label'],
                            'correlation_group':q.get('split_group_id') or q.get('family_id'),
                            'B0_cached_rank':baseline['after_rank'],'B0_cached_top1':baseline['after_top1'],
                            'source_manifest_checksum':m['checksum']})
    source='data/rshb-query-admission-v1/annotation-v2/manifest.json';m=verify(read_json(ROOT/source))
    for q in m['queries']:
        if q['image_sha256'] in held_sha or q.get('source_image_sha256') in held_sha:
            excluded.append({'query_id':q['query_id'],'dataset':'rshb_v2','reason':'current_source_identity_hold'});continue
        if not q['exact_evaluation_eligible']:
            excluded.append({'query_id':q['query_id'],'dataset':'rshb_v2','reason':'identity_hold'});continue
        r=next((r for r in by_sha.get(q['source_image_sha256'],[]) if r.get('known_slug')==q['ground_truth_slug']),None)
        if not r or r.get('owner_pool','active')!='active':
            excluded.append({'query_id':q['query_id'],'dataset':'rshb_v2','reason':'missing_active_binding'});continue
        if r['allocation']!='development_no_fit':raise ValueError('Refusing non-development query')
        label=r['approved_front_label']
        if label and label.get('bbox'):
            label=dict(label,bbox=[v*q['input_size'][j%2]/q['source_size'][j%2] for j,v in enumerate(label['bbox'])])
        records.append({'query_id':q['query_id'],'dataset':'rshb_v2','image_path':q['image_path'],
                        'image_sha256':q['image_sha256'],'ground_truth_slug':q['ground_truth_slug'],
                        'target_bbox':q['target_bbox'],'approved_label':label,'correlation_group':q['photograph_id'],
                        'correlation_risk_ids':q.get('correlation_risk_ids',[]),'source_manifest_checksum':m['checksum']})
    return records,excluded


def compare(data, gallery, output):
    allocation=verify(read_json(data/'allocation.json'));release=verify(read_json(gallery/'manifest.json'))
    if release['build']['allocation_checksum']!=allocation['checksum']:raise ValueError('Wrong gallery data')
    for name,value in release['files'].items():
        if sha256(gallery/name)!=value:raise ValueError('Gallery changed')
    enc_id,encoder=encoder_setup()
    if enc_id!=release['encoder_id']:raise ValueError('Wrong encoder')
    index=LabelFirstIndex(np.load(gallery/'vectors.npy'),read_json(gallery/'references.json'),enc_id)
    queries,excluded=query_manifest(allocation)
    signature=seal({'allocation_checksum':allocation['checksum'],'gallery_checksum':release['checksum'],
                    'runner_sha256':sha256(__file__),'core_sha256':sha256(ROOT/'rshb_vine/visual_core.py'),
                    'mode':'manual_label_oracle_development_not_automatic_endpoint','queries':queries,'excluded':excluded})
    if output.exists():
        if verify(read_json(output/'protocol.json'))!=signature:raise ValueError('Stale development comparison')
    else:write_json(output/'protocol.json',signature)
    # Offline B0 replay for revised RSHB v2; no parent Pipeline, resolver, matcher.
    from rshb_vine.models import OCR
    from rshb_vine.preprocessing import label_box
    from rshb_vine.retrieval import visual_search
    ocr=None;base=ROOT/'artifacts/baseline-v0/data/index/source-research-v2'
    base_manifest=verify(read_json(base/'manifest.json'))
    for name in ('vectors.npy','references.json'):
        if sha256(base/name)!=base_manifest['files'][name]:raise ValueError('B0 base changed')
    multi=ROOT/'runs/multiview-v1';mr=verify(read_json(multi/'gallery-release.json'))
    for name,key in [('added-vectors.npy','added_vectors_sha256'),('added-references.json','added_references_sha256')]:
        if sha256(multi/name)!=mr[key]:raise ValueError('B0 augmentation changed')
    b0vectors=np.concatenate([np.load(base/'vectors.npy'),np.load(multi/'added-vectors.npy')]);b0refs=read_json(base/'references.json')+read_json(multi/'added-references.json')
    results=[]
    def rank(rows,slug):return next((i for i,r in enumerate(rows,1) if r['slug']==slug),None)
    for i,q in enumerate(queries):
        dest=output/'records'/(q['query_id']+'.json')
        if dest.exists():record=verify(read_json(dest))
        else:
            start=time.perf_counter();path=local_path(ROOT,q['image_path'])
            if sha256(path)!=q['image_sha256']:raise ValueError('Query source changed')
            image,_=decode(path.read_bytes())
            views,images=prepare_views(image,q['target_bbox'],q['approved_label'])
            vectors=encoder.encode(images);result=index.search(vectors,views,enc_id)
            b1_ms=(time.perf_counter()-start)*1000
            if q['dataset']=='rshb_v2':
                if ocr is None:ocr=OCR(ROOT)
                roi=images[0];box=label_box(ocr.read(roi),roi.size)
                old_vectors=[vectors[0]]
                if box:old_vectors.append(encoder.encode([roi.crop(box)])[0])
                old=visual_search(b0vectors,b0refs,np.asarray(old_vectors),20)
                before_rank=rank(old,q['ground_truth_slug']);before_top1=old[0]['slug']
            else:before_rank=q['B0_cached_rank'];before_top1=q['B0_cached_top1']
            record=seal({'protocol_checksum':signature['checksum'],'query_id':q['query_id'],'dataset':q['dataset'],
                         'correlation_group':q['correlation_group'],'B0_rank':before_rank,'B0_top1':before_top1,
                         'B1_rank':rank(result['ranked_candidates'],q['ground_truth_slug']),
                         'context_rank':rank(result['channel_top20']['context'],q['ground_truth_slug']),
                         'label_rank':rank(result['channel_top20']['front_label'],q['ground_truth_slug']),
                         'union_hit':q['ground_truth_slug'] in result['candidate_union'],
                         'label_available':len(views)==2,'B1_decode_encode_search_ms':b1_ms,'result':result})
            write_json(dest,record)
        if record['protocol_checksum']!=signature['checksum']:raise ValueError('Wrong query checkpoint')
        results.append(record)
        if (i+1)%25==0:print('queries',i+1,'/',len(queries),flush=True)
    summaries={}
    for dataset in ('kultovo_open','rshb_v2'):
        rows=[r for r in results if r['dataset']==dataset]
        summaries[dataset]={'n':len(rows),'label_available':sum(r['label_available'] for r in rows),
             'hits':{field:{str(k):sum(r[field] is not None and r[field]<=k for r in rows) for k in (1,5,20)}
                     for field in ('B0_rank','B1_rank','context_rank','label_rank')},
             'candidate_union_hits':sum(r['union_hit'] for r in rows),
             'fixed':sum(r['B0_rank']!=1 and r['B1_rank']==1 for r in rows),
             'regressed':sum(r['B0_rank']==1 and r['B1_rank']!=1 for r in rows),
             'disagreements':sum(r['result']['channel_disagreement'] for r in rows)}
    report=seal({'protocol_checksum':signature['checksum'],'datasets':summaries,'excluded_count':len(excluded),
                 'claims':['development only','manual label oracle, not automatic localization','no fit','no final-test access',
                           'B0 cached on identical Kultovo queries; fresh B0 replay on revised eligible RSHB v2',
                           'union hit is coverage of up to40 slugs, not Recall@20','latency is offline path, not endpoint SLA']})
    write_json(output/'report.json',report);print(summaries,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','compare'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--gallery',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder',type=Path,help='Completed metric trial encoder directory; build only')
    a=p.parse_args()
    if a.action=='build':build(a.data,a.output,a.encoder)
    else:
        if a.encoder:p.error('--encoder is build-only; adapted paired comparison requires its own fixed protocol')
        if not a.gallery:p.error('--gallery required')
        compare(a.data,a.gallery,a.output)
