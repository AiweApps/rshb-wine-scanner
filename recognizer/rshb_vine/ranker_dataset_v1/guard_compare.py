"""Rebuild the actual frozen receipt evidence for paired old/new resolver checks."""
from pathlib import Path

from rshb_vine.candidate_listwise import _representative
from rshb_vine.candidate_discriminators import enrich_candidate_features
from rshb_vine.io import digest,read_json,verify
from rshb_vine.learned_selection_features import build_candidate_features
from rshb_vine.product_evidence_resolution import ProductEvidenceResolution
from rshb_vine.product_selection import ProductSelection
from rshb_vine.ranker_dataset_v1.model import rank


def _payload(root,row):
    path=root/row['receipt_path']
    raw_source=verify(read_json(path))
    sid=str(row['instance_id'])
    if row['source'] in ('historical_open_saved','historical_error42'):
        uid=row['query_id'].split('/target/')[0]
        source_id=uid.split('/',1)[1]
        visual=raw_source
        evidence=verify(read_json(root/'runs/evidence-selector-v1/evidence'/(digest(source_id)+'.json')))
        raw=next(t for t in visual['targets'] if str(t['instance_id'])==sid)
        packet=next(t for t in evidence['targets'] if str(t['instance_id'])==sid)
        control=next(t['best_candidate'] for t in evidence['S0_control_output']['targets']
                     if str(t['instance_id'])==sid)
        return {'control_slug':control,'raw_visual':raw,
                'observations':packet['ocr_packet'].get('observations',[]),
                'control_candidates':packet['returned_candidate_rows']}
    if raw_source.get('kind')=='recognition-single-image-replay-v1':
        result=raw_source['result']
    elif raw_source.get('result') is not None:
        result=raw_source['result']
    else:
        raise ValueError('Unsupported actual runtime receipt kind')
    packet=next(t for t in result['product_identity_evidence']['targets']
                if str(t['instance_id'])==sid)
    return {'control_slug':packet['control_slug'],'raw_visual':packet['raw_visual'],
            'observations':packet['ocr_observations'],
            'control_candidates':packet['control_candidates']}


def _proposal(row,ranking,enhanced,model):
    candidates={c['candidate_id']:c for c in enhanced['candidates']}
    if {r['candidate_id'] for r in ranking}!=set(candidates):
        raise ValueError('New ranker changed actual candidate pool')
    input_index={c['candidate_id']:i for i,c in enumerate(enhanced['candidates'])}
    ranked=[]
    for position,item in enumerate(ranking,1):
        c=candidates[item['candidate_id']]
        representative,source=_representative(c)
        if item['product_id']!=c['product_id'] or representative not in c['card_slugs']:
            raise ValueError('New ranker changed candidate/card identity')
        ranked.append({'rank':position,'input_pool_index':input_index[item['candidate_id']],
            'candidate_id':item['candidate_id'],'product_id':c['product_id'],
            'identity_status':c['identity_status'],'card_slugs':list(c['card_slugs']),
            'representative_slug':representative,
            'representative_source':source,'score':item['score'],
            'is_control_proposal':bool(c['features']['control.is_proposal'])})
    winner=ranked[0] if ranked else None
    return {'schema_version':'ranker-real-runtime-proposal-v1',
        'model_checksum':model['checksum'],'ranked_candidates':ranked,
        'candidate_id':winner['candidate_id'] if winner else None,
        'product_id':winner['product_id'] if winner else None,
        'representative_slug':winner['representative_slug'] if winner else None,
        'best_candidate':winner['representative_slug'] if winner else None,
        'representative_source':winner['representative_source'] if winner else None,
        'score':winner['score'] if winner else None,
        'score_margin':ranked[0]['score']-ranked[1]['score'] if len(ranked)>1 else None,
        'exact_slug':None,'probability':None,'release_admitted':False,
        'reason':'single_admitted_real_runtime_listwise_ranker'}


def compare_one(root,row,model,old):
    root=Path(root)
    payload=_payload(root,row)
    if (payload['control_slug']!=row['control_slug']
            or payload['raw_visual']['views']!=row['views']
            or payload['raw_visual']['rotation_ccw']!=row['rotation_ccw']):
        raise ValueError('Receipt/compact ranking task differs')
    base=build_candidate_features(control_slug=payload['control_slug'],
        raw_visual=payload['raw_visual'],observations=payload['observations'],
        cards=old.registry.cards,claims=old.registry.claims,
        control_candidates=payload['control_candidates'])
    enhanced=enrich_candidate_features(base)
    if digest(enhanced)!=row['feature_digest']:
        raise ValueError('103-feature runtime parity failed')
    old_selection=old.select(**payload)
    if digest(old_selection['features'])!=row['feature_digest']:
        raise ValueError('Old103 runtime feature parity failed')
    ranking=rank(row,model)
    proposal=_proposal(row,ranking,enhanced,model)
    resolved=ProductEvidenceResolution(root).resolve(base,proposal)
    return {'query_id':row['query_id'],'old103_raw':old_selection['proposal'].get('existing_evidence_resolution',{}).get('proposal'),
        'old103_guarded':old_selection['proposal'].get('representative_slug'),
        'old103_saved_public':row['selected_slug'],
        'new_raw':proposal['representative_slug'],'new_guarded':resolved['representative_slug'],
        'new_guard_blocked':resolved['existing_evidence_resolution']['blocked'],
        'new_guard_reasons':resolved['existing_evidence_resolution']['reasons'],
        'actual_pool_candidates':len(ranking),
        'old103_parity':old_selection['proposal'].get('representative_slug')==row['selected_slug'],
        'public_HTTP_parity_status':'not_run'}
