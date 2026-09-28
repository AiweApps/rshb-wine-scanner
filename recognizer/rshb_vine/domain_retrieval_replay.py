"""Replace only retrieval embeddings in a frozen, source-bound full-frame trace.

The caller owns source SHA verification and model/gallery identity. No detector,
object gate, crop geometry, target count or OCR is recomputed by this component.
"""
import copy
from rshb_vine.visual_core import View
from rshb_vine.preprocessing import checked_box


def replay(image,baseline,encoder,index,encoder_id,text_policy):
    result=copy.deepcopy(baseline)
    target_ids=[str(t['instance_id']) for t in result.get('targets',[])]
    observations=copy.deepcopy(baseline.get('variant_text',{}).get('observations',[]))
    use_text=baseline.get('variant_text',{}).get('performed',False)
    if use_text and len(target_ids)!=1:
        raise ValueError('Unexpected baseline OCR/target contract')
    for target in result.get('targets',[]):
        views=[View(**v) for v in target['retrieval']['views']]
        images=[image.crop(checked_box(v.bbox,image.size)) for v in views]
        angle=target.get('retrieval_pixel_rotation_ccw',0)
        if angle not in (0,90,270):raise ValueError('Unsupported frozen rotation')
        if angle:images=[im.rotate(angle,expand=True) for im in images]
        retrieval=index.search(encoder.encode(images),views,encoder_id)
        if use_text:
            visual=copy.deepcopy(retrieval['ranked_candidates'])
            ranked,evidence=text_policy.rerank(copy.deepcopy(visual),copy.deepcopy(observations))
            retrieval.update(visual_ranked_candidates=visual,ranked_candidates=ranked,
                             best_candidate=ranked[0]['slug'] if ranked else None)
            result['variant_text']['evidence']=evidence
        target['retrieval']=retrieval
    if len(target_ids)==1:
        retrieval=result['targets'][0]['retrieval']
        result.update(best_candidate=retrieval['best_candidate'],
                      ranked_candidates=retrieval['ranked_candidates'])
        if use_text:result['visual_ranked_candidates']=retrieval['visual_ranked_candidates']
    assert target_ids==[str(t['instance_id']) for t in result.get('targets',[])]
    # Cached old timing is not the new encoder's full HTTP latency.
    result.pop('timing_ms',None)
    result['domain_retrieval_replay']=dict(encoder_id=encoder_id,source='frozen_localization_and_ocr',
                                          full_http_latency_measured=False)
    return result
