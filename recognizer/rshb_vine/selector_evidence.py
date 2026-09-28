"""Lossless receipt adapter for the fixed-pool selector experiment; no decisions."""
from copy import deepcopy
from rshb_vine.io import digest
from rshb_vine.resolution.identity import norm


def target_evidence(result, target, receipt_sha):
    sid = str(target['instance_id'])
    packet = result.get('variant_text', {}) if len(result.get('targets', [])) == 1 else next(
        (p for p in result.get('instance_text', {}).get('targets', []) if str(p['instance_id']) == sid), {})
    primary_box = packet.get('bbox_original') or packet.get('bbox')
    reads = []
    conditional = result.get('conditional_label_ocr', {})
    for i, read in enumerate(conditional.get('reads', [])):
        if str(read.get('instance_id')) == sid:
            reads.append({'path':f'conditional_label_ocr.reads.{i}', 'reader':conditional.get('reader'),
                          'bbox':read.get('bbox'), 'rotation':read.get('rotation'),
                          'coordinate_space':'individual_rotated_crop_pixels', 'observations':read.get('observations', [])})
    alternate = result.get('alternative_ocr', {})
    if len(result.get('targets', [])) == 1 and alternate.get('observations') is not None:
        reads.append({'path':'alternative_ocr', 'reader':alternate.get('reader'), 'bbox':alternate.get('bbox'),
                      'rotation':target.get('retrieval_pixel_rotation_ccw',0),
                      'coordinate_space':'individual_rotated_crop_pixels', 'observations':alternate['observations']})
    observations = []
    for i, line in enumerate(packet.get('observations', [])):
        matches = []
        # Equal text/score is a candidate association, not proof of the same physical reading.
        for read in reads:
            for j, raw in enumerate(read['observations']):
                if raw['raw_text'] == line['raw_text'] and raw['score'] == line['score']:
                    matches.append({k:v for k,v in read.items() if k!='observations'} | {'observation_index':j, 'polygon':raw.get('polygon')})
        observations.append({'raw':deepcopy(line), 'merged_index':i,
                             'source_receipt_sha':receipt_sha,
                             'lineage_candidates':matches,
                             'lineage_status':'candidate_match_not_proven' if matches else 'primary_packet_reader_requires_runtime_binding',
                             'original_packet_bbox':primary_box,
                             'original_packet_rotation':packet.get('rotation_ccw'),
                             'original_coordinate_space':packet.get('polygon_coordinate_space'),
                             'text_group':digest([sid,norm(line['raw_text'])]),
                             'group_semantics':'same normalized text, not independent evidence; no spatial equivalence asserted'})
    ret = target['retrieval']
    return {'instance_id':sid,
            'geometry':{k:deepcopy(v) for k,v in target.items() if k!='retrieval'},
            'fixed_pool':deepcopy(ret.get('candidate_union', [])),
            'returned_candidate_rows':deepcopy(ret.get('ranked_candidates', [])),
            'returned_rows_limit':'contain prior selection evidence/order; not raw visual tie-break',
            'visual_ranked_candidates':deepcopy(ret.get('visual_ranked_candidates', [])),
            'channel_top20':deepcopy(ret.get('channel_top20', {})),
            'visual_model_provenance':'must resolve through bound receipt lineage, do not assume B0',
            'views':deepcopy(ret.get('views', [])),
            'ocr_packet':deepcopy(packet), 'supplemental_reads':reads, 'observations':observations}
