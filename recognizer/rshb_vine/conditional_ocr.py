"""Candidate-directed OCR on original label pixels, with uncalibrated evidence."""
import math
from rshb_vine.io import digest
from rshb_vine.preprocessing import checked_box
from rshb_vine.resolution.identity import IdentityReranker, GRAPES, SUGAR, COLOR, STYLE, values, norm


class ConditionalOCR:
    def __init__(self, catalog, reader, reader_id, *, apply_reranking=False):
        self.catalog = {r['slug']: r for r in catalog if not r.get('excluded_from_retrieval')}
        self.reader = reader
        self.apply_reranking = apply_reranking
        self.reranker = IdentityReranker(list(self.catalog.values()), policy='contradictions_only')
        self.policy_id = digest({'policy': 'candidate-discriminating-ocr-v1', 'reader_id': reader_id,
                                 'catalog': list(self.catalog.values()), 'minimum_ocr_score': .85,
                                 'apply_reranking': apply_reranking})

    def plan(self, retrieval):
        ranked = retrieval['ranked_candidates']
        if len(ranked) < 2:
            return {'run': False, 'reason': 'fewer_than_two_candidates'}
        if any(r['slug'] not in self.catalog for r in ranked):
            return {'run': False, 'reason': 'catalog_snapshot_mismatch'}
        a, b = [self.catalog[r['slug']]['fields'] for r in ranked[:2]]
        fields = {}
        for key, source, aliases in [('grape', 'Сорт винограда', GRAPES),
                                     ('sugar', 'Название вина', SUGAR),
                                     ('color', 'Категория', COLOR),
                                     ('style', 'Название вина', STYLE)]:
            left, right = values(a.get(source, ''), aliases), values(b.get(source, ''), aliases)
            # An absent field is unknown, not evidence of a difference.
            if left and right and left != right:
                fields[key] = [sorted(left), sorted(right)]
        producers = [norm(f.get('Винодельня', '')) for f in (a, b)]
        same_producer = bool(producers[0]) and producers[0] == producers[1]
        if retrieval.get('channel_disagreement') and all(producers) and not same_producer:
            fields['producer'] = [[p] for p in producers]
        run = bool(fields) and (same_producer or retrieval.get('channel_disagreement', False))
        return {'run': run, 'reason': 'candidate_fields_differ' if run else 'no_supported_discriminating_field',
                'candidate_pair': [r['slug'] for r in ranked[:2]], 'discriminating_fields': fields}

    def apply(self, image, bbox, retrieval):
        result = dict(retrieval)
        result.update(slug=None, probability_correct=None,
                      decision='uncertain' if retrieval['ranked_candidates'] else 'unknown')
        plan = self.plan(retrieval)
        trace = {'policy_id': self.policy_id, 'plan': plan, 'performed': False, 'ranking_changed': False}
        result['ocr'] = trace
        if not plan['run']:
            return result
        box = checked_box(bbox, image.size)
        trace['bbox_original'] = box
        trace['coordinate_space'] = 'EXIF-oriented original pixels'
        try:
            observations = self.reader.read(image.crop(box))
            original_observations = []
            for observation in observations:
                if not math.isfinite(observation['score']) or not 0 <= observation['score'] <= 1:
                    raise ValueError('Invalid OCR score')
                converted = dict(observation)
                if observation.get('polygon') is not None:
                    if any(not math.isfinite(float(v)) for point in observation['polygon'] for v in point):
                        raise ValueError('Invalid OCR polygon')
                    converted['polygon_original'] = [[float(x) + box[0], float(y) + box[1]]
                                                      for x, y in observation['polygon']]
                    converted['polygon_coordinate_space'] = 'label crop pixels'
                    converted['polygon_original_coordinate_space'] = 'EXIF-oriented original pixels'
                original_observations.append(converted)
            ranked = self.reranker.rerank(retrieval['ranked_candidates'], observations)
            if {r['slug'] for r in ranked} != {r['slug'] for r in retrieval['ranked_candidates']}:
                raise ValueError('OCR must preserve candidate pool')
        except Exception as exc:
            trace.update(error_type=type(exc).__name__, reason='ocr_stage_failed')
            result['reasons'] = list(retrieval['reasons']) + ['ocr_unavailable']
            return result
        trace.update(performed=True, observations=original_observations,
                     candidate_evidence=[{'slug': r['slug'], 'fields': r['identity_evidence']} for r in ranked],
                     proposed_order=[r['slug'] for r in ranked],
                     limits=['OCR/catalog field trust is provisional, not a calibrated acceptance rule.',
                             'Years and ABV are not inferred from arbitrary numbers or slug spelling.'])
        evidence = {r['slug']: r['identity_evidence'] for r in ranked}
        pair = plan['candidate_pair']
        trace['has_one_sided_field_evidence'] = any(
            {evidence[s][field]['state'] for s in pair} == {'matched', 'contradicted'}
            for field in plan['discriminating_fields'])
        if self.apply_reranking:
            trace['ranking_changed'] = trace['proposed_order'] != [r['slug'] for r in retrieval['ranked_candidates']]
            result['visual_ranked_candidates'] = retrieval['ranked_candidates']
            result['ranked_candidates'] = ranked
            result['best_candidate'] = ranked[0]['slug'] if ranked else None
        result['slug'] = None
        result['probability_correct'] = None
        result['decision'] = 'uncertain' if ranked else 'unknown'
        result['reasons'] = list(retrieval['reasons']) + ['ocr_evidence_uncalibrated']
        if not trace['has_one_sided_field_evidence']:
            result['reasons'].append('distinguishing_text_not_established')
        return result
