"""Typed label-first visual retrieval. No OCR, resolver, or local matcher imports."""
from dataclasses import asdict, dataclass

import numpy as np
from rshb_vine.preprocessing import checked_box


@dataclass(frozen=True)
class View:
    kind: str
    bbox: list
    source: str
    complete: bool | None
    original_size: list
    cropper_version: str = 'owner-front-label-v1'
    coordinate_space: str = 'EXIF-oriented original pixels'


def prepare_views(image, context_bbox=None, approved_label=None):
    """Keep independent context and complete label, cropped from original pixels."""
    context = checked_box(context_bbox or [0, 0, *image.size], image.size)
    views = [View('context', context, 'target_roi' if context_bbox else 'full_frame', True, list(image.size))]
    if approved_label and approved_label.get('visibility') in ('complete', 'partial'):
        bbox = checked_box(approved_label['bbox'], image.size)
        complete = approved_label['visibility'] == 'complete'
        views.append(View('front_label' if complete else 'partial_label', bbox, 'owner_approved', complete, list(image.size)))
    return views, [image.crop(v.bbox) for v in views]


def validate_vectors(vectors, n):
    if vectors.ndim != 2 or vectors.shape != (n, 768) or not np.isfinite(vectors).all():
        raise ValueError('Expected finite N x 768 embeddings')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Embeddings must be L2 normalized')


class LabelFirstIndex:
    """One shared encoder space; independent cosine channels, at most40 slugs."""
    def __init__(self, vectors, references, encoder_id):
        self.vectors = np.asarray(vectors, dtype='float32')
        validate_vectors(self.vectors, len(references))
        if any(r['kind'] not in ('context', 'front_label') for r in references):
            raise ValueError('text_detail cannot enter identity index')
        if any(r['kind']=='front_label' and not r.get('complete') for r in references):
            raise ValueError('Partial label cannot enter complete-label channel')
        self.references, self.encoder_id = references, encoder_id
        self.channel_ids = {kind:np.array([i for i,r in enumerate(references) if r['kind']==kind],dtype=int)
                            for kind in ('context','front_label')}

    def search(self, query, views, encoder_id):
        if encoder_id != self.encoder_id:
            raise ValueError('Query/gallery encoder mismatch; rebuild both channels')
        query=np.asarray(query,dtype='float32'); validate_vectors(query,len(views))
        scores={}; lists={}
        for kind in ('context','front_label'):
            qids=[i for i,v in enumerate(views) if v.kind==kind or (kind=='front_label' and v.kind in ('partial_label', 'detected_label'))]
            ids=self.channel_ids[kind]
            per_slug={}
            if qids and len(ids):
                values=self.vectors[ids] @ query[qids].T
                for offset,i in enumerate(ids):
                    j=int(np.argmax(values[offset]));value=float(values[offset,j]);slug=self.references[i]['slug']
                    if slug not in per_slug or value>per_slug[slug]['score']:
                        per_slug[slug]={'slug':slug,'score':value,'reference_index':int(i),'query_view_index':qids[j]}
            scores[kind]=per_slug
            lists[kind]=sorted(per_slug.values(),key=lambda r:(-r['score'],r['slug']))[:20]
        primary='front_label' if lists['front_label'] else 'context'
        pool={r['slug'] for branch in lists.values() for r in branch}
        ranked=[]
        for slug in pool:
            evidence={kind:scores[kind].get(slug) for kind in scores}
            primary_score=evidence[primary]['score'] if evidence[primary] else None
            ranked.append({'slug':slug,'score':primary_score,'channels':evidence})
        ranked.sort(key=lambda r:(r['score'] is None,-r['score'] if r['score'] is not None else 0,r['slug']))
        disagreement=bool(lists['context'] and lists['front_label'] and lists['context'][0]['slug']!=lists['front_label'][0]['slug'])
        return {'ranked_candidates':ranked,'best_candidate':ranked[0]['slug'] if ranked else None,
                'channel_top20':lists,'candidate_union':sorted(pool),'primary_channel':primary,
                'channel_disagreement':disagreement,'decision':'uncertain' if ranked else 'unknown',
                'slug':None,'probability_correct':None,
                'reasons':['confidence_not_calibrated']+(['partial_label_evidence'] if any(v.kind=='partial_label' for v in views) else [])+(['channel_disagreement'] if disagreement else [])+
                          (['label_visibility_unassessed'] if any(v.kind=='detected_label' for v in views) else [])+
                          (['complete_label_unavailable_context_fallback'] if primary=='context' else []),
                'views':[asdict(v) for v in views]}
