"""Opt-in layout-consistency evidence at ProductID level; never selects, the existing resolver decides."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from rshb_vine.geometric_reference_probe_v1 import POLICY, prepare
from rshb_vine.geometric_reference_probe_v1.layout import CRITERIA, assessable, correspond, layout_pass
from rshb_vine.io import digest, sha256, write_json

GALLERY = 'runs/catalog-training-data-v2/stage5/evaluator/actual-fit-v1/B4343/gallery/gallery.json'
PRODUCT_BUNDLE = 'config/product-identity-current.json'
LABEL_KINDS = ('detected_label', 'front_label', 'partial_label')
TRIGGER = {'id': 'sparse-layout-trigger-v1',
           'label_view': 'target has a detected/front/partial label view',
           'visual_disagreement': 'B3 channel_top20 front_label[0] and context[0] belong to different ProductIDs (unregistered slugs stay card-local)',
           'no_full_text_identity': 'recognition_repair_v1.ocr_candidate_injection (ocr-catalog-phrase-injection-v2, full name + producer) for this instance has matched_products == 0; a missing entry is not eligible',
           'budget': '2..40 ProductIDs with an eligible front_label reference in the saved candidate_union'}


AMBIGUITY = {'id': 'layout-competing-product-ambiguity-v4',
             'rule': 'a disagreement promotes nothing when any other ProductID has unique_inliers >= CRITERIA min_unique_inliers '
                     'and a plausible homography; winner ratio and hull coverage cannot erase such a competing layout',
             'source': 'root bridge 13/15 after W0233 (sibling with 28 plausible inliers, reference hull .198)'}


class ReferenceMetadataError(ValueError):
    pass


def competing(row):
    return row['unique_inliers'] >= CRITERIA['min_unique_inliers'] and row['homography_plausible']


def frozen_paths(gallery):
    from rshb_vine.catalog_training_evaluation_v2 import pins
    frozen = pins.read('references')
    if len(frozen) != len(gallery) or not all(f['index'] == i and all(f[k] == g[k] for k in ('slug', 'kind', 'image_sha256', 'bbox'))
                                               for i, (f, g) in enumerate(zip(frozen, gallery))):
        raise ValueError('frozen references differ from the B3 gallery')
    return {f['index']: f['source_path'] for f in frozen}


def product_of(registry, slug):
    return registry.product_id(slug) if slug in registry.cards else 'unregistered-card:' + slug


def label_view(target):
    return next((v for v in target['retrieval'].get('views', []) if v['kind'] in LABEL_KINDS), None)


def sparse_trigger(body, target, registry):
    """Frozen trigger over a saved response; returns (eligible, conditions)."""
    tops = target['retrieval'].get('channel_top20') or {}
    label, context = (tops.get('front_label') or [None])[0], (tops.get('context') or [None])[0]
    sid = str(target['instance_id'])
    injection = next((e for e in (body.get('recognition_repair_v1') or {}).get('ocr_candidate_injection', []) if str(e.get('instance_id')) == sid), None)
    conditions = {'label_view': label_view(target) is not None,
                  'label_top1': label and label['slug'], 'context_top1': context and context['slug'],
                  'visual_disagreement': bool(label and context and product_of(registry, label['slug']) != product_of(registry, context['slug'])),
                  'injection_status': injection and injection.get('status'),
                  'no_full_text_identity': bool(injection is not None and injection.get('matched_products') == 0)}
    return conditions['label_view'] and conditions['visual_disagreement'] and conditions['no_full_text_identity'], conditions


class LayoutConsistencyVerifier:
    """Evidence only: output keeps the incumbent; any use must pass through the existing resolver with unchanged observations."""

    def __init__(self, root, cache_dir=None, max_products=40):
        from rshb_vine.catalog.product_registry import ProductRegistry
        from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
        self.root = Path(root)
        gallery_path = self.root / GALLERY
        gallery = json.loads(gallery_path.read_text())
        self.references = gallery['references']
        self.gallery = {'path': GALLERY, 'sha256': sha256(gallery_path), 'checksum': gallery['checksum']}
        self.paths = frozen_paths(self.references)
        self.registry = ProductRegistry.from_bundle(self.root, PRODUCT_BUNDLE)
        conflicts = ReferenceConflicts(self.root)
        self.excluded = {e['reference_index']: e for e in conflicts.excluded_references(self.references)}
        self.conflicts_checksum = conflicts.checksum
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_products = max_products
        self._prepared = {}
        here = Path(__file__).resolve().parent
        self.code = {p.name: sha256(p) for p in sorted(here.glob('*.py'))}
        self.identity = digest({'policy': POLICY, 'criteria': CRITERIA, 'ambiguity': AMBIGUITY, 'trigger': TRIGGER, 'gallery': self.gallery, 'code': self.code,
                                'registry': self.registry.checksum, 'conflicts': self.conflicts_checksum})

    def reference_metadata(self, index):
        """Bytes, EXIF-oriented size, coordinate space and bbox must match the frozen gallery row."""
        from rshb_vine.preprocessing import decode
        ref = self.references[index]
        data = (self.root / self.paths[index]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['image_sha256']:
            raise ReferenceMetadataError(f'reference {index}: bytes differ from image_sha256')
        image, meta = decode(data, max_bytes=1 << 30, max_pixels=36_000_000)
        if list(image.size) != list(ref['original_size']) or ref.get('coordinate_space') != meta['coordinate_space']:
            raise ReferenceMetadataError(f"reference {index}: oriented size {list(image.size)} / space {ref.get('coordinate_space')} differ from frozen metadata")
        x1, y1, x2, y2 = ref['bbox']
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ReferenceMetadataError(f'reference {index}: bbox outside oriented image')
        return image

    def _reference(self, index):
        if index not in self._prepared:
            image = self.reference_metadata(index)
            self._prepared[index] = prepare(image.crop(tuple(round(v) for v in self.references[index]['bbox'])))
        return self._prepared[index]

    def eligible_references(self, target):
        retrieval = target['retrieval']
        union = set(retrieval.get('candidate_union') or [])
        response_excluded = {e['reference_index'] for e in (retrieval.get('reference_conflicts') or {}).get('excluded_references', [])}
        return [i for i, r in enumerate(self.references)
                if r['kind'] == 'front_label' and r['slug'] in union and i not in self.excluded and i not in response_excluded]

    def evaluate_target(self, image, body, target, force=False):
        sid = str(target['instance_id'])
        incumbent = target['retrieval'].get('best_candidate')
        base = {'instance_id': sid, 'incumbent': incumbent, 'incumbent_retained': True, 'resolver_required': True}
        eligible, conditions = sparse_trigger(body, target, self.registry)
        base['trigger'] = conditions
        if not (eligible or force):
            return dict(base, action='not_triggered')
        indices = self.eligible_references(target)
        products = {product_of(self.registry, self.references[i]['slug']) for i in indices}
        if len(products) < 2 or len(products) > self.max_products:
            return dict(base, action='skipped', reason=f'{len(products)} ProductIDs with eligible references outside [2,{self.max_products}]')
        view = label_view(target)
        crop = image.crop(tuple(round(v) for v in view['bbox']))
        angle = target.get('retrieval_pixel_rotation_ccw') or 0
        if angle:
            crop = crop.rotate(angle, expand=True)
        query = prepare(crop)
        best = {}
        for i in indices:
            row = {'reference_index': i, 'slug': self.references[i]['slug'], **correspond(query, self._reference(i))}
            pid = product_of(self.registry, row['slug'])
            if pid not in best or (row['unique_inliers'], row['hull_query']) > (best[pid]['unique_inliers'], best[pid]['hull_query']):
                best[pid] = dict(row, product_id=pid)
        ranking = sorted(best.values(), key=lambda r: (-r['unique_inliers'], -r['hull_query'], r['product_id']))
        winner, runner = ranking[0], ranking[1]
        ratio = winner['unique_inliers'] / runner['unique_inliers'] if runner['unique_inliers'] else None
        supported = layout_pass(winner) and (ratio is None or ratio >= CRITERIA['min_winner_to_next'])
        brief = lambda r: {k: r[k] for k in ('product_id', 'slug', 'reference_index', 'reference_size', 'reference_features', 'unique_inliers',
                                             'hull_query', 'hull_reference', 'span_query', 'span_reference', 'homography_plausible')} | {
            'layout_pass': layout_pass(r), 'assessable': assessable(r, query)}
        incumbent_pid = product_of(self.registry, incumbent) if incumbent else None
        inc = best.get(incumbent_pid)
        rivals = [r for r in ranking[1:] if competing(r)]
        if not supported:
            action = 'no_layout_support'
        elif winner['product_id'] == incumbent_pid:
            action = 'agrees_with_incumbent'
        elif inc is None or not assessable(inc, query):
            action = 'abstain_diagnostic'
        elif rivals:
            action = 'ambiguous_competing_layout'
        else:
            action = 'layout_disagrees_with_incumbent'
        out = dict(base, action=action, winner_to_next=ratio, query_size=list(query['size']), query_features=len(query['keypoints']),
                   label_view={k: view[k] for k in ('kind', 'bbox', 'source')}, rotation_ccw=angle, pairs=len(indices),
                   decision_rule=AMBIGUITY['id'], competing_rivals=[brief(r) for r in rivals],
                   top5=[brief(r) for r in ranking[:5]], incumbent_row=brief(inc) if inc else None)
        if supported:
            out['layout_evidence'] = {'product_id': winner['product_id'], 'card_slugs_in_union': sorted(
                s for s in target['retrieval'].get('candidate_union', []) if product_of(self.registry, s) == winner['product_id']),
                'kind': 'positive layout consistency, not exact identity or vintage'}
        return out

    def evaluate(self, image_bytes, body, image=None, force=False):
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        key = digest({'image': image_sha, 'response': digest(body), 'verifier': self.identity, 'force': force})
        path = self.cache_dir / f'{key}.json' if self.cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text())
        if image is None:
            from rshb_vine.preprocessing import decode
            image, _ = decode(image_bytes)
        start = time.process_time()
        targets = [self.evaluate_target(image, body, t, force) for t in body.get('targets', [])]
        trace = {'policy': 'layout-consistency-v4-evidence', 'decision_rule': AMBIGUITY, 'applied': False, 'calibrated': False, 'cache_key': key, 'targets': targets,
                 'provenance': {'image_sha256': image_sha, 'response_digest': digest(body), 'gallery': self.gallery,
                                'reference_paths': 'catalog_training_evaluation_v2.pins references; bytes, oriented size, coordinate space and bbox verified',
                                'identity_registry_checksum': self.registry.checksum, 'product_bundle': PRODUCT_BUNDLE,
                                'reference_conflicts_checksum': self.conflicts_checksum, 'excluded_reference_indices': sorted(self.excluded),
                                'code_sha256': self.code, 'policy': POLICY, 'criteria': CRITERIA, 'trigger': TRIGGER, 'verifier_identity': self.identity},
                 'cpu_seconds': round(time.process_time() - start, 3),
                 'limits': ['positive layout evidence, not calibrated exact-SKU proof', 'variant/vintage not decided by layout',
                            'abstain_diagnostic keeps the incumbent; poor-resolution absence is not contradiction',
                            'text contradiction and final selection belong to the existing resolver with unchanged observations']}
        if path:
            write_json(path, trace)
        return trace

    def apply(self, image_bytes, body, image=None, force=False):
        result = deepcopy(body)
        result['layout_consistency_v1'] = self.evaluate(image_bytes, body, image, force)
        return result
