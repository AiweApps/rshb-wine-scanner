"""Sparse layout-consistency stage bound to the running selector: same registry, same target pixels, unchanged resolver.

The verifier is constructed explicitly with the selector's ProductRegistry (not product-identity-current), so the
trigger, per-product ranking and winner -> pool candidate mapping use the ProductIDs the publisher uses. The
label view is cropped with the runtime's checked_box (floor/ceil), the same pixels the B3 arm encodes. A supported
disagreement moves the winner's pool candidate to the top of the raw order (learned scores and ranks preserved)
and passes the unchanged existing-evidence resolver; anything else keeps the incumbent and records why.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from rshb_vine.geometric_reference_probe_v1 import POLICY
from rshb_vine.geometric_reference_probe_v1 import verifier as V
from rshb_vine.geometric_reference_probe_v1.layout import CRITERIA
from rshb_vine.io import digest, sha256

STAGE = 'sparse-layout-geometry-stage-v1'
PACKAGE = 'rshb_vine/geometric_reference_probe_v1'
SOURCES = tuple(f'{PACKAGE}/{n}' for n in ('__init__.py', 'layout.py', 'verifier.py'))
PROPOSING_ACTION = 'layout_disagrees_with_incumbent'
PREPARED_LIMIT = 512
SCOPE = {'id': 'single-sku-target-pass-v1',
         'rule': 'run only when the selection pass has exactly one SKU target (control result targets == product '
                 'evidence targets, checked on all 593 saved responses); multi-target passes keep the incumbent unmatched',
         'source': 'root bridge 22: 92 multi-target requests added p95 ~5 s with no gain in class321'}
LIMITS = ('layout is style/layout evidence, not exact SKU or vintage; vintage stays with the post-selection resolver',
          'abstain when the incumbent reference is not assessable; absent SIFT support on a small reference is not contradiction',
          'geometry is not run on multi-target passes')


class _Bounded(dict):
    """Prepared-reference cache of a long-lived service; oldest SIFT descriptors are dropped first."""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        while len(self) > PREPARED_LIMIT:
            del self[next(iter(self))]


class SelectorBoundLayoutVerifier(V.LayoutConsistencyVerifier):
    """LayoutConsistencyVerifier state with the selector registry supplied by the caller instead of PRODUCT_BUNDLE."""

    def __init__(self, root, registry, max_products=40):
        from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
        self.root = Path(root)
        gallery_path = self.root / V.GALLERY
        gallery = json.loads(gallery_path.read_text())
        self.references = gallery['references']
        self.gallery = {'path': V.GALLERY, 'sha256': sha256(gallery_path), 'checksum': gallery['checksum']}
        self.paths = V.frozen_paths(self.references)
        self.registry = registry
        conflicts = ReferenceConflicts(self.root)
        self.excluded = {e['reference_index']: e for e in conflicts.excluded_references(self.references)}
        self.conflicts_checksum = conflicts.checksum
        self.cache_dir = None
        self.max_products = max_products
        self._prepared = _Bounded()
        self.code = {p: sha256(self.root / p) for p in SOURCES}
        self.identity = digest({'policy': POLICY, 'criteria': CRITERIA, 'ambiguity': V.AMBIGUITY, 'trigger': V.TRIGGER, 'gallery': self.gallery,
                                'code': self.code, 'registry': registry.checksum, 'conflicts': self.conflicts_checksum})


def _pool_union(base):
    return sorted({s for c in base['candidates'] for s in c['card_slugs']})


def target_view(result_target, out, instance_id):
    """The published-equivalent target the verifier reads: control views/channels, selector pool union, incumbent."""
    retrieval = result_target['retrieval']
    return {'instance_id': instance_id,
            'retrieval_pixel_rotation_ccw': result_target.get('retrieval_pixel_rotation_ccw') or 0,
            'retrieval': {'views': deepcopy(retrieval.get('views', [])),
                          'channel_top20': retrieval.get('channel_top20') or {},
                          'reference_conflicts': retrieval.get('reference_conflicts'),
                          'candidate_union': _pool_union(out['base']),
                          'best_candidate': out['proposal']['representative_slug'] if out.get('proposal') else None}}


def injection_body(decision, instance_id):
    return {'recognition_repair_v1': {'ocr_candidate_injection': [
        {'instance_id': instance_id, 'status': decision.get('status'), 'matched_products': decision.get('matched_products')}]}}


def pixel_view(target, image_size):
    """Replace the label view bbox by the runtime crop box so verifier rounding reproduces the encoded pixels."""
    from rshb_vine.preprocessing import checked_box
    view = V.label_view(target)
    original = list(view['bbox'])
    view['bbox'] = checked_box(original, image_size)
    return {'bbox_original': original, 'bbox_pixels': list(view['bbox']), 'crop': 'preprocessing.checked_box',
            'rotation_ccw': target['retrieval_pixel_rotation_ccw']}


def promote(raw, candidate_id, evidence):
    """Raw proposal with ``candidate_id`` first; learned scores kept, learned rank recorded, layout order explicit."""
    raw = deepcopy(raw)
    for row in raw['ranked_candidates']:
        row['learned_rank'] = row.get('learned_rank', row['rank'])
    raw['ranked_candidates'].sort(key=lambda r: r['candidate_id'] != candidate_id)
    for rank, row in enumerate(raw['ranked_candidates'], 1):
        row['rank'] = rank
        row['order_source'] = 'layout_consistency_promotion' if rank == 1 else 'learned_score_order'
    top = raw['ranked_candidates'][0]
    raw.update(candidate_id=top['candidate_id'], product_id=top['product_id'],
               representative_slug=top['representative_slug'], best_candidate=top['representative_slug'],
               representative_source=top['representative_source'], score=top['score'], score_margin=None,
               order_source='layout_consistency_promotion', reason='sparse_layout_consistency', exact_slug=None,
               probability=None, release_admitted=False, layout_consistency=evidence)
    return raw


class GeometryStage:
    trace_key = 'layout_geometry'
    position = 'post_guard'

    def __init__(self, root, registry):
        self.verifier = SelectorBoundLayoutVerifier(root, registry)
        self.registry = registry
        self.identity = {'stage': STAGE, 'verifier_identity': self.verifier.identity, 'trigger': V.TRIGGER['id'],
                         'criteria': CRITERIA['id'], 'decision_rule': V.AMBIGUITY['id'], 'policy': POLICY['id'],
                         'identity_registry_checksum': registry.checksum,
                         'product_bundle_checksum': registry.bundle_checksum,
                         'product_bundle': 'selector registry passed by the caller; verifier PRODUCT_BUNDLE not read',
                         'sources_sha256': self.verifier.code, 'scope': SCOPE['id'], 'limits': LIMITS}

    def apply(self, selection, out, call, request):
        sid = str(call['raw_visual']['instance_id'])
        trace = {'stage': STAGE, 'instance_id': sid, 'applied': False, 'incumbent_retained': True,
                 'identity_registry_checksum': self.registry.checksum, 'product_bundle_checksum': self.registry.bundle_checksum,
                 'incumbent': out['proposal']['representative_slug'] if out.get('proposal') else None}
        out[self.trace_key] = trace
        if not out.get('raw_proposal') or not out['raw_proposal']['ranked_candidates']:
            trace['action'] = 'empty_pool'
            return out
        if request is None:
            trace['action'] = 'request_context_missing'
            return out
        if len(request.result.get('targets', [])) != 1:
            trace['action'] = 'outside_single_target_scope'
            return out
        source = request.target(sid)
        if source is None:
            trace['action'] = 'target_missing_in_request_result'
            return out
        target = target_view(source, out, sid)
        body = injection_body(out['ocr_candidate_injection'], sid)
        eligible, conditions = V.sparse_trigger(body, target, self.registry)
        trace['trigger'] = conditions
        if not eligible:
            trace['action'] = 'not_triggered'
            return out
        started, cached = time.process_time(), len(self.verifier._prepared)
        image = request.image()
        trace['pixels'] = pixel_view(target, image.size)
        result = self.verifier.evaluate_target(image, body, target)
        trace['cpu_seconds'] = round(time.process_time() - started, 3)
        trace['references_cached_before'] = cached
        trace['verifier'] = result
        trace['action'] = result['action']
        if result['action'] != PROPOSING_ACTION:
            return out
        slugs = set(result['layout_evidence']['card_slugs_in_union'])
        pool = [c for c in out['base']['candidates'] if slugs & set(c['card_slugs'])]
        if len(pool) != 1:
            trace['action'] = 'abstain_winner_not_one_pool_candidate'
            trace['pool_matches'] = [c['candidate_id'] for c in pool]
            return out
        winner = pool[0]['candidate_id']
        evidence = {'stage': STAGE, 'product_id': result['layout_evidence']['product_id'], 'candidate_id': winner,
                    'winner_to_next': result['winner_to_next'], 'kind': result['layout_evidence']['kind'],
                    'vintage': 'not decided by layout'}
        raw = promote(out['raw_proposal'], winner, evidence)
        resolved = selection.legacy.resolver.resolve(out['base'], raw)
        trace['resolver'] = resolved['existing_evidence_resolution']
        if resolved.get('candidate_id') != winner:
            trace['action'] = 'abstain_existing_resolver_blocked'
            return out
        trace.update(applied=True, incumbent_retained=False, proposed_candidate=winner,
                     selected=resolved['representative_slug'], action='promoted_through_resolver')
        out['raw_proposal_before_geometry'] = out['raw_proposal']
        out['proposal_before_geometry'] = out['proposal']
        out['raw_proposal'], out['proposal'] = raw, resolved
        return out
