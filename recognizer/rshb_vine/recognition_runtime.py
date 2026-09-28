"""One ordinary recognition entrypoint with explicit diagnostic selector modes.

Service retains frozen8175; an explicit candidate_service profile may publish a
separately evaluated selector. Diagnostics never silently replace answers. Both visual models belong to the single
control instance, and only exact within-request batches can be reused.
"""
from pathlib import Path
import time

from rshb_vine.catalog.product_registry import ProductRegistry
from rshb_vine.io import digest, local_path, read_json, seal, sha256, verify
from rshb_vine.product_first.year_evidence import describe_year
from rshb_vine.request_visual_cache import RequestVisualCache


PROFILE_KIND = 'recognition-runtime-profile-v1'
MODES = ('service', 'candidate_service', 'diagnostic_learned', 'diagnostic_competitive')
OWN_SOURCES = ('rshb_vine/recognition_runtime.py', 'rshb_vine/request_visual_cache.py',
               'scripts/run_recognition.py', 'rshb_vine/recognition_api.py')


def load_profile(root, profile_path):
    """Verify configuration and referenced files without creating any model."""
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != PROFILE_KIND or profile.get('mode') not in MODES:
        raise ValueError('Unsupported recognition profile or mode')
    if type(profile.get('memoize_visual_batches')) is not bool:
        raise ValueError('Profile must explicitly choose visual memo mode')
    sources = profile.get('sources', {})
    required = {*OWN_SOURCES, profile['control_protocol'], profile['product_bundle']}
    if profile['mode'] != 'service':
        selector = profile['selector']
        required.update((selector['protocol'], selector['model']))
        if profile['mode'] == 'candidate_service':
            required.add('rshb_vine/selected_product_output.py')
            required.add('rshb_vine/runtime_product_selection.py')
    elif profile.get('selector') is not None:
        raise ValueError('Service mode must not install a diagnostic selector')
    if not required <= set(sources):
        raise ValueError('Profile does not pin its runtime/configuration sources')
    admission = profile.get('admission')
    if admission:
        if profile['mode'] != 'candidate_service' or admission['path'] not in sources:
            raise ValueError('Admission requires a pinned selected-service decision')
        decision = verify(read_json(local_path(root, admission['path'])))
        if (decision['checksum'] != admission['checksum'] or decision.get('status') != 'selected'
                or decision.get('model_checksum') != profile['selector']['model_checksum']
                or not decision.get('quality_gate_passed') or not decision.get('speed_gate_passed')
                or not decision.get('functional_passed')):
            raise ValueError('Selected service has no matching quality/HTTP/speed admission')
    for path, expected in sources.items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('Recognition profile source changed: ' + path)
    control = verify(read_json(local_path(root, profile['control_protocol'])))
    bundle = verify(read_json(local_path(root, profile['product_bundle'])))
    if (bundle['checksum'] != profile['product_bundle_checksum']
            or bundle['registry']['checksum'] != profile['registry_checksum']):
        raise ValueError('Profile product identity changed')
    for kind in ('registry', 'claims'):
        if sha256(local_path(root, bundle[kind]['path'])) != bundle[kind]['sha256']:
            raise ValueError('Profile product bundle bytes changed: ' + kind)
    if profile['mode'] != 'service':
        model = verify(read_json(local_path(root, profile['selector']['model'])))
        if model['checksum'] != profile['selector']['model_checksum']:
            raise ValueError('Profile selector model changed')
    return profile, control, bundle


def describe_profile(root, profile_path):
    profile, control, bundle = load_profile(root, profile_path)
    return {'kind': PROFILE_KIND, 'profile_checksum': profile['checksum'],
            'mode': profile['mode'], 'public_answer': 'selected_product' if profile['mode'] == 'candidate_service' else 'frozen8175',
            'control_protocol': profile['control_protocol'], 'control_protocol_checksum': control['checksum'],
            'product_bundle_checksum': bundle['checksum'], 'product_registry_checksum': bundle['registry']['checksum'],
            'claims_checksum': bundle['claims']['checksum'], 'counts': bundle['counts'],
            'memoize_visual_batches': profile['memoize_visual_batches'],
            'calibration_status': 'unknown', 'probability': None, 'models_loaded': False,
            'diagnostic_selector': profile.get('selector')}


def _control_holders(control, root):
    core = control.expanded.core
    square, title, instance = core.parent, core.parent.parent, core.parent.parent.parent
    base = instance.base
    for obj, name in ((core, 'GuardedConsensus'), (square, 'SquareRescue'),
                      (title, 'TitleCandidate'), (instance, 'InstanceTextProfile'), (base, 'LabelFirstPipeline')):
        if type(obj).__name__ != name:
            raise ValueError('Frozen model ownership changed: expected ' + name)
    arms, bindings = {}, {}
    for arm, holder, encoder_id in (('B0', base, base.encoder_id), ('B3', core, core.eid)):
        folder = root / 'data/catalog-additions-20260921' / arm
        gallery = verify(read_json(folder / 'gallery.json'))
        if (gallery['encoder_id'] != encoder_id or holder.index.encoder_id != encoder_id
                or holder.index.references != gallery['references']
                or sha256(folder / 'vectors.npy') != gallery['vectors_sha']
                or holder.index.vectors.shape != (len(gallery['references']), 768)):
            raise ValueError('Control holder/gallery binding changed: ' + arm)
        arms[arm] = holder
        bindings[arm] = {'encoder_id': encoder_id, 'gallery_checksum': gallery['checksum'],
                         'vectors_sha256': gallery['vectors_sha'], 'references': len(gallery['references']),
                         'ownership': 'existing_frozen_control_holder'}
    if base.encoder_id == core.eid or instance.spec['encoder_id'] != base.encoder_id:
        raise ValueError('Retrieval spaces or the B0 object-gate binding changed')
    return arms, bindings


class RecognitionRuntime:
    def __init__(self, root, profile_path):
        root = Path(root).resolve()
        profile, _, _ = load_profile(root, profile_path)
        self.profile = profile
        self.mode = profile['mode']
        self.selection = None
        if self.mode == 'service':
            self.registry = ProductRegistry.from_bundle(root, profile['product_bundle'])
        else:
            from rshb_vine.product_selection import ProductSelection
            from rshb_vine.competitive_selection import CompetitiveSelection
            if self.mode == 'candidate_service':
                from rshb_vine.runtime_product_selection import RuntimeProductSelection
                cls = RuntimeProductSelection
            else:
                cls = CompetitiveSelection if self.mode == 'diagnostic_competitive' else ProductSelection
            selector = profile['selector']
            self.selection = cls(root, profile['product_bundle'], selector['protocol'], selector['model'])
            self.registry = self.selection.registry
        if self.registry.checksum != profile['registry_checksum']:
            raise ValueError('Unexpected live registry')
        from rshb_vine.color_verified_recognition import ColorVerifiedRecognition
        self.control = ColorVerifiedRecognition(root, local_path(root, profile['control_protocol']))
        self.holders, bindings = _control_holders(self.control, root)
        self.visual_cache = RequestVisualCache(capture_retrieval=self.selection is not None)
        self.arms = {}
        for arm, holder in self.holders.items():
            encoder_id = bindings[arm]['encoder_id']
            holder.encoder = self.visual_cache.wrap(holder.encoder, encoder_id)
            if self.selection is not None:
                holder.index = self.visual_cache.wrap_index(holder.index, encoder_id)
            self.arms[arm] = (encoder_id, holder.encoder, holder.index)
        self.manifest = seal({'kind': 'shared-recognition-runtime-v1', 'profile_checksum': profile['checksum'],
            'runtime_descriptor_checksum': profile['checksum'],
            'profile_mode': self.mode, 'control': self.control.manifest,
            'product_registry': self.registry.checksum, 'product_bundle': self.registry.bundle_checksum,
            'claims': self.registry.claims_checksum, 'visual_holders': bindings,
            'memoize_visual_batches': profile['memoize_visual_batches'],
            'public_answer': 'selected_product' if self.mode == 'candidate_service' else 'frozen8175',
            'selector_accepted': bool(profile.get('admission')), 'calibration_status': 'unknown', 'probability': None,
            'diagnostic_model': self.selection.model['checksum'] if self.selection else None})

    def _describe(self, slug, observations, sid):
        registry = self.registry
        result = describe_year({'representative_slug': slug, 'product_id': registry.product_id(slug)},
                               observations, registry.claims, {'kind': 'same_instance_ocr', 'instance_id': sid})
        result.update(identity_registry_checksum=registry.checksum, claims_checksum=registry.claims_checksum,
            identity_binding_status=registry.cards[slug]['binding_status'] if slug else 'no_product',
            catalog_card_slugs=registry.members[registry.product_id(slug)] if slug else [])
        return result

    def _attach_products(self, result, data, control_seconds):
        from rshb_vine.selector_evidence import target_evidence
        source_digest = digest(result)
        outputs, evidence, selections = [], [], {}
        image = None
        if self.selection is not None and result.get('targets'):
            from rshb_vine.preprocessing import decode
            image = decode(data)[0]
        for target in result.get('targets', []):
            sid = str(target['instance_id'])
            packet = target_evidence(result, target, source_digest)
            observed = packet['ocr_packet'].get('observations', [])
            control_slug = target['retrieval'].get('best_candidate')
            target['product_resolution'] = self._describe(control_slug, observed, sid)
            target['product_candidates'] = self.registry.group_ranked_cards(packet['returned_candidate_rows'])
            if self.selection is None:
                continue
            from rshb_vine.preprocessing import checked_box
            from rshb_vine.visual_core import View
            views = [View(**view) for view in target['retrieval']['views']]
            crops = [image.crop(checked_box(view.bbox, image.size)) for view in views]
            angle = target.get('retrieval_pixel_rotation_ccw', 0)
            if angle not in (0, 90, 270):
                raise ValueError('Unsupported frozen rotation')
            if angle:
                crops = [crop.rotate(angle, expand=True) for crop in crops]
            raw = {'instance_id': sid, 'views': target['retrieval']['views'], 'rotation_ccw': angle, 'arms': {}}
            for arm, (eid, encoder, index) in self.arms.items():
                retrieved = index.reuse(crops, views, eid)
                if retrieved is None:
                    retrieved = index.search(encoder.encode(crops), views, eid)
                raw['arms'][arm] = {'encoder_id': eid, 'retrieval': retrieved}
            selected = self.selection.select(control_slug=control_slug, raw_visual=raw,
                observations=observed, control_candidates=packet['returned_candidate_rows'])
            selections[sid] = selected
            outputs.append({'instance_id': sid, 'control_slug': control_slug, 'proposal': selected['proposal'],
                'candidate_count': len(selected['features']['candidates']), 'feature_digest': digest(selected['features']),
                'views': raw['views'], 'rotation_ccw': angle})
            evidence.append({'instance_id': sid, 'raw_visual': raw, 'ocr_observations': observed,
                             'control_slug': control_slug, 'control_candidates': packet['returned_candidate_rows']})
        if self.mode == 'candidate_service':
            from rshb_vine.selected_product_output import publish_selected_products
            result = publish_selected_products(result, selections)
            by_id = {str(e['instance_id']): e for e in evidence}
            for target in result.get('targets', []):
                sid = str(target['instance_id'])
                target['product_resolution'] = self._describe(target['retrieval']['best_candidate'],
                    by_id[sid]['ocr_observations'], sid)
                target['product_candidates'] = self.registry.group_ranked_cards(target['retrieval']['ranked_candidates'])
            self.control.refresh_aliases(result)
            result['selected_product_output'].update(requires_alias_refresh=False, requires_product_year_refresh=False)
        result['product_resolution'] = {
            'identity_registry_checksum': self.registry.checksum, 'bundle_checksum': self.registry.bundle_checksum,
            'claims_checksum': self.registry.claims_checksum, 'product_grouping_uses_vintage': False,
            'year_evidence_stage': 'post_selection', 'inherited8175_policy_can_use_year': True,
            'targets': [{'instance_id': str(t['instance_id']), 'resolution': t['product_resolution']}
                        for t in result.get('targets', [])],
            'selection_source': 'selected_product' if self.mode == 'candidate_service' else 'retained8175', 'probability': None}
        if self.selection is not None:
            result['learned_selector_execution' if self.mode == 'candidate_service' else 'learned_selector_shadow'] = {
                'mode': self.mode, 'accepted': False, 'model': self.selection.model['checksum'],
                'identity_registry_checksum': self.registry.checksum, 'targets': outputs,
                'control_seconds': control_seconds, 'public_answer_unchanged': self.mode != 'candidate_service',
                'cost_note': 'Diagnostic retrieval reuses exact request batches only; not a release speed claim.'}
            result['product_identity_evidence'] = {'source_control_digest': source_digest, 'targets': evidence}
        return result

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        with self.visual_cache.request(enabled=self.profile['memoize_visual_batches']) as request:
            result = self.control.recognize(data, roi)
            control_seconds = time.perf_counter() - started
            if result.get('decision') != 'invalid_image':
                result = self._attach_products(result, data, control_seconds)
            total = time.perf_counter() - started
            timing = result.setdefault('timing_ms', {})
            timing['frozen8175_total'] = timing.get('total', control_seconds * 1000)
            timing['total'] = total * 1000
            result['recognition_runtime'] = {
                'profile_checksum': self.profile['checksum'], 'mode': self.mode,
                'public_answer': 'selected_product' if self.mode == 'candidate_service' else 'frozen8175',
                'public_answer_unchanged': self.mode != 'candidate_service',
                'control_seconds': control_seconds, 'total_seconds': total,
                'visual_memo': request.snapshot(), 'calibration_status': 'unknown'}
            if self.selection is not None and 'learned_selector_shadow' in result:
                result['learned_selector_shadow']['total_seconds'] = total
            return result
