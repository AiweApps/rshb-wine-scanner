"""Scan use cases: readiness against the expected profile, a bounded queue, the public view and the eval slug."""
import asyncio
import math
import re
import time

from wine_scanner.contracts import DECISIONS, FORMATS, MAX_BYTES, MAX_PIXELS, Rejected, box, code, codes, slug

MAX_WAITING = 2
QUEUE_WAIT_S = 90
READY_TTL_S = 15
ALTERNATIVES = 4
INSTANCE = re.compile(r'^[A-Za-z0-9_.:-]{1,64}$')


def instance(value):
    value = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else None
    return value if value and INSTANCE.match(value) else None


class Gate:
    """One request in flight to the backend, at most MAX_WAITING queued behind it; the rest get 503."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.admitted = 0
        self.served = 0
        self.last_seconds = None

    def state(self):
        return {'in_flight': int(self.lock.locked()), 'waiting': max(0, self.admitted - int(self.lock.locked())),
                'capacity_waiting': MAX_WAITING, 'served': self.served, 'last_seconds': self.last_seconds}

    async def run(self, job):
        if self.admitted >= 1 + MAX_WAITING:
            raise Rejected(503, 'busy', 'Сервис занят другими фото, повторите чуть позже', retry_after=5)
        self.admitted += 1
        try:
            try:
                await asyncio.wait_for(self.lock.acquire(), QUEUE_WAIT_S)
            except asyncio.TimeoutError:
                raise Rejected(503, 'busy', 'Очередь не освободилась, повторите позже', retry_after=10)
            started = time.perf_counter()
            try:
                return await job()
            finally:
                self.lock.release()
                self.served += 1
                self.last_seconds = round(time.perf_counter() - started, 2)
        finally:
            self.admitted -= 1


def _entries(block):
    entries = list(block.get('bottles') or [])
    observation = block.get('roi_observation')
    if observation and all(e.get('instance_id') != observation.get('instance_id') for e in entries):
        entries.append(observation)
    return entries


def build_view(result, assets, image):
    """UI view of one backend answer: every target with its card and alternatives, no scores or internals."""
    block = result['target_contract']
    targets = {str(t.get('instance_id')): t for t in result.get('targets') or []}
    entries = _entries(block)
    entries.sort(key=lambda e: (box(e.get('geometry')) is None, (box(e.get('geometry')) or [0])[0]))
    bottles = []
    for number, entry in enumerate(entries, 1):
        source = targets.get(str(entry.get('roi_pass_instance_id') or entry.get('instance_id')))
        ranked = ((source or {}).get('retrieval') or {}).get('ranked_candidates') or []
        best = slug(entry.get('card'))
        seen, alternatives = {best}, []
        if source is not None and (entry.get('card_source') == 'this_request' or entry.get('roi_pass_instance_id')):
            for candidate in ranked:
                candidate = slug((candidate or {}).get('slug'))
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    alternatives.append(assets.card(candidate))
                if len(alternatives) == ALTERNATIVES:
                    break
        decision = code(entry.get('target_decision'))
        bottles.append({'number': number, 'instance_id': instance(entry.get('instance_id')),
                        'geometry': box(entry.get('geometry')), 'label_bbox': box(entry.get('label_bbox')),
                        'select_roi': box(entry.get('select_roi')), 'selectable': code(entry.get('selectable')),
                        'primary': entry.get('primary') is True, 'addressed': entry.get('addressed') is True,
                        'decision': decision, 'decision_text': DECISIONS.get(decision),
                        'reasons': codes(entry.get('reasons')), 'best': assets.card(best),
                        'alternatives': alternatives, 'confirmed': False, 'probability': None})
    decision = code(result.get('decision'))
    backend_ms = (block.get('timing_ms') or {}).get('total')
    passes = ((result.get('atlas_repair_release_v1') or {}).get('geometry') or {}).get('passes') or []
    if passes and all(isinstance(p.get('ms'), (int, float)) and math.isfinite(p['ms']) and p['ms'] >= 0
                      for p in passes):
        backend_ms = sum(p['ms'] for p in passes)
    if not (isinstance(backend_ms, (int, float)) and math.isfinite(backend_ms)):
        backend_ms = None
    frame = block.get('frame_size')
    count = block.get('bottle_count')
    return {'decision': decision, 'decision_text': DECISIONS.get(decision),
            'raw_slug': slug(result.get('slug')), 'best_slug': slug(result.get('best_candidate')),
            'published_card': assets.card(result.get('best_candidate')), 'mode': code(block.get('mode')),
            'roi': box(block.get('roi')), 'bottle_count': count if isinstance(count, int) else None,
            'primary_instance_id': instance(block.get('primary_instance_id')),
            'primary_basis': code(block.get('primary_basis')), 'primary_confirmed': False,
            'requires_target_selection': block.get('requires_target_selection') is True,
            'frame_size': frame if isinstance(frame, list) and len(frame) == 2
            and all(isinstance(v, int) for v in frame) else None,
            'image': image, 'bottles': bottles, 'reasons': codes(result.get('reasons')),
            'calibrated': False, 'probability': None,
            'profile_checksum': block.get('profile_checksum'), 'backend_ms': backend_ms}


def target_slug(result, roi):
    """Card of the addressed target for a ROI request, else of the primary target; None when there is none."""
    block = result['target_contract']
    if roi is not None:
        wanted, flag = block.get('addressed_instance_id'), 'addressed'
        if block.get('mode') != 'explicit_roi':
            return None
    else:
        wanted, flag = block.get('primary_instance_id'), 'primary'
        if block.get('mode') != 'automatic':
            return None
    if wanted is None:
        return None
    observation = block.get('roi_observation') or {}
    for entry in _entries(block):
        if entry.get('instance_id') == wanted and (entry.get(flag) is True or entry is observation):
            return slug(entry.get('card'))
    return None


class ScanService:
    def __init__(self, backend, assets, expected_profile):
        self.backend, self.assets, self.expected = backend, assets, expected_profile
        self.gate = Gate()
        self._ready = {'at': 0.0, 'value': None}

    async def readiness(self, fresh=False):
        cached = self._ready['value']
        if not fresh and cached and time.monotonic() - self._ready['at'] < READY_TTL_S:
            return cached
        reachable, descriptor = await self.backend.health()
        state = {'backend_reachable': reachable, 'expected_profile': self.expected,
                 'runtime_descriptor': descriptor, 'profile_match': descriptor == self.expected}
        state['ready'] = reachable and state['profile_match']
        self._ready.update(at=time.monotonic(), value=state)
        return state

    async def status(self):
        return {**await self.readiness(fresh=True), 'queue': self.gate.state(),
                'limits': {'max_bytes': MAX_BYTES, 'max_pixels': MAX_PIXELS, 'formats': sorted(FORMATS.values())},
                'catalog': self.assets.summary()}

    async def _recognize(self, data, info, roi):
        state = await self.readiness()
        if not state['ready']:
            state = await self.readiness(fresh=True)
        if not state['ready']:
            reason = 'Сервис распознавания недоступен' if not state['backend_reachable'] else \
                'Сервис временно недоступен'
            raise Rejected(503, 'backend_not_ready', reason, retry_after=15)
        try:
            result = await self.gate.run(lambda: self.backend.recognize(data, info, roi))
        except Rejected as error:
            if error.code == 'backend_unreachable':
                self._ready['value'] = None
            raise
        block = result.get('target_contract')
        if not isinstance(block, dict) or block.get('profile_checksum') != self.expected:
            self._ready['value'] = None
            raise Rejected(502, 'profile_mismatch', 'Сервис временно недоступен; повторите позже')
        return result

    async def recognize(self, data, info, roi):
        return build_view(await self._recognize(data, info, roi), self.assets, info)

    async def predict(self, data, info, roi):
        return {'slug': target_slug(await self._recognize(data, info, roi), roi)}
