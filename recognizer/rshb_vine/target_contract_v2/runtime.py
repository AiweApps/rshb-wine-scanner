"""Opt-in target-contract runtime: byte-pinned systemic-v2 release profile -> unchanged responses + target_contract.

The parent is config/recognition-systemic-v2-release.json loaded through the shared factory, never the mutable
current pointer. An ROI request is one parent ROI pass; bottles=all adds one no-ROI pass under the same lock.
"""
import json
import math
from pathlib import Path
import threading
import time

from rshb_vine.io import digest, local_path, read_json, seal, sha256, verify
from rshb_vine.target_contract_v2 import contract

KIND = 'recognition-target-contract-v2-profile'
PROFILE = 'config/target-contract-v2-candidate.json'
CURRENT_POINTER = 'config/recognition-current.json'
BOTTLES = ('addressed', 'all')


def load_profile(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile.get('policy') != contract.POLICY:
        raise ValueError('Unsupported target-contract profile')
    if profile['parent_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError('Target-contract profile must not depend on the mutable current pointer')
    if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
        raise ValueError('Parent profile bytes changed')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('Target-contract source changed: ' + path)
    return profile


def parse_roi(text):
    if text is None:
        return None
    try:
        roi = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError('ROI must be a JSON array') from exc
    if (not isinstance(roi, list) or len(roi) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in roi)):
        raise ValueError('ROI must be [x1, y1, x2, y2] finite numbers')
    if not (roi[2] > roi[0] and roi[3] > roi[1]):
        raise ValueError('ROI must have x2 > x1 and y2 > y1')
    return roi


def parse_bottles(value):
    value = value or 'addressed'
    if value not in BOTTLES:
        raise ValueError('bottles must be addressed or all')
    return value


class TargetContractRecognition:
    def __init__(self, root, profile_path=PROFILE):
        from rshb_vine.recognition_factory import load_runtime
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        self.inner = load_runtime(root, profile['parent_profile'])
        if self.inner.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Parent runtime differs from the target-contract profile')
        self.profile = profile
        self.manifest = seal({'kind': 'target-contract-v2-runtime', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': self.inner.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'policy': contract.POLICY, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        started = time.perf_counter()
        result = self.inner.recognize(data, roi)
        first = time.perf_counter()
        if result.get('decision') == 'invalid_image':
            return result
        full = self.inner.recognize(data, None) if roi is not None and bottles == 'all' else None
        second = time.perf_counter()
        before = digest(result)
        block = contract.build(result, roi, full)
        if digest(result) != before:
            raise ValueError('Target contract changed the parent response')
        block.update(profile_checksum=self.profile['checksum'],
                     parent_profile_checksum=self.profile['parent_profile_checksum'],
                     timing_ms={'request_pass': (first - started) * 1000,
                                'full_pass': (second - first) * 1000 if full is not None else None,
                                'contract': (time.perf_counter() - second) * 1000,
                                'total': (time.perf_counter() - started) * 1000})
        result['target_contract'] = block
        return result


def create_app(pipeline):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from starlette.responses import JSONResponse
    app = FastAPI(title='Local RSHB recognition · target contract v2 (opt-in)')
    lock = threading.Lock()

    def run(image, roi_text=None, bottles=None):
        try:
            roi, scope = parse_roi(roi_text), parse_bottles(bottles)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        data = image.file.read(20 * 1024 * 1024 + 1)
        if not lock.acquire(blocking=False):
            raise HTTPException(503, 'Inference busy; retry later')
        try:
            result = pipeline.recognize(data, roi, scope)
        except Exception as exc:
            raise HTTPException(503, 'Inference failed: ' + type(exc).__name__) from exc
        finally:
            lock.release()
        if result['decision'] == 'invalid_image':
            raise HTTPException(422, result['reasons'])
        return result

    @app.get('/health/ready')
    def ready():
        return {'ready': True, 'snapshot': pipeline.manifest['checksum'],
                'runtime_descriptor_checksum': pipeline.manifest.get('runtime_descriptor_checksum'),
                'parent_runtime': pipeline.manifest.get('parent_runtime'), 'policy': contract.POLICY}

    @app.post('/v1/recognize')
    def recognize(image: UploadFile = File(...), target_roi: str | None = Form(None), bottles: str | None = Form(None)):
        return JSONResponse(run(image, target_roi, bottles))

    @app.post('/v2/targets')
    def targets(image: UploadFile = File(...), target_roi: str | None = Form(None), bottles: str | None = Form(None)):
        result = run(image, target_roi, bottles)
        return JSONResponse(contract.compact(result, result['target_contract']))

    @app.post('/v1/eval/predict')
    def predict(image: UploadFile = File(...)):
        result = run(image)
        if result['best_candidate'] is None:
            raise HTTPException(422, 'No candidate; refusing a fabricated slug')
        return JSONResponse({'slug': result['best_candidate']})

    @app.post('/v2/eval/predict')
    def predict_target(image: UploadFile = File(...), target_roi: str | None = Form(None)):
        result = run(image, target_roi)
        block = result['target_contract']
        if result['best_candidate'] is None:
            raise HTTPException(422, {'detail': 'No published candidate; refusing a fabricated slug',
                                      'primary_basis': block['primary_basis'], 'bottle_count': block['bottle_count']})
        return JSONResponse({'slug': result['best_candidate'], 'instance_id': block['primary_instance_id'],
                             'basis': block['primary_basis'], 'calibrated': False, 'probability': None})

    return app


def serve(root, port, profile_path=PROFILE):
    import socket
    import uvicorn
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = TargetContractRecognition(root, profile_path)
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()
