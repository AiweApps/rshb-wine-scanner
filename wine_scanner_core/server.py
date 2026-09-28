"""Recognition API of the core: the vendored target-contract routes over the Linux V6 runtime.

Readiness and every answer carry the core descriptor, which names what actually runs (Linux CPU, PP-OCRv6 medium,
this asset pack and vendored tree); the parent release profile and runtime are reported as provenance only.
"""
import time

from wine_scanner_core.linux_v6 import LEDGER, LinuxV6Runtime


class CoreRecognition:
    def __init__(self, core, threads):
        self.core = core
        self.runtime = LinuxV6Runtime(core.root, threads, core.provenance)
        self.descriptor = core.descriptor
        self.profile = {'checksum': self.descriptor['checksum'], 'kind': self.descriptor['kind']}
        self.manifest = {'checksum': self.descriptor['checksum'],
                         'runtime_descriptor_checksum': self.descriptor['checksum'],
                         'parent_runtime': self.descriptor['parent']['runtime_checksum']}
        self.loaded_at = time.time()

    def recognize(self, data, roi=None, bottles='addressed'):
        result, request = self.runtime.recognize(data, roi, bottles)
        if request['health']['vision_leaks'] or request['health']['blocked_exec']:
            raise RuntimeError('Apple Vision was reached during a request')
        block = result.get('target_contract')
        if isinstance(block, dict):
            block['core_parent_profile_checksum'] = block.get('profile_checksum')
            block['profile_checksum'] = self.descriptor['checksum']
        result['wine_scanner_core'] = {
            'descriptor_checksum': self.descriptor['checksum'], 'execution': self.descriptor['execution'],
            'parent': self.descriptor['parent'], 'request': request, 'calibrated': False}
        return result

    def status(self):
        return {'descriptor': self.descriptor, 'load_seconds': self.runtime.load_seconds,
                'memory': self.runtime.memory, 'execution': self.runtime.execution,
                'vision_leaks': len(LEDGER.vision_leaks)}


def create_app(pipeline):
    from rshb_vine.target_contract_v2.runtime import create_app as target_app
    app = target_app(pipeline)
    app.title = 'Wine scanner core · Linux CPU · PP-OCRv6'

    @app.get('/health/live')
    def live():
        return {'live': True}

    @app.get('/v1/core')
    def core():
        return pipeline.status()

    return app
