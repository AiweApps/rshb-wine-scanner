"""Opt-in HTTP clone of the byte-pinned current target-contract closure with the v2 geometry association installed.

Loaded from the explicit release profile, never written to the current pointer. The frozen orphan owner is the single
``inner.parent.route.runtime.control.orphans`` object; it is replaced only inside this process.
"""
from copy import deepcopy
from pathlib import Path

from rshb_vine import orphan_label_recovery_v2
from rshb_vine.io import seal, sha256
from rshb_vine.roskachestvo_geometry_v2 import association as A

PARENT_PROFILE = 'config/recognition-target-contract-v2-release.json'
PARENT_CHECKSUM = '0d1c22944024be2307202998f533ff26b5212bd8e99bd952bbe0921507609473'
SOURCES = ('rshb_vine/roskachestvo_geometry_v2/__init__.py', 'rshb_vine/roskachestvo_geometry_v2/association.py',
           'rshb_vine/roskachestvo_geometry_v2/runtime.py', 'rshb_vine/orphan_label_recovery_v2.py')


def locate(target):
    from rshb_vine.target_misses_v1.route import CanvasCloseupRoute
    coherent = target.inner.parent
    route = coherent.route
    control = route.runtime.control
    orphans = control.orphans
    if (not isinstance(route, CanvasCloseupRoute) or route.runtime is not coherent.runtime
            or type(orphans) is not orphan_label_recovery_v2.OrphanLabelRecovery
            or orphans.retry != control.recognize_without_orphans):
        raise ValueError('Unexpected runtime graph for the geometry candidate')
    return control, orphans


class GeometryCandidateRecognition:
    def __init__(self, root, profile_path=PARENT_PROFILE):
        from rshb_vine.target_contract_v2.runtime import TargetContractRecognition
        root = Path(root).resolve()
        inner = TargetContractRecognition(root, profile_path)
        if inner.profile['checksum'] != PARENT_CHECKSUM:
            raise ValueError('Parent target-contract profile differs from the protocol')
        control, frozen = locate(inner)
        self.recovery = A.GeometryOrphanRecovery(frozen)
        control.orphans = self.recovery
        self.inner, self.profile = inner, inner.profile
        self.descriptor = seal({'kind': 'roskachestvo-geometry-v2-candidate', 'policy': A.POLICY,
                                'constants': A.CONSTANTS, 'parent_profile': profile_path,
                                'parent_profile_checksum': PARENT_CHECKSUM,
                                'parent_runtime': inner.manifest['checksum'],
                                'sources_sha256': {p: sha256(root / p) for p in SOURCES},
                                'weights_changed': False, 'identity_links_changed': False,
                                'public_slug': 'never confirmed', 'activated': False, 'calibrated': False})
        self.manifest = seal({'kind': 'roskachestvo-geometry-v2-service', 'policy': A.POLICY,
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': inner.manifest['checksum'], 'activated': False})

    def recognize(self, data, roi=None, bottles='addressed'):
        self.recovery.events = []
        result = self.inner.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        result['roskachestvo_geometry_v2'] = {'policy': A.POLICY, 'descriptor_checksum': self.descriptor['checksum'],
                                              'events': deepcopy(self.recovery.events),
                                              'recovered': [e['instance_id'] for e in self.recovery.events if e['used']]}
        return result


def serve(root, port):
    import socket
    import uvicorn
    from rshb_vine.target_contract_v2.runtime import create_app
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = GeometryCandidateRecognition(root)
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()
