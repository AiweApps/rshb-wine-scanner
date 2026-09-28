"""Separate HTTP challenger composing frozen8174 with admitted color facts."""
from pathlib import Path
import time
from rshb_vine.reference_dual_recognition import ReferenceDualRecognition
from rshb_vine.reference_color_evidence import ReferenceColorEvidence
from rshb_vine.unified_recognition import verify_protocol
from rshb_vine.io import read_json, verify, seal


class ColorVerifiedRecognition(ReferenceDualRecognition):
    def __init__(self, root, protocol_path):
        super().__init__(root, protocol_path)
        protocol = verify_protocol(root, protocol_path)
        ledger = verify(read_json(Path(root)/protocol['color_facts']))
        self.color_evidence = ReferenceColorEvidence(ledger['records'])
        self.manifest = seal({**{k:v for k,v in self.manifest.items() if k != 'checksum'},
                              'kind':'color-verified-recognition-v1','color_facts':ledger['checksum']})

    def recognize(self, data, roi=None):
        start = time.perf_counter()
        result = self.color_evidence.apply(super().recognize(data, roi))
        self.refresh_aliases(result)
        result.setdefault('timing_ms', {})['total'] = 1000*(time.perf_counter()-start)
        return result
