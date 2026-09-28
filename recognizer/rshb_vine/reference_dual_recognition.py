"""Integrated dual-model candidate preserving source-bound reference text."""
from pathlib import Path
from rshb_vine.verified_dual_recognition import VerifiedDualRecognition
from rshb_vine.reference_phrase_preservation import ReferencePhrasePreservation
from rshb_vine.unified_recognition import verify_protocol
from rshb_vine.io import seal


class ReferenceDualRecognition(VerifiedDualRecognition):
    def __init__(self, root, protocol_path):
        super().__init__(root, protocol_path)
        protocol = verify_protocol(root, protocol_path)
        self.dual = ReferencePhrasePreservation(
            self.dual, root, Path(root) / protocol['reference_phrase_protocol'],
            Path(root) / protocol['reference_phrase_ledger'],
            protocol['reference_phrase_ledger_checksum'])
        self.manifest = seal({**{k: v for k, v in self.manifest.items() if k != 'checksum'},
                              'kind': 'reference-dual-recognition-v2',
                              'reference_phrase_ledger': self.dual.ledger['checksum']})
