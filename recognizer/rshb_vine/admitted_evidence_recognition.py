"""Candidate using previously admitted reference facts in positive selectors."""
from pathlib import Path
from rshb_vine.integrated_recognition import IntegratedRecognition
from rshb_vine.admitted_grape_evidence import AdmittedGrapeEvidence
from rshb_vine.orphan_label_recovery_v2 import OrphanLabelRecovery
from rshb_vine.io import read_jsonl, seal


class AdmittedEvidenceRecognition(IntegratedRecognition):
    def __init__(self, root, protocol_path):
        super().__init__(root, protocol_path)
        from rshb_vine.unified_recognition import verify_protocol
        protocol = verify_protocol(root, protocol_path)
        catalog = read_jsonl(Path(root) / protocol['catalog_directory'] / 'catalog.jsonl')
        adapter = AdmittedGrapeEvidence(root, Path(root) / protocol['grape_protocol'])
        receipt = adapter.install(self, catalog)
        previous = self.orphans
        self.orphans = OrphanLabelRecovery(previous.labels, previous.bottles,
                                           self.recognize_without_orphans)
        self.manifest = seal({**{k: v for k, v in self.manifest.items() if k != 'checksum'},
                              'kind': 'admitted-evidence-recognition-v3',
                              'admitted_grape_evidence': receipt,
                              'orphan_text_contract': 'v2'})
