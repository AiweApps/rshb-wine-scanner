"""Expanded recognition with one evidence-preserving bottle retry."""
from pathlib import Path
import time
from rshb_vine.io import read_jsonl, seal
from rshb_vine.unified_recognition import UnifiedRecognition, verify_protocol
from rshb_vine.bottle_isolation_evidence import BottleIsolationEvidence


class IntegratedRecognition(UnifiedRecognition):
    def __init__(self, root, protocol_path):
        protocol = verify_protocol(root, protocol_path)
        super().__init__(root, protocol_path)
        catalog = read_jsonl(Path(root) / protocol['catalog_directory'] / 'catalog.jsonl')
        self.isolation = BottleIsolationEvidence(catalog, self.recognize_without_orphans)
        self.manifest = seal({**{k: v for k, v in self.manifest.items() if k != 'checksum'},
                              'kind': 'integrated-recognition-v2'})

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        result = self.recognize_without_orphans(data, roi)
        result = self.isolation.apply(data, result, roi)
        result = self.orphans.apply(data, result, roi)
        self.refresh_aliases(result)
        result['unified_recognition'] = {
            'manifest': self.manifest['checksum'], 'transport': 'in_process',
            'stages': ['expanded_catalog', 'series', 'sugar',
                       'bottle_isolation_evidence', 'orphan_label_recovery'],
            'retry': 'expanded_catalog_series_sugar_only',
        }
        result.setdefault('timing_ms', {})['total'] = 1000 * (time.perf_counter() - started)
        return result
