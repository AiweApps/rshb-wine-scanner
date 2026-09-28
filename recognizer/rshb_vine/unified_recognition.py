"""In-process expanded catalogue, typed evidence and physical orphan recovery.

An experimental composition, not an admitted release. The orphan retry calls
only the non-orphan path, so a recovered crop cannot recursively spawn retries.
"""
from collections import defaultdict
from pathlib import Path
import time

from rshb_vine.io import read_json, read_jsonl, verify, seal, sha256
from rshb_vine.sugar_evidence_challenger import SugarEvidence


class CatalogSugarEvidence(SugarEvidence):
    """Use the same verified ledger with an explicitly supplied catalogue.

New catalogue metadata is not promoted into verified sweetness evidence.
"""
    def __init__(self, root, catalog):
        self.catalog = {row['slug']: row for row in catalog}
        self.ledger = verify(read_json(
            Path(root) / 'runs/sugar-evidence-v1/reference-ledger.json'))['records']


def verify_protocol(root, protocol_path):
    root = Path(root)
    protocol = verify(read_json(protocol_path))
    for filename, checksum in protocol['sources'].items():
        if sha256(root / filename) != checksum:
            raise ValueError('Unified frozen dependency changed: ' + filename)
    return protocol


class UnifiedRecognition:
    def __init__(self, root, protocol_path=None):
        root = Path(root)
        protocol_path = protocol_path or root / 'runs/unified-recognition-v1/protocol.json'
        protocol = verify_protocol(root, protocol_path)
        import torch
        from rshb_vine.catalog_additions import ExpandedCatalogPipeline
        from rshb_vine.series_evidence_challenger import SeriesEvidenceChallenger
        from rshb_vine.orphan_label_recovery import OrphanLabelRecovery
        from rshb_vine.models import Detector
        from rshb_vine.label_detector import LabelDetector
        from rshb_vine.bottle_rescue import LowBottleDetector

        torch.set_num_threads(2)
        directory = root / protocol['catalog_directory']
        catalog = read_jsonl(directory / 'catalog.jsonl')
        self.expanded = ExpandedCatalogPipeline(root, directory)
        self.series = SeriesEvidenceChallenger(catalog)
        self.sugar = CatalogSugarEvidence(root, catalog)
        self.reverse_aliases = defaultdict(list)
        for alias, canonical in self.expanded.aliases.items():
            self.reverse_aliases[canonical].append(alias)
        # CPU is intentionally identical to the 8167 geometry experiment.
        # The expanded parent's MPS detector is retained for its existing path.
        detector = Detector(root, 'cpu')
        detector.processor.size = {'height': 640, 'width': 640}
        self.orphans = OrphanLabelRecovery(
            LabelDetector(root / 'runs/label-detector-real-v2', 'cpu'),
            LowBottleDetector(detector), self.recognize_without_orphans)
        self.manifest = seal({
            'kind': 'unified-recognition-v1', 'protocol': protocol['checksum'],
            'parent': self.expanded.manifest, 'weights_changed': False,
            'canonical_slugs': self.expanded.manifest['canonical_slugs'],
            'sources': protocol['sources'], 'calibrated': False,
            'admission': 'pending', 'transport': 'in_process',
        })

    def refresh_aliases(self, result):
        for target in result.get('targets', []):
            retrieval = target['retrieval']
            retrieval['related_catalog_slugs'] = sorted(
                self.reverse_aliases.get(retrieval.get('best_candidate'), []))
        result.setdefault('catalog_additions', {}).update(
            manifest=self.expanded.manifest['catalog_additions'], weights_changed=False,
            related_catalog_slugs=sorted(
                self.reverse_aliases.get(result.get('best_candidate'), [])))
        return result

    def recognize_without_orphans(self, data, roi=None):
        result = self.expanded.recognize(data, roi)
        if result.get('decision') == 'invalid_image':
            return result
        result = self.sugar.apply(self.series.apply(result))
        return self.refresh_aliases(result)

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        result = self.recognize_without_orphans(data, roi)
        result = self.orphans.apply(data, result, roi)
        self.refresh_aliases(result)
        result['unified_recognition'] = {
            'manifest': self.manifest['checksum'], 'transport': 'in_process',
            'stages': ['expanded_catalog', 'series', 'sugar', 'orphan_label_recovery'],
            'retry': 'expanded_catalog_series_sugar_without_orphan_recursion',
        }
        result.setdefault('timing_ms', {})['total'] = 1000 * (time.perf_counter() - started)
        return result
