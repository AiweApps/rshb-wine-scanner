"""Core assembly: verified vendored tree + asset pack + provenance receipt -> the runtime descriptor.

The descriptor is model-free and deterministic, so the gateway's expected profile can be computed before the
recognizer is loaded; the loaded runtime reports exactly this checksum on /health/ready and in every answer.
"""
from pathlib import Path

from wine_scanner_core.pack import AssetPack, SourceLedger, canonical, sha256_file
from wine_scanner_core.provenance import Provenance

KIND = 'wine-scanner-core-linux-v6-v1'
PACKAGE = Path(__file__).resolve().parent
PARENT_ROLE = ('provenance: the vendored release this core executes; its device=mps and Apple Vision metadata are '
               'construction values, not what runs here')


def code_sha():
    return {'wine_scanner_core/' + p.name: sha256_file(p) for p in sorted(PACKAGE.glob('*.py'))}


class Core:
    def __init__(self, root, manifest_path, manifest_sha256, threads, workers=4):
        self.root = Path(root).resolve(strict=True)
        self.pack = AssetPack(manifest_path, self.root, manifest_sha256, workers)
        self.ledger = SourceLedger(self.root)
        if self.pack.manifest.get('source_ledger') != self.ledger.ledger['checksum']:
            raise ValueError('Asset pack was exported for another vendored source ledger')
        self.provenance = Provenance(self.root, self.pack.manifest['provenance'], self.ledger)
        self.threads = threads
        parent = self.pack.manifest['parent']
        self.descriptor = dict(kind=KIND, assets_manifest_sha256=manifest_sha256,
                               assets_checksum=self.pack.manifest['checksum'],
                               source_ledger_checksum=self.ledger.ledger['checksum'], core_code=code_sha(),
                               provenance_checksum=self.provenance.receipt['checksum'],
                               v6_descriptor=self.provenance.expected['v6_descriptor'],
                               parent=dict(parent, role=PARENT_ROLE),
                               execution={'platform': 'linux', 'device': 'cpu', 'paddle_threads': threads,
                                          'torch_threads': threads,
                                          'ocr_slots': 'PP-OCRv6 medium det+rec (v6_medium_native) in Vision slots A and B',
                                          'sparse_ocr': 'PP-OCRv5 mobile det + eslav_PP-OCRv5_mobile_rec',
                                          'apple_vision_executed': False, 'calibrated': False})
        self.descriptor['checksum'] = canonical(self.descriptor)

    def verify(self):
        return {'sources': self.ledger.verify(mounted=self.pack.top_level()), 'assets': self.pack.verify(),
                'provenance': self.provenance.verify()}

    def preflight(self):
        """Model-free: the vendored profile chain and the three receipt guards, as the loader runs them."""
        self.provenance.install()
        from rshb_vine.b3_only_v1 import runtime as R
        from rshb_vine.catalog_training_evaluation_v2 import reindex
        from rshb_vine.io import sha256
        from rshb_vine.ocr_v5_v6_v1 import pipeline as P
        from rshb_vine.reference_additions_v1 import admission
        from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
        from rshb_vine.release_next_v1 import release as RN
        if sha256(self.root / P.PROFILE) != P.PROFILE_FILE_SHA256:
            raise ValueError('Active profile file changed')
        profile = RN.load_profile(self.root, P.PROFILE)
        base = RN.load_base_profile(self.root, profile['parent_profile'])
        a = {k: str(self.root / v) for k, v in R.ARM_INPUTS.items()}
        arm, _ = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                     a['fit_admission'])
        if arm['encoder_id'] != base['b3_encoder_id']:
            raise ValueError('Restored arm is not the pinned B3 encoder')
        gallery, _ = RN._gallery(self.root, base, arm)
        admission.load(self.root)
        conflicts = ReferenceConflicts(self.root)
        descriptor = P.adapter_descriptor('v6_medium', 'v6_medium_native')
        if descriptor['checksum'] != self.provenance.expected['v6_descriptor']:
            raise ValueError('V6 adapter descriptor differs from the exported one')
        return {'profile_checksum': profile['checksum'], 'base_profile_checksum': base['checksum'],
                'encoder_id': arm['encoder_id'], 'gallery_references': len(gallery['references']),
                'reference_conflicts_checksum': conflicts.checksum, 'v6_descriptor': descriptor['checksum'],
                'receipt': self.provenance.check_used(), 'models_loaded': False}

