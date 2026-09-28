"""Development candidate: the current fa317ef2 TextEvidenceRelease with exactly one of R or O installed.

The release is loaded as the factory loads it; the component is installed once on that graph. Every request
runs inside the R request scope (bounded to outer no-ROI requests; ROI and its bottles=all pass unchanged).
The response keeps the fa317ef2 identity fields of the graph and gains a ``reading_recovery_v1`` block naming
the frozen descriptor; it is never activated from here. Public slug and probability stay None.
"""
from pathlib import Path

from rshb_vine.io import local_path, read_json, seal, sha256, verify

PARENT_PROFILE = 'config/recognition-current.json'
PARENT_CHECKSUM = 'fa317ef284b0cc0ffc93e38a7297f4ed45139fb7e544d2888d53e246c92ff62a'
COMPONENTS = ('R', 'O')
DESCRIPTOR_KIND = 'reading-recovery-v1-candidate-descriptor'
OUT = 'runs/evidence-consistency-v1/reading'
SOURCES = ('rshb_vine/reading_recovery_v1/__init__.py', 'rshb_vine/reading_recovery_v1/route.py',
           'rshb_vine/reading_recovery_v1/rotated_ocr.py', 'rshb_vine/reading_recovery_v1/census.py',
           'rshb_vine/reading_recovery_v1/candidate.py', 'scripts/reading_recovery_v1.py')
BLOCK = 'reading_recovery_v1'


def descriptor_path(component):
    return f'{OUT}/candidate-{component}/descriptor.json'


def freeze_descriptor(root, component):
    root = Path(root).resolve()
    if component not in COMPONENTS:
        raise ValueError('component must be R or O')
    parent = verify(read_json(root / PARENT_PROFILE))
    if parent['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Current profile is not fa317ef2')
    census = verify(read_json(root / OUT / 'census.json'))
    return seal({'kind': DESCRIPTOR_KIND, 'component': component, 'parent_profile': PARENT_PROFILE,
                 'parent_checksum': PARENT_CHECKSUM, 'parent_profile_sha256': sha256(root / PARENT_PROFILE),
                 'census_checksum': census['checksum'], 'sources_sha256': {p: sha256(root / p) for p in SOURCES},
                 'calibrated': False, 'release_admitted': False, 'weights_changed': False})


def load_descriptor(root, path):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if descriptor.get('kind') != DESCRIPTOR_KIND or descriptor.get('component') not in COMPONENTS:
        raise ValueError('Not a reading-recovery candidate descriptor')
    changed = [p for p, s in descriptor['sources_sha256'].items() if sha256(root / p) != s]
    if changed or sha256(root / descriptor['parent_profile']) != descriptor['parent_profile_sha256']:
        raise ValueError('Candidate sources or parent profile changed since freeze: ' + ', '.join(changed[:3]))
    return descriptor


class ReadingRecoveryCandidate:
    def __init__(self, root, component):
        from rshb_vine.reading_recovery_v1 import route as R, rotated_ocr as O
        from rshb_vine.text_evidence_repair_v2.release import TextEvidenceRelease
        root = Path(root).resolve()
        self.descriptor = load_descriptor(root, descriptor_path(component))
        release = TextEvidenceRelease(root, self.descriptor['parent_profile'])
        if release.profile['checksum'] != PARENT_CHECKSUM:
            raise ValueError('Loaded release is not fa317ef2')
        self.component, self.release, self.profile = component, release, self.descriptor
        self.installation = R.install(release) if component == 'R' else O.install(release, root)
        self.manifest = seal({'kind': 'reading-recovery-v1-candidate-runtime',
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': release.manifest['checksum'], 'parent_profile_checksum': PARENT_CHECKSUM,
                              'component': component, 'installation': self.installation, 'activated': False,
                              'calibration_status': 'unknown'})

    def __getattr__(self, name):
        return getattr(self.release, name)

    def recognize(self, data, roi=None, bottles='addressed'):
        from rshb_vine.reading_recovery_v1 import route as R
        with R.request_scope(roi, bottles) as scope:
            result = self.release.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        truthful = self._label_view_contract(result) if self.component == 'R' else []
        result[BLOCK] = {'label_view_contract_entries': truthful,
                         'component': self.component, 'descriptor_checksum': self.descriptor['checksum'],
                         'runtime_checksum': self.manifest['checksum'], 'parent_profile_checksum': PARENT_CHECKSUM,
                         'route_events': scope['events'], 'calibrated': False, 'probability': None,
                         'release_admitted': False}
        return result

    @staticmethod
    def _label_view_contract(result):
        """The frozen target contract publishes a target's bbox as label_bbox; for an R label view (the full bottle,
        no localized label) that entry is republished with label_bbox None and the bottle-sized view kept apart."""
        views = {str(t['instance_id']): t for t in result.get('targets') or [] if t.get('label_localized') is False}
        changed = []

        def walk(node, path):
            if isinstance(node, dict):
                target = views.get(str(node.get('instance_id')))
                if target is not None and 'label_bbox' in node and node['label_bbox'] is not None:
                    node.update(label_view_bbox=node['label_bbox'], label_bbox=None, label_localized=False,
                                label_view_source=target.get('label_view_source'))
                    changed.append(path)
                for k, v in node.items():
                    walk(v, path + '/' + str(k))
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, path + '[%d]' % i)

        if views:
            walk(result.get('target_contract'), '/target_contract')
        return changed
