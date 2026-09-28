"""One deterministic T_catalog → selector-v4 ``extra_candidates`` contract for runtime and training.

Input is only the same-instance OCR observations already present in a runtime receipt
(``product_identity_evidence.targets[*].ocr_observations``). No reread, no labels, no GT.
The text score is an uncalibrated match strength, never a confidence; absent or weak OCR
yields no candidates and no veto.
"""
from pathlib import Path

from rshb_vine.io import digest, sha256
from rshb_vine.system_retrieval_v4.adapter import pool_entry
from rshb_vine.system_retrieval_v4.text_candidates import CHANNEL, LIMIT, LINE_MIN_SCORE, CatalogueTextCandidates

CONTRACT_VERSION = 'text-runtime-integration-v1-contract'
SOURCE = CHANNEL
OBSERVATION_POLICY = 'same_instance_runtime_ocr_observations_only_no_reread'
SCORE_KIND = 'uncalibrated_text_match_strength_not_confidence'
SOURCE_PATHS = (
    'rshb_vine/text_runtime_integration_v1/contract.py',
    'rshb_vine/system_retrieval_v4/text_candidates.py', 'rshb_vine/system_retrieval_v4/normalize.py',
    'rshb_vine/system_retrieval_v4/adapter.py', 'config/producer-brand-aliases.json',
    'rshb_vine/frozen_candidate_selector.py', 'rshb_vine/gallery_variant_text.py',
    'rshb_vine/positive_variant_text.py', 'rshb_vine/resolution/identity.py',
)


class TextCandidateContract:
    def __init__(self, registry, root):
        root = Path(root)
        self.registry = registry
        self.generator = CatalogueTextCandidates(registry, root)
        self.source_checksums = {path: sha256(root / path) for path in SOURCE_PATHS}

    def describe(self):
        body = {'version': CONTRACT_VERSION, 'source': SOURCE, 'observation_policy': OBSERVATION_POLICY,
                'line_min_score': LINE_MIN_SCORE, 'limit': LIMIT, 'score_kind': SCORE_KIND,
                'identity_registry_checksum': self.registry.checksum, 'sources_sha256': self.source_checksums,
                'cards_indexed': len(self.generator.cards), 'hard_veto': False, 'calibrated': False}
        return body | {'checksum': digest(body)}

    def extra_candidates(self, observations):
        """[{'slug','source','rank','score','evidence'}] exactly as SystemSelectionV4.select expects."""
        output = []
        for candidate in self.generator.generate(list(observations or [])):
            slug = candidate['slug']
            if slug not in self.registry.cards:
                raise ValueError('T_catalog card absent from identity registry: ' + slug)
            entry = pool_entry(candidate)
            output.append({'slug': slug, 'source': SOURCE, 'rank': candidate['rank'], 'score': candidate['score'],
                           'evidence': {'contract': CONTRACT_VERSION, 'score_kind': SCORE_KIND,
                                        'candidate_id': entry['candidate_id'], 'product_id': entry['product_id'],
                                        'text_catalog': entry['provenance']['text_catalog']}})
        return output
