"""Runtime adapter: attach T_catalog candidates and optional targeted reread to a recognition result as separate channels.

Existing result fields, the control answer and selector output are never modified; the consumer (selector) decides.
"""
import time

from rshb_vine.system_retrieval_v4.text_candidates import CHANNEL, CatalogueTextCandidates
from rshb_vine.system_retrieval_v4.targeted_ocr import TargetedLabelReread

VERSION = 'system-retrieval-v4'


def pool_entry(candidate):
    """Pool row shaped like learned_selection_features candidates, with T_catalog as its own source channel."""
    return {'candidate_id': candidate['candidate_id'], 'product_id': candidate['product_id'],
            'card_slugs': [candidate['slug']], 'representative_slug': candidate['slug'],
            'channels': {CHANNEL: {'rank': candidate['rank'], 'raw': {'slug': candidate['slug'], 'score': candidate['score']}}},
            'provenance': {'text_catalog': {k: candidate[k] for k in (
                'rule', 'name_coverage', 'matched_name_tokens', 'unmatched_name_tokens', 'matched_producer_tokens',
                'matched_grape_tokens', 'support_lines', 'calibrated')}}}


class SystemRetrievalV4:
    def __init__(self, registry, root, reader=None):
        self.text = CatalogueTextCandidates(registry, root)
        self.reread = TargetedLabelReread(reader) if reader is not None else None

    def attach(self, result, image=None):
        started = time.perf_counter()
        if self.reread is not None and image is not None:
            self.reread.apply(image, result)
        supplemental = {t['instance_id']: t['supplemental_observations']
                        for t in (result.get('targeted_label_ocr') or {}).get('targets', [])}
        targets = []
        for evidence in (result.get('product_identity_evidence') or {}).get('targets') or []:
            instance = str(evidence.get('instance_id'))
            observations = list(evidence.get('ocr_observations') or [])
            extra = supplemental.get(instance, [])
            candidates = self.text.generate(observations + extra)
            targets.append({'instance_id': instance, 'ocr_lines_used': len(observations) + len(extra),
                            'targeted_lines_used': len(extra), 'candidates': candidates,
                            'pool_extension': [pool_entry(c) for c in candidates]})
        result[VERSION] = {'version': VERSION, 'channel': CHANNEL, 'targets': targets, 'calibrated': False,
                           'answer_changed': False, 'ms': round((time.perf_counter() - started) * 1000, 2)}
        return result
