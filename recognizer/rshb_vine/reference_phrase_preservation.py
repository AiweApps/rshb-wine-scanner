"""Preserve positive producer-exclusive evidence from frozen bound references."""
from copy import deepcopy
from pathlib import Path
from rshb_vine.io import read_json, verify, sha256
from rshb_vine.typed_catalog_lexicon import normalize
from rshb_vine.catalog_phrase_text import phrases


class ReferencePhrasePreservation:
    def __init__(self, parent, root, protocol_path, ledger_path, expected_ledger_checksum):
        self.parent = parent
        root = Path(root)
        protocol = verify(read_json(protocol_path))
        for path, checksum in protocol['sources'].items():
            if sha256(root / path) != checksum:
                raise ValueError('Reference anchor source changed: ' + path)
        self.ledger = verify(read_json(ledger_path))
        if self.ledger['checksum'] != expected_ledger_checksum or self.ledger['protocol'] != protocol['checksum']:
            raise ValueError('Unexpected reference anchor ledger')

    def apply(self, baseline, raw_b3):
        candidate = self.parent.apply(baseline, raw_b3)
        result = deepcopy(baseline)
        proposed = {str(t['instance_id']): t for t in candidate.get('targets', [])}
        traces = []
        for target, old_trace in zip(result.get('targets', []), candidate['alternative_b3_consensus']['targets']):
            trace = deepcopy(old_trace)
            traces.append(trace)
            if not trace['changed']:
                continue
            sid = str(target['instance_id'])
            before, after = trace['before'], trace['after']
            owner = self.ledger['producer_by_slug'].get(before)
            other = self.ledger['producer_by_slug'].get(after)
            # Same producer is not an identity conflict; missing metadata is not negative.
            blocked = []
            if owner and other and owner != other:
                packets = [('standard', self.parent.parent.observations(baseline, sid))]
                isolation = baseline.get('bottle_isolation_evidence', {})
                if isolation.get('attempted') and str(isolation.get('instance_id')) == sid:
                    retry_obs = isolation.get('retry_observations', [])
                    if retry_obs:
                        packets.append(('isolation_retry_separate_coordinates', retry_obs))
                # Never combine polygons belonging to different crop coordinate spaces.
                phrase_sources = {}
                token_sources = {}
                for source, obs in packets:
                    for phrase in phrases(obs):
                        phrase_sources.setdefault(phrase, []).append(source)
                    for o in obs:
                        if o['score'] >= .85:
                            for token in normalize(o['raw_text']).split():
                                token_sources.setdefault(token, []).append(source)
                for token in sorted(set(token_sources) & set(self.ledger['token_anchors_by_slug'].get(before, []))):
                    anchor = self.ledger['token_anchors'][token]
                    if anchor['producer'] == owner:
                        blocked.append({'kind': 'token', 'text': token, 'producer': owner,
                                        'distinct_reference_images': anchor['distinct_reference_images'],
                                        'query_sources': sorted(set(token_sources[token]))})
                for phrase in sorted(set(phrase_sources) & set(self.ledger['phrase_anchors_by_slug'].get(before, []))):
                    anchor = self.ledger['phrase_anchors'][phrase]
                    if anchor['producer'] == owner:
                        blocked.append({'kind': 'phrase', 'text': phrase, 'producer': owner,
                                        'distinct_reference_images': anchor['distinct_reference_images'],
                                        'query_sources': sorted(set(phrase_sources[phrase]))})
            trace['reference_anchor_evidence'] = blocked
            if blocked:
                trace.update(after=before, changed=False, reason='preserve_read_bound_reference_producer_anchor')
            else:
                target['retrieval'] = deepcopy(proposed[sid]['retrieval'])
        if len(result.get('targets', [])) == 1 and any(t['changed'] for t in traces):
            retrieval = result['targets'][0]['retrieval']
            result.update(best_candidate=retrieval['best_candidate'], ranked_candidates=retrieval['ranked_candidates'])
            if result.get('slug') is not None:
                result['slug'] = None
            if 'catalog_additions' in result:
                result['catalog_additions']['related_catalog_slugs'] = retrieval.get('related_catalog_slugs', [])
        result['alternative_b3_consensus'] = {'policy': 'reference-phrase-rank-fused-b3-v2',
                                             'targets': traces, 'calibrated': False}
        result['reference_phrase_preservation'] = {'ledger': self.ledger['checksum'], 'no_new_ocr': True}
        result.pop('timing_ms', None)
        return result
