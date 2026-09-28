"""Selector payload and OCR provenance side-car from a saved receipt or from a live runtime result.

Receipt replay and the live adapter call the same ``trace_result``; historical evidence packets (the
older fit sources) use the packet description of ``ocr_provenance_v1``. No OCR or model is run here.
"""
from rshb_vine.io import digest, read_json, verify
from rshb_vine.ocr_provenance_v1 import contract
from rshb_vine.ranker_dataset_v1.guard_compare import _payload

HISTORICAL = ('historical_open_saved', 'historical_error42')


def trace_result(result, instance_id, observations):
    primary, supplements = contract.from_runtime_result(result, instance_id)
    return contract.trace(observations, primary, supplements)


def trace_packet(packet, observations):
    primary, supplements = contract.from_evidence_packet(packet)
    return contract.trace(observations, primary, supplements)


def row_input(root, row):
    """(payload, provenance records, trace report, source kind) of one roster row."""
    payload = _payload(root, row)
    observations = payload['observations']
    if row['source'] in HISTORICAL:
        source_id = row['query_id'].split('/target/')[0].split('/', 1)[1]
        evidence = verify(read_json(root / 'runs/evidence-selector-v1/evidence' / (digest(source_id) + '.json')))
        packet = next(t for t in evidence['targets'] if str(t['instance_id']) == str(row['instance_id']))
        if packet['ocr_packet'].get('observations', []) != observations:
            raise ValueError('Evidence packet differs from payload observations: ' + row['query_id'])
        records, report = trace_packet(packet, observations)
        return payload, records, report, 'historical_evidence_packet'
    result = verify(read_json(root / row['receipt_path']))['result']
    records, report = trace_result(result, row['instance_id'], observations)
    return payload, records, report, 'runtime_receipt'


def check_trace(records, report, observations):
    problems = []
    if len(records) != len(observations) or any(r is None for r in records):
        problems.append('untraced_line')
    if report['supplement_blocks_unmatched']:
        problems.append('unmatched_supplement')
    if report['primary_lines_with_empty_polygon']:
        problems.append('empty_polygon_in_primary')
    if any(r['native_score'] != o.get('score') for r, o in zip(records, observations)):
        problems.append('native_score_differs')
    return problems
