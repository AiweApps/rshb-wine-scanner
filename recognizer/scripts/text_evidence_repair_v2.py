"""Text-evidence repair v2: fixed protocol and bounded CPU arms of component D over the current selector consumer.

protocol  -- pins, arms, gates and budget (--draft writes the review draft; the sealed protocol refuses overwrite)
run       -- ROOT ONLY: one arm (G0v2 | BD | final) through the consumer on all 890 open targets
decide    -- after BD (and final when run): the candidate fixed by the protocol rule and its release gates
Every step writes an exclusive started marker and refuses to overwrite; there are no retries, sweeps or reruns.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import text_evidence_repair_v1 as V1  # noqa: E402

OUT = ROOT / 'runs/text-evidence-improve-v2'
INTEGRATION = OUT / 'integration'
PROTOCOL = OUT / 'protocol-v2.json'
PROTOCOL_DRAFT = OUT / 'protocol-v2-draft.json'
K2_ADMISSION = INTEGRATION / 'k2-bd-admission.json'
V1_OUT = 'runs/text-evidence-improve-v1'
V1_PROTOCOL = V1_OUT + '/protocol-v1.json'
V1_AUDIT = V1_OUT + '/integration/audit-open593.json'
V1_G0 = V1_OUT + '/integration/G0'
V1_DECISION = V1_OUT + '/integration/root-v1-decision.json'
TABLE_CHECKSUM = '22a4e4f8eb605b4a164d3da5b2139456e9554e7a813c59fb11fcf51db2de1566'
ARM_COMPONENTS = {'G0v2': (), 'BD': ('B', 'D'), 'final': ('A', 'B', 'D', 'M')}
M_SOURCE = 'rshb_vine/text_evidence_repair_v2/reader_preservation.py'
ARMS = tuple(ARM_COMPONENTS)
ARM_SECONDS = 600
CUMULATIVE_SECONDS = 1800
REAL_GEOMETRY_CALLS_PER_ARM = 80
OWN = ('rshb_vine/text_evidence_repair_v2/__init__.py', 'rshb_vine/text_evidence_repair_v2/producer_consistency.py',
       'rshb_vine/text_evidence_repair_v2/composition.py', 'rshb_vine/text_evidence_repair_v2/experiment.py',
       'scripts/text_evidence_repair_v2.py')
V1_FILES = (*V1.OWN, 'rshb_vine/text_evidence_repair_v1/normalization.py',
            'rshb_vine/text_evidence_repair_v1/producer_names.py', V1_PROTOCOL, V1_AUDIT, V1_DECISION,
            V1_G0 + '/summary.json', V1_G0 + '/rows.jsonl', V1_G0 + '/geometry-outputs.jsonl',
            'runs/text-evidence-improve-v1/identity/short-producer-forms-v1.json')


def E():
    from rshb_vine.text_evidence_repair_v2 import experiment
    return experiment


def write_new(path, obj):
    from rshb_vine.io import write_json
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, obj)


def pins():
    from rshb_vine.io import read_json, sha256
    paths = list(dict.fromkeys((*OWN, *V1_FILES, *V1.FROZEN, *V1.data_inputs())))
    if (ROOT / M_SOURCE).exists():
        paths.append(M_SOURCE)
    table = read_json(ROOT / V1_FILES[-1])
    for q, value in table['inputs_sha256'].items():
        if (ROOT / q).is_file():
            if sha256(ROOT / q) != value:
                raise SystemExit('attested alias table input changed since the table was built: ' + q)
            paths.append(q)
    return {p: sha256(ROOT / p) for p in dict.fromkeys(paths)}


def prefreeze_diagnostics():
    """Development CPU smokes run before the freeze, disclosed with their bytes; none is an acceptance result."""
    from rshb_vine.io import sha256
    folder = OUT / 'diagnosis'
    files = sorted(p for p in folder.rglob('*') if p.is_file()) if folder.exists() else []
    return {'meaning': 'pre-freeze development CPU smoke / causal inspection, not acceptance and not a prior candidate '
                       'evaluation; no rule, weight or threshold was chosen from them',
            'files': {str(p.relative_to(ROOT)): sha256(p) for p in files},
            'unsaved': ['W0293 only: G0v2 consumer parity with v1 G0 (selected/raw order/digest equal), BD wiring and '
                        'A+B+D+M wiring (M reason a_unchanged), author construction checks'],
            'other_diagnostics': 'reviewer M raw-OCR checks (W0470 class) are diagnostic as well, not a gate'}


def build_protocol():
    from rshb_vine.io import read_json, verify
    e = E()
    current = verify(read_json(ROOT / V1.CURRENT_PROFILE))
    if current['checksum'] != V1.CURRENT_CHECKSUM:
        raise SystemExit('current release profile is not 24b67962')
    audit, v1 = verify(read_json(ROOT / V1_AUDIT)), verify(read_json(ROOT / V1_PROTOCOL))
    g0 = verify(read_json(ROOT / V1_G0 / 'summary.json'))
    if not audit['passed'] or not g0['G0'] or v1['component_tables']['B'] != TABLE_CHECKSUM:
        raise SystemExit('v1 audit, v1 G0 or the pinned table differ from the admitted v1 state')
    return {
        'kind': 'text-evidence-repair-v2-protocol', 'written_before_full_gate': True,
        'prefreeze_diagnostics': prefreeze_diagnostics(),
        'hypothesis': 'W0293 diagnosis: the 18 attested short forms are read by the injection only; the scoring path '
                      'reads producer forms from v4 _producer_forms, so an injected card keeps prod.phrase 0. D makes '
                      'the same forms producer forms of v4 phrase, systemic evidence and producer_role_v2 for every '
                      'card of their producers; nothing else changes.',
        'diagnosis': 'runs/text-evidence-improve-v2/diagnosis/w0293-b-arm-inspection.json',
        'current': {'profile': V1.CURRENT_PROFILE, 'checksum': V1.CURRENT_CHECKSUM, 'ranker': e.ABLATED_CHECKSUM},
        'v1': {'protocol': v1['checksum'], 'g0_summary': g0['checksum'], 'audit': audit['checksum'],
               'decision': V1_DECISION, 'kept': 'v1 results, gates and K5 rejection of C unchanged; no GT edited'},
        'population': dict(v1['population'], admission=v1['population_admission']),
        'consumer': {'selection': 'TextEvidenceSelectionV2(RepairSelectionV2(guarded v1 + 5c0be, [producer_role, '
                                  'layout_geometry])); with D the producer_role stage of a per-instance parent copy is '
                                  'AliasProducerStage',
                     'order': 'A normalization -> injection (B proxy, frozen index/context) -> frozen ranker -> '
                              'resolver -> producer_role (D: alias-aware v4/E phrase, frozen producer_role_v2 apply '
                              'over the alias view, one re-rank) -> guard v2 -> layout geometry',
                     'inputs': v1['consumer']['inputs'], 'geometry': v1['consumer']['geometry'].replace('G0', 'G0v2')},
        'component_D': {'rule': 'catalogue-attested-producer-aliases-v1', 'table': V1_FILES[-1],
                        'table_checksum': TABLE_CHECKSUM,
                        'meaning': 'machine-attested producer surface forms (catalogue uniqueness, distinct print on '
                                   'own references, not printed on other producers); not owner-verified identity, no '
                                   'catalogue merge',
                        'invariant': 'RuntimeError (arm failure) when the alias-aware build changes any systemic '
                                     'feature except prod.phrase or catalogue specificities change'},
        'arms': {k: list(v) for k, v in ARM_COMPONENTS.items()},
        'arm_conditions': {
            'G0v2': 'first; no components: all 890 targets equal combined-r2 and the v1 G0 rows in selected, raw order '
                    'and feature digest',
            'BD': 'only after G0v2 passed',
            'final': 'only after BD passed S and root wrote %s confirming >= 1 trusted front-label fix of BD (K2); '
                     'components A+B+D+M (M = reader preservation of A, reviewer module), no C'
                     % K2_ADMISSION.relative_to(ROOT)},
        'gates': dict({k: v1['gates'][k] for k in ('K1', 'K2', 'K3', 'K4', 'K5', 'L1', 'descriptive')},
                      S='arm: 0 labelled correct -> wrong over every labelled slice and 0 contract violations',
                      L2='component-only p95 (normalization + short-form propose + alias producer stage, excluding '
                         'SIFT) <= 150 ms'),
        'ledger': 'all changes including unlabelled; positive-candidate support increases and decreases, contradiction '
                  'increases, the best GT score and the top-2 margin before/after; labelled changes tagged with '
                  'same-producer / wrong-sibling / GT-in-pool before and after; every alias phrase addition lists all '
                  'candidates of the producer that received it',
        'candidate_rule': 'final when it passed S and every labelled fix of BD is still correct in final; otherwise BD '
                          'when it passed S; otherwise none (current stays)',
        'failure': v1['failure'],
        'budget': {'arm_wall_seconds': ARM_SECONDS, 'cumulative_seconds': CUMULATIVE_SECONDS, 'arms_max': len(ARMS),
                   'real_geometry_calls_per_arm': REAL_GEOMETRY_CALLS_PER_ARM, 'retries': 0,
                   'corrections': v1['budget']['corrections'], 'models': v1['budget']['models']},
        'recovery': v1['recovery'], 'live_later': v1['live_later'],
        'component_tables': {'B': TABLE_CHECKSUM, 'D': TABLE_CHECKSUM},
        'component_sources_missing': [] if (ROOT / M_SOURCE).exists() else [M_SOURCE],
        'no_change_after_results': True, 'pins_sha256': pins()}


def cmd_protocol(a):
    from rshb_vine.io import seal, write_json
    if a.draft:
        protocol = seal(dict(build_protocol(), status='draft_for_review_not_admitted'))
        write_json(PROTOCOL_DRAFT, protocol, replace=True)
    else:
        protocol = build_protocol()
        if protocol['component_sources_missing']:
            raise SystemExit('component sources missing: ' + ', '.join(protocol['component_sources_missing']))
        write_new(PROTOCOL, seal(dict(protocol, admission=a.admission)))
        protocol = json.loads(PROTOCOL.read_text())
    print(json.dumps({k: protocol[k] for k in ('arms', 'budget', 'component_tables', 'checksum')}, indent=1))


def load_protocol():
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(PROTOCOL))
    changed = [p for p, v in protocol['pins_sha256'].items() if sha256(ROOT / p) != v]
    if changed:
        raise SystemExit('pinned file changed since protocol: ' + ', '.join(changed))
    return protocol


def spent_seconds():
    from rshb_vine.io import read_json
    total = 0.
    for arm in ARMS:
        folder = INTEGRATION / arm
        if not (folder / 'started.json').exists():
            continue
        done = [folder / n for n in ('summary.json', 'failure.json') if (folder / n).exists()]
        if not done:
            raise SystemExit('arm %s started without summary or failure record; resolve it before new work' % arm)
        total += read_json(done[0])['seconds']
    return total


class Budget(V1.Budget):
    def __init__(self):
        self.started = time.perf_counter()
        self.before = spent_seconds()
        if self.before >= CUMULATIVE_SECONDS:
            raise SystemExit('cumulative replay budget already spent')
        self.geometry = None

    def check(self, done):
        step = time.perf_counter() - self.started
        if step > ARM_SECONDS or self.before + step > CUMULATIVE_SECONDS:
            raise SystemExit('replay budget exceeded (arm %.0fs, cumulative %.0fs) after %s' % (step, self.before + step, done))
        if self.geometry is not None and len(self.geometry.calls) > REAL_GEOMETRY_CALLS_PER_ARM:
            raise SystemExit('real geometry calls exceed %d after %s' % (REAL_GEOMETRY_CALLS_PER_ARM, done))


def arm_rows(arm):
    from rshb_vine.io import read_json, sha256, verify
    summary = verify(read_json(INTEGRATION / arm / 'summary.json'))
    for name, value in summary['files_sha256'].items():
        if sha256(INTEGRATION / arm / name) != value:
            raise SystemExit('arm output changed: %s/%s' % (arm, name))
    return summary, E().jsonl(INTEGRATION / arm / 'rows.jsonl')


def check_condition(arm):
    if arm == 'G0v2':
        return
    g0, _ = arm_rows('G0v2')
    if not g0['G0']:
        raise SystemExit('G0v2 failed; no component arm')
    if arm == 'final':
        bd, _ = arm_rows('BD')
        if not bd['S'] or not K2_ADMISSION.exists():
            raise SystemExit('final needs BD to pass S and the root K2 admission of BD')


def g0_parity(rows):
    """Per target: v2 G0 equals combined-r2 (current_parity) and the v1 G0 row."""
    from collections import Counter
    v1 = {(r['key'], r['instance_id']): r['outcome'] for r in E().jsonl(ROOT / V1_G0 / 'rows.jsonl')}
    counts = Counter()
    for r in rows:
        o, w = r['outcome'], v1[(r['key'], r['instance_id'])]
        for k, v in r['current_parity'].items():
            counts[f'combined_r2.{k}:{v}'] += 1
        for k in ('selected', 'raw_order', 'feature_digest'):
            counts[f'v1_g0.{k}:{o[k] == w[k]}'] += 1
    ok = len(v1) == len(rows) and all(k.endswith(':True') for k in counts)
    return dict(counts), ok


def cmd_run(a):
    from rshb_vine.io import seal, sha256
    e = E()
    protocol = load_protocol()
    check_condition(a.arm)
    components = tuple(protocol['arms'][a.arm])
    reference = None
    if a.arm != 'G0v2':
        reference = {(r['key'], r['instance_id']): r for r in e.jsonl(INTEGRATION / 'G0v2/geometry-outputs.jsonl')}
    folder = INTEGRATION / a.arm
    budget = Budget()
    V1.start(folder, protocol)
    journal = None
    try:
        consumer = e.Consumer(components, reference, protocol['component_tables'])
        budget.geometry = consumer.geometry
        journal = V1.Journal(folder, consumer.geometry)
        rows = e.run_arm(consumer, budget, journal)
        journal.close()
    except BaseException as error:
        import traceback
        if journal is not None:
            journal.close()
        write_new(folder / 'failure.json', {
            'arm': a.arm, 'protocol_checksum': protocol['checksum'], 'error': repr(error),
            'traceback': traceback.format_exc(), 'seconds': budget.seconds(),
            'last_completed_image': journal and journal.last_key, 'targets_written': journal and journal.targets,
            'rerun': 'only after a root-recorded technical correction'})
        raise
    files = ['rows.jsonl', 'geometry-calls.jsonl']
    summary = {'kind': 'text-evidence-repair-v2-arm', 'arm': a.arm, 'components': list(components),
               'identity': consumer.selection.identity, 'protocol_checksum': protocol['checksum'],
               'targets': len(rows), 'geometry': dict(consumer.geometry.stats, logged_calls=len(consumer.geometry.calls)),
               'contract_violations': sum(1 for r in rows if r['outcome']['exact_or_probability']),
               'components_seconds_p95': V1.p95([r['outcome']['stage_seconds']['components'] for r in rows])}
    if 'D' in components:
        aliases = [r['outcome']['text'].get('producer_aliases') or {} for r in rows]
        summary['alias'] = {'targets_alias_v4_added': sum(bool(x.get('alias_v4_added')) for x in aliases),
                            'targets_phrase_changed': sum(bool(x.get('phrase_changes')) for x in aliases),
                            'targets_applied': sum(bool(x.get('applied')) for x in aliases),
                            'targets_revoked': sum(bool(x.get('revoked')) for x in aliases),
                            'extra_reranks': sum(bool(x.get('extra_rerank')) for x in aliases)}
    if a.arm == 'G0v2':
        V1.write_jsonl(folder / 'geometry-outputs.jsonl', [dict(v, key=k[0], instance_id=k[1])
                                                           for k, v in consumer.geometry.outputs.items()])
        files.append('geometry-outputs.jsonl')
        parity, ok = g0_parity(rows)
        summary.update(parity=parity, G0=ok and not summary['contract_violations'])
    else:
        g0, g0_rows = arm_rows('G0v2')
        comparison = e.compare(g0_rows, rows, e.Producers(consumer.inner.registry, consumer.rp.mapping))
        V1.write_jsonl(folder / 'changes.jsonl', comparison.pop('changes'))
        files.append('changes.jsonl')
        summary.update(comparison=comparison, g0_checksum=g0['checksum'],
                       S=comparison['counts']['regression'] == 0 and not comparison['contract_violations'])
    summary['files_sha256'] = {n: sha256(folder / n) for n in files}
    summary['seconds'] = budget.seconds()
    write_new(folder / 'summary.json', seal(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != 'identity'}, ensure_ascii=False, indent=1))


def fixes(arm):
    return {(c['key'], c['instance_id']) for c in E().jsonl(INTEGRATION / arm / 'changes.jsonl') if c['kind'] == 'fix'}


def cmd_decide(a):
    from rshb_vine.io import seal
    protocol = load_protocol()
    bd, _ = arm_rows('BD')
    candidate, reason = None, 'BD failed S'
    if (INTEGRATION / 'final/summary.json').exists():
        final, final_rows = arm_rows('final')
        correct = {(r['key'], r['instance_id']) for r in final_rows if r['outcome']['correct']}
        lost = sorted(fixes('BD') - correct)
        if final['S'] and not lost:
            candidate, reason = 'final', 'final passed S and keeps every BD fix'
        elif bd['S']:
            candidate, reason = 'BD', 'final failed S or lost BD fixes %s' % lost
    elif bd['S']:
        candidate, reason = 'BD', 'final not run'
    if candidate is None:
        write_new(INTEGRATION / 'decision.json', seal({'kind': 'text-evidence-repair-v2-decision',
                                                       'protocol_checksum': protocol['checksum'], 'candidate': None,
                                                       'reason': reason}))
        raise SystemExit('no candidate: ' + reason + '; current stays')
    summary, _ = arm_rows(candidate)
    c = summary['comparison']
    gates = {'S': summary['S'], 'K1': c['counts']['regression'] == 0,
             'K2_candidate_fixes_pending_front_scope': c['candidate_fixes_pending_front_scope'],
             'K2': 'root review of source/condition/raw pixels must confirm >= 1 front-label product fix; not a numeric pass',
             'K3': not c['protected_broken'] and not c['fix_listed_changed'], 'K4': not c['contract_violations'],
             'K5_review_items': c['changed_unlabelled'] + c['counts']['changed_same_label'],
             'L2_components_p95_seconds': summary['components_seconds_p95'],
             'L2': summary['components_seconds_p95'] is not None and summary['components_seconds_p95'] <= 0.150}
    result = seal({'kind': 'text-evidence-repair-v2-decision', 'protocol_checksum': protocol['checksum'],
                   'candidate': candidate, 'reason': reason, 'components': summary['components'],
                   'arm_summary': summary['checksum'], 'gates': gates, 'by_slice': c['by_slice'],
                   'semantic': c['semantic'],
                   'admission': 'pending root K2 front review and K5 causal review; live L1 later'})
    write_new(INTEGRATION / 'decision.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('protocol')
    p.add_argument('--draft', action='store_true')
    p.add_argument('--admission', help='root admission reference (bridge message); required unless --draft')
    p.set_defaults(fn=cmd_protocol)
    p = sub.add_parser('run')
    p.add_argument('--arm', required=True, choices=ARMS)
    p.set_defaults(fn=cmd_run)
    sub.add_parser('decide').set_defaults(fn=cmd_decide)
    a = parser.parse_args()
    if a.cmd == 'protocol' and not a.draft and not a.admission:
        parser.error('--admission is required for the sealed protocol')
    a.fn(a)


if __name__ == '__main__':
    main()
