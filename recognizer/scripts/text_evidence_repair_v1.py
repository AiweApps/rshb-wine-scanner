"""Text-evidence repair v1: open593 audit, fixed protocol and bounded CPU arms over the current selector consumer.

audit     -- open593 image SHAs vs closed/held/calibration/consumed pools (metadata only, before any body read)
protocol  -- pins, arms, gates, budget and combination rule (--draft writes the review draft; sealed refuses overwrite)
run       -- ROOT ONLY: one arm (G0 | A | B | C | combo) through the consumer on all 890 open targets
decide    -- after the single arms: the combination fixed by the protocol rule; after combo: the release gates
Every step writes an exclusive started marker and refuses to overwrite; there are no retries, sweeps or reruns.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/text-evidence-improve-v1'
INTEGRATION = OUT / 'integration'
AUDIT = INTEGRATION / 'audit-open593.json'
PROTOCOL = OUT / 'protocol-v1.json'
PROTOCOL_DRAFT = OUT / 'protocol-draft.json'
C_ADMISSION = INTEGRATION / 'c-admission.json'
COMBO_DECISION = INTEGRATION / 'combo-decision.json'
SINGLE_ARMS = ('A', 'B', 'C')
ARMS = ('G0', *SINGLE_ARMS, 'combo')
ARM_SECONDS = 600
CUMULATIVE_SECONDS = 3000
REAL_GEOMETRY_CALLS_PER_ARM = 80
CURRENT_PROFILE = 'config/zero-target-release-v1-profile.json'
CURRENT_CHECKSUM = '24b67962c1032c2cd9e8cd381edb54a6448484933e952e9eb9802706120b5920'
OWN = ('rshb_vine/text_evidence_repair_v1/__init__.py', 'rshb_vine/text_evidence_repair_v1/composition.py',
       'rshb_vine/text_evidence_repair_v1/experiment.py', 'scripts/text_evidence_repair_v1.py')
FROZEN = ('scripts/recognition_repair_v2.py', 'scripts/zero_target_release_v1.py',
          'rshb_vine/recognition_repair_v2/composition.py', 'rshb_vine/recognition_repair_v2/geometry.py',
          'rshb_vine/recognition_repair_v2/producer.py', 'rshb_vine/recognition_repair_v2/runtime.py',
          'rshb_vine/ocr_candidate_repair_v1/discriminator.py', 'rshb_vine/ocr_candidate_repair_v1/selection.py',
          'rshb_vine/ocr_candidate_repair_v1/guarded.py', 'rshb_vine/ocr_candidate_repair_v1/injection.py',
          'rshb_vine/producer_role_v2/__init__.py', 'rshb_vine/producer_role_v2/roles.py',
          'rshb_vine/producer_role_v2/selection.py', 'rshb_vine/systemic_ranking_v2/evidence.py',
          'rshb_vine/systemic_ranking_v2/selection.py', 'rshb_vine/systemic_ranking_v2/model.py',
          'rshb_vine/system_selection_v4/features.py', 'rshb_vine/system_selection_v4/text.py',
          'rshb_vine/system_selection_v4/context.py', 'rshb_vine/interaction_ranker_v1/evidence.py',
          'rshb_vine/interaction_ranker_v1/source.py', 'rshb_vine/ocr_provenance_v1/contract.py',
          'rshb_vine/learned_selection_features.py', 'rshb_vine/geometric_reference_probe_v1/verifier.py',
          'rshb_vine/b3_only_v1/runtime.py', 'rshb_vine/combined_ranker_v1/identity.py',
          'rshb_vine/b3_ranker_refit_v1/census.py', 'rshb_vine/coherent_challenger_v1/scope.py',
          'rshb_vine/data_protection.py', 'rshb_vine/zero_target_release_v1/route.py',
          'rshb_vine/zero_target_release_v1/release.py', 'runs/systemic-ranking-v2/final/model.json',
          'runs/recognition-repair-v2/class-v3/class-triggered.json', 'runs/recognition-repair-v2/candidate/profile.json',
          'config/recognition-repair-v2-release.json', CURRENT_PROFILE, 'config/recognition-current.json')


def E():
    from rshb_vine.text_evidence_repair_v1 import experiment
    return experiment


def data_inputs():
    e = E()
    return (e.GATE + '/protocol.json', e.GATE + '/receipts/ledger.jsonl', e.GATE + '/compare/paired-rows.jsonl',
            e.COMBINED_R2, e.OPEN_ROWS, e.LEDGER)


def component_sources():
    from rshb_vine.text_evidence_repair_v1.composition import COMPONENT_SOURCES
    return COMPONENT_SOURCES


def write_new(path, obj):
    from rshb_vine.io import write_json
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, obj)


def write_jsonl(path, rows):
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def start(folder, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (folder / 'started.json').open('x') as f:
            json.dump({'protocol_checksum': protocol['checksum'], 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, f)
    except FileExistsError:
        raise SystemExit(str(folder.relative_to(ROOT)) + ' already started; no automatic rerun')


def cmd_audit(a):
    from rshb_vine.io import seal
    result = seal(E().audit_open593(ROOT))
    write_new(AUDIT, result)
    print(json.dumps({k: result[k] for k in ('images', 'pools', 'protection_closed', 'passed')}, indent=1))
    print('hits:', len(result['hits']), 'unknown_role_images:', result['unknown_role_images']['count'])
    if not result['passed']:
        raise SystemExit('open593 overlaps a protected pool; stop before any body read')


def pins(require_components):
    from rshb_vine.io import sha256
    paths = list(OWN + FROZEN + data_inputs())
    missing = []
    for name, files in component_sources().items():
        for p in files:
            if (ROOT / p).exists():
                paths.append(p)
            else:
                missing.append(p)
    if AUDIT.exists():
        paths.append(str(AUDIT.relative_to(ROOT)))
    paths += [p for p in table_inputs() if (ROOT / p).exists()]
    if require_components and missing:
        raise SystemExit('component sources missing: ' + ', '.join(missing))
    return {p: sha256(ROOT / p) for p in dict.fromkeys(paths)}, missing


def table_inputs():
    """Data files a sealed component table was built from; their current bytes must equal the table's record."""
    from rshb_vine.io import read_json, sha256
    paths = []
    for files in component_sources().values():
        for p in files:
            if p.endswith('.json') and (ROOT / p).exists():
                for q, value in (read_json(ROOT / p).get('inputs_sha256') or {}).items():
                    if (ROOT / q).is_file():
                        if sha256(ROOT / q) != value:
                            raise SystemExit('component table input changed since the table was built: ' + q)
                        paths.append(q)
    return paths


def component_tables():
    """Sealed table checksum per table-driven component, pinned into the protocol and passed to the consumer."""
    from rshb_vine.io import read_json
    return {name: read_json(ROOT / p)['checksum'] for name, files in component_sources().items()
            for p in files if p.endswith('.json') and (ROOT / p).exists()}


def build_protocol(require_components):
    from rshb_vine.io import read_json, verify
    e = E()
    current = verify(read_json(ROOT / CURRENT_PROFILE))
    if current['checksum'] != CURRENT_CHECKSUM:
        raise SystemExit('current release profile is not 24b67962')
    audit = verify(read_json(AUDIT)) if AUDIT.exists() else None
    if require_components and (audit is None or not audit['passed']):
        raise SystemExit('open593 audit missing or failed')
    pinned, missing = pins(require_components)
    return {
        'kind': 'text-evidence-repair-v1-protocol', 'written_before_evaluation': True,
        'current': {'profile': CURRENT_PROFILE, 'checksum': CURRENT_CHECKSUM, 'ranker': e.ABLATED_CHECKSUM,
                    'selector': 'repair-v2 99451df0 selection (producer_role, layout_geometry) under the zero-target route'},
        'audit': {'path': str(AUDIT.relative_to(ROOT)), 'checksum': audit and audit['checksum'],
                  'passed': audit and audit['passed'], 'unknown_role_images': audit and audit['unknown_role_images']['count']},
        'population': {'images': e.OPEN_IMAGES, 'targets': e.OPEN_TARGETS, 'roster': e.GATE + '/protocol.json',
                       'sets': ['train137', 'supplemental456'],
                       'use': 'open development/regression gate only; never fit, threshold, weight or rule choice after results; '
                              'validation/test/closed/calibration/protected never read'},
        'consumer': {'selection': 'TextEvidenceSelection(RepairSelectionV2(guarded v1 + 5c0be, [producer_role, layout_geometry]))',
                     'inputs': 'saved raw OCR observations, provenance traced on the originals, original scene line keys; '
                               'saved bodies and observations verified unchanged after every target',
                     'order': 'A normalization -> injection (B proxy) -> frozen ranker -> resolver -> producer_role -> '
                              'name roles (C) -> guard v2 -> layout geometry',
                     'geometry': 'G0: class_v3 cache (parity-proven); other arms reuse the G0 verifier output only under an '
                                 'identical input fingerprint (incumbent, eligible references, trigger conditions, label view, '
                                 'rotation), otherwise the CPU SIFT verifier runs and is logged'},
        'arms': {'G0': [], 'A': ['A'], 'B': ['B'], 'C': ['C'], 'combo': 'fixed rule below over the single-arm results'},
        'arm_conditions': {'C': 'runs only with a root admission file %s (source justification); otherwise not run'
                                % C_ADMISSION.relative_to(ROOT),
                           'single arms': 'only after G0 passed', 'combo': 'only when the rule selects >= 2 components'},
        'gates': {
            'G0': 'no components: all 890 targets equal combined-r2 in selected, raw order and feature digest; observations '
                  'unchanged; exact/probability None',
            'S': 'single arm: 0 labelled correct -> wrong over every labelled slice (focal and non-focal)',
            'K1': 'combo (or the single qualifying arm): 0 labelled correct -> wrong',
            'K2': 'candidate fixes = wrong -> correct in exact_train or real_confirmed (candidate_fixes_pending_front_scope); '
                  'root checks the existing condition/source or raw pixels of every candidate and must confirm >= 1 '
                  'front-label product fix before admission; a back-label fix is no required benefit, while back-label '
                  'regressions still count in K1',
            'K3': 'every G0-correct ledger fixed_vs_old / recovered_after_b3only / guard-blocked / mandatory row stays correct; '
                  'every repair_v1_fix_listed row keeps its selection',
            'K4': 'exact_slug and probability stay None; pools only change by injection; geometry/guard changes listed',
            'K5': 'every changed unlabelled or same-label target reviewed by root before any release step',
            'L1': 'live paired candidate p95 <= max(baseline p95 * 1.2, baseline p95 + 0.3 s) on the predeclared live roster',
            'L2': 'component-only p95 (normalization + short-form propose + name-role stage, excluding SIFT) <= 150 ms; '
                  'CPU replay reports it per target (stage_seconds), live confirms it',
            'descriptive': 'semantic positive-candidate text deltas, per-target replay seconds, geometry calls'},
        'failure': 'any failed gate: diagnose and record the cause; current 24b67962 stays; no rule change after results',
        'combo_rule': {
            'member': 'component arm with 0 labelled regressions and (>= 1 labelled fix, or for A only: >= 1 target with a '
                      'positive-candidate support increase and 0 targets with a positive-candidate contradiction increase)',
            'requires': 'at least one member with >= 1 labelled fix; A never forms a combination alone',
            'C': 'only if admitted and run', 'single': 'one member: that arm is the candidate, no combo run'},
        'budget': {'arm_wall_seconds': ARM_SECONDS, 'cumulative_seconds': CUMULATIVE_SECONDS, 'arms_max': len(ARMS),
                   'real_geometry_calls_per_arm': REAL_GEOMETRY_CALLS_PER_ARM, 'retries': 0,
                   'corrections': 'at most one recorded fix of an actual technical bug per arm; no rule, threshold or gate change',
                   'models': 'no NN, OCR, HTTP or GPU in this protocol; CPU SIFT only through the logged verifier'},
        'recovery': 'an interrupted arm keeps its started marker and partial nothing; a rerun needs a root-recorded technical '
                    'correction; live HTTP/browser and rollback are a later root-driven step',
        'live_later': {'when': 'root-driven, only after K1-K5 and L2',
                       'roster_rule': 'every open real image whose selection changed in the candidate arm, plus the '
                                      'unchanged representatives fixed now: all images with a text trace applied but no '
                                      'selection change, 10 unchanged labelled real images (first by key), 5 negative_nonwine, '
                                      'all real_supplied_roi, W0353 (zero-target route retention), compact API; the rule is '
                                      'not changed after results',
                       'checks': 'paired baseline 8175 vs candidate: selection/features/geometry equal except the replay '
                                 'changes, raw OCR packet unchanged, route/contract fields, L1 latency, browser card'},
        'population_admission': {'unknown_role_images': 'root msg14: the 226 supplemental456 images without protection/'
                                 'unified role are accepted only as inherited admitted open development of the earlier '
                                 'open-http-gate protocol; not independent, never fit; no new role classification'},
        'component_tables': component_tables(),
        'component_sources_missing': missing, 'no_change_after_results': True, 'pins_sha256': pinned}


def cmd_protocol(a):
    from rshb_vine.io import seal, write_json
    if a.draft:
        protocol = seal(dict(build_protocol(False), status='draft_for_review_not_admitted'))
        write_json(PROTOCOL_DRAFT, protocol, replace=True)
    else:
        write_new(PROTOCOL, seal(dict(build_protocol(True), admission=a.admission)))
        protocol = json.loads(PROTOCOL.read_text())
    print(json.dumps({k: protocol[k] for k in ('component_sources_missing', 'budget', 'audit')}, indent=1))


def load_protocol():
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(PROTOCOL))
    changed = [p for p, v in protocol['pins_sha256'].items() if sha256(ROOT / p) != v]
    if changed:
        raise SystemExit('pinned file changed since protocol: ' + ', '.join(changed))
    return protocol


def spent_seconds():
    """Seconds of every finished (summary) or failed (failure) arm; an unresolved started arm blocks new work."""
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


class Budget:
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

    def seconds(self):
        return round(time.perf_counter() - self.started, 1)


class Journal:
    """Append-only per-image rows and geometry calls, flushed after every image; kept on any failure."""

    def __init__(self, folder, geometry):
        self.rows = (folder / 'rows.jsonl').open('x')
        self.calls = (folder / 'geometry-calls.jsonl').open('x')
        self.geometry, self.written_calls, self.last_key, self.targets = geometry, 0, None, 0

    def __call__(self, key, rows):
        for r in rows:
            self.rows.write(json.dumps(r, ensure_ascii=False) + '\n')
        for c in self.geometry.calls[self.written_calls:]:
            self.calls.write(json.dumps(c, ensure_ascii=False) + '\n')
        self.written_calls = len(self.geometry.calls)
        self.rows.flush()
        self.calls.flush()
        self.last_key, self.targets = key, self.targets + len(rows)

    def close(self):
        self.rows.close()
        self.calls.close()


def arm_rows(arm):
    from rshb_vine.io import read_json, sha256, verify
    summary = verify(read_json(INTEGRATION / arm / 'summary.json'))
    for name, value in summary['files_sha256'].items():
        if sha256(INTEGRATION / arm / name) != value:
            raise SystemExit('arm output changed: %s/%s' % (arm, name))
    return summary, E().jsonl(INTEGRATION / arm / 'rows.jsonl')


def components_of(arm):
    from rshb_vine.io import read_json, verify
    if arm == 'G0':
        return ()
    if arm == 'C' and not C_ADMISSION.exists():
        raise SystemExit('component C is not admitted by root')
    if arm == 'combo':
        decision = verify(read_json(COMBO_DECISION))
        if len(decision['members']) < 2:
            raise SystemExit('combination rule selected fewer than two components; no combo arm')
        return tuple(decision['members'])
    return (arm,)


def cmd_run(a):
    from collections import Counter
    from rshb_vine.io import seal, sha256
    e = E()
    protocol = load_protocol()
    components = components_of(a.arm)
    reference = None
    if a.arm != 'G0':
        g0, _ = arm_rows('G0')
        if not g0['G0']:
            raise SystemExit('G0 failed; no component arm')
        reference = {(r['key'], r['instance_id']): r for r in e.jsonl(INTEGRATION / 'G0/geometry-outputs.jsonl')}
    folder = INTEGRATION / a.arm
    budget = Budget()
    start(folder, protocol)
    journal = None
    try:
        consumer = e.Consumer(components, reference, protocol['component_tables'])
        budget.geometry = consumer.geometry
        journal = Journal(folder, consumer.geometry)
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
    summary = {'kind': 'text-evidence-repair-v1-arm', 'arm': a.arm, 'components': list(components),
               'identity': consumer.selection.identity, 'protocol_checksum': protocol['checksum'],
               'targets': len(rows), 'geometry': dict(consumer.geometry.stats, logged_calls=len(consumer.geometry.calls)),
               'contract_violations': sum(1 for r in rows if r['outcome']['exact_or_probability']),
               'components_seconds_p95': p95([r['outcome']['stage_seconds']['components'] for r in rows])}
    if a.arm == 'G0':
        write_jsonl(folder / 'geometry-outputs.jsonl', [dict(v, key=k[0], instance_id=k[1])
                                                         for k, v in consumer.geometry.outputs.items()])
        files.append('geometry-outputs.jsonl')
        parity = Counter(f'{k}:{v}' for r in rows for k, v in r['current_parity'].items())
        summary.update(current_parity=dict(parity),
                       G0=all(all(r['current_parity'].values()) for r in rows) and not summary['contract_violations'])
    else:
        g0, g0_rows = arm_rows('G0')
        comparison = e.compare(g0_rows, rows)
        write_jsonl(folder / 'changes.jsonl', comparison.pop('changes'))
        files.append('changes.jsonl')
        summary.update(comparison=comparison, g0_checksum=g0['checksum'],
                       S=comparison['counts']['regression'] == 0 and not comparison['contract_violations'])
    summary['files_sha256'] = {n: sha256(folder / n) for n in files}
    summary['seconds'] = budget.seconds()
    write_new(folder / 'summary.json', seal(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != 'identity'}, ensure_ascii=False, indent=1))


def p95(values):
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))] if values else None


def member(arm, comparison):
    if comparison['counts']['regression'] or comparison['contract_violations']:
        return False
    if comparison['counts']['fix'] >= 1:
        return True
    s = comparison['semantic']
    return arm == 'A' and s.get('targets_positive_support_up', 0) >= 1 and s.get('targets_positive_contra_up', 0) == 0


def cmd_decide(a):
    from rshb_vine.io import seal
    protocol = load_protocol()
    if not (INTEGRATION / 'combo/summary.json').exists():
        singles = {arm: arm_rows(arm)[0] for arm in SINGLE_ARMS if (INTEGRATION / arm / 'summary.json').exists()}
        if 'A' not in singles or 'B' not in singles or (C_ADMISSION.exists() and 'C' not in singles):
            raise SystemExit('single arms incomplete: ' + ', '.join(sorted(singles)))
        members = [arm for arm, s in singles.items() if member(arm, s['comparison'])]
        gainful = [arm for arm in members if singles[arm]['comparison']['counts']['fix'] >= 1]
        members = members if gainful else []
        decision = {'kind': 'text-evidence-repair-v1-combo-decision', 'protocol_checksum': protocol['checksum'],
                    'rule': protocol['combo_rule'], 'members': members, 'gainful': gainful,
                    'arms': {arm: {'checksum': s['checksum'], 'counts': s['comparison']['counts'],
                                   'semantic': s['comparison']['semantic'], 'member': arm in members}
                             for arm, s in singles.items()},
                    'candidate': ('combo' if len(members) >= 2 else members[0] if members else None)}
        write_new(COMBO_DECISION, seal(decision))
        print(json.dumps(decision, ensure_ascii=False, indent=1))
        if decision['candidate'] in SINGLE_ARMS:
            candidate_gates(protocol)
        return
    candidate_gates(protocol)


def candidate_gates(protocol):
    """K1-K5 and L2 of the candidate fixed by the combo decision (the combo arm or the single qualifying arm)."""
    from rshb_vine.io import read_json, seal, verify
    decision = verify(read_json(COMBO_DECISION))
    arm = decision['candidate']
    if arm is None:
        raise SystemExit('no candidate: no component passed the combination rule; current stays')
    path = INTEGRATION / ('gates-%s.json' % arm)
    summary, _ = arm_rows(arm)
    c = summary['comparison']
    gates = {'K1': c['counts']['regression'] == 0,
             'K2_candidate_fixes_pending_front_scope': c['candidate_fixes_pending_front_scope'],
             'K2': 'root review of source/condition/raw pixels must confirm >= 1 front-label product fix; not a numeric pass',
             'K3': not c['protected_broken'] and not c['fix_listed_changed'], 'K4': not c['contract_violations'],
             'K5_review_items': c['changed_unlabelled'] + c['counts']['changed_same_label'],
             'L2_components_p95_seconds': summary['components_seconds_p95'],
             'L2': summary['components_seconds_p95'] is not None and summary['components_seconds_p95'] <= 0.150}
    result = seal({'kind': 'text-evidence-repair-v1-candidate-gates', 'protocol_checksum': protocol['checksum'],
                   'combo_decision': decision['checksum'], 'candidate_arm': arm, 'components': summary['components'],
                   'arm_summary': summary['checksum'], 'gates': gates, 'by_slice': c['by_slice'],
                   'admission': 'pending root K2 front review and K5 causal review; live L1 later'})
    write_new(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('audit').set_defaults(fn=cmd_audit)
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
