"""Decision consistency v1: fixed protocol and bounded CPU arms of components I and G over the fa317ef2 selector.

protocol  -- pins, arms, gates and budget (--draft writes the review draft; the sealed protocol refuses overwrite)
run       -- ROOT ONLY: one replay, arms G0 -> I -> G -> IG (IG only when I and G pass S) on all 890 open targets
Every arm writes an exclusive started marker and refuses to overwrite; there are no retries, sweeps or reruns.
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

OUT = ROOT / 'runs/evidence-consistency-v1/selection'
PROTOCOL = OUT / 'protocol.json'
PROTOCOL_DRAFT = OUT / 'protocol-draft.json'
SCOPE = 'runs/evidence-consistency-v1/scope.json'
ADMISSION = 'runs/evidence-consistency-v1/open593-admission.json'
V2_OUT = 'runs/text-evidence-improve-v2'
V2_PROTOCOL = V2_OUT + '/protocol-v2.json'
V2_FINAL = V2_OUT + '/integration/final'
V2_G0 = V2_OUT + '/integration/G0v2'
LIVE = {'pointer': 'config/recognition-current.json', 'profile': 'config/text-evidence-release-v2-profile.json',
        'manifest': 'config/text-evidence-release-v2-manifest.json'}
LIVE_CHECKSUM = 'fa317ef284b0cc0ffc93e38a7297f4ed45139fb7e544d2888d53e246c92ff62a'
HISTORICAL_POINTER = 'config/recognition-current.json'
ARM_COMPONENTS = {'G0': (), 'I': ('I',), 'G': ('G',), 'IG': ('I', 'G')}
ARMS = tuple(ARM_COMPONENTS)
ARM_SECONDS = 600
REAL_GEOMETRY_CALLS_PER_ARM = 80
OWN = ('rshb_vine/decision_consistency_v1/__init__.py', 'rshb_vine/decision_consistency_v1/identity.py',
       'rshb_vine/decision_consistency_v1/pairwise.py', 'rshb_vine/decision_consistency_v1/composition.py',
       'rshb_vine/decision_consistency_v1/experiment.py', 'scripts/decision_consistency_v1.py')
V2_OUTPUTS = (V2_PROTOCOL, V2_FINAL + '/summary.json', V2_FINAL + '/rows.jsonl', V2_G0 + '/summary.json',
              V2_G0 + '/geometry-outputs.jsonl', V2_OUT + '/integration/root-gate.json')


def E():
    from rshb_vine.decision_consistency_v1 import experiment
    return experiment


def write_new(path, obj):
    from rshb_vine.io import write_json
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, obj)


def live_release():
    from rshb_vine.io import read_json, sha256
    pointer = read_json(ROOT / LIVE['pointer'])
    if pointer.get('checksum') != LIVE_CHECKSUM:
        raise SystemExit('live pointer is not fa317ef2')
    return {'checksum': LIVE_CHECKSUM, 'sha256': {p: sha256(ROOT / p) for p in LIVE.values()},
            'meaning': 'the running 8175 release; never switched or modified by this experiment'}


def historical_pins():
    """v2 protocol pins (written under 24b67962): all must be unchanged except the live pointer, reported separately."""
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(ROOT / V2_PROTOCOL))
    changed = sorted(p for p, v in protocol['pins_sha256'].items() if p != HISTORICAL_POINTER and sha256(ROOT / p) != v)
    if changed:
        raise SystemExit('v2 algorithm/harness pins changed since v2 protocol: ' + ', '.join(changed))
    return protocol, {'protocol': V2_PROTOCOL, 'checksum': protocol['checksum'],
                      'pins_checked': len(protocol['pins_sha256']) - 1,
                      'excluded_pointer': {'path': HISTORICAL_POINTER,
                                           'v2_protocol_sha256': protocol['pins_sha256'][HISTORICAL_POINTER],
                                           'meaning': 'pointer pinned at 24b67962 time; the live pointer is pinned '
                                                      'under live_release, not compared with this value'}}


def pins():
    from rshb_vine.io import sha256
    return {p: sha256(ROOT / p) for p in dict.fromkeys((*OWN, *V2_OUTPUTS, SCOPE, ADMISSION))}


def build_protocol():
    from rshb_vine.io import read_json, verify
    v2, historical = historical_pins()
    final = verify(read_json(ROOT / V2_FINAL / 'summary.json'))
    admission = verify(read_json(ROOT / ADMISSION))
    if final['components'] != ['A', 'B', 'D', 'M'] or not final['S']:
        raise SystemExit('v2 final arm is not the admitted A+B+D+M arm')
    return {
        'kind': 'decision-consistency-v1-protocol', 'written_before_results': True,
        'scope': SCOPE, 'population_admission': {'path': ADMISSION, 'checksum': admission['checksum']},
        'population': 'exact open593 / 890 targets of text-evidence-improve-v2 (437 labelled); development, not '
                      'independent quality; web50/closed75/test46/RESET locked never read',
        'live_release': live_release(), 'historical_v2': historical,
        'hypotheses': {
            'I': 'resolver identity preservation compares NFKC strings, so a phrase shared by both cards in different '
                 'scripts is reported lost (web50 C030 diagnosis); removing only such spans changes no other reason',
            'G': 'guard v2 exclusivity is pool-global, so a word the agreed visual candidate does not claim is '
                 '"shared" when unrelated products in the pool carry it (web50 C027 diagnosis); a pairwise test '
                 'releases such blocks while the other four conditions stay'},
        'consumer': {'base': 'TextEvidenceSelectionV2(A,B,D,M) over RepairSelectionV2(guarded v1 + 5c0be, '
                              '[producer_role, layout_geometry]) = v2 final arm consumer, built once',
                     'compose': 'rshb_vine.decision_consistency_v1.composition.compose (instance-local copy; I: '
                                'legacy/resolver copy + first pre_guard stage; G: first post_guard, release-only)',
                     'order': 'A -> injection(B) -> ranker -> resolver -> [I re-resolve] -> producer_role(D) -> M -> '
                              'guard v2 (frozen) -> [G pairwise release] -> layout geometry',
                     'install': 'owners runtime/target/repair/inner/release.selection all replaced by the composed '
                                'copy; T.block publishes the composed identity with parent_adapter; parent sources '
                                'unchanged; serial single consumer only',
                     'geometry': 'fresh fingerprint-keyed cache per arm over %s/geometry-outputs.jsonl; real CPU '
                                 'verifier only on a fingerprint miss, logged' % V2_G0,
                     'inputs': 'saved raw OCR, provenance traced on the originals, original scene line keys; bodies '
                               'and bytes SHA-checked; no body edited'},
        'arms': {k: list(v) for k, v in ARM_COMPONENTS.items()},
        'arm_conditions': {'G0': 'first; parity with %s rows in %s for all 890 targets' % (V2_FINAL, ', '.join(E().PARITY_FIELDS)),
                           'I': 'only after G0 parity', 'G': 'only after G0 parity',
                           'IG': 'only when I and G both pass S'},
        'gates': {'S': '0 labelled correct -> wrong over every slice and 0 contract violations (probability/exact slug)',
                  'K3': 'no protected row broken, no repair-v1 fix-listed row changed',
                  'review': 'every labelled change, every unlabelled change and every census row (I recovered spans; '
                            'G compared targets by class) reviewed by root before any admission',
                  'benefit': 'scope benefit gate (>=1 verified front fix or source-proven consistency defect with 0 '
                             'regressions and explicitly unmeasured quality benefit)',
                  'latency': 'I+G stage p95 reported; live paired HTTP only later under the scope latency gate',
                  'regression': 'any significant regression rejects the arm; causal explanation, no threshold sweep'},
        'review_corrections': {
            'I_script': 'homoglyph/translit only for cyrillic-vs-latin or mixed-script token pairs; same-script '
                        'skeleton equality never recovers; any token with a digit matches only exact; per-token '
                        'script evidence and card_start traced; contiguous same-length sequence; no fuzzy/aliases',
            'I_census': 'per-target history of every I resolve (first stage, producer/M, guard, pairwise, geometry) '
                        'with caller and deep-copied snapshot; census counts all recovered spans, also when the '
                        'final selection is unchanged',
            'G_producer': 'grape-only count vs closed monosort also requires one shared non-empty catalogue producer '
                          'key over all winner and agreed member cards; producer relation same/different/unknown '
                          'traced and split in class counts; other name tokens keep the generic rule',
            'install_owner': 'release.selection added to the checked and replaced owners'},
        'explicit_classes': {'common_name': 'G class released_common_name (counted token claimed by >=1 other pool '
                                            'product, not by the agreed candidate)',
                             'monosort_vs_blend': 'G classes released_closed_monosort_grape (same producer only) and '
                                                  'diagnostic new_block_open_or_blend_grape / '
                                                  'new_block_monosort_grape_producer_not_same (never applied)'},
        'candidate_rule': 'none automatic: root selects after review; IG only if it keeps every fix of I and G and '
                          'passes S; otherwise at most the single arm that passed S with a verified benefit',
        'budget': {'arm_wall_seconds': ARM_SECONDS, 'arms_max': len(ARMS), 'real_geometry_calls_per_arm':
                   REAL_GEOMETRY_CALLS_PER_ARM, 'retries': 0, 'sweeps': 0, 'models': 'no NN/OCR/HTTP; CPU SIFT only '
                   'on geometry fingerprint misses'},
        'component_tables': v2['component_tables'],
        'no_change_after_results': True, 'pins_sha256': pins()}


def cmd_protocol(a):
    from rshb_vine.io import seal, write_json
    if a.draft:
        protocol = seal(dict(build_protocol(), status='draft_for_review_not_admitted'))
        write_json(PROTOCOL_DRAFT, protocol, replace=True)
    else:
        write_new(PROTOCOL, seal(dict(build_protocol(), admission=a.admission)))
        protocol = json.loads(PROTOCOL.read_text())
    print(json.dumps({k: protocol[k] for k in ('arms', 'budget', 'component_tables', 'checksum')}, indent=1))


def load_protocol():
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(PROTOCOL))
    changed = [p for p, v in protocol['pins_sha256'].items() if sha256(ROOT / p) != v]
    if changed:
        raise SystemExit('pinned file changed since protocol: ' + ', '.join(changed))
    historical_pins()
    if live_release() != protocol['live_release']:
        raise SystemExit('live release files changed since protocol')
    return protocol


class Budget:
    def __init__(self, geometry):
        self.started, self.geometry = time.perf_counter(), geometry

    def check(self, done):
        step = time.perf_counter() - self.started
        if step > ARM_SECONDS:
            raise SystemExit('arm budget exceeded (%.0fs) after %s' % (step, done))
        if len(self.geometry.calls) > REAL_GEOMETRY_CALLS_PER_ARM:
            raise SystemExit('real geometry calls exceed %d after %s' % (REAL_GEOMETRY_CALLS_PER_ARM, done))

    def seconds(self):
        return round(time.perf_counter() - self.started, 1)


def arm_rows(arm):
    from rshb_vine.io import read_json, sha256, verify
    summary = verify(read_json(OUT / arm / 'summary.json'))
    for name, value in summary['files_sha256'].items():
        if sha256(OUT / arm / name) != value:
            raise SystemExit('arm output changed: %s/%s' % (arm, name))
    return summary, E().jsonl(OUT / arm / 'rows.jsonl')


def check_condition(arm):
    if arm == 'G0':
        return
    g0, _ = arm_rows('G0')
    if not g0['G0']:
        raise SystemExit('G0 parity failed; no component arm')
    if arm == 'IG' and not all(arm_rows(a)[0]['S'] for a in ('I', 'G')):
        raise SystemExit('IG needs I and G to pass S')


def run_one(consumer, arm, protocol):
    from rshb_vine.io import seal, sha256
    e = E()
    check_condition(arm)
    folder = OUT / arm
    selection = consumer.arm(ARM_COMPONENTS[arm])
    budget = Budget(consumer.geometry)
    V1.start(folder, protocol)
    journal = None
    try:
        journal = V1.Journal(folder, consumer.geometry)
        rows = e.run_arm(consumer, budget, journal)
        journal.close()
    except BaseException as error:
        import traceback
        if journal is not None:
            journal.close()
        write_new(folder / 'failure.json', {
            'arm': arm, 'protocol_checksum': protocol['checksum'], 'error': repr(error),
            'traceback': traceback.format_exc(), 'seconds': budget.seconds(),
            'last_completed_image': journal and journal.last_key, 'targets_written': journal and journal.targets,
            'rerun': 'only after a root-recorded technical correction'})
        raise
    files = ['rows.jsonl', 'geometry-calls.jsonl']
    stage = lambda name: V1.p95([r['outcome']['stage_seconds'].get(name) for r in rows])  # noqa: E731
    summary = {'kind': 'decision-consistency-v1-arm', 'arm': arm, 'components': list(ARM_COMPONENTS[arm]),
               'identity': selection.identity, 'protocol_checksum': protocol['checksum'], 'targets': len(rows),
               'geometry': dict(consumer.geometry.stats, logged_calls=len(consumer.geometry.calls)),
               'contract_violations': sum(1 for r in rows if r['outcome']['exact_or_probability']),
               'components_seconds_p95': V1.p95([r['outcome']['stage_seconds']['components'] for r in rows]),
               'stage_seconds_p95': {'cross_script_identity': stage('cross_script_identity'),
                                     'pairwise_guard': stage('pairwise_guard')}}
    if arm == 'G0':
        parity = e.g0_parity(rows, e.jsonl(ROOT / V2_FINAL / 'rows.jsonl'))
        summary.update(parity=parity, G0=parity['passed'] and not summary['contract_violations'])
    else:
        g0, g0_rows = arm_rows('G0')
        comparison, census_i, census_g = e.compare(g0_rows, rows, e.V2.Producers(consumer.inner.registry,
                                                                                  consumer.rp.mapping))
        for name, data in (('changes.jsonl', comparison.pop('changes')), ('census-i.jsonl', census_i),
                           ('census-g.jsonl', census_g)):
            V1.write_jsonl(folder / name, data)
            files.append(name)
        summary.update(comparison=comparison, g0_checksum=g0['checksum'],
                       S=comparison['counts']['regression'] == 0 and not comparison['contract_violations'])
    summary['files_sha256'] = {n: sha256(folder / n) for n in files}
    summary['seconds'] = budget.seconds()
    write_new(folder / 'summary.json', seal(summary))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('identity', 'comparison')}, ensure_ascii=False, indent=1))
    return summary


def cmd_run(a):
    protocol = load_protocol()
    e = E()
    reference = {(r['key'], r['instance_id']): r for r in e.jsonl(ROOT / V2_G0 / 'geometry-outputs.jsonl')}
    consumer = e.Consumer(reference, protocol['component_tables'])
    for arm in a.arms:
        if arm == 'IG' and not all(arm_rows(x)[0]['S'] for x in ('I', 'G')):
            print('IG skipped: I or G failed S')
            continue
        run_one(consumer, arm, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('protocol')
    p.add_argument('--draft', action='store_true')
    p.add_argument('--admission', help='root admission reference (bridge message); required unless --draft')
    p.set_defaults(fn=cmd_protocol)
    p = sub.add_parser('run')
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    p.set_defaults(fn=cmd_run)
    a = parser.parse_args()
    if a.cmd == 'protocol' and not a.draft and not a.admission:
        parser.error('--admission is required for the sealed protocol')
    if a.cmd == 'run' and list(a.arms) != [x for x in ARMS if x in a.arms]:
        parser.error('arms run in the order G0, I, G, IG')
    a.fn(a)


if __name__ == '__main__':
    main()
