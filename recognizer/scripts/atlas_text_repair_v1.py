"""Atlas text repair v1 (component T): static census, protocol and bounded CPU replay over saved actual responses.

census    -- exact static admission of every gallery-eligible catalogue card (no query data, no replay)
protocol  -- pins, arms, gates and budget (--draft writes the review draft; the sealed protocol refuses overwrite)
run       -- ROOT ONLY: arms I (parity with the saved I rows) then IT on the open593 890 targets
atlas     -- ROOT ONLY: targeted saved-body replay of atlas Talu frames, arms I (parity with live) and IT; explanatory
Arm names: I = current cd0b9910 selector (A+B+D+M+I), IT = I + component T; the rejected pairwise guard G is never used.
Every run writes an exclusive started marker and refuses to overwrite; there are no retries, sweeps or reruns.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'runs/atlas-repair-v1/text'
CENSUS = OUT / 'census.json'
PROTOCOL = OUT / 'protocol.json'
PROTOCOL_DRAFT = OUT / 'protocol-draft.json'
SCOPE = 'runs/atlas-repair-v1/scope.json'
DC_PROTOCOL = 'runs/evidence-consistency-v1/selection/protocol.json'
I_ARM = 'runs/evidence-consistency-v1/selection/I'
G0V2_GEOMETRY = 'runs/text-evidence-improve-v2/integration/G0v2/geometry-outputs.jsonl'
PARENT_PROFILE = 'cd0b9910133f4c17df53d7e171ae3c0432a5b8a1252163d42413ef6808d81ece'
FROZEN_INDEX = 'fc866ab8d17192c9181b224e2f036d1ca3eb62769982418ab99a19d3fd8d8c6a'
SERVED_INDEX = '826ad35d9bea79567a180116ff7235523b2f56bcec96be8de285bdc3e72fd075'
TABLE_B = '22a4e4f8eb605b4a164d3da5b2139456e9554e7a813c59fb11fcf51db2de1566'
ATLAS_TALU = (211, 212, 213, 214, 215, 216, 329, 330, 331, 332, 333, 341, 342, 343, 344, 345, 346,
              450, 451, 452, 453, 455, 456)
OWN = ('rshb_vine/atlas_text_repair_v1/__init__.py', 'rshb_vine/atlas_text_repair_v1/index.py',
       'rshb_vine/atlas_text_repair_v1/composition.py', 'rshb_vine/atlas_text_repair_v1/experiment.py',
       'scripts/atlas_text_repair_v1.py')
ARM_SECONDS = 600
REAL_GEOMETRY_CALLS_PER_ARM = 40
ARMS = ('I', 'IT')


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


def static_index():
    """Frozen index, live B proposer and T exactly as the live selector builds them from the catalogue (no models)."""
    from rshb_vine.atlas_text_repair_v1.index import CompositeIdentityIndex
    from rshb_vine.ocr_candidate_repair_v1 import injection as I
    from rshb_vine.text_evidence_repair_v1 import composition as C1, producer_names as P
    selection = P.catalogue_selection(ROOT)
    frozen = I.CatalogPhraseIndex(selection, I.eligible_slugs(ROOT, selection.registry))
    parent = P.build_injection_index(frozen, selection.registry, selection.context, selection.stats, root=ROOT,
                                     table_checksum=TABLE_B)
    served = C1.ProducerNameIndex(frozen, parent, P.RULE['version'])
    if frozen.checksum != FROZEN_INDEX or served.checksum != SERVED_INDEX:
        raise SystemExit('frozen/served index differs from the cd0b9910 responses')
    return selection, frozen, CompositeIdentityIndex(parent, ROOT)


def build_census():
    from collections import Counter
    from rshb_vine.io import digest
    from rshb_vine.atlas_text_repair_v1.index import CLAIM_ROLES
    from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
    selection, frozen, t = static_index()
    conflicts = ReferenceConflicts(ROOT)
    rows = []
    for slug, r in sorted(t.census.items()):
        conflict = conflicts.injection_conflict(slug)
        rows.append(dict(r, reference_conflict=conflict,
                         klass=('rejected' if r['status'] != 'admitted' else
                                'never_added:' + r['never_added'] if r['never_added'] else
                                'addable_alias_only' if r['frozen_indexed'] else 'addable_new')))
    classes = Counter(r['klass'] for r in rows)
    reasons = Counter(x for r in rows for x in r['reasons'])
    affected = [r for r in rows if r['klass'] == 'addable_new']
    claims = {role: {'%s|unknown=%s' % (k[0], k[1]): v for k, v in sorted(
                  Counter((r['claims'][role]['status'], r['claims'][role].get('unknown')) for r in affected).items(),
                  key=str)} for role in CLAIM_ROLES}
    conflict = {role: [r['slug'] for r in affected if r['claims'][role]['status'] == 'conflict'] for role in CLAIM_ROLES}
    alias_cards = [r['slug'] for r in rows if r['aliases'] and r['klass'].startswith('addable')]
    blocking = Counter()
    for r in rows:
        if r['never_added'] == 'more_specific_sibling':
            blocking[('frozen_indexed' if r['frozen_indexed'] else 'not_indexed',
                      'name_roles:' + '+'.join(sorted(set(r['name_roles']))))] += 1
    return {'kind': 'atlas-text-repair-v1-census', 'parent_profile': PARENT_PROFILE,
            'frozen_index_checksum': frozen.checksum, 'served_index_checksum': SERVED_INDEX,
            'composite_checksum': t.checksum, 'registry_checksum': selection.registry.checksum,
            'reference_conflicts_checksum': conflicts.checksum, 'rule': t.rule,
            'eligible_cards': len(t.census), 'universe_cards': len(t.universe), 'frozen_indexed': len(frozen.entries),
            'composite_entries': len(t.entries),
            'classes': dict(sorted(classes.items())), 'reject_reasons': dict(sorted(reasons.items())),
            'more_specific_blocked_by_roles': {'%s|%s' % k: v for k, v in sorted(blocking.items())},
            'addable_new_claim_status': claims,
            'addable_new_claim_conflicts': {k: v for k, v in conflict.items() if v},
            'addable_new_reference_conflicts': [[r['slug'], r['reference_conflict']] for r in affected
                                                if r['reference_conflict']],
            'addable_with_attested_alias': {'cards': len(alias_cards), 'slugs': alias_cards},
            'rows_digest': digest(rows), 'rows': rows,
            'limits': 'static catalogue admission only; no query, replay, model, OCR or HTTP; not a quality claim'}


def cmd_census(a):
    from rshb_vine.io import seal
    census = seal(build_census())
    write_new(CENSUS, census)
    print(json.dumps({k: census[k] for k in ('composite_checksum', 'eligible_cards', 'composite_entries', 'classes',
                                              'reject_reasons', 'more_specific_blocked_by_roles',
                                              'addable_new_claim_status', 'checksum')}, ensure_ascii=False, indent=1))
    print('alias cards:', census['addable_with_attested_alias']['cards'],
          'conflicts:', len(census['addable_new_reference_conflicts']))


def pins():
    from rshb_vine.io import read_json, sha256, verify
    from rshb_vine.decision_consistency_v1.composition import SOURCES as DC_SOURCES
    from rshb_vine.text_evidence_repair_v2.composition import COMPONENT_SOURCES, SOURCES as V2_SOURCES
    i_summary = verify(read_json(ROOT / I_ARM / 'summary.json'))
    paths = [*OWN, *DC_SOURCES, *V2_SOURCES, *(p for v in COMPONENT_SOURCES.values() for p in v),
             'rshb_vine/ocr_candidate_repair_v1/injection.py', 'rshb_vine/positive_variant_text.py',
             'rshb_vine/systemic_ranking_v2/evidence.py', 'rshb_vine/system_selection_v4/text.py',
             'rshb_vine/decision_consistency_v1/experiment.py', 'rshb_vine/text_evidence_repair_v2/experiment.py',
             'rshb_vine/text_evidence_repair_v1/experiment.py', 'config/reference-conflicts-v1.json',
             SCOPE, DC_PROTOCOL, I_ARM + '/summary.json', G0V2_GEOMETRY, 'runs/wine-atlas-20260927/stable-run/'
             'per-image-evaluation.jsonl', str(CENSUS.relative_to(ROOT))]
    out = {p: sha256(ROOT / p) for p in dict.fromkeys(paths)}
    for name, value in i_summary['files_sha256'].items():
        if sha256(ROOT / I_ARM / name) != value:
            raise SystemExit('saved I arm output changed: ' + name)
        out[f'{I_ARM}/{name}'] = value
    return out


def build_protocol():
    from rshb_vine.io import read_json, verify
    census = verify(read_json(CENSUS))
    dc = verify(read_json(ROOT / DC_PROTOCOL))
    scope = verify(read_json(ROOT / SCOPE))
    return {
        'kind': 'atlas-text-repair-v1-protocol', 'written_before_results': True, 'no_change_after_results': True,
        'scope': {'path': SCOPE, 'checksum': scope['checksum']},
        'parent': {'profile': PARENT_PROFILE, 'recipe': ['A', 'B', 'D', 'M', 'I'],
                   'decision_consistency_protocol': {'path': DC_PROTOCOL, 'checksum': dc['checksum']},
                   'frozen_index_checksum': FROZEN_INDEX, 'served_index_checksum': SERVED_INDEX},
        'component': {'T': census['rule'], 'composite_checksum': census['composite_checksum'],
                      'census': {'path': str(CENSUS.relative_to(ROOT)), 'checksum': census['checksum'],
                                 'classes': census['classes']}},
        'hypothesis': 'the frozen 0.6 core-name gate drops cards whose complete identity (full name + specific producer) '
                      'is read on the target (atlas Talu CS Reserve: core rezerv 0.4233, plus RESERVE/Резерв); '
                      'catalogue composite admission with sibling-specificity adds such cards without loosening the gate',
        'consumer': {'base': 'decision_consistency_v1.experiment.Consumer (A+B+D+M over RepairSelectionV2 guarded v1 + '
                             '5c0be, [producer_role, layout_geometry]) built once',
                     'I': 'compose(base, (I,)) = the cd0b9910 selector',
                     'IT': 'fresh compose(base, (I,)) with atlas_text_repair_v1.composition.attach on that same object '
                           '(the one install used by the release on all five owners)',
                     'geometry': 'current layout-geometry stage (not the rejected pairwise guard G): saved verifier '
                                 'outputs of the text-evidence-v2 base arm G0v2 under an identical input fingerprint; a '
                                 'miss runs the same frozen CPU verifier on the original pixels only within '
                                 'budget.real_geometry_calls_per_arm, cached by (target, fingerprint) for later arms of the '
                                 'same process and logged; beyond the budget the target is blocked and listed. No NN/OCR/HTTP',
                     'inputs': 'saved raw OCR, provenance traced on the originals, original scene line keys; SHA-checked'},
        'arms': {'I': 'first; parity with %s/rows.jsonl on %s for all 890 targets, 0 blocked' % (
                     I_ARM, ', '.join(__import__('rshb_vine.atlas_text_repair_v1.experiment',
                                                 fromlist=['PARITY_FIELDS']).PARITY_FIELDS)),
                 'IT': 'only after I parity; same 890 targets'},
        'atlas_diagnostic': {'indices': list(ATLAS_TALU), 'arms': ['I', 'IT'],
                             'I_condition': 'replayed selected, feature digest and injection equal the saved live '
                                            'cd0b9910 record per target (geometry-triggered targets are blocked)',
                             'role': 'already-consumed owner-external diagnostic; explains Talu only, never an '
                                     'independent or quality-labelled result'},
        'gates': {'S': '0 labelled correct -> wrong over every slice, 0 contract violations, 0 protected broken, '
                       '0 fix-listed changed',
                  'review': 'every labelled and unlabelled selection change and every T census row (proposed/admitted/'
                            'skipped with reasons) reviewed by root with source before any admission',
                  'blocked': 'blocked targets are reported, not counted as unchanged; root decides whether saved '
                             'geometry sources are needed',
                  'benefit': 'scope benefit gate: >=1 visually confirmed fix, otherwise no utility claim',
                  'latency': 'T propose p95 reported; live paired HTTP only later under the scope latency gate',
                  'regression': 'any significant regression rejects T; causal explanation, no threshold sweep'},
        'budget': {'arm_wall_seconds': ARM_SECONDS, 'arms_max': len(ARMS),
                   'real_geometry_calls_per_arm': REAL_GEOMETRY_CALLS_PER_ARM,
                   'real_geometry': 'root-only; same frozen CPU verifier, exact input fingerprint caching, no NN/OCR; '
                                    'set to 0 at seal to block every miss instead',
                   'retries': 0, 'sweeps': 0, 'text_fixed_designs': 1},
        'candidate_rule': 'none automatic: root selects after review',
        'protected': 'no web50/closed75/web232 test46/RESET locked reading',
        'pins_sha256': pins()}


def cmd_protocol(a):
    from rshb_vine.io import seal, write_json
    if a.draft:
        protocol = seal(dict(build_protocol(), status='draft_for_review_not_admitted'))
        write_json(PROTOCOL_DRAFT, protocol, replace=True)
    else:
        write_new(PROTOCOL, seal(dict(build_protocol(), admission=a.admission)))
        protocol = json.loads(PROTOCOL.read_text())
    print(json.dumps({k: protocol[k] for k in ('arms', 'budget', 'component', 'checksum')}, ensure_ascii=False, indent=1))


def load_protocol():
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(PROTOCOL))
    changed = [p for p, v in protocol['pins_sha256'].items() if sha256(ROOT / p) != v]
    if changed:
        raise SystemExit('pinned file changed since protocol: ' + ', '.join(changed))
    return protocol


def consumer(protocol):
    from rshb_vine.io import read_json, verify
    from rshb_vine.atlas_text_repair_v1 import experiment as E
    reference = {(r['key'], r['instance_id']): r for r in E.X.jsonl(ROOT / G0V2_GEOMETRY)}
    tables = verify(read_json(ROOT / DC_PROTOCOL))['component_tables']
    real = protocol['budget']['real_geometry_calls_per_arm'] if protocol else 0
    return E, E.Consumer(reference, tables, ROOT, real)


def check_identity(c, protocol):
    t = c.selection.composite_identity
    if t.checksum != protocol['component']['composite_checksum']:
        raise SystemExit('attached T checksum differs from the protocol census')


class Budget:
    def __init__(self):
        self.started = time.perf_counter()

    def check(self, done):
        if time.perf_counter() - self.started > ARM_SECONDS:
            raise SystemExit('arm budget exceeded after ' + done)

    def seconds(self):
        return round(time.perf_counter() - self.started, 1)


def run_arm(E, c, arm, protocol, folder):
    from rshb_vine.io import seal, sha256
    selection = c.arm(arm)
    if arm == 'IT':
        check_identity(c, protocol)
    start(folder, protocol)
    budget = Budget()
    stream = (folder / 'rows.jsonl').open('x')

    def sink(key, rows):
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
        stream.flush()

    try:
        rows = E.open_rows(c, budget, sink)
    except BaseException as error:
        import traceback
        stream.close()
        write_new(folder / 'failure.json', {'arm': arm, 'protocol_checksum': protocol['checksum'], 'error': repr(error),
                                            'traceback': traceback.format_exc(), 'seconds': budget.seconds(),
                                            'rerun': 'only after a root-recorded technical correction'})
        raise
    stream.close()
    write_jsonl(folder / 'geometry-blocked.jsonl', c.geometry.blocked)
    write_jsonl(folder / 'geometry-calls.jsonl', c.geometry.calls)
    files = ['rows.jsonl', 'geometry-blocked.jsonl', 'geometry-calls.jsonl']
    stage = [r['outcome']['stage_seconds'].get('producer_names_propose') for r in rows if r['outcome']]
    summary = {'kind': 'atlas-text-repair-v1-arm', 'arm': arm, 'identity': selection.identity,
               'protocol_checksum': protocol['checksum'], 'targets': len(rows),
               'blocked': sum(1 for r in rows if r['outcome'] is None),
               'geometry': dict(c.geometry.stats, blocked=len(c.geometry.blocked)),
               'contract_violations': sum(1 for r in rows if r['outcome'] and r['outcome']['exact_or_probability']),
               'propose_seconds_p95': _p95(stage)}
    if arm == 'I':
        saved = E.X.jsonl(ROOT / I_ARM / 'rows.jsonl')
        result = E.parity(rows, saved)
        summary.update(parity=result, I=result['passed'] and not summary['contract_violations'])
    else:
        _, base = arm_rows('I', need_parity=True)
        comparison, census = E.compare(base, rows, E.V2.Producers(c.inner.registry, c.rp.mapping))
        write_jsonl(folder / 'changes.jsonl', comparison.pop('changes'))
        write_jsonl(folder / 'census-t.jsonl', census)
        files += ['changes.jsonl', 'census-t.jsonl']
        summary.update(comparison=comparison,
                       S=(comparison['counts']['regression'] == 0 and not comparison['contract_violations']
                          and not comparison['protected_broken'] and not comparison['fix_listed_changed']))
    summary['files_sha256'] = {n: sha256(folder / n) for n in files}
    summary['seconds'] = budget.seconds()
    write_new(folder / 'summary.json', seal(summary))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('identity', 'comparison')}, ensure_ascii=False, indent=1))
    if arm == 'IT':
        print(json.dumps({k: summary['comparison'][k] for k in ('counts', 'by_slice', 'changed_unlabelled', 'census',
                                                                'blocked')}, ensure_ascii=False, indent=1))


def _p95(values):
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))] if values else None


def arm_rows(arm, need_parity=False):
    from rshb_vine.io import read_json, sha256, verify
    from rshb_vine.atlas_text_repair_v1 import experiment as E
    folder = OUT / 'open' / arm
    summary = verify(read_json(folder / 'summary.json'))
    for name, value in summary['files_sha256'].items():
        if sha256(folder / name) != value:
            raise SystemExit('arm output changed: %s/%s' % (arm, name))
    if need_parity and not summary.get('I'):
        raise SystemExit('arm I parity failed; no IT arm')
    return summary, E.X.jsonl(folder / 'rows.jsonl')


def cmd_run(a):
    protocol = load_protocol()
    E, c = consumer(protocol)
    for arm in a.arms:
        run_arm(E, c, arm, protocol, OUT / 'open' / arm)


def cmd_atlas(a):
    from rshb_vine.io import seal, sha256
    protocol = load_protocol()
    E, c = consumer(protocol)
    items = E.atlas_items(ROOT, protocol['atlas_diagnostic']['indices'])
    folder = OUT / 'atlas'
    start(folder, protocol)
    result = {}
    for arm in ARMS:
        c.arm(arm)
        if arm == 'IT':
            check_identity(c, protocol)
        rows = E.atlas_rows(c, ROOT, items)
        write_jsonl(folder / f'{arm}.jsonl', rows)
        write_jsonl(folder / f'{arm}-geometry-calls.jsonl', c.geometry.calls)
        result[arm] = rows
    live = [r for r in result['I'] if r['replay'] is not None]
    mismatch = [[r['index'], r['instance_id']] for r in live
                if (r['replay']['selected'], r['replay']['feature_digest'], r['replay']['injection']['status'],
                    r['replay']['injection']['index_checksum'])
                != (r['live']['selected'], r['live']['feature_digest'], (r['live']['injection'] or {}).get('status'),
                    (r['live']['injection'] or {}).get('index_checksum'))]
    changes = []
    for i, t in zip(result['I'], result['IT']):
        if i['replay'] and t['replay'] and (i['replay']['selected'] != t['replay']['selected']
                                            or t['replay']['composite']['status'] != 'no_composite_match'):
            changes.append({'index': i['index'], 'instance_id': i['instance_id'], 'gt': i['gt_candidates'][:1],
                            'before': i['replay']['selected'], 'after': t['replay']['selected'],
                            'composite': {k: t['replay']['composite'][k] for k in ('status', 'applied', 'counts',
                                                                                  'proposed', 'admitted', 'skipped')}})
    summary = seal({'kind': 'atlas-text-repair-v1-atlas-diagnostic', 'protocol_checksum': protocol['checksum'],
                    'indices': [r['index'] for r in items], 'targets': len(result['I']),
                    'blocked': {arm: [[r['index'], r['instance_id'], r['blocked']] for r in rows if r['replay'] is None]
                                for arm, rows in result.items()},
                    'live_parity': {'compared': len(live), 'mismatch': mismatch, 'passed': not mismatch},
                    'changes': changes,
                    'files_sha256': {n: sha256(folder / n) for arm in ARMS
                                     for n in (f'{arm}.jsonl', f'{arm}-geometry-calls.jsonl')},
                    'role': protocol['atlas_diagnostic']['role']})
    write_new(folder / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('targets', 'blocked', 'live_parity', 'changes')}, ensure_ascii=False,
                     indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('census').set_defaults(fn=cmd_census)
    p = sub.add_parser('protocol')
    p.add_argument('--draft', action='store_true')
    p.add_argument('--admission', help='root admission reference (bridge message); required unless --draft')
    p.set_defaults(fn=cmd_protocol)
    p = sub.add_parser('run')
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    p.set_defaults(fn=cmd_run)
    sub.add_parser('atlas').set_defaults(fn=cmd_atlas)
    a = parser.parse_args()
    if a.cmd == 'protocol' and not a.draft and not a.admission:
        parser.error('--admission is required for the sealed protocol')
    if a.cmd == 'run' and list(a.arms) != [x for x in ARMS if x in a.arms]:
        parser.error('arms run in the order I, IT')
    a.fn(a)


if __name__ == '__main__':
    main()
