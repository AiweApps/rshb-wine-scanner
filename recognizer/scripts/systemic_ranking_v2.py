"""Systemic ranking v2: table → diagnose → monotone → protocol; cv → final → evaluate need root approval."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/systemic-ranking-v2'
V1 = ROOT / 'runs/interaction-ranker-v1'
ROSTER = V1 / 'fit-roster.jsonl'
ROSTER_SHA256 = '46e3ba0b028f966d2d600072793a975c951fdf70b26131d4dc03d75393f62552'
TABLE = OUT / 'tables/fit.jsonl'
TABLE_SUMMARY = OUT / 'tables/fit.summary.json'
APPROVAL = OUT / 'root-approval.json'


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))


def _full_rows():
    from rshb_vine.combined_ranker_v1 import roster as combined
    from rshb_vine.io import read_json, verify
    frozen = verify(read_json(ROOT / 'runs/combined-ranker-v1/roster.json'))
    return {r['query_id']: r for r in combined.load_rows(ROOT, frozen, 'fit')}


def _roster_meta():
    from rshb_vine.io import sha256
    if sha256(ROSTER) != ROSTER_SHA256:
        raise SystemExit('interaction-ranker-v1 fit roster changed; immutable input')
    return {r['query_id']: r for r in _jsonl(ROSTER)}


def _selection():
    from rshb_vine.interaction_ranker_v1 import evaluate as V
    from rshb_vine.systemic_ranking_v2.selection import SystemicSelection
    selection = SystemicSelection.from_profile(ROOT)
    return selection, V.Projection(ROOT, selection.registry)


def table(_args):
    from rshb_vine.io import digest, seal, sha256, write_json
    from rshb_vine.systemic_ranking_v2 import evidence as E, table as T
    started = time.perf_counter()
    full, metas = _full_rows(), _roster_meta()
    selection, projection = _selection()
    records = T.build(ROOT, selection, projection, [full[q] for q in sorted(metas)], metas)
    _write_jsonl(TABLE, records)
    parity = Counter(f'{k}:{"ok" if v else "FAIL"}' for r in records for k, v in r['parity'].items())
    positives = Counter()
    for r in records:
        pos = [c for c in r['candidates'] if c['positive']]
        positives['rows_with_positive'] += bool(pos)
        positives['rows_multi_positive'] += len(pos) > 1
        positives['rows_positive_differs_from_frozen'] += any(c['positive'] != c['positive_frozen'] for c in r['candidates'])
    summary = seal({'kind': 'systemic-ranking-v2-fit-table', 'feature_schema': E.SCHEMA_VERSION,
                    'features': list(E.FEATURE_NAMES), 'signs': E.SIGNS, 'rows': len(records),
                    'labels_attached_after_features': True, 'new_visual_or_ocr_inference': False,
                    'positive_semantics': 'snapshot-03 product of any card in the candidate', 'positives': dict(positives),
                    'parity': dict(parity), 'catalog_stats': selection.stats.describe(), 'fit_admitted': False,
                    'roster_sha256': ROSTER_SHA256, 'rows_checksum': digest([r['feature_digest'] for r in records]),
                    'table_sha256': sha256(TABLE), 'seconds': time.perf_counter() - started})
    write_json(TABLE_SUMMARY, summary, replace=True)
    print(json.dumps({k: summary[k] for k in ('rows', 'parity', 'positives', 'catalog_stats', 'seconds')}, indent=1))


PROBES = ('web232:W0213/', 'web232:W0470/', 'web232:W0646/', 'real-extra2-048/', 'rti1-d0e76fd77b5e9598/',
          'rti1-bc617fa47e59232d/')


def _units(rows):
    return {'rows': len(rows), 'components': len({r['group'] for r in rows})}


def _name_cover(c):
    return max((n['cover'] for n in c['evidence']['name']), default=0.)


def diagnose(_args):
    """Once over fit471: where the v1 representation loses information, shown with matched tokens and rivals."""
    from rshb_vine.io import seal, sha256, write_json
    v1 = {r['query_id']: r for r in _jsonl(V1 / 'tables/fit.jsonl')}
    records = _jsonl(TABLE)
    found = defaultdict(list)
    grades = defaultdict(Counter)
    for r in records:
        old = v1[r['query_id']]
        oc = {c['candidate_id']: c for c in old['candidates']}
        gts = [c for c in r['candidates'] if c['positive']]
        negs = [c for c in r['candidates'] if not c['positive'] and not c['loss_ignore']]
        if not gts:
            found['no_positive_in_pool'].append(r)
            found['old103_wrong_pre_resolver'].append(r)
            found['old103_wrong_post_resolver_snapshot03'].append(r)
            continue
        gt = max(gts, key=lambda c: -c['old103_rank'])
        g1 = oc[gt['candidate_id']]
        f, keys = gt['features'], set(gt['evidence']['producer_keys'])
        sib = [c for c in negs if keys & set(c['evidence']['producer_keys'])]
        lines = set(g1['support_lines'].get('name', []))
        if lines and any(lines & set(oc[c['candidate_id']]['support_lines'].get('name', [])) for c in sib):
            found['v1_name_support_shared_with_sibling'].append(r)
            if _name_cover(gt) > max(_name_cover(c) for c in sib):
                found['v1_name_shared__v2_cover_separates'].append(r)
        if g1['features'].get('txt.producer.support', 0.) == 0. and f.get('prod.cover', 0.) > 0:
            found['v1_producer_absent__tokens_read'].append(r)
            if any(not keys & set(c['evidence']['producer_keys']) for c in negs):
                found['v1_producer_absent__tokens_read__other_producer_in_pool'].append(r)
        atoms = {a for n in gt['evidence']['name'] for a in n['atoms']}
        read = {t['token'] for n in gt['evidence']['name'] for t in n['tokens'] if any(t['grades'].values())}
        if read and read <= atoms and any(not keys & set(c['evidence']['producer_keys'])
                                          and read & {t['token'] for n in c['evidence']['name'] for t in n['tokens'] if any(t['grades'].values())}
                                          for c in negs):
            found['gt_name_read_only_typed_atoms_shared_across_producers'].append(r)
        if any(len(l['sources']) > 1 for l in r['query']['lines']):
            found['line_read_by_several_crops_or_readers'].append(r)
        if any(oc[c['candidate_id']]['features'].get('txt.name.sources', 0.) > 1 for c in [gt, *negs]):
            found['v1_name_sources_count_gt1'].append(r)
        for field, tokens in (('name', [t for n in gt['evidence']['name'] for t in n['tokens']]),
                              ('producer', gt['evidence']['producer']['tokens'] or [])):
            for reader in ('v', 'p'):
                best = max((t['grades'][reader] for t in tokens), default=0.)
                grades[f'gt_best_{field}_grade.{reader}'][str(round(best, 1))] += 1
        winner = min(r['candidates'], key=lambda c: c['old103_rank'])
        if not winner['positive']:
            found['old103_wrong_pre_resolver'].append(r)
        if not any(r['old103_selected'] in c['card_slugs'] and c['positive'] for c in r['candidates']):
            found['old103_wrong_post_resolver_snapshot03'].append(r)
        contra = [k for k in winner['features'] if '.contra_g.' in k]
        if not winner['positive'] and contra and not any('.contra_g.' in k for k in gt['features']):
            found['old103_wrong_winner_has_contradiction_gt_none'].append(r)
        above = [c for c in negs if c['old103_rank'] < gt['old103_rank']]
        if any('.contra_g.' in k for c in above for k in c['features']) and not any('.contra_g.' in k for k in gt['features']):
            found['negative_with_contradiction_ranked_above_gt'].append(r)
        for role in ('grape_blend', 'color', 'sugar', 'style'):
            state = gt['evidence']['typed'][role]['state']
            if state == 'unknown_claim' and any(c['features'].get(f'typ.{role}.support_g.v') or c['features'].get(f'typ.{role}.support_g.p') for c in negs):
                found[f'gt_unknown_{role}_claim__negative_supported'].append(r)
    probes = []
    for r in records:
        if not any(p in r['query_id'] for p in PROBES):
            continue
        cands = sorted(r['candidates'], key=lambda c: c['old103_rank'])
        show = [c for c in cands[:3]] + [c for c in cands[3:] if c['positive']]
        probes.append({'query_id': r['query_id'], 'gt03': r['positive_products03'], 'old103': r['old103_selected'],
                       'lines': [(l['raw'][0], l['grades'], l['sources']) for l in r['query']['lines'][:8]],
                       'candidates': [{'id': c['candidate_id'], 'positive': c['positive'], 'old103_rank': c['old103_rank'],
                                       'producer': [(t['token'], t['weight'], t['grades']) for t in c['evidence']['producer']['tokens'] or []],
                                       'name': [[(t['token'], t['weight'], t['grades']) for t in n['tokens']] for n in c['evidence']['name']],
                                       'typed': {k: v['state'] for k, v in c['evidence']['typed'].items() if v['state'] != 'not_observed'},
                                       'visual_ranks': c['evidence']['visual_ranks'],
                                       'features': c['features']} for c in show]})
    result = seal({'kind': 'systemic-ranking-v2-diagnosis', 'scope': 'fit471 only (saved receipts); no model fitted',
                   'denominator': len(records),
                   'baseline_note': 'post-resolver = stored old103_selected projected to snapshot-03 (formal baseline); '
                                    'pre-resolver = learned old103 order top1; the absent-positive row is wrong in both',
                   'counts': {k: _units(v) for k, v in sorted(found.items())},
                   'ids': {k: sorted(r['query_id'] for r in v) for k, v in sorted(found.items()) if len(v) <= 40},
                   'grades': {k: dict(v) for k, v in grades.items()}, 'probes': probes,
                   'inputs_sha256': {'v1_table': sha256(V1 / 'tables/fit.jsonl'), 'v2_table': sha256(TABLE)}})
    write_json(OUT / 'diagnosis.json', result, replace=True)
    print(json.dumps({'counts': result['counts'], 'grades': result['grades']}, indent=1))


def queries(records, exclude=frozenset()):
    import numpy as np
    from rshb_vine.systemic_ranking_v2 import evidence as E
    return [{'id': r['query_id'], 'group': r['group'], 'fold': r['cv_fold'],
             'x': np.asarray([[c['features'].get(n, 0.) for n in E.FEATURE_NAMES] for c in r['candidates']]),
             'positive': [c['positive'] for c in r['candidates']], 'keep': [not c['loss_ignore'] for c in r['candidates']]}
            for r in records if r['fit_eligible'] and r['query_id'] not in exclude]


def readiness(_args):
    """Real FIT matrices: shapes, finiteness, masks, analytic vs finite-difference gradient; nothing is fitted."""
    import numpy as np
    from rshb_vine.io import seal, sha256, write_json
    from rshb_vine.systemic_ranking_v2 import evidence as E, model as M
    excluded = _excluded_from_fit()
    qs = queries(_jsonl(TABLE), excluded)
    kept, skipped, mass, groups = M.prepare(qs)
    x_all = np.concatenate([q['x'] for q in kept])
    scale = np.maximum(x_all.std(axis=0), M.Config().scale_floor)
    blocks = [(q['x'] / scale, q['positive'], m) for q, m in zip(kept, mass)]
    w = np.asarray([.1 * E.SIGNS[n] if E.SIGNS[n] else .05 for n in E.FEATURE_NAMES])
    loss, grad = M._objective(w, blocks, M.Config().l2)
    errors = []
    for j in range(len(w)):
        step = np.zeros(len(w))
        step[j] = 1e-6
        numeric = (M._objective(w + step, blocks, 1.)[0] - M._objective(w - step, blocks, 1.)[0]) / 2e-6
        errors.append(abs(numeric - grad[j]) / max(1., abs(numeric)))
    per_fold = {}
    for k in range(3):
        tk, ts, _, tg = M.prepare([q for q in qs if q['fold'] != k])
        per_fold[str(k)] = {'train_queries': len(tk), 'train_groups': tg, 'skipped': len(ts),
                            'held_rows': sum(q['fold'] == k for q in qs)}
    usage = {n: int((x_all[:, j] != 0).sum()) for j, n in enumerate(E.FEATURE_NAMES)}
    result = seal({'kind': 'systemic-ranking-v2-readiness', 'queries': len(qs), 'excluded_from_fit_loss': len(excluded),
                   'kept': len(kept), 'skipped': skipped,
                   'groups': groups, 'candidates': int(len(x_all)), 'features': len(E.FEATURE_NAMES),
                   'finite': bool(all(np.isfinite(q['x']).all() for q in qs)),
                   'gradient_check_max_relative_error': float(max(errors)), 'loss_at_check_point': float(loss),
                   'nonzero_candidates_per_feature': usage, 'scales': dict(zip(E.FEATURE_NAMES, map(float, scale))),
                   'loss_masked_candidates': int(sum((~np.asarray(q['keep'])).sum() for q in qs)),
                   'folds': per_fold, 'fits_run': 0, 'table_sha256': sha256(TABLE)})
    write_json(OUT / 'readiness.json', result, replace=True)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checksum', 'scales', 'nonzero_candidates_per_feature')}, indent=1))


def parity(_args):
    """Live select() path vs receipt evaluate() on every runtime-receipt fit row (same features, same old103)."""
    from rshb_vine.interaction_ranker_v1.source import row_input
    from rshb_vine.io import read_json, seal, verify, write_json
    full, metas = _full_rows(), _roster_meta()
    table_rows = {r['query_id']: r for r in _jsonl(TABLE)}
    selection, _ = _selection()
    counts, failures = Counter(), []
    for qid in sorted(metas):
        row = full[qid]
        payload, provenance, report, kind = row_input(ROOT, row)
        if kind != 'runtime_receipt':
            counts['historical_packet_not_live_shaped'] += 1
            continue
        selection.current_result = deepcopy(verify(read_json(ROOT / row['receipt_path']))['result'])
        try:
            live = selection.select(**{k: deepcopy(v) for k, v in payload.items()})
        finally:
            selection.current_result = None
        record = selection.records.pop()
        ok = (record['systemic_feature_digest'] == table_rows[qid]['feature_digest']
              and record['legacy_selected'] == row['selected_slug'] == live['proposal']['representative_slug'])
        counts['live_equals_replay:' + ('ok' if ok else 'FAIL')] += 1
        if not ok:
            failures.append(qid)
    result = seal({'kind': 'systemic-ranking-v2-live-replay-parity', 'counts': dict(counts), 'failures': failures,
                   'note': 'live select() with the saved in-flight result vs evaluate() on the replay payload; CPU only'})
    write_json(OUT / 'parity.json', result, replace=True)
    print(json.dumps(result['counts']))


CODE = ('rshb_vine/systemic_ranking_v2/__init__.py', 'rshb_vine/systemic_ranking_v2/evidence.py',
        'rshb_vine/systemic_ranking_v2/model.py', 'rshb_vine/systemic_ranking_v2/selection.py',
        'rshb_vine/systemic_ranking_v2/table.py', 'scripts/systemic_ranking_v2.py',
        'rshb_vine/interaction_ranker_v1/evidence.py', 'rshb_vine/interaction_ranker_v1/source.py',
        'rshb_vine/interaction_ranker_v1/evaluate.py', 'rshb_vine/system_selection_v4/features.py',
        'rshb_vine/system_selection_v4/text.py', 'rshb_vine/system_selection_v4/context.py',
        'rshb_vine/combined_ranker_v1/identity.py', 'rshb_vine/combined_ranker_v1/roster.py',
        'rshb_vine/ranker_integration_v3/masks.py', 'rshb_vine/ocr_provenance_v1/contract.py',
        'rshb_vine/learned_selection_features.py', 'rshb_vine/candidate_discriminators.py',
        'rshb_vine/runtime_product_selection.py', 'rshb_vine/training_admission.py', 'rshb_vine/data_protection.py',
        'rshb_vine/systemic_ranking_v2/runtime.py', 'rshb_vine/candidate_listwise.py', 'rshb_vine/product_selection.py',
        'rshb_vine/catalog/product_registry.py', 'rshb_vine/recognition_runtime.py', 'rshb_vine/recognition_api.py',
        'rshb_vine/coherent_challenger_v1/active.py', 'rshb_vine/coherent_challenger_v1/identity.py',
        'rshb_vine/target_misses_v1/route.py', 'rshb_vine/system_selection_v4/roster.py')
INPUTS = ('runs/interaction-ranker-v1/fit-roster.jsonl', 'runs/combined-ranker-v1/roster.json',
          'runs/system-selection-v4/roster.json', 'runs/all-open-errors-v1/register.json',
          'config/recognition-old103-v1.json', 'runs/selector-discriminators-v1/model.json',
          'runs/identity-link-review-v1/candidate-bundle-handoff.json',
          'data/product-identity-v1/snapshot-03-candidate/manifest.json',
          'runs/systemic-ranking-v2/tables/fit.jsonl', 'runs/systemic-ranking-v2/fit-cohort-intersections.json',
          'runs/selector-discriminators-v1/protocol.json', 'config/product-identity-current.json',
          'data/product-identity-v1/snapshot-03-candidate/registry.json',
          'data/product-identity-v1/snapshot-03-candidate/claims.json', 'config/recognition-coherent-v1.json')


def _pins():
    from rshb_vine.io import sha256
    from rshb_vine.system_selection_v4.context import SOURCE_PATHS
    return {p: sha256(ROOT / p) for p in sorted({*CODE, *INPUTS, *SOURCE_PATHS})}


def _excluded_from_fit():
    from rshb_vine.io import read_json, verify
    doc = verify(read_json(OUT / 'fit-cohort-intersections.json'))
    if doc.get('resolution', {}).get('decision') != 'excluded_from_fit_loss_scored_oof':
        raise SystemExit('fit-cohort intersection resolution missing')
    return {r['query_id'] for r in doc['rows']}


def protocol(_args):
    from rshb_vine.io import read_json, seal, verify, write_json
    from rshb_vine.systemic_ranking_v2 import evidence as E, model as M
    from dataclasses import asdict
    ready = verify(read_json(OUT / 'readiness.json'))
    table_summary = verify(read_json(TABLE_SUMMARY))
    live = verify(read_json(OUT / 'parity.json'))
    document = seal({
        'kind': 'systemic-ranking-v2-protocol', 'date': '2026-09-24', 'current_changed': False,
        'model': {'version': M.MODEL_VERSION, 'config': asdict(M.Config()), 'objective':
                  'multi-positive listwise softmax L_q = logsumexp(s_kept) - logsumexp(s_pos), group mass Q/(G n_g), '
                  'L2 1.0 on scaled weights, L-BFGS-B from zero; no sweep, no seed',
                  'constraints': 'sign bounds per feature (SIGNS, including every per-reader contradiction and verified-'
                                 'contradiction dimension <= 0); pool.control_proposal effective weight in [0, 1] logit as a '
                                 'fixed experiment assumption (its real contribution/dominance is measured after fit, not assumed)',
                  'features': list(E.FEATURE_NAMES), 'signs': E.SIGNS, 'feature_schema': E.SCHEMA_VERSION,
                  'tie_policy': M.TIE_POLICY, 'resolver': 'unchanged old103 existing-evidence resolver after ranking'},
        'data': {'fit': 'frozen interaction-ranker-v1 fit roster: 471 rows, 46 components, 3 folds as stored (cv_fold); '
                        'prospective fit445: 26 downloaded117/rshb_query_v2 role-conflict rows quarantined from every fit '
                        'loss, scaler and the CV gate (same groups/folds); historical 471 roster immutable',
                 'positives': 'snapshot-03 product of any candidate card vs snapshot-03 projection of frozen GT',
                 'loss_masks': 'combined identity.loss_ignore (positives never masked)', 'skipped': ready['skipped'],
                 'new_rows': 'none; admission-review proposals (18 fit / 5 internal) wait for a separate root decision and are not read',
                 'not_used': ['locked/consumed/calibration/protected data', 'new eval labels', 'held rows',
                              'canonical/confusability pairs', 'synthetic972']},
        'trial_budget': {'cv_fits': 3, 'final_fits': 'at most 1', 'hyperparameter_search': 'none', 'retries': 'none'},
        'rules': {
            'final_fit': 'OOF post-resolver (snapshot-03 product) fixes - regressions >= 1 vs old103 on the identical fit445 '
                         'rows (old103 not retrained); excluded26 reported only as role-conflict diagnostics',
            'release_candidate': ['live==replay parity on every replayed row', 'internal117: fixes >= regressions (post-resolver)',
                                  'dev108: fixes >= regressions (post-resolver)',
                                  'open register out-of-roster single-target rows: fixes >= regressions',
                                  'open register out-of-roster REAL single-target rows: fixes > regressions (strict)',
                                  'monotone check on real rows: 0 violations', 'selection p95 <= old103 p95 + 0.25 s'],
            'stop': 'if the CV rule fails: no final fit, diagnosis only; if a release gate fails: no release, diagnosis only; '
                    'no retraining, no threshold/feature change on evaluated outcomes',
            'reporting': 'every regression id reported for root decision; all fixes/regressions with ids, rows/components/lineage, pre- and post-resolver, closeup/ordinary, '
                         'real/synthetic/catalog slices of the register separately; target choice unchanged (same targets)'},
        'evidence': {'readiness': ready['checksum'], 'fit_table': table_summary['checksum'],
                     'table_parity': table_summary['parity'], 'live_replay_parity': live['counts'],
                     'diagnosis': verify(read_json(OUT / 'diagnosis.json'))['checksum'],
                     'gradient_check_max_relative_error': ready['gradient_check_max_relative_error']},
        'contrast': {'interaction_ranker_v1': 'depth-3 GBRT over any-token support x reader best scores, no control: '
                                              'OOF 457->454; replaced by specificity-weighted coverage, rival-aware '
                                              'producer/exclusivity, per-reader grades, sign-bounded linear head',
                     'ranker_criterion_v2': 'v4 features + free-sign unbounded control, L2 0.1: OOF 458, dev108 103->99 '
                                            '(cause of those regressions not established here); here control is bounded '
                                            'and the text evidence representation differs'},
        'fit_cohort_intersections': {'path': 'runs/systemic-ranking-v2/fit-cohort-intersections.json',
                                     'rows': 26, 'policy': 'quarantined until explicit superseding fit authority is evidenced; '
                                     'not fit/scaler/loss/gate; replayed only as role-conflict diagnostics, never an independent test'},
        'pin_verification': 'every cv/final/evaluate call re-hashes all sources_sha256 and refuses on any change',
        'sources_sha256': _pins()})
    write_json(OUT / 'protocol.json', document, replace=True)
    print(document['checksum'])


def _approved():
    from rshb_vine.io import read_json, verify
    if not APPROVAL.exists():
        raise SystemExit('Fit/evaluation blocked: no root approval (runs/systemic-ranking-v2/root-approval.json)')
    approval = read_json(APPROVAL)
    protocol = verify(read_json(OUT / 'protocol.json'))
    if approval.get('decision') != 'fit_approved' or approval.get('protocol_checksum') != protocol['checksum']:
        raise SystemExit('Root approval does not name the current protocol checksum')
    changed = {p: h for p, h in _pins().items() if protocol['sources_sha256'].get(p) != h}
    if changed:
        raise SystemExit('Pinned sources changed after the protocol: ' + ', '.join(sorted(changed)))
    return protocol


def _new(path, document):
    from rshb_vine.io import write_json
    if path.exists():
        raise SystemExit(f'{path.relative_to(ROOT)} exists; immutable')
    write_json(path, document)


def _fit(qs, label):
    from rshb_vine.io import seal
    from rshb_vine.systemic_ranking_v2 import evidence as E, model as M
    started = time.perf_counter()
    model = seal(M.fit(qs, M.Config(), E.FEATURE_NAMES, E.SIGNS) | {'label': label, 'fit_seconds': time.perf_counter() - started})
    M.check_loadable(model)
    return model


def _meta_fn(metas):
    return lambda row: {'group': metas[row['query_id']]['group'], 'lineage': metas[row['query_id']]['lineage_group'],
                        'slice': 'closeup' if str(row['instance_id']).startswith('closeup') else 'ordinary'}


def _control_contribution(model, records):
    """Rows where zeroing the bounded 8175 term alone changes the pre-resolver top1 (real candidates)."""
    import numpy as np
    from rshb_vine.systemic_ranking_v2 import evidence as E, model as M
    j = E.FEATURE_NAMES.index('pool.control_proposal')
    out = Counter()
    for r in records:
        x = np.asarray([[c['features'].get(n, 0.) for n in E.FEATURE_NAMES] for c in r['candidates']])
        s = M.score(model, x)
        y = x.copy()
        y[:, j] = 0.
        t = M.score(model, y)
        a, b = int(np.argmax(s)), int(np.argmax(t))
        out['rows'] += 1
        if a != b:
            out['control_decisive'] += 1
            out['decisive_toward_positive' if r['candidates'][a]['positive'] else 'decisive_toward_negative'] += 1
        margin = np.sort(s)[-1] - np.sort(s)[-2] if len(s) > 1 else 0.
        out['top1_margin_lt_control_weight'] += bool(margin < model['effective_weights']['pool.control_proposal'])
    return dict(out, effective_weight=model['effective_weights']['pool.control_proposal'])


def cv(_args):
    from rshb_vine.interaction_ranker_v1 import evaluate as V
    from rshb_vine.io import read_json, seal, verify, write_json
    from rshb_vine.training_admission import require_training_admission
    protocol = _approved()
    full, metas = _full_rows(), _roster_meta()
    require_training_admission(list(full.values()), ROOT)
    (OUT / 'cv').mkdir(parents=True, exist_ok=True)
    _new(OUT / 'cv/started.json', {'protocol_checksum': protocol['checksum'], 'started': time.time(), 'fits': 3})
    table_rows = _jsonl(TABLE)
    qs = queries(table_rows, _excluded_from_fit())
    selection, projection = _selection()
    items, fits, parity = [], [], {}
    for k in range(3):
        model = _fit([q for q in qs if q['fold'] != k], f'cv-fold-{k}')
        write_json(OUT / f'cv/fold-{k}.json', model)
        fits.append({'fold': k, 'model': model['checksum'], 'training': model['training'],
                     'effective_weights': model['effective_weights'], 'at_upper_bound': model['at_upper_bound']})
        selection.use_ranker(model)
        held = [full[q] for q, m in sorted(metas.items()) if m['cv_fold'] == k]
        fold_items, parity[str(k)] = V.roster_items(ROOT, selection, held, _meta_fn(metas), projection)
        fits[-1]['control_contribution_held'] = _control_contribution(
            model, [r for r in table_rows if r['cv_fold'] == k])
        items += fold_items
    result = V.tally(items)
    crossing = {r['query_id'] for r in verify(read_json(OUT / 'fit-cohort-intersections.json'))['rows']}
    gate = V.tally([i for i in items if i['row'] not in crossing])
    result = {'gate_fit445': gate, 'all471_descriptive': result,
              'excluded26_role_conflict_diagnostic': V.tally([i for i in items if i['row'] in crossing])}
    post = gate['post_resolver']
    allowed = post['fixes']['rows'] - post['regressions']['rows'] >= 1
    decision = seal({'kind': 'systemic-ranking-v2-cv-decision', 'protocol_checksum': protocol['checksum'], 'fits': fits,
                     'oof': result, 'replay_parity': parity, 'final_fit_allowed': allowed,
                     'identity_for_correctness': projection.describe(),
                     'changed': [{k: i[k] for k in ('row', 'old_slug', 'new_slug', 'old_correct', 'new_correct', 'slice')}
                                 for i in items if i['changed']],
                     'rule': protocol['rules']['final_fit'],
                     'note': 'train-side grouped CV on previously analysed rows; not independent quality'})
    _new(OUT / 'cv/decision.json', decision)
    print(json.dumps({'final_fit_allowed': allowed, 'post': {k: post[k] for k in ('old_top1', 'new_top1', 'fixes', 'regressions', 'fix_ids', 'regression_ids')},
                      'pre': {k: gate['pre_resolver'][k] for k in ('old_top1', 'new_top1')}}, indent=1))


def final(_args):
    from rshb_vine.io import read_json, verify
    from rshb_vine.training_admission import require_training_admission
    protocol = _approved()
    decision = verify(read_json(OUT / 'cv/decision.json'))
    if decision['protocol_checksum'] != protocol['checksum'] or not decision['final_fit_allowed']:
        raise SystemExit('Final fit not allowed by the frozen CV rule')
    require_training_admission(list(_full_rows().values()), ROOT)
    (OUT / 'final').mkdir(parents=True, exist_ok=True)
    _new(OUT / 'final/started.json', {'protocol_checksum': protocol['checksum'], 'started': time.time()})
    table_rows = _jsonl(TABLE)
    model = _fit(queries(table_rows, _excluded_from_fit()), 'final')
    _new(OUT / 'final/model.json', model)
    _new(OUT / 'final/control-contribution-fit471.json', _control_contribution(model, table_rows))
    print(model['checksum'], json.dumps(model['effective_weights'], indent=1))


def monotone(args):
    """Real fit rows: raising a contradiction never raises, raising support never lowers a candidate's score."""
    import numpy as np
    from rshb_vine.io import read_json, seal, verify, write_json
    from rshb_vine.systemic_ranking_v2 import evidence as E, model as M
    model = verify(read_json(ROOT / args.model))
    checks, violations = Counter(), []
    for r in _jsonl(TABLE):
        x = np.asarray([[c['features'].get(n, 0.) for n in E.FEATURE_NAMES] for c in r['candidates']])
        base = M.score(model, x)
        for j, n in enumerate(E.FEATURE_NAMES):
            sign = E.SIGNS[n]
            if not sign:
                continue
            y = x.copy()
            y[:, j] += .1 if n.endswith('.gap') else .5
            delta = M.score(model, y) - base
            bad = delta > 1e-12 if sign < 0 else delta < -1e-12
            checks['contradiction' if sign < 0 else 'support'] += len(delta)
            if bad.any():
                violations.append({'query_id': r['query_id'], 'feature': n, 'candidates': int(bad.sum())})
    result = seal({'kind': 'systemic-ranking-v2-monotone-check', 'model': model['checksum'], 'checks': dict(checks),
                   'violations': violations[:50], 'violation_count': len(violations)})
    write_json(OUT / f'monotone-{model["label"]}.json', result, replace=True)
    print(json.dumps({'checks': dict(checks), 'violations': len(violations)}))


def _rosters():
    from rshb_vine.io import read_json, verify
    from rshb_vine.system_selection_v4 import roster as v4_roster
    combined = verify(read_json(ROOT / 'runs/combined-ranker-v1/roster.json'))
    v4 = verify(read_json(ROOT / 'runs/system-selection-v4/roster.json'))
    return combined, v4, {'in_fit': {r['image_sha256'] for r in combined['rows'] if r['role'] == 'fit'},
                          'internal117': {r['image_sha256'] for r in combined['rows'] if r['role'] == 'internal_selection'},
                          'dev108': {r['image_sha256'] for r in v4_roster.load_rows(ROOT, v4, 'development108')}}


def evaluate(_args):
    """Once, after the final fit: internal117, dev108, open register by membership and data type, latency, gates."""
    from rshb_vine.combined_ranker_v1 import roster as combined_roster
    from rshb_vine.interaction_ranker_v1 import evaluate as V
    from rshb_vine.io import read_json, seal, verify
    from rshb_vine.system_selection_v4 import roster as v4_roster
    protocol = _approved()
    model = verify(read_json(OUT / 'final/model.json'))
    monotone_check = verify(read_json(OUT / 'monotone-final.json'))
    (OUT / 'evaluation').mkdir(parents=True, exist_ok=True)
    _new(OUT / 'evaluation/started.json', {'protocol_checksum': protocol['checksum'], 'model': model['checksum'],
                                           'started': time.time()})
    combined, v4, rosters = _rosters()
    selection, projection = _selection()
    selection.use_ranker(model)
    component = {r['query_id']: r['component'] for r in combined['rows']}

    def meta(row):
        return {'group': component.get(row['query_id'], row['query_id']), 'lineage': row['image_sha256'],
                'slice': 'closeup' if str(row['instance_id']).startswith('closeup') else 'ordinary'}

    internal, internal_parity = V.roster_items(ROOT, selection, combined_roster.load_rows(ROOT, combined, 'internal_selection'),
                                               meta, projection)
    dev, dev_parity = V.roster_items(ROOT, selection, v4_roster.load_rows(ROOT, v4, 'development108'), meta, projection)
    reg, reg_skipped = V.register_items(ROOT, selection, rosters, with_labels=True, projection=projection)
    by = {k: V.tally([i for i in reg if i['membership'] == k]) for k in ('in_fit', 'internal117', 'dev108', 'out_of_roster')}
    out_types = {t: V.tally([i for i in reg if i['membership'] == 'out_of_roster' and i['slice'] == t])
                 for t in sorted({i['slice'] for i in reg if i['membership'] == 'out_of_roster'})}
    ti, td, to = V.tally(internal), V.tally(dev), by['out_of_roster']

    def p95(values):
        values = sorted(values)
        return values[int(.95 * (len(values) - 1))] if values else None

    def net_ok(t):
        return t['post_resolver']['fixes']['rows'] >= t['post_resolver']['regressions']['rows']

    seconds = [i['seconds'] for i in internal + dev]
    latency = {'old103_p95': p95([x['legacy'] for x in seconds]), 'systemic_p95': p95([x['total'] for x in seconds])}
    gates = {'parity': all(i['parity'] for i in internal + dev), 'internal117_net': net_ok(ti), 'dev108_net': net_ok(td),
             'out_of_roster_net': net_ok(to), 'monotone': monotone_check['violation_count'] == 0,
             'out_of_roster_real_strict_gain': 'real' in out_types and (
                 out_types['real']['post_resolver']['fixes']['rows'] > out_types['real']['post_resolver']['regressions']['rows']),
             'latency': latency['systemic_p95'] is not None and latency['systemic_p95'] <= latency['old103_p95'] + .25}
    changed = [{k: i[k] for k in ('row', 'old_slug', 'new_slug', 'old_correct', 'new_correct', 'slice')}
               for i in internal + dev + reg if i['changed'] and 'old_correct' in i]
    decision = seal({'kind': 'systemic-ranking-v2-evaluation', 'protocol_checksum': protocol['checksum'],
                     'model': model['checksum'], 'identity_for_correctness': projection.describe(),
                     'internal117': ti, 'dev108': td, 'register': by, 'register_out_of_roster_by_data_type': out_types,
                     'register_skipped': reg_skipped, 'replay_parity': {'internal117': internal_parity, 'dev108': dev_parity},
                     'changed_answers': changed, 'latency_seconds': latency, 'gates': gates,
                     'release_candidate': all(gates.values()),
                     'live_http': 'separate step under the inference slot',
                     'independence': 'register overlaps training and earlier analysis; no independent-test claim',
                     'current_changed': False})
    _new(OUT / 'evaluation/decision.json', decision)
    print(json.dumps({'gates': gates, 'internal117': {k: ti['post_resolver'][k] for k in ('old_top1', 'new_top1', 'fix_ids', 'regression_ids')},
                      'dev108': {k: td['post_resolver'][k] for k in ('old_top1', 'new_top1', 'fix_ids', 'regression_ids')},
                      'out_of_roster': {k: to['post_resolver'][k] for k in ('n', 'old_top1', 'new_top1', 'fixes', 'regressions')}}, indent=1))


PROFILE = ROOT / 'config/systemic-ranking-v2-candidate.json'
PARENT = 'config/recognition-coherent-v1.json'


def profile(_args):
    """Opt-in candidate profile over the byte-frozen coherent-v1 composite; never the current pointer."""
    from rshb_vine.io import read_json, seal, sha256, verify, write_json
    from rshb_vine.systemic_ranking_v2.runtime import KIND
    model = verify(read_json(OUT / 'final/model.json'))
    evaluation = verify(read_json(OUT / 'evaluation/decision.json'))
    parent = verify(read_json(ROOT / PARENT))
    pins = [*CODE, 'rshb_vine/systemic_ranking_v2/runtime.py', 'runs/systemic-ranking-v2/final/model.json']
    document = seal({'kind': KIND, 'name': 'systemic-ranking-v2-candidate', 'parent_profile': PARENT,
                     'parent_profile_sha256': sha256(ROOT / PARENT), 'parent_profile_checksum': parent['checksum'],
                     'model': 'runs/systemic-ranking-v2/final/model.json', 'model_checksum': model['checksum'],
                     'evaluation_checksum': evaluation['checksum'], 'release_candidate': evaluation['release_candidate'],
                     'release_status': 'not_released_pending_root', 'calibrated': False,
                     'pins_sha256': {p: sha256(ROOT / p) for p in pins},
                     'rollback': 'do not load this profile; current pointer untouched'})
    if PROFILE.exists():
        raise SystemExit('candidate profile exists; immutable')
    write_json(PROFILE, document)
    print(document['checksum'])


def serve(args):
    import socket
    from rshb_vine.recognition_api import create_app
    from rshb_vine.systemic_ranking_v2.runtime import SystemicRecognition
    import uvicorn
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', args.port))
        listener.listen(16)
        pipeline = SystemicRecognition(ROOT, PROFILE.relative_to(ROOT))
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=args.port)).run(sockets=[listener])
    finally:
        listener.close()


def main():
    commands = {'table': table, 'diagnose': diagnose, 'readiness': readiness, 'parity': parity, 'protocol': protocol,
                'cv': cv, 'final': final, 'monotone': monotone, 'evaluate': evaluate,
                'profile': profile, 'serve': serve}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=sorted(commands))
    parser.add_argument('--model', default='runs/systemic-ranking-v2/final/model.json')
    parser.add_argument('--port', type=int, default=8176)
    args = parser.parse_args()
    commands[args.command](args)


if __name__ == '__main__':
    main()
