"""CPU replay of decision consistency v1 over the saved admitted open593 (890 targets), one replay for all arms.

The consumer is the text-evidence v2 one (A+B+D+M over RepairSelectionV2 with the 5c0be ranker, the fa317ef2 recipe),
built once. G0 is that selector itself and must equal the v2 ``final`` rows; I, G and IG are ``compose`` copies of it.
Every arm gets a fresh fingerprint-keyed geometry cache over the G0v2 verifier outputs (a different incumbent or
input runs the real CPU verifier and is logged). The ledger keeps every labelled and unlabelled change with the
resolver reasons, the frozen and effective guard decisions and the proposal of each stage; the census lists every
target where I recovered a span or G compared a winner with an agreed visual candidate.
"""
from collections import Counter

from rshb_vine.decision_consistency_v1 import identity as I
from rshb_vine.decision_consistency_v1.composition import BASE_RECIPE, compose
from rshb_vine.text_evidence_repair_v2 import experiment as V2

X = V2.X
jsonl, run_arm = V2.jsonl, V2.run_arm
PARITY_FIELDS = ('selected', 'raw_order', 'feature_digest', 'guard', 'geometry', 'injection', 'producer_reranked', 'top3')
STAGE_PROPOSALS = ('raw_proposal_before_guard', 'proposal_before_cross_script', 'proposal_before_pairwise',
                   'proposal_before_geometry')


class Consumer(V2.Consumer):
    def __init__(self, geometry_reference, tables):
        super().__init__(BASE_RECIPE, geometry_reference, tables)
        self.base_selection, self.reference = self.selection, geometry_reference

    def arm(self, components):
        """Selector and fresh geometry cache of one arm; the replay, parent and base selector are reused."""
        self.selection = compose(self.base_selection, components) if components else self.base_selection
        self.geometry = X.GeometryCache(self.rp.stage, self.inner.registry, self.geometry.cached, self.reference)
        self.rp.stage.verifier.evaluate_target = self.geometry
        return self.selection

    def outcome(self, out, gt):
        s = super().outcome(out, gt)
        res = out['proposal'].get('existing_evidence_resolution') or {}
        cross = res.get('cross_script_identity') or {}
        s['resolver'] = {'reason': res.get('reason'), 'reasons': res.get('reasons'), 'blocked': res.get('blocked'),
                         'lost_identity_phrases': res.get('lost_identity_phrases'),
                         'literal_lost': cross.get('literal'), 'recovered': [r['span'] for r in cross.get('recovered', [])]}
        stage = out.get('cross_script_identity')
        s['cross_script_stage'] = stage and {k: stage.get(k) for k in ('reason', 'applied', 'selected', 'reasons')}
        if stage and stage.get('identity'):
            s['cross_script_stage']['recovered'] = stage['identity']['recovered']
        s['cross_script_calls'] = stage and [{k: c[k] for k in ('index', 'caller', 'raw_top', 'selected', 'reason',
                                                                'recovered')}
                                             for c in I.recovered_calls(stage.get('calls') or [])]
        if stage and 'calls' in stage:
            s['cross_script_resolves'] = len(stage['calls'])
        pg = out.get('pairwise_guard')
        s['pairwise_guard'] = pg and _pairwise_view(pg)
        frozen = out.get('discriminator_guard') or {}
        s['guard_effective'] = (pg or {}).get('effective') or {'blocked': bool(frozen.get('blocked')),
                                                              'decided_by': 'frozen_guard_v2'}
        s['stage_choices'] = {k: out[k]['representative_slug'] for k in STAGE_PROPOSALS if out.get(k)}
        return s


def _pairwise_view(pg):
    view = {k: pg.get(k) for k in ('reason', 'class', 'effective', 'changed_conditions', 'applied', 'selected',
                                   'agreed_candidate', 'text_winner', 'guard_pool', 'guard_pairwise')}
    detail = pg.get('pairwise') or {}
    view['agreed_closed_monosort'] = detail.get('agreed_closed_monosort')
    view['producer_relation'] = (detail.get('producer_relation') or {}).get('relation')
    view['producer_keys'] = detail.get('producer_relation')
    view['agreed_claims'] = detail.get('agreed_claims')
    view['names'] = [{'slug': n['slug'], 'exclusive_pool': n['exclusive_pool'], 'exclusive_pairwise': n['exclusive_pairwise'],
                      'applied': n['applied'], 'tokens': [t for t in n['tokens'] if t['read']]}
                     for n in detail.get('names', [])]
    return view


def g0_parity(rows, final_rows):
    """G0 against the saved v2 final arm: every parity field of every target."""
    final = {(r['key'], r['instance_id']): r['outcome'] for r in final_rows}
    counts, differ = Counter(), []
    for r in rows:
        f = final.get((r['key'], r['instance_id']))
        for k in PARITY_FIELDS:
            same = f is not None and r['outcome'][k] == f[k]
            counts[f'{k}:{same}'] += 1
            if not same:
                differ.append([r['key'], r['instance_id'], k])
    ok = len(final) == len(rows) == X.OPEN_TARGETS and not differ
    return {'counts': dict(counts), 'differ': differ[:50], 'passed': ok}


def _side(o):
    return {'selected': o['selected'], 'correct': o['correct'], 'resolver': o.get('resolver'),
            'guard': o['guard'], 'guard_effective': o.get('guard_effective'), 'stage_choices': o.get('stage_choices')}


def compare(base_rows, arm_rows, producers):
    """v2 ledger (all changes) plus resolver/guard paths, the I census and the G class census."""
    result = V2.compare(base_rows, arm_rows, producers)
    base = {(r['key'], r['instance_id']): r for r in base_rows}
    arm = {(r['key'], r['instance_id']): r for r in arm_rows}
    for c in result['changes']:
        k = (c['key'], c['instance_id'])
        c['path'] = {'before': _side(base[k]['outcome']), 'after': _side(arm[k]['outcome'])}
        c['pairwise_guard'] = arm[k]['outcome'].get('pairwise_guard')
        c['cross_script_stage'] = arm[k]['outcome'].get('cross_script_stage')
    census_i, census_g, classes, callers, relations = [], [], Counter(), Counter(), Counter()
    for k, r in arm.items():
        o, b = r['outcome'], base[k]['outcome']
        row = {'key': k[0], 'instance_id': k[1], 'slice': r['slice'], 'gt_products': r['gt_products'],
               'before': {'selected': b['selected'], 'correct': b['correct']},
               'after': {'selected': o['selected'], 'correct': o['correct']}}
        stage = o.get('cross_script_stage') or {}
        calls = o.get('cross_script_calls') or []
        if calls:
            spans = [r for c in calls for r in c['recovered']]
            callers.update(c['caller'] for c in calls)
            census_i.append(dict(row, first_stage_reason=stage.get('reason'), reasons=stage.get('reasons'),
                                 first_stage_recovered=stage.get('recovered') or [], calls=calls,
                                 recovered_spans=len(spans), cross_script_spans=sum(r['cross_script'] for r in spans),
                                 final_selection_changed=b['selected'] != o['selected']))
        pg = o.get('pairwise_guard') or {}
        if pg.get('class'):
            label = 'unlabelled' if not r['gt_products'] else 'correct_after' if o['correct'] else 'wrong_after'
            relation = pg.get('producer_relation') or 'not_computed'
            relations[(pg['class'], relation)] += 1
            classes[(pg['class'], pg['reason'], relation, label, b['correct'] == o['correct'] or not r['gt_products'])] += 1
            census_g.append(dict(row, **{f: pg.get(f) for f in ('class', 'reason', 'effective', 'changed_conditions',
                                                                'agreed_closed_monosort', 'producer_relation',
                                                                'producer_keys', 'names')}))
    result['census'] = {
        'cross_script_recovered_targets': len(census_i),
        'cross_script_recovered_spans': sum(x['recovered_spans'] for x in census_i),
        'cross_script_recovered_final_unchanged': sum(not x['final_selection_changed'] for x in census_i),
        'cross_script_recovering_calls_by_caller': dict(sorted(callers.items())),
        'pairwise_compared_targets': len(census_g),
        'pairwise_classes': [{'class': c, 'reason': reason, 'producer_relation': rel, 'label_after': label,
                              'label_unchanged': same, 'targets': n}
                             for (c, reason, rel, label, same), n in sorted(classes.items())],
        'pairwise_class_by_producer_relation': [{'class': c, 'producer_relation': rel, 'targets': n}
                                                for (c, rel), n in sorted(relations.items())],
        'monosort_vs_blend': {
            'meaning': 'new_block_open_or_blend_grape = pairwise would block a winner whose pool-exclusive word is a '
                       'grape absent from an open agreed list or blend-listed by it; release-only G never applies it',
            'targets': [[x['key'], x['instance_id'], x['after']['correct']] for x in census_g
                        if x['class'] == 'new_block_open_or_blend_grape'],
            'released_closed_monosort_grape': [[x['key'], x['instance_id'], x['producer_relation'],
                                                x['before']['correct'], x['after']['correct']]
                                               for x in census_g if x['class'] == 'released_closed_monosort_grape'],
            'monosort_grape_producer_not_same': [[x['key'], x['instance_id'], x['producer_relation'], x['reason']]
                                                 for x in census_g if any(
                                                     t['verdict'].startswith('grape_vs_closed_monosort_producer_')
                                                     for n in x['names'] or [] for t in n['tokens'])],
            'released_common_name': [[x['key'], x['instance_id'], x['before']['correct'], x['after']['correct']]
                                     for x in census_g if x['class'] == 'released_common_name']}}
    return result, census_i, census_g
