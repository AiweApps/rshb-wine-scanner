"""Producer role of the target's text: which readings may speak for a manufacturer.

An observation of the target is an eligible manufacturer-name reading when
- it has its own target-bound provenance (injection-v2 target sources); no provenance gives no reading;
- its line is not an origin-designation line: a line carrying a protected origin marker (ЗГУ, ЗНМП, PGI, PDO,
  IGP, AOC/AOP, DOC/DOCG/DOP, IGT, or the words «географическое указание» / «место происхождения» /
  appellation) states where the wine comes from, not who made it;
- on a line whose text is also read on another target of the scene, only an own-label read (instance label
  or label view crop) counts; a context-crop read of such a line stays unowned.
Grades (per reader, never mixed) come only from eligible observations of the line.

A producer form is read only when at least one of its specific tokens (catalogue producer specificity >= 0.5,
skeleton >= 3, skeleton not in the existing generic vocabulary such as винодельня, вино, шато, усадьба) is read
non-fuzzy on an eligible observation. Its other tokens (generic or short) then count as in the frozen evidence;
without such an anchor the form contributes nothing, so a generic word alone is no maker signal. Partial
specific readings (SONS of Nikolaev & Sons) stay soft evidence at any native confidence.

The candidate's own prod.read_w / prod.read_g.* and the rival contrast prod.other_g.* are recomputed from the
same eligible readings with the frozen aggregation (best form, specificity-weighted sum, cap, rival = other
product of another producer). prod.phrase keeps its frozen value only when one producer form is completely read
(every token of skeleton >= 3, non-fuzzy, on eligible observations, with a specific token); otherwise 0.
Raw OCR, other features, weights, ranker, resolver and the guard are unchanged.
"""
from rshb_vine.interaction_ranker_v1 import evidence as E1
from rshb_vine.ocr_candidate_repair_v1 import injection as I
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E

ORIGIN_MARKERS = ('zgu', 'znmp', 'pgi', 'pdo', 'igp', 'aoc', 'aop', 'doc', 'docg', 'dop', 'igt')
ORIGIN_STEMS = ('geografichesk', 'geographical', 'proiskhozhden', 'appellation')
OWN_LABEL = ('instance_label_crop', 'label_view_crop')
FEATURES = ('prod.read_w', *(f'prod.read_g.{r}' for r in E.READERS), 'prod.phrase',
            *(f'prod.other_g.{r}' for r in E.READERS))
RULE = {
    'version': 'producer-role-v2',
    'target_sources': I.RULE['target_sources'],
    'origin_markers': list(ORIGIN_MARKERS),
    'origin_stems': list(ORIGIN_STEMS),
    'scene_ownership': 'a line also read on another target counts only through own-label observations',
    'specific_token': {'min_producer_weight': 0.5, 'min_skeleton': 3,
                       'generic_vocabulary': 'context.generic, compared by skeleton (шато = chateau)'},
    'anchor': 'a producer form counts only when one of its specific tokens is read on an eligible observation',
    'phrase': 'frozen prod.phrase kept only with a complete non-fuzzy eligible reading of one producer form',
    'features': list(FEATURES),
    'confidence': 'no native-confidence requirement; grades per reader from eligible observations',
}
FULL = {'origin': True, 'ownership': True, 'anchor': True, 'phrase': True}
NONE = dict.fromkeys(FULL, False)


def origin_line(tokens):
    return any(t.forms[0] in ORIGIN_MARKERS or t.forms[0].startswith(ORIGIN_STEMS) for t in tokens)


class TargetProducerText:
    """Eligible observations and per-reader grades of one target, over the frozen line grouping."""

    def __init__(self, observations, provenance, other_line_keys=(), flags=FULL):
        if len(provenance) != len(observations):
            raise ValueError('Provenance side-car does not cover the observation packet')
        self.lines = V4._lines(observations)
        self.index = E._Index(self.lines)
        other = set(other_line_keys)
        self.grades, self.eligible, self.excluded = {}, {}, {}
        for line in self.lines:
            lid = line['line_id']
            origin = flags['origin'] and origin_line(line['tokens'])
            shared = flags['ownership'] and any(I._line_key(r) in other for r in line['raw'])
            grade, ids = dict.fromkeys(E.READERS, 0.), []
            for i in line['observation_ids']:
                record = provenance[i]
                reason = ('no_provenance' if record is None else
                          'not_target_bound' if record['crop_source'] not in RULE['target_sources'] else
                          'origin_designation_line' if origin else
                          'shared_line_context_crop' if shared and record['crop_source'] not in OWN_LABEL else None)
                if reason:
                    self.excluded.setdefault(lid, {})[i] = reason
                    continue
                ids.append(i)
                score = record['native_score']
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    tag = E.READER_SHORT[E1.READER_TAGS[record['reader']]]
                    grade[tag] = max(grade[tag], float(score))
            self.grades[lid] = grade
            if ids:
                self.eligible[lid] = ids

    def token(self, token):
        """(per-reader grade, status, eligible lines) of one claim token."""
        hits = self.index.matches(token.forms[0])
        strong = [l for l, k in hits.items() if k in E.MATCHED]
        lines = sorted(l for l in strong if l in self.eligible)
        grade = {r: max((self.grades[l][r] for l in lines), default=0.) for r in E.READERS}
        if lines:
            status = 'read'
        elif strong:
            status = 'excluded:' + ','.join(sorted({x for l in strong for x in self.excluded.get(l, {}).values()}))
        else:
            status = 'fuzzy_only' if hits else 'unread'
        return grade, status, lines, sorted(strong)

    def trace(self):
        return [{'line_id': l['line_id'], 'raw': l['raw'], 'tokens': [t.forms[0] for t in l['tokens']],
                 'eligible': self.eligible.get(l['line_id'], []),
                 'excluded': {str(i): r for i, r in self.excluded.get(l['line_id'], {}).items()}}
                for l in self.lines if l['line_id'] in self.excluded]


class ProducerRoles:
    def __init__(self, context, stats):
        self.context, self.stats = context, stats
        self.generic = {text.skeleton(g) for g in context.generic}

    def specific(self, token):
        return (self.stats.producer_weight(token) >= RULE['specific_token']['min_producer_weight']
                and len(token.skeletons[0]) >= RULE['specific_token']['min_skeleton']
                and token.skeletons[0] not in self.generic)

    def coverage(self, target, form, anchor):
        tokens = text.tokenize(form)
        out = {'form': form, 'cover': 0., 'read_w': 0., **{f'read_g.{r}': 0. for r in E.READERS},
               'anchored': False, 'complete': False, 'tokens': []}
        total = sum(self.stats.producer_weight(t) for t in tokens)
        if not tokens or total <= 0:
            return out
        reads = []
        for t in tokens:
            grade, status, lines, strong = target.token(t)
            reads.append((t, self.stats.producer_weight(t), grade, any(grade.values())))
            out['tokens'].append({'token': t.forms[0], 'weight': round(self.stats.producer_weight(t), 4),
                                  'specific': self.specific(t), 'status': status, 'lines': lines,
                                  'strong_lines': strong, 'grades': grade})
        long = [x for x, t in zip(out['tokens'], tokens) if len(t.skeletons[0]) >= RULE['specific_token']['min_skeleton']]
        out['complete'] = bool(long) and any(x['specific'] for x in long) and all(x['status'] == 'read' for x in long)
        out['anchored'] = any(read and self.specific(t) for t, _, _, read in reads)
        if anchor and not out['anchored']:
            return out
        for t, w, grade, read in reads:
            out['cover'] += w * read / total
            out['read_w'] += w * read
            for r in E.READERS:
                out[f'read_g.{r}'] += w * grade[r]
        return out

    def producers(self, candidate, target, anchor):
        members, _ = E._members(candidate, self.context)
        reads = [self.coverage(target, form, anchor) for m in members.values() for form in m['producers']]
        best = max(reads, key=E._rank, default=None) or dict(self.coverage(target, '', anchor), form=None)
        return best, reads

    def features(self, out, target, flags=FULL):
        """New producer feature rows aligned with ``out['rows']`` and per-candidate evidence."""
        best, forms = zip(*(self.producers(c, target, flags['anchor']) for c in out['base']['candidates'])) \
            if out['base']['candidates'] else ((), ())
        evidence, result = out['evidence'], []
        for i, (row, ev) in enumerate(zip(out['rows'], evidence)):
            own = best[i]
            f = {'prod.read_w': E._cap(own['read_w']), **{f'prod.read_g.{r}': E._cap(own[f'read_g.{r}']) for r in E.READERS}}
            proven = [p['form'] for p in forms[i] if p['complete']]
            f['prod.phrase'] = row['prod.phrase'] if proven or not flags['phrase'] else 0.
            rivals = [j for j, o in enumerate(evidence) if j != i and not set(o['products']) & set(ev['products'])
                      and o['producer_keys'] and not set(o['producer_keys']) & set(ev['producer_keys'])]
            rival = {}
            for r in E.READERS:
                j = max(rivals, key=lambda k: best[k][f'read_g.{r}'], default=None)
                f[f'prod.other_g.{r}'] = max(0., E._cap(best[j][f'read_g.{r}']) - E._cap(own[f'read_g.{r}'])) \
                    if j is not None else 0.
                rival[r] = j is not None and {'candidate_id': evidence[j]['candidate_id'], 'form': best[j]['form'],
                                              'read_g': best[j][f'read_g.{r}']}
            excluded = [dict(t, form=p['form']) for p in forms[i] for t in p['tokens'] if t['status'].startswith('excluded:')]
            result.append((f, {'own': own, 'complete_forms': proven, 'rival': rival, 'excluded_tokens': excluded}))
        return result
