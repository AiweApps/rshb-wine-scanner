"""Catalogue-derived short producer forms for the OCR catalogue-phrase injection (component B).

The frozen injection (``ocr-catalog-phrase-injection-v2``) needs every specific token of one full producer
form, so a label that prints only the distinctive part of a legal name (``Эльбузд`` for ``Донское
винодельческое хозяйство "Эльбузд"``) cannot add its card. A short form is a proper subset of one catalogue
producer form's specific tokens that the whole catalogue and the gallery reference OCR tie to that producer
alone. The table is built once offline from catalogue metadata and existing reference OCR; no query data, no
hand alias. The reference evidence must show the short form printed as a distinct name: an OCR line whose only
non-generic words are the short form. At request time a per-instance wrapper around the frozen index runs the
parent unchanged and only then tries short forms for cards the parent did not match, with the same core-name,
target-crop, ownership, blend and conflict checks; short forms never revoke a parent decision and are skipped
when they would push the target over the parent's product budget. A short-form producer word may not be the
same OCR word that already reads a core-name token. Scoring is unchanged: the systemic producer evidence
already credits a partial read of a full form by token specificity.
"""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

from rshb_vine.io import digest, read_json, sha256, verify
from rshb_vine.ocr_candidate_repair_v1 import injection as I
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E

RULE = {
    'version': 'catalogue-short-producer-forms-v1',
    'parent_rule': I.RULE['version'],
    'form_tokens': 'tokens of one catalogue producer form that the parent injection requires '
                   '(skeleton >= min_token_skeleton, producer weight >= min_producer_token_weight)',
    'specific': 'form tokens whose skeleton is outside the existing generic vocabulary (frozen_candidate_selector.GENERIC '
                'via SelectionContext.generic) and that are not a typed grape/colour/sugar/style atom',
    'short_form': 'a proper subset of the form tokens made only of specific tokens; minimal by inclusion',
    'min_single_token_skeleton': 5,
    'metadata_unique': 'no other producer key has all short-form skeletons among its producer-form tokens',
    'name_unique': 'no card of another producer key has all short-form skeletons among its name/series tokens',
    'not_region_only': 'at least one short-form token is absent from every card Регион value',
    'catalogue_text_exclusive': 'at least one short-form token occurs in no producer form and no text field '
                                '(TEXT_FIELDS, commercial name, series) of any card of another producer key',
    'distinct_printed_name': 'a gallery reference of this producer has a media-audit OCR line at native score >= '
                             'min_reference_score whose words outside the generic vocabulary and digits are exactly '
                             'the short form (each word read non-fuzzy by a short-form token and vice versa); every OCR '
                             'observation on that reference that reads another specific token of the source form, even '
                             'fuzzily, is at most max_rest_height_ratio of that line height (smaller legal print, not a '
                             'name split over two lines)',
    'max_rest_height_ratio': 0.6,
    'distinct_majority': 'references with a distinct printed short form outnumber the producer references that read '
                         'every short-form token together with another specific token of the source form otherwise',
    'min_reference_score': 0.5,
    'reference_exclusive': 'no gallery reference of another producer key has every short-form token read '
                           '(non-fuzzy, any score)',
    'request_time': 'parent decision kept verbatim; short forms tried only for cards the parent did not match; '
                    'a short-form token must be read at an OCR word that no core-name token of the same card uses; '
                    'short additions are all skipped when parent plus short products exceed the parent budget or '
                    'the parent is already over budget',
}
TEXT_FIELDS = ('Описание', 'Название вина', 'Регион', 'Сорт винограда', 'Категория', 'Цвет', 'Название фото')
GALLERY = I.GALLERY
REFERENCE_OCR = 'data/media-audit/machine.jsonl'
TABLE = 'runs/text-evidence-improve-v1/identity/short-producer-forms-v1.json'
SOURCES = (GALLERY, REFERENCE_OCR, 'rshb_vine/ocr_candidate_repair_v1/injection.py',
           'rshb_vine/systemic_ranking_v2/evidence.py')


def input_checksums(root, context):
    """SHA-256 of every table input: gallery, reference OCR, catalogue context sources and the builder code."""
    root = Path(root)
    return {**{p: sha256(root / p) for p in SOURCES}, **context.source_checksums}


def _producer_key(card, slug):
    return I._producer_key(card) or 'card:' + slug


def _skeletons(value):
    return {t.skeletons[0] for t in text.tokenize(value or '')}


def _reference_ocr(root, shas):
    """{image sha: OCR lines} for gallery originals only; other audit rows are skipped unparsed."""
    found, key = {}, re.compile(r'"sha256": "([0-9a-f]{64})"')
    with (Path(root) / REFERENCE_OCR).open() as stream:
        for row in stream:
            m = key.search(row)
            if not m or m.group(1) not in shas:
                continue
            record = json.loads(row)
            if record.get('sha256') in shas:
                lines = V4._lines(record.get('ocr') or [])
                found[record['sha256']] = (lines, E._Index(lines), record.get('ocr') or [])
    return found


def _reads(reference, tokens):
    """True when every token (forms[0]) is read non-fuzzy somewhere on one reference."""
    _, index, _ = reference
    return all(any(k in E.MATCHED for k in index.matches(t.forms[0]).values()) for t in tokens)


def _height(observation):
    box = observation.get('bbox_bottom_left_normalized') or [0., 0., 0., 0.]
    return max(0., box[3] - box[1])


def _distinct_lines(reference, tokens, others, generic, min_score):
    """OCR observations printing exactly the short form (generic words and digits aside) as the dominant name.

    ``others`` are the source form's remaining specific tokens; any observation reading one of them (fuzzy included)
    must be at most ``max_rest_height_ratio`` of the short-form observation's height.
    """
    observations = reference[2]
    rest = [_height(o) for o in observations
            if any(text.token_match(q, t.forms[0]) for q in text.tokenize(o.get('raw_text', '')) for t in others)]
    found = []
    for o in observations:
        score = o.get('score')
        if not isinstance(score, (int, float)) or score < min_score:
            continue
        words = [q for q in text.tokenize(o.get('raw_text', '')) if q.skeletons[0] not in generic and not q.forms[0].isdigit()]
        height = _height(o)
        if words and all(any(text.token_match(q, t.forms[0]) in E.MATCHED for t in tokens) for q in words) \
                and all(any(text.token_match(q, t.forms[0]) in E.MATCHED for q in words) for t in tokens) \
                and all(h <= RULE['max_rest_height_ratio'] * height for h in rest):
            found.append({'raw': o['raw_text'], 'score': round(float(score), 4), 'height': round(height, 4),
                          'rest_heights': [round(h, 4) for h in rest]})
    return found


def build_table(root, selection):
    """Every candidate short form of every catalogue producer form with the reason it was admitted or rejected.

    ``selection`` supplies the live selector registry, context and catalogue stats (the injection's own inputs).
    """
    root = Path(root)
    registry, context, stats = selection.registry, selection.context, selection.stats
    producer_tokens, producer_forms, name_tokens, region = defaultdict(set), defaultdict(set), defaultdict(set), set()
    vocabulary = defaultdict(set)
    slugs_of = defaultdict(list)
    for slug, card in registry.cards.items():
        profile, meta = registry.claims.get(slug), card.get('raw_metadata', {})
        key = _producer_key(card, slug)
        slugs_of[key].append(slug)
        forms = V4._producer_forms(card, profile, context)
        producer_forms[key].update(forms)
        producer_tokens[key].update(s for f in forms for s in _skeletons(f))
        names = (V4._claim_values(profile, 'commercial_name')[0] or [meta.get('Название вина', '')]) \
            + V4._claim_values(profile, 'series')[0] + [meta.get('Название вина', '')]
        for value in names:
            if value:
                name_tokens[key].add(frozenset(_skeletons(value)))
        region |= _skeletons(meta.get('Регион', ''))
        vocabulary[key] |= producer_tokens[key] | {s for v in names for s in _skeletons(v)} \
            | {s for f in TEXT_FIELDS for s in _skeletons(meta.get(f, ''))}
    gallery = read_json(root / GALLERY)['references']
    refs = defaultdict(set)
    for r in gallery:
        if r['slug'] in registry.cards:
            refs[_producer_key(registry.cards[r['slug']], r['slug'])].add(r['image_sha256'])
    ocr = _reference_ocr(root, {s for v in refs.values() for s in v})

    generic = {text.skeleton(g) for g in context.generic}

    def specific(t):
        return (t.skeletons[0] not in generic and not t.forms[0].isdigit()
                and not E._atoms([t], context.lexicon))

    candidates, admitted = [], []
    for key in sorted(producer_forms):
        for form in sorted(producer_forms[key]):
            full = [t for t in text.tokenize(form) if len(t.skeletons[0]) >= I.RULE['min_token_skeleton']
                    and stats.producer_weight(t) >= I.RULE['min_producer_token_weight']]
            full = list({t.skeletons[0]: t for t in full}.values())
            spec = [t for t in full if specific(t)]
            accepted_sets = []
            for size in range(1, len(full)):
                for subset in combinations(spec, size):
                    skel = frozenset(t.skeletons[0] for t in subset)
                    if any(a <= skel for a in accepted_sets):
                        continue
                    checks = {
                        'single_token_length': size > 1 or len(subset[0].skeletons[0]) >= RULE['min_single_token_skeleton'],
                        'metadata_unique': sorted(k for k, v in producer_tokens.items() if k != key and skel <= v),
                        'name_unique': sorted(k for k, v in name_tokens.items() if k != key and any(skel <= n for n in v)),
                        'not_region_only': not skel <= region,
                        'exclusive_tokens': sorted(x for x in skel
                                                   if not any(x in v for k, v in vocabulary.items() if k != key)),
                    }
                    support, rest = {}, [t for t in spec if t not in subset]
                    for sha in sorted(refs[key]):
                        if sha in ocr and (read := _distinct_lines(ocr[sha], subset, rest, generic,
                                                                   RULE['min_reference_score'])):
                            support[sha] = read
                    together = sorted(sha for sha in refs[key] if sha in ocr and sha not in support and rest
                                      and _reads(ocr[sha], subset)
                                      and any(text.token_match(q, t.forms[0]) for l in ocr[sha][0]
                                              for q in l['tokens'] for t in rest))
                    foreign = {}
                    for other, shas in refs.items():
                        if other == key:
                            continue
                        hit = sorted(s for s in shas if s in ocr and _reads(ocr[s], subset))
                        if hit:
                            foreign[other] = hit
                    reasons = [name for name, ok in (
                        ('single_token_too_short', checks['single_token_length']),
                        ('shared_with_other_producer_forms', not checks['metadata_unique']),
                        ('shared_with_other_producer_names', not checks['name_unique']),
                        ('region_words_only', checks['not_region_only']),
                        ('no_catalogue_exclusive_token', checks['exclusive_tokens']),
                        ('not_printed_as_distinct_name', support),
                        ('mostly_printed_with_rest_of_form', len(support) > len(together)),
                        ('printed_on_other_producer_references', not foreign)) if not ok]
                    record = {'producer_key': key, 'source_form': form, 'form_tokens': [t.forms[0] for t in full],
                              'short_tokens': [t.forms[0] for t in subset], 'skeletons': sorted(skel),
                              'weights': {t.forms[0]: round(stats.producer_weight(t), 4) for t in subset},
                              'other_producer_forms': checks['metadata_unique'],
                              'other_producer_names': checks['name_unique'],
                              'catalogue_exclusive_tokens': checks['exclusive_tokens'],
                              'distinct_printed_name': support, 'printed_with_rest_of_form': together,
                              'reference_images_of_producer': len(refs[key]),
                              'reference_images_with_ocr': sum(s in ocr for s in refs[key]),
                              'printed_on_other_producers': foreign,
                              'status': 'admitted' if not reasons else 'rejected', 'reasons': reasons}
                    candidates.append(record)
                    if not reasons:
                        accepted_sets.append(skel)
                        admitted.append({k: record[k] for k in ('producer_key', 'source_form', 'form_tokens',
                                                                  'short_tokens', 'skeletons', 'weights')}
                                        | {'cards': len(slugs_of[key]), 'reference_support_images': len(support)})
    return {
        'kind': 'short-producer-forms', 'rule': RULE,
        'registry_checksum': registry.checksum,
        'inputs_sha256': input_checksums(root, context),
        'producers': len(producer_forms), 'gallery_references_with_ocr': len(ocr),
        'candidates': candidates, 'admitted': admitted,
    }


def load_table(root, context, registry, expected_checksum):
    """The sealed table, refused unless its checksum is the pinned one and every input still has its SHA."""
    table = verify(read_json(Path(root) / TABLE))
    if table['checksum'] != expected_checksum or table['rule'] != RULE \
            or table['registry_checksum'] != registry.checksum:
        raise ValueError('Short producer table differs from the pinned table, rule or registry')
    if table['inputs_sha256'] != input_checksums(root, context):
        raise ValueError('Short producer table inputs changed since the table was built')
    return table


def build_injection_index(base_index, registry, context, stats, *, root, table_checksum):
    """Wrap a constructed frozen ``CatalogPhraseIndex`` with the pinned short-form table (no rebuild, no mutation)."""
    if base_index.registry is not registry or base_index.stats is not stats or base_index.context is not context:
        raise ValueError('Base index was built from another registry, context or stats')
    return ShortProducerIndex(base_index, load_table(root, context, registry, table_checksum))


class ShortProducerIndex:
    """The frozen phrase index plus admitted short producer forms; parent decisions and profiles are its own."""

    rule = RULE

    def __init__(self, base, table):
        self.base = base
        self.registry, self.context, self.stats, self.entries = base.registry, base.context, base.stats, base.entries
        self.profile = base.profile
        self.table_checksum = table['checksum']
        by_key = defaultdict(list)
        for row in table['admitted']:
            by_key[row['producer_key']].append(row)
        self.short = {}
        for slug, entry in base.entries.items():
            forms, seen = [], set()
            for row in by_key.get(entry['producer_key'], ()):
                tokens = text.tokenize(' '.join(row['short_tokens']))
                if sorted(t.skeletons[0] for t in tokens) != row['skeletons']:
                    raise ValueError('Short producer form does not re-tokenize to its skeletons')
                if tuple(row['skeletons']) not in seen:
                    seen.add(tuple(row['skeletons']))
                    forms.append((row, [(t, self.stats.producer_weight(t)) for t in tokens]))
            if forms:
                self.short[slug] = forms
        self.checksum = digest({'parent': base.checksum, 'rule': RULE, 'table': self.table_checksum,
                                'short': {s: [r['short_tokens'] for r, _ in f] for s, f in sorted(self.short.items())}})

    def describe(self):
        return dict(self.base.describe(), short_rule=RULE['version'], parent_checksum=self.base.checksum,
                    table_checksum=self.table_checksum, cards_with_short_forms=len(self.short), checksum=self.checksum)

    @staticmethod
    def _places(index, claim_norm):
        """Non-fuzzy (line_id, pos) places of one claim token; the same candidate places as ``E._Index.matches``."""
        skeleton = text.skeleton(claim_norm)
        places = set(index.exact.get(claim_norm, ())) | set(index.skeleton.get(skeleton, ()))
        if skeleton and len(skeleton) >= 5:
            places |= {p for p in index.by_initial.get(skeleton[0], ())
                       if abs(len(index.lines[p[0]]['tokens'][p[1]].skeletons[0]) - len(skeleton)) <= 2}
        return {p for p in places if text.token_match(index.lines[p[0]]['tokens'][p[1]], claim_norm) in E.MATCHED}

    def propose(self, observations, provenance, pool_slugs, other_line_keys=(), conflicts=None):
        decision = self.base.propose(observations, provenance, pool_slugs, other_line_keys, conflicts)
        trace = {'rule': RULE['version'], 'table_checksum': self.table_checksum, 'tried': [], 'matched': [],
                 'injected': [], 'skipped': [], 'status': 'no_short_match'}
        decision['short_producer'] = trace
        done = {m['slug'] for m in decision['matched']}
        todo = {s: f for s, f in self.short.items() if s not in done}
        if not todo:
            return decision
        bound = [i for i, r in enumerate(provenance) if r is not None and r['crop_source'] in I.RULE['target_sources']]
        lines = V4._lines([observations[i] for i in bound])
        index = E._Index(lines)
        new = {}
        for slug, forms in sorted(todo.items()):
            entry, core, core_places = self.entries[slug], [], set()
            for t in entry['core']:
                places = self._places(index, t.forms[0])
                if not places:
                    break
                core_places |= places
                strong = {l: k for l, k in index.matches(t.forms[0]).items() if k in E.MATCHED}
                core.append({'token': t.forms[0], 'lines': sorted(strong), 'kinds': sorted(set(strong.values()))})
            else:
                for row, form in forms:
                    reads, separate = [], True
                    for t, w in form:
                        places = self._places(index, t.forms[0])
                        reads.append({'token': t.forms[0], 'weight': round(w, 4), 'lines': sorted({l for l, _ in places})})
                        separate &= bool(places - core_places)
                    if not all(r['lines'] for r in reads):
                        continue
                    trace['tried'].append({'slug': slug, 'short_tokens': row['short_tokens'],
                                           'source_form': row['source_form'], 'separate_from_core': separate})
                    if separate:
                        new[slug] = {'slug': slug, 'candidate_key': entry['key'], 'product_id': entry['product_id'],
                                     'name_kind': entry['kind'], 'core_name_weight': round(entry['weight'], 4),
                                     'core': core, 'producer': reads, 'producer_form_kind': 'short',
                                     'short_form': {k: row[k] for k in ('short_tokens', 'source_form', 'producer_key')},
                                     'blend_explained_by': self.base._blend_explanation(entry, pool_slugs)}
                        break
        if not new:
            return decision
        grades = E._line_grades(lines, [provenance[i] for i in bound])
        line_raw = {l['line_id']: {'raw': l['raw'], 'grades': grades[l['line_id']],
                                   'observation_ids': [bound[i] for i in l['observation_ids']]} for l in lines}
        for m in new.values():
            used = sorted({l for part in (m['core'], m['producer']) for x in part for l in x['lines']})
            m['lines'] = {l: line_raw[l] for l in used}
            m['ownership_uncertain'] = any(I._line_key(r) in other_line_keys
                                           for x in m['core'] for l in x['lines'] for r in line_raw[l]['raw'])
            m['conflict'] = conflicts(m['slug']) if conflicts else None
        trace['matched'] = sorted(new)
        parent_keys = {m['candidate_key'] for m in decision['matched']}
        keys = parent_keys | {m['candidate_key'] for m in new.values()}
        if decision['status'] == 'ambiguous_phrase_over_budget' or len(keys) > I.RULE['budget_products']:
            trace['status'] = 'short_form_over_budget'
            trace['skipped'] = [{'slug': s, 'reason': 'short_form_over_budget'} for s in sorted(new)]
            return decision
        pool_keys = {I._candidate_key(s, self.registry.cards) for s in pool_slugs}
        injected_keys = {I._candidate_key(s, self.registry.cards) for s in decision['injected']}
        for slug in sorted(new):
            m = new[slug]
            reason = ('already_in_pool' if m['candidate_key'] in pool_keys else
                      'already_injected' if m['candidate_key'] in injected_keys else
                      'ownership_uncertain' if m['ownership_uncertain'] else
                      'explained_by_pool_blend' if m['blend_explained_by'] else
                      'reference_conflict' if m['conflict'] and m['conflict'].get('blocks_injection') else None)
            if reason:
                trace['skipped'].append({'slug': slug, 'reason': reason})
            else:
                trace['injected'].append(slug)
        trace['status'] = 'injected' if trace['injected'] else 'matched_not_injected'
        decision['matched'] = [*decision['matched'], *(new[s] for s in sorted(new))]
        decision['matched_products'] = len(keys)
        decision['injected'] = [*decision['injected'], *trace['injected']]
        decision['skipped'] = [*decision['skipped'], *trace['skipped']]
        decision['status'] = ('injected' if decision['injected'] else 'matched_not_injected')
        return decision


def catalogue_selection(root):
    """Registry, context and stats exactly as the live selector builds them, without models."""
    from types import SimpleNamespace
    from rshb_vine.catalog.product_registry import ProductRegistry
    from rshb_vine.recognition_repair_v2 import runtime
    from rshb_vine.system_selection_v4.context import SelectionContext
    from rshb_vine.systemic_ranking_v2.selection import BASELINE_PROFILE
    registry = ProductRegistry.from_bundle(root, verify(read_json(Path(root) / BASELINE_PROFILE))['product_bundle'])
    if registry.checksum != runtime.SELECTOR_REGISTRY_CHECKSUM:
        raise ValueError('selector registry differs from the live profile value')
    context = SelectionContext(root)
    return SimpleNamespace(registry=registry, context=context, stats=E.CatalogStats(registry, context))


if __name__ == '__main__':
    from rshb_vine.io import seal, write_json
    ROOT = Path(__file__).resolve().parents[2]
    table = seal(build_table(ROOT, catalogue_selection(ROOT)))
    write_json(ROOT / TABLE, table)
    print(json.dumps({'table': TABLE, 'checksum': table['checksum'], 'candidates': len(table['candidates']),
                      'admitted': len(table['admitted'])}, ensure_ascii=False))
