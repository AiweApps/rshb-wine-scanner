"""Component A: typed-surface normalization of already read OCR observations.

One versioned rule repairs two reading defects of role words (sugar/colour/style/curated grape aliases) that the
v4 tokenizer cannot see, and nothing else:

- ``confusable_word``: a token written with Latin (or mixed) glyphs whose complete homoglyph Cyrillic reading, via
  the v4 tokenizer tables, is exactly one complete single-token curated alias (CYXOE -> СУХОЕ);
- ``split_word``: 2-3 whitespace-separated pieces of one observation whose concatenation is exactly one complete
  single-token curated alias while no piece alone is one (СУХО Е -> СУХОЕ).

Only curated ``alias_sources`` of the supplied typed lexicon form the vocabulary; catalogue rows, candidates, GT and
the ranking are never read. There is no fuzzy matching and no merging across observations, lines or targets. An
observation holding a negation or a modifier of a role word (НЕ, ПОЛУ, EXTRA, SEMI, ...) is left unchanged, so a
changed token can never turn полусухое/не сухое/extra brut into a bare dry/brut reading. Latin readings that are
themselves alias words (BRUT, ROSE, DRY) and words with a glyph outside the tables stay untouched.

The effective packet is a copy of equal length and order in which only ``raw_text`` of a repaired observation
differs; score, polygon, order and every other key are copied. Provenance is neither modified nor re-traced: callers
keep the original side-car (``contract.trace`` matches by raw_text/score), so every effective reading keeps its
reader, crop source and native score. This is a surface repair of existing reads, not a new or better OCR reading.
"""
from copy import deepcopy
import unicodedata

from rshb_vine.io import digest
from rshb_vine.ocr_candidate_repair_v1 import injection as I
from rshb_vine.system_selection_v4 import text

ROLES = ('grape_blend', 'color', 'sugar', 'style')
NEGATIONS = ('не', 'not', 'non', 'no', 'без')
MIN_FORM = 5
VERTICAL = frozenset('\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029')
RULE = {
    'version': 'ocr-typed-surface-normalization-v1',
    'vocabulary': 'complete single-token curated alias forms of TypedLexicon.alias_sources for the given roles; '
                  'a form owned by two keys of one role is ambiguous and excluded; only forms of at least '
                  'min_form_chars normalized characters may be produced',
    'min_form_chars': MIN_FORM,
    'confusable_word': 'token with Latin glyphs; every glyph mapped by system_selection_v4.text _TO_CYR '
                       '(and the _CYR_ALT variant); all mapped variants agree on exactly one vocabulary form; the '
                       'original normalized token is not any curated alias token',
    'split_word': '2-3 adjacent digit-free pieces of one observation separated only by horizontal whitespace (no line '
                  'or paragraph separator); the concatenation '
                  '(confusable reading when it has Latin glyphs) is exactly one vocabulary form; no piece alone is one',
    'blockers': 'an observation with a negation word or a role modifier (other tokens of a multi-token curated alias '
                'that contains a single-token form, e.g. extra/semi; prefixes joining two single-token forms, e.g. '
                'полу) in any reading is never changed',
    'negations': list(NEGATIONS),
    'scope': 'per observation; no fuzzy matching, numbers, cross-observation, cross-line or cross-target joins',
    'span_basis': 'edit spans index the NFC form of the original raw_text',
    'grades': 'reader, crop source and native score of the observation are kept; provenance side-car unchanged',
}


def line_key(raw):
    return I._line_key(raw)


def _plain(raw):
    return ''.join(c for c in unicodedata.normalize('NFKD', raw) if not unicodedata.combining(c))


def _norm(raw):
    return ' '.join(t.forms[0] for t in text.tokenize(raw, drop_years=False))


def _cyrillic_readings(raw):
    """Distinct normalized complete Cyrillic homoglyph readings of one Latin/mixed token, with their text."""
    chars = _plain(raw)
    if not any(map(text._is_lat, chars)):
        return {}
    readings = {}
    for alternates in (None, text._CYR_ALT):
        converted = text._convert(chars, text._TO_CYR, text._is_cyr, alternates)
        if converted:
            readings.setdefault(_norm(converted), converted)
    return readings


class Vocabulary:
    """Role vocabulary and blocker words derived only from the lexicon's curated alias maps."""

    def __init__(self, lexicon, roles=ROLES):
        unknown = set(roles) - set(ROLES)
        if unknown:
            raise ValueError(f'Unknown typed roles {sorted(unknown)}')
        self.roles = tuple(r for r in ROLES if r in roles)
        owners, alias_tokens, multi, entries = {}, set(), [], []
        for role in self.roles:
            for source, mapping in lexicon.alias_sources[role]:
                for key, aliases in mapping.items():
                    for alias in aliases:
                        norms = [t.forms[0] for t in text.tokenize(alias, drop_years=False)]
                        alias_tokens.update(norms)
                        if len(norms) == 1:
                            owners.setdefault(norms[0], set()).add((role, key))
                            entries.append((norms[0], role, key, source))
                        elif norms:
                            multi.append(norms)
        by_role = {}
        for form, pairs in owners.items():
            for role, key in pairs:
                by_role.setdefault((form, role), set()).add(key)
        ambiguous = {form for (form, _), keys in by_role.items() if len(keys) > 1}
        self.forms = {form: sorted(pairs) for form, pairs in owners.items()
                      if form not in ambiguous and len(form) >= MIN_FORM}
        self.ambiguous = sorted(ambiguous)
        self.all_forms = frozenset(owners)
        self.alias_tokens = frozenset(alias_tokens)
        modifiers = {t for norms in multi if set(norms) & set(owners) for t in norms}
        for form in owners:
            for other in owners:
                if len(other) > len(form) and other.endswith(form) and other[:-len(form)]:
                    modifiers.add(other[:-len(form)])
        negations = {_norm(w) for w in NEGATIONS}
        self.blockers = frozenset((modifiers - set(owners)) | negations)
        self.digest = digest({'rule': RULE, 'roles': list(self.roles), 'entries': sorted(entries),
                              'ambiguous': self.ambiguous, 'blockers': sorted(self.blockers)})

    def resolve(self, raw, *, pieces=False):
        """(effective text, vocabulary form) when ``raw`` reads as exactly one form, else None."""
        readings = _cyrillic_readings(raw)
        if readings:
            if not pieces and _norm(raw) in self.alias_tokens:
                return None
            hits = {form: converted for form, converted in readings.items() if form in self.forms}
            if len(hits) != 1 or len(readings) != len(hits):
                return None
            form, converted = next(iter(hits.items()))
            return converted, form
        form = _norm(raw)
        return (raw, form) if form in self.forms else None

    def is_form(self, raw):
        return any(form in self.all_forms for form in (_norm(raw), *_cyrillic_readings(raw)))

    def is_blocker(self, raw):
        return any(form in self.blockers for form in (_norm(raw), *_cyrillic_readings(raw)))

    def describe(self):
        return {'rule': RULE, 'roles': list(self.roles), 'digest': self.digest, 'forms': len(self.forms),
                'ambiguous_forms': self.ambiguous, 'blockers': sorted(self.blockers)}


def _horizontal_gap(gap):
    return bool(gap) and gap.isspace() and not any(c in VERTICAL for c in gap)


def _words(raw):
    return [(m.start(), m.end(), m.group()) for m in text._WORD.finditer(raw)]


def _edits(raw, vocabulary):
    words = _words(raw)
    if not words or any(vocabulary.is_blocker(w) for _, _, w in words):
        return []
    edits, i = [], 0
    while i < len(words):
        found = None
        for n in (3, 2):
            run = words[i:i + n]
            if len(run) < n or any(any(c.isdigit() for c in w) for _, _, w in run):
                continue
            if not all(_horizontal_gap(raw[a[1]:b[0]]) for a, b in zip(run, run[1:])):
                continue
            if any(vocabulary.is_form(w) for _, _, w in run):
                continue
            resolved = vocabulary.resolve(''.join(w for _, _, w in run), pieces=True)
            if resolved:
                found = ('split_word', run, resolved)
                break
        if found is None:
            start, end, word = words[i]
            if not word.isdigit() and _cyrillic_readings(word):
                resolved = vocabulary.resolve(word)
                if resolved and resolved[0] != word:
                    found = ('confusable_word', [words[i]], resolved)
        if found is None:
            i += 1
            continue
        kind, run, (converted, form) = found
        edits.append({'kind': kind, 'span_basis': 'nfc_raw_text', 'span': [run[0][0], run[-1][1]],
                      'from': raw[run[0][0]:run[-1][1]],
                      'to': converted, 'form': form, 'roles_keys': [list(p) for p in vocabulary.forms[form]]})
        i += len(run)
    return edits


def normalize_text(raw, vocabulary):
    """(effective raw text, edits) of one observation text; the text is returned unchanged without edits."""
    if not isinstance(raw, str) or not raw:
        return raw, []
    nfc = unicodedata.normalize('NFC', raw)
    edits = _edits(nfc, vocabulary)
    if not edits:
        return raw, []
    out, cursor = [], 0
    for e in edits:
        out.append(nfc[cursor:e['span'][0]])
        out.append(e['to'])
        cursor = e['span'][1]
    out.append(nfc[cursor:])
    return ''.join(out), edits


def normalize_observations(observations, provenance, lexicon, *, roles=ROLES, vocabulary=None):
    """(effective observations, trace) for one target packet; inputs are never mutated.

    ``provenance`` must be the original side-car aligned with ``observations`` (entries may be None); it is only
    read for the trace and must be passed on unchanged together with the effective packet.
    """
    if len(provenance) != len(observations):
        raise ValueError('Provenance side-car does not cover the observation packet')
    vocabulary = vocabulary or Vocabulary(lexicon, roles)
    effective, changed, counts = [], [], {'confusable_word': 0, 'split_word': 0}
    for index, (observation, record) in enumerate(zip(observations, provenance)):
        copy = deepcopy(observation)
        raw = observation.get('raw_text')
        new, edits = normalize_text(raw, vocabulary)
        if edits:
            copy['raw_text'] = new
            for e in edits:
                counts[e['kind']] += 1
            changed.append({'index': index, 'reader': (record or {}).get('reader'),
                            'crop_source': (record or {}).get('crop_source'),
                            'native_score': observation.get('score'), 'original': raw, 'effective': new,
                            'original_line_key': line_key(raw), 'effective_line_key': line_key(new),
                            'nfc_differs_from_original': unicodedata.normalize('NFC', raw) != raw, 'edits': edits})
        effective.append(copy)
    if len(effective) != len(observations) or any(
            {k: v for k, v in e.items() if k != 'raw_text'} != {k: v for k, v in o.items() if k != 'raw_text'}
            for e, o in zip(effective, observations)):
        raise ValueError('typed-surface normalization invariant failed')
    trace = {'rule': RULE['version'], 'roles': list(vocabulary.roles), 'vocabulary_digest': vocabulary.digest,
             'observations': len(observations), 'changed_observations': len(changed), 'edit_counts': counts,
             'changed': changed, 'input_digest': digest(observations), 'effective_digest': digest(effective),
             'claim': 'surface repair of existing reads; reader, native score and provenance unchanged; '
                      'not a new OCR reading'}
    return effective, trace


def counterpart_line_keys(result, instance_id, lexicon, *, roles=ROLES, vocabulary=None):
    """Other-target line keys of one result: the raw keys of ``injection.scene_line_keys`` plus the keys of the
    same observations after normalization, so an effective own line never escapes a cross-target ownership check."""
    vocabulary = vocabulary or Vocabulary(lexicon, roles)
    sid, keys = str(instance_id), set(I.scene_line_keys(result, instance_id))
    packets = [p for p in (result.get('instance_text') or {}).get('targets', [])
               if str(p.get('instance_id')) != sid]
    packets += [r for r in (result.get('conditional_label_ocr') or {}).get('reads', [])
                if str(r.get('instance_id')) != sid]
    for packet in packets:
        for o in packet.get('observations', []):
            keys.add(line_key(normalize_text(o.get('raw_text', ''), vocabulary)[0]))
    return keys - {''}
