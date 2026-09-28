"""Component I: an identity phrase printed on both cards in different scripts is not lost.

The frozen ``BottleIsolationEvidence.lost_evidence`` compares NFKC/casefold strings, so «di caspico» read on the label
is lost when the proposed card spells it «Ди Каспико». I keeps that literal result and removes a lost span only when
its whole token sequence matches, contiguously and in order, one field (producer, name, grape) of the proposed card
under the catalogue normal form of selector v4 (``text.tokenize``: Cyrillic transliteration, homoglyphs, skeleton)
with match kinds exact/homoglyph/translit only; fuzzy similarity never recovers a span. A non-exact token pair counts
only as an actual script correspondence (Cyrillic vs Latin, or a mixed-script token on either side): two same-script
spellings with one phonetic skeleton are not identity. Tokens carrying a digit match only exactly. Nothing is
added to the lost set, every other resolver reason (typed colour/grape/sugar/style guard, positive variant, verified
sugar) and the OCR packet are untouched. The resolver used is an instance-local copy; the frozen one is not modified.
"""
import copy
import sys
import time
import unicodedata

from rshb_vine.system_selection_v4 import text

STAGE = 'decision-consistency-cross-script-identity-v1'
FIELDS = ('Винодельня', 'Название вина', 'Сорт винограда')
MATCHED = ('exact', 'homoglyph', 'translit')
RULE = {
    'version': 'cross-script-identity-phrase-v1',
    'recover': 'literal lost span whose full token sequence matches one proposed-card field contiguously with '
               'system_selection_v4.text token_match kinds exact/homoglyph/translit (catalogue normalize + skeleton)',
    'script': 'homoglyph/translit only between different scripts (cyrillic vs latin) or with a mixed-script token; '
              'same-script skeleton equality is not recovered',
    'never': 'fuzzy match, same-script translit, new lost spans, other resolver reasons, OCR packet, frozen resolver object',
    'numbers': 'kept (drop_years=False); a token with any digit matches only exact; standalone years already removed by '
               'the frozen v2 comparison projection',
}


def script(raw):
    letters = [c for c in unicodedata.normalize('NFKD', raw) if c.isalpha()]
    lat, cyr = any(c.isascii() for c in letters), any('\u0400' <= c <= '\u04ff' for c in letters)
    return 'mixed' if lat and cyr else 'latin' if lat else 'cyrillic' if cyr else 'other' if letters else 'none'


def _match(a, b):
    """Strongest admitted kind between two tokens in either direction with its script evidence, or None."""
    kinds = [k for k in (text.token_match(a, b.forms[0]), text.token_match(b, a.forms[0])) if k in MATCHED]
    if not kinds:
        return None
    kind = min(kinds, key=text.RANK.__getitem__)
    scripts = (script(a.raw), script(b.raw))
    if kind != 'exact':
        if any(c.isdigit() for c in a.raw + b.raw):
            return None
        if not ('mixed' in scripts or set(scripts) == {'latin', 'cyrillic'}):
            return None
    return {'kind': kind, 'span_raw': a.raw, 'card_raw': b.raw, 'span_script': scripts[0], 'card_script': scripts[1]}


def find(span_tokens, field_tokens):
    """(start, per-token evidence) of the first contiguous same-length match of ``span_tokens``, or None."""
    n = len(span_tokens)
    for start in range(len(field_tokens) - n + 1):
        pairs = []
        for a, b in zip(span_tokens, field_tokens[start:start + n]):
            pair = _match(a, b)
            if pair is None:
                break
            pairs.append(pair)
        else:
            return start, pairs
    return None


class CrossScriptIdentity:
    """``lost_evidence`` of the frozen identity component, minus spans the proposed card carries in another script."""

    def __init__(self, frozen):
        self.frozen, self.catalog, self.retry = frozen, frozen.catalog, frozen.retry
        self.last = None

    def lost_evidence(self, observations, before, after):
        literal = self.frozen.lost_evidence(observations, before, after)
        self.last = {'rule': RULE['version'], 'before': before, 'after': after, 'literal': literal,
                     'kept': literal, 'recovered': []}
        if not literal:
            return literal
        fields = {k: text.tokenize(self.catalog[after].get(k, ''), drop_years=False) for k in FIELDS}
        kept, recovered = [], []
        for span in literal:
            tokens = text.tokenize(span, drop_years=False)
            hit = None
            if len(tokens) == len(span.split()):
                hit = next(((k, found) for k, toks in fields.items() for found in [find(tokens, toks)] if found), None)
            if hit is None:
                kept.append(span)
                continue
            field, (start, pairs) = hit
            recovered.append({'span': span, 'field': field, 'card_value': self.catalog[after].get(field, ''),
                              'card_start': start, 'card_tokens': [t.raw for t in fields[field][start:start + len(tokens)]],
                              'span_forms': [t.forms[0] for t in tokens], 'kinds': [p['kind'] for p in pairs],
                              'scripts': pairs,
                              'cross_script': any('mixed' in (p['span_script'], p['card_script']) or
                                                  {p['span_script'], p['card_script']} == {'latin', 'cyrillic'}
                                                  for p in pairs)})
        self.last.update(kept=kept, recovered=recovered)
        return kept


def _caller(frame):
    path = frame.f_code.co_filename.replace('\\', '/')
    return '%s:%s' % (path.split('rshb_vine/', 1)[-1], frame.f_code.co_name)


class CrossScriptResolution:
    """Instance-local copy of the frozen ProductEvidenceResolution whose identity component is CrossScriptIdentity.

    Between ``begin`` and ``end`` (one target, serial use only) every resolve is recorded with its direct caller and a
    deep-copied snapshot of the I detail, so recoveries in producer/M/guard/pairwise/geometry resolves are counted too.
    """

    def __init__(self, frozen):
        self.frozen = frozen
        self.identity = CrossScriptIdentity(frozen.identity)
        self.copy = copy.copy(frozen)
        self.copy.identity = self.identity
        self.calls = None

    def begin(self):
        if self.calls is not None:
            raise RuntimeError('I resolver call history is already open (serial use only)')
        self.calls = []

    def end(self):
        calls, self.calls = self.calls, None
        return calls or []

    def resolve(self, features, proposal):
        self.identity.last = None
        out = self.copy.resolve(features, proposal)
        detail = self.identity.last
        out['existing_evidence_resolution']['cross_script_identity'] = detail
        if self.calls is not None:
            self.calls.append(copy.deepcopy({
                'index': len(self.calls), 'caller': _caller(sys._getframe(1)),
                'raw_top': proposal.get('representative_slug'), 'selected': out.get('representative_slug'),
                'reason': out['existing_evidence_resolution'].get('reason'),
                'literal': detail and detail['literal'], 'recovered': detail['recovered'] if detail else []}))
        return out

    def __getattr__(self, name):
        return getattr(self.frozen, name)


def recovered_calls(calls):
    return [c for c in calls if c['recovered']]


def legacy_view(legacy):
    """Shallow copy of the running legacy selection with the I resolver; the running object is not modified."""
    view = copy.copy(legacy)
    view.resolver = CrossScriptResolution(legacy.resolver)
    return view


def _without(resolution):
    out = copy.deepcopy(resolution)
    out['existing_evidence_resolution'].pop('cross_script_identity', None)
    return out


class CrossScriptStage:
    """First pre_guard stage: the initial proposal (resolved by the frozen inner resolver) is resolved again with I."""

    trace_key = 'cross_script_identity'
    position = 'pre_guard'

    def __init__(self):
        self.identity = {'stage': STAGE, 'rule': RULE, 'position': 'first pre_guard; later resolves use the I copy',
                         'calls': 'every I resolve of the target: caller, raw top, selected, recovered spans'}
        self.last_seconds = None

    def apply(self, selection, out, call, request):
        started = time.perf_counter()
        trace = {'stage': STAGE, 'applied': False}
        out[self.trace_key] = trace
        try:
            if out.get('raw_proposal') is None:
                trace['reason'] = 'no_proposal'
                return out
            before = out['proposal']
            resolved = selection.legacy.resolver.resolve(out['base'], out['raw_proposal'])
            detail = resolved['existing_evidence_resolution'].get('cross_script_identity')
            trace['identity'] = detail
            if not (detail and detail['recovered']):
                if _without(resolved) != before:
                    raise RuntimeError('I resolver differs from the frozen resolver without a recovered span')
                trace['reason'] = 'no_recovered_span'
                return out
            trace.update(reasons={'before': before['existing_evidence_resolution'].get('reasons'),
                                  'after': resolved['existing_evidence_resolution'].get('reasons')},
                         selected={'before': before['representative_slug'], 'after': resolved['representative_slug']})
            trace['applied'] = resolved['representative_slug'] != before['representative_slug']
            trace['reason'] = 'selection_changed' if trace['applied'] else 'other_reasons_keep_selection'
            out['proposal_before_cross_script'] = before
            out['proposal'] = resolved
            return out
        finally:
            self.last_seconds = time.perf_counter() - started
