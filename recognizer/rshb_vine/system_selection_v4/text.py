"""Token readings and typed catalogue lexicon for selector v4.

Match kinds are ordered reading quality, never identity proof. Typed aliases come
only from existing runtime maps and literal catalogue field values; no query fit.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from rshb_vine.typed_catalog_lexicon import normalize, TypedCatalogLexicon
from rshb_vine.positive_variant_text import ALIASES
from rshb_vine.resolution.identity import GRAPES, SUGAR, COLOR, STYLE
from rshb_vine.gallery_variant_text import STYLE_ALIASES

VERSION = 'system-selection-v4-text-v1'
YEAR_MIN, YEAR_MAX = 1800, 2099
MATCH_ORDER = ('exact', 'homoglyph', 'translit', 'fuzzy')
RANK = {kind: i for i, kind in enumerate(MATCH_ORDER)}

# Ordered; both OCR and catalogue tokens always pass through the same sequence.
SKELETON_RULES = (
    ('tsch|tch', 'ch'), ('shch|sch', 'sh'), ('ch', 'sh'), ('ph', 'f'), ('th', 't'), ('rh', 'r'),
    ('gh', 'g'), ('kh', 'h'), ('dzh', 'g'), ('gi(?=[aou])', 'g'), ('gn', 'n'), ('eaux?', 'o'),
    ('au', 'o'), ('x', 'ks'), ('(?<=[a-z]{4})(?<!t)s$', ''), ('(?<=n)c$', ''), ('(?<=[aeiou])t$', ''),
    ('ck', 'k'), ('qu?', 'k'), ('c(?=[eiy])', 's'), ('c', 'k'), ('tz', 'ts'), ('w', 'v'), ('ou', 'u'),
    ('j', 'y'), ('ye', 'e'), ('yo', 'o'), ('y', 'i'), (r'([a-z])\1+', r'\1'),
    ('iu', 'u'), ('ia', 'a'), ('ie', 'i'), ('(?<=[aeiou])h$', ''), (r'([a-z])\1+', r'\1'),
)
_RULES = tuple((re.compile(p), r) for p, r in SKELETON_RULES)

# Case matters: H/B/N/M look like Н/В/И/М only in upper case.
_TO_CYR = dict(zip('AaBbCcEeHKkMmNnOoPprTtuXxYyg0346', 'АаВьСсЕеНКкМмИпОоРргТтиХхУудозчб'))
_CYR_ALT = {'n': 'л', 'm': 'т'}
_TO_LAT = dict(zip('АаВвЕеКкМмНнОоРрСсТтУуХхипгьИіІ0', 'AaBbEeKkMmHhOoPpCcTtYyXxunrbNiIo'))
_WORD = re.compile(r'(?:[^\W_]|[\u0300-\u036f])+')
_RUNS = re.compile(r'\d+|\D+')
_GRAPE_SPLIT = re.compile(r'[,;/+]|\s+-\s+|\s+и\s+', re.I)
_GRAPE_FILLER = frozenset(normalize('сорт сорта сортов виноград винограда').split())


def is_year(token):
    return len(token) == 4 and token.isascii() and token.isdigit() and YEAR_MIN <= int(token) <= YEAR_MAX


def base_norm(text):
    return normalize(text)


@lru_cache(maxsize=1 << 16)
def skeleton(norm_token):
    text = norm_token
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


@dataclass(frozen=True)
class Token:
    raw: str
    forms: tuple
    skeletons: tuple


def _is_lat(c):
    return c.isascii() and c.isalpha()


def _is_cyr(c):
    return '\u0400' <= c <= '\u04ff'


def _convert(chars, table, native, alternates=None):
    out = []
    for c in chars:
        if native(c):
            out.append(c)
        elif c in table:
            out.append((alternates or {}).get(c, table[c]))
        else:
            return None
    return ''.join(out)


def _homoglyph_forms(raw, form):
    chars = ''.join(c for c in unicodedata.normalize('NFKD', raw) if not unicodedata.combining(c))
    lat, cyr = sum(map(_is_lat, chars)), sum(map(_is_cyr, chars))
    if not (lat and cyr or (lat or cyr) and any(c in '0346' for c in chars)):
        return ()
    cyrillic = [_convert(chars, _TO_CYR, _is_cyr), _convert(chars, _TO_CYR, _is_cyr, _CYR_ALT)]
    latin = [_convert(chars, _TO_LAT, _is_lat)]
    forms = []
    for variant in (cyrillic + latin if cyr >= lat else latin + cyrillic):
        norm = normalize(variant) if variant else ''
        if norm and ' ' not in norm and norm != form and norm not in forms:
            forms.append(norm)
    return tuple(forms[:2])


def _pieces(raw):
    # A lone 0/3/4/6 inside letters is a glyph candidate; other digit runs are numbers.
    runs = _RUNS.findall(raw)
    glue = [r.isdigit() and len(r) == 1 and r in '0346' for r in runs]
    pieces = [runs[0]]
    for i in range(1, len(runs)):
        if glue[i] or glue[i - 1]:
            pieces[-1] += runs[i]
        else:
            pieces.append(runs[i])
    return pieces


@lru_cache(maxsize=1 << 15)
def _tokenize(text, drop_years):
    tokens = []
    for word in _WORD.findall(unicodedata.normalize('NFC', text)):
        for raw in _pieces(word):
            norm = normalize(raw)
            if not norm or drop_years and is_year(norm):
                continue
            if ' ' in norm:
                tokens.extend(Token(raw, (n,), (skeleton(n),)) for n in norm.split()
                              if not (drop_years and is_year(n)))
                continue
            forms = (norm, *_homoglyph_forms(raw, norm))
            tokens.append(Token(raw, forms, tuple(skeleton(f) for f in forms)))
    return tuple(tokens)


def tokenize(text, *, drop_years=True):
    return _tokenize(str(text), bool(drop_years))


def _within(a, b, limit):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


@lru_cache(maxsize=1 << 18)
def _fuzzy(a, b):
    if len(a) < 5 or len(b) < 5 or a[0] != b[0] or any(c.isdigit() for c in a + b):
        return False
    limit = 2 if len(a) >= 9 and len(b) >= 9 else 1
    return abs(len(a) - len(b)) <= limit and _within(a, b, limit)


def token_match(query, claim_norm):
    forms = query.forms
    if claim_norm == forms[0]:
        return 'exact'
    if not claim_norm:
        return None
    if claim_norm in forms:
        return 'homoglyph'
    target = skeleton(claim_norm)
    if len(target) >= 3 and target in query.skeletons:
        return 'translit'
    if len(target) >= 5 and any(_fuzzy(s, target) for s in query.skeletons):
        return 'fuzzy'
    return None


_LINE_INDEX = {}


def _line_index(line):
    entry = _LINE_INDEX.get(id(line))
    if entry is not None and entry[0] is line:
        return entry[1]
    forms, skeletons, initials = {}, {}, {}
    for pos, token in enumerate(line):
        for form in token.forms:
            forms.setdefault(form, set()).add(pos)
        for s in token.skeletons:
            skeletons.setdefault(s, set()).add(pos)
            if len(s) >= 5:
                initials.setdefault(s[0], set()).add(pos)
    index = forms, skeletons, initials
    if isinstance(line, tuple):
        if len(_LINE_INDEX) >= 4096:
            _LINE_INDEX.clear()
        # The stored reference keeps id(line) unique while the entry lives.
        _LINE_INDEX[id(line)] = (line, index)
    return index


def phrase_match(line, claim_norms):
    norms = tuple(n for n in claim_norms if n and not is_year(n))
    if not norms or len(norms) > len(line):
        return None
    forms, skeletons, initials = _line_index(line)
    first = skeleton(norms[0])
    starts = forms.get(norms[0], set()) | skeletons.get(first, set())
    if len(first) >= 5:
        starts = starts | initials.get(first[0], set())
    best = None
    for start in sorted(p for p in starts if p + len(norms) <= len(line)):
        kinds = []
        for offset, norm in enumerate(norms):
            kind = token_match(line[start + offset], norm)
            if kind is None:
                break
            kinds.append(kind)
        else:
            ranks = [RANK[k] for k in kinds]
            score = (max(ranks), sum(ranks), start)
            if best is None or score < best[0]:
                best = (score, start, kinds)
    if best is None:
        return None
    _, start, kinds = best
    return {'start': start, 'end': start + len(norms), 'kinds': kinds,
            'weakest': max(kinds, key=RANK.__getitem__)}


@dataclass(frozen=True)
class _Entry:
    role: str
    norms: tuple
    key: str
    source: str
    mapping: str
    curated: bool


def _raw_key(tokens):
    return 'raw:' + '_'.join(t.skeletons[0] for t in tokens)


class TypedLexicon:
    ROLES = ('grape_blend', 'color', 'sugar', 'style')

    def __init__(self, catalog_rows, root):
        grape_maps = (('identity.GRAPES', GRAPES),
                      ('positive_variant_text.ALIASES', {k: v for k, v in ALIASES.items() if k != 'reserve'}),
                      ('TypedCatalogLexicon.grapes', TypedCatalogLexicon([], Path(root)).grapes))
        self.alias_sources = {'grape_blend': grape_maps, 'color': (('identity.COLOR', COLOR),),
                              'sugar': (('identity.SUGAR', SUGAR),),
                              'style': (('identity.STYLE', STYLE), ('gallery_variant_text.STYLE_ALIASES', STYLE_ALIASES))}
        self._entries, self._seen = [], set()
        for role, maps in self.alias_sources.items():
            for source, mapping in maps:
                for key, aliases in mapping.items():
                    for alias in aliases:
                        self._add(role, tokenize(alias), key, source, 'alias', True)
        self._add_unique_heads()
        self._fragments = {(e.role, s[i:j]) for e in self._entries for s in [tuple(map(skeleton, e.norms))]
                           for i in range(len(s)) for j in range(i + 1, len(s) + 1) if j - i < len(s)}
        self.skipped_fragments = 0
        self._reindex()
        rows = [r for r in catalog_rows if not r.get('excluded_from_retrieval')]
        self.catalog_rows, self.catalog_excluded = len(rows), len(catalog_rows) - len(rows)
        for value in sorted({r['fields'].get('Сорт винограда', '') for r in rows} - {''}):
            for piece in _GRAPE_SPLIT.split(value):
                keys = self._grape_piece(piece)
                main = self._grape_main(piece)
                if len({k['key'] for k in keys}) == 1 and main:
                    self._add('grape_blend', main, keys[0]['key'], 'catalog:Сорт винограда', keys[0]['mapping'], False)
        for value in sorted({r['fields'].get('Категория', '') for r in rows} - {''}):
            for item in self._color_raw(tokenize(value)):
                self._add('color', item[1], item[0], 'catalog:Категория', 'catalog_raw_form', False)
        self._reindex()
        self._cache = {}

    def _add(self, role, tokens, key, source, mapping, curated):
        norms = tuple(t.forms[0] for t in tokens)
        # A catalogue piece that is only part of a curated alias (Совиньон) is an OCR-line fragment risk.
        if not curated and (role, tuple(map(skeleton, norms))) in self._fragments:
            self.skipped_fragments += 1
            return
        if norms and (role, norms, key) not in self._seen:
            self._seen.add((role, norms, key))
            self._entries.append(_Entry(role, norms, key, source, mapping, curated))

    def _add_unique_heads(self):
        # A grape alias head used by no other key anywhere may stand alone (Шенен for Шенен Блан).
        owners = {}
        for e in self._entries:
            for norm in e.norms:
                owners.setdefault(skeleton(norm), set()).add(e.key)
        for e in list(self._entries):
            head = skeleton(e.norms[0])
            if e.role == 'grape_blend' and len(e.norms) > 1 and len(head) >= 5 and owners[head] == {e.key}:
                self._add(e.role, tokenize(e.norms[0]), e.key, e.source, 'alias', True)

    def _reindex(self):
        self._by_form, self._by_skeleton, self._by_initial = {}, {}, {}
        for i, e in enumerate(self._entries):
            head = skeleton(e.norms[0])
            self._by_form.setdefault(e.norms[0], []).append(i)
            self._by_skeleton.setdefault(head, []).append(i)
            if len(head) >= 5:
                self._by_initial.setdefault(head[0], []).append((len(head), i))

    def _candidates(self, line, roles, curated_only):
        found = []
        for start, token in enumerate(line):
            ids = set()
            for form in token.forms:
                ids.update(self._by_form.get(form, ()))
            for s in token.skeletons:
                ids.update(self._by_skeleton.get(s, ()))
                if len(s) >= 5:
                    ids.update(i for n, i in self._by_initial.get(s[0], ()) if abs(n - len(s)) <= 2)
            for i in ids:
                e = self._entries[i]
                end = start + len(e.norms)
                if e.role not in roles or curated_only and not e.curated or end > len(line):
                    continue
                kinds = [token_match(line[start + j], norm) for j, norm in enumerate(e.norms)]
                if None in kinds:
                    continue
                # Single catalogue words may be ordinary vocabulary (менее ~ Менье); no short folded reading.
                if not e.curated and len(e.norms) == 1 and RANK[kinds[0]] >= RANK['translit'] and \
                        len(skeleton(e.norms[0])) < (7 if kinds[0] == 'fuzzy' else 6):
                    continue
                found.append((e, start, end, kinds))
        return found

    def _select(self, found):
        result = {role: [] for role in self.ROLES}
        def quality(m):
            ranks = [RANK[k] for k in m[3]]
            return max(ranks), sum(ranks)
        ranked = sorted(found, key=lambda m: (m[1] - m[2], *quality(m), -len(' '.join(m[0].norms)), m[1], m[0].key))
        tied = {}
        for m in found:
            tied.setdefault((m[0].role, m[1], m[2], *quality(m)), set()).add(m[0].key)
        used = {role: set() for role in self.ROLES}
        for e, start, end, kinds in ranked:
            span = set(range(start, end))
            if span & used[e.role]:
                continue
            # Equal readings of different keys on one span are ambiguous, not evidence.
            if len(tied[(e.role, start, end, *quality((e, start, end, kinds)))]) > 1:
                continue
            used[e.role] |= span
            result[e.role].append({'key': e.key, 'start': start, 'end': end, 'form': ' '.join(e.norms),
                                   'match_kind': max(kinds, key=RANK.__getitem__), 'source': e.source,
                                   'mapping': e.mapping})
        for items in result.values():
            items.sort(key=lambda m: m['start'])
        return result

    def entities(self, line):
        line = tuple(line)
        cached = self._cache.get(line)
        if cached is None:
            if len(self._cache) >= 8192:
                self._cache.clear()
            cached = self._cache[line] = self._select(self._candidates(line, set(self.ROLES), False))
        return {role: [dict(m) for m in items] for role, items in cached.items()}

    @staticmethod
    def _grape_alternatives(piece):
        return [tokenize(part) for part in re.split(r'[()]', piece) if tokenize(part)]

    def _grape_main(self, piece):
        alternatives = self._grape_alternatives(piece)
        return alternatives[0] if alternatives else ()

    def _grape_piece(self, piece):
        alternatives = self._grape_alternatives(piece)
        if not alternatives or any(t.forms[0] in _GRAPE_FILLER for t in alternatives[0]):
            return []
        for tokens in alternatives:
            matches = self._select(self._candidates(tokens, {'grape_blend'}, True))['grape_blend']
            if matches and matches[0]['start'] == 0:
                covered = {i for m in matches for i in range(m['start'], m['end'])}
                qualifier = ' '.join(t.forms[0] for i, t in enumerate(tokens) if i not in covered)
                return [{'key': m['key'], 'raw_piece': piece.strip(), 'mapping': 'alias',
                         'match_kind': m['match_kind'], 'qualifier': qualifier} for m in matches]
        return [{'key': _raw_key(alternatives[0]), 'raw_piece': piece.strip(), 'mapping': 'catalog_raw_form',
                 'match_kind': None, 'qualifier': ''}]

    def _color_raw(self, tokens):
        # A category word outside every colour/sugar/style alias is a literal colour (Оранжевое).
        found = self._select(self._candidates(tokens, {'color', 'sugar', 'style'}, True))
        covered = {i for items in found.values() for m in items for i in range(m['start'], m['end'])}
        rest = [t for i, t in enumerate(tokens) if i not in covered and not t.forms[0].isdigit()]
        return [(_raw_key(rest), tuple(rest))] if not found['color'] and len(rest) == 1 else []

    def claim_keys(self, role, raw_value):
        if role not in self.ROLES:
            raise ValueError(f'Unknown typed role {role}')
        if role == 'grape_blend':
            keys = [k for piece in _GRAPE_SPLIT.split(str(raw_value)) for k in self._grape_piece(piece)]
        else:
            tokens = tokenize(raw_value)
            keys = [{'key': m['key'], 'raw_piece': ' '.join(t.raw for t in tokens[m['start']:m['end']]),
                     'mapping': 'alias', 'match_kind': m['match_kind'], 'qualifier': ''}
                    for m in self._select(self._candidates(tokens, {role}, True))[role]]
            if role == 'color' and not keys:
                keys = [{'key': key, 'raw_piece': ' '.join(t.raw for t in rest), 'mapping': 'catalog_raw_form',
                         'match_kind': None, 'qualifier': ''} for key, rest in self._color_raw(tokens)]
        unique, seen = [], set()
        for k in keys:
            if k['key'] not in seen:
                seen.add(k['key'])
                unique.append(k)
        return unique

    def describe(self):
        roles = {}
        for role in self.ROLES:
            entries = [e for e in self._entries if e.role == role]
            roles[role] = {'forms': len(entries), 'keys': len({e.key for e in entries}),
                           'curated_forms': sum(e.curated for e in entries),
                           'catalog_forms': sum(not e.curated for e in entries),
                           'raw_keys': len({e.key for e in entries if e.key.startswith('raw:')}),
                           'alias_sources': [s for s, _ in self.alias_sources[role]]}
        return {'version': VERSION, 'roles': roles, 'catalog_rows': self.catalog_rows,
                'catalog_rows_excluded': self.catalog_excluded, 'skipped_alias_fragments': self.skipped_fragments,
                'skeleton_rules': [list(r) for r in SKELETON_RULES], 'max_homoglyph_forms': 2,
                'match_order': list(MATCH_ORDER)}
