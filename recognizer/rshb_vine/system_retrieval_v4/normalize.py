"""Shared script-agnostic token folding for OCR lines and catalogue fields; no learned parameters."""
import re
import unicodedata

_CYR_LAT = dict(zip('абвгдежзиклмнопрстуфхцчшщъыьэюя',
                    ['a', 'b', 'v', 'g', 'd', 'e', 'zh', 'z', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't',
                     'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya']))
_LAT_TO_CYR = dict(zip('aceopxykmhbtn', 'асеорхукмнвтп'))
_CYR_TO_LAT = {v: k for k, v in _LAT_TO_CYR.items()}
_ITALIC_TO_CYR = {'m': 'т', 'n': 'л', 'u': 'и'}
_DIGIT_TO_CYR = {'6': 'б', '3': 'з', '0': 'о', '4': 'ч'}
_DIGIT_TO_LAT = {'0': 'o', '1': 'l', '5': 's'}
_UPPER_LOOKALIKE = set('ABCEHKMOPTXY')
_PHONETIC = [('eau', 'o'), ('shch', 'sh'), ('sch', 'sh'), ('kh', 'h'), ('ph', 'f'), ('yu', 'u'), ('ya', 'a'), ('ye', 'e'),
             ('iy', 'i'), ('yy', 'i'), ('ij', 'i'), ('y', 'i'), ('j', 'i'), ('w', 'v'), ('q', 'k'), ('x', 'ks')]
TOKEN = re.compile(r'[^\W_]+', re.UNICODE)


def _base(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold().replace('ё', 'е').replace('й', 'и')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))


def _script(ch):
    if 'а' <= ch <= 'я':
        return 'cyr'
    if 'a' <= ch <= 'z':
        return 'lat'
    return 'digit' if ch.isdigit() else 'other'


def skeleton(token):
    """Latin phonetic skeleton of one already-folded token (Cyrillic transliterated)."""
    text = ''.join(_CYR_LAT.get(c, c) for c in token)
    text = text.replace('ch', 'sh')
    text = re.sub(r'c(?=[eiy])', 's', text).replace('ck', 'k').replace('c', 'k')
    for a, b in _PHONETIC:
        text = text.replace(a, b)
    text = re.sub(r'(.)\1+', r'\1', text)
    return re.sub('[^a-z0-9]', '', text)


def token_variants(raw_token):
    """All skeletons of one OCR token: as read plus homoglyph-folded readings of mixed-script tokens."""
    folded = _base(raw_token)
    variants = {folded}
    scripts = {_script(c) for c in folded}
    letters = [c for c in folded if _script(c) in ('cyr', 'lat')]
    if 'cyr' in scripts and ('lat' in scripts or 'digit' in scripts):
        variants.add(''.join(_LAT_TO_CYR.get(c, _DIGIT_TO_CYR.get(c, c)) for c in folded))
        variants.add(''.join(_ITALIC_TO_CYR.get(c, _LAT_TO_CYR.get(c, _DIGIT_TO_CYR.get(c, c))) for c in folded))
    if 'lat' in scripts and 'cyr' in scripts and sum(_script(c) == 'lat' for c in letters) > len(letters) / 2:
        variants.add(''.join(_CYR_TO_LAT.get(c, _DIGIT_TO_LAT.get(c, c)) for c in folded))
    raw = unicodedata.normalize('NFKC', str(raw_token))
    if len(raw) >= 3 and raw.isupper() and all(c in _UPPER_LOOKALIKE or c in '036' for c in raw):
        variants.add(''.join(_LAT_TO_CYR.get(c, _DIGIT_TO_CYR.get(c, c)) for c in raw.casefold()))
    return {s for s in (skeleton(v) for v in variants) if s}


def tokens(text):
    return [t for t in TOKEN.findall(_base(text))]


def catalogue_skeletons(text):
    return [s for s in (skeleton(t) for t in tokens(text)) if s]


def ocr_skeletons(text):
    """List of per-token variant sets, order preserved."""
    return [v for v in (token_variants(t) for t in TOKEN.findall(str(text))) if v]


def edit_limit(length):
    return 0 if length < 5 else 1 if length < 9 else 2


def levenshtein_within(a, b, limit):
    if abs(len(a) - len(b)) > limit:
        return False
    if limit == 0:
        return a == b
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        low = i
        for j, cb in enumerate(b, 1):
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            current.append(value)
            low = min(low, value)
        if low > limit:
            return False
        previous = current
    return previous[-1] <= limit


def fuzzy_equal(catalogue_skeleton, variants):
    limit = edit_limit(len(catalogue_skeleton))
    return any(levenshtein_within(catalogue_skeleton, v, limit) for v in variants)
