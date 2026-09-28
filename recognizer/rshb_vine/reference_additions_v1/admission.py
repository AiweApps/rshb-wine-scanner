"""Admission of the release-next-v1 reference additions: sources, receipts, card identity, protection and budget.

Everything here is metadata and byte checks; no model is loaded and no pixel is reinterpreted. The admission config
names each added catalogue image by its site-snapshot receipt, the card API body the image URL belongs to, and the
two views whose geometry was fixed by the root-reviewed crop manifest before any encoding.
"""
import json
from pathlib import Path

from rshb_vine.io import local_path, read_json, sha256, verify

CONFIG = 'config/reference-additions-release-next-v1.json'
CONFIG_KIND = 'reference-additions-release-next-v1'
ROOT_ADMISSION_KIND = 'release-next-v1-reference-root-admission'
ROOT_ADMISSION_CHECKSUM = 'f798b158f7b1f7566c85c50b76233e464fbd23b1368ee18cdee696508b4f1274'
CATALOG = 'data/normalized/catalog.jsonl'
PARENT_GALLERY = 'runs/catalog-training-data-v2/stage5/evaluator/actual-fit-v1/B4343/gallery'
PARENT_GALLERY_SHA = '76e12a9b7298e942f85f3b71dc64e1eacadf81075eb27d6914b12ba248b43802'
PARENT_VECTORS_SHA = '8f0ba26e71ab645fd599a57b2110a559bdddc4fc430c4d746d95147af51eeba7'
PARENT_ROWS = 4206
KINDS = ('context', 'front_label')
SHA_LISTS = ('data/web-validation50-20260927/frozen-manifest.json', 'data/evaluation/web232-v1/all663.json',
             'data/evaluation/reset-v1/membership.json')
IMAGE_URL = 'https://api.vino-svoe.ru/v1/img/str-api/0/0/resize'


def _pinned(root, ref, key='path', digest='sha256'):
    path = local_path(root, ref[key])
    if sha256(path) != ref[digest]:
        raise ValueError('Pinned admission input changed: ' + ref[key])
    return path


def catalog_cards(root, slugs):
    cards = {}
    with local_path(root, CATALOG).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['slug'] in slugs:
                cards[row['slug']] = row
    return cards


def protected(root, shas):
    """Every SHA that appears in the protection closure or in the listed evaluation membership files."""
    from rshb_vine.data_protection import load_protection
    from rshb_vine.evaluation.reset_access import check_read
    closure = set(load_protection(root)['forbidden_sha256'])
    hits = {s: ['protection_closure'] for s in shas if s in closure}
    for name in SHA_LISTS:
        path = local_path(root, name)
        check_read(path)
        data = path.read_bytes()
        for s in shas:
            if s.encode() in data:
                hits.setdefault(s, []).append(name)
    return hits


def _check_record(root, record, rows):
    image = record['image']
    receipt = read_json(_pinned(root, image, 'receipt', 'receipt_sha256'))
    path = _pinned(root, image)
    if receipt['sha256'] != image['sha256'] or receipt['body_path'] != image['path'] or receipt['status'] != 200:
        raise ValueError('Image receipt does not bind the admitted bytes: ' + record['slug'])
    api = record['card_api']
    api_receipt = read_json(_pinned(root, api, 'receipt', 'receipt_sha256'))
    body = json.loads(_pinned(root, api).read_bytes())
    if (api_receipt['sha256'] != api['sha256'] or body['slug'] != record['slug']
            or body['title'] != record['catalog_title'] or body['alcohol'] != record['card_api_alcohol']):
        raise ValueError('Card API snapshot does not name this slug/title/alcohol: ' + record['slug'])
    if receipt['url'] != IMAGE_URL + body['image']['url']:
        raise ValueError('Admitted image is not the card image of ' + record['slug'])
    if sha256(local_path(root, 'data/extracted/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads/'
                         + Path(body['image']['url']).name)) != record['pixel_twin']['sha256']:
        raise ValueError('Archive twin of the card image changed: ' + record['slug'])
    if any(r['slug'] == record['slug'] for r in rows):
        raise ValueError('Slug already has parent gallery rows: ' + record['slug'])
    from PIL import Image
    with Image.open(path) as im:
        size = list(im.size)
    views = record['views']
    if size != views['size'] or views['context'] != [0, 0, *size]:
        raise ValueError('Context view is not the full admitted frame: ' + record['slug'])
    x1, y1, x2, y2 = views['front_label']
    if not (x1 == 0 and x2 == size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError('Front label is not the full-width reviewed band: ' + record['slug'])
    return size


def load(root, path=CONFIG):
    """(config, report) after every byte, identity, protection and budget check; raises on the first violation."""
    root = Path(root).resolve()
    config = verify(read_json(local_path(root, path)))
    if config.get('kind') != CONFIG_KIND or config.get('no_query_promotion') is not True:
        raise ValueError('Unsupported reference additions config')
    admission = verify(read_json(_pinned(root, config['root_admission'])))
    if admission.get('kind') != ROOT_ADMISSION_KIND or admission['checksum'] != ROOT_ADMISSION_CHECKSUM:
        raise ValueError('Root admission is not the sealed f798b158 decision')
    for key in ('source_audit', 'amendment'):
        _pinned(root, config[key])
    crop = config['crop_manifest']
    reviewed = {i['sha256']: i for i in read_json(_pinned(root, crop))['items']}
    _pinned(root, crop, 'overlay', 'overlay_sha256')
    records = config['records']
    for r in records:
        item, views = reviewed.get(r['image']['sha256']), r['views']
        if (item is None or item.get('role') != 'new_candidate' or item['path'] != r['image']['path']
                or [item['size'], item['context_bbox'], item['front_label_bbox']]
                != [views['size'], views['context'], views['front_label']]):
            raise ValueError('Views differ from the root-reviewed crop manifest: ' + r['slug'])
    shas = [r['image']['sha256'] for r in records]
    if sorted(shas) != sorted(admission['source_sha256']) or len(set(shas)) != len(shas):
        raise ValueError('Config images differ from the root-admitted source SHAs')
    budget = admission['build_budget']
    if len(records) > budget['canonical_images'] or len(records) * len(KINDS) > budget['B3_views']:
        raise ValueError('Config exceeds the root build budget')
    if any(r['gallery_only'] is not True or r['fit_admitted'] is not False for r in records):
        raise ValueError('Every addition must be gallery-only and not fit-admitted')
    parent = config['parent_gallery']
    folder = local_path(root, parent['path'])
    if (parent['path'] != PARENT_GALLERY or sha256(folder / 'gallery.json') != PARENT_GALLERY_SHA
            or sha256(folder / 'vectors.npy') != PARENT_VECTORS_SHA or parent['rows'] != PARENT_ROWS):
        raise ValueError('Parent B3 gallery is not the frozen 76e12a9b/4206 rows')
    rows = verify(read_json(folder / 'gallery.json'))['references']
    if len(rows) != PARENT_ROWS:
        raise ValueError('Parent gallery row count changed')
    cards = catalog_cards(root, {r['slug'] for r in records})
    missing = [r['slug'] for r in records if r['slug'] not in cards
               or cards[r['slug']]['fields']['Название вина'] != r['catalog_title']
               or cards[r['slug']].get('excluded_from_retrieval')]
    if missing:
        raise ValueError('Catalogue card absent, renamed or excluded: ' + ', '.join(missing))
    sizes = {r['slug']: _check_record(root, r, rows) for r in records}
    family = sorted({s for r in records for s in r['future_independent_query_excluded_sha256']})
    hits = protected(root, [*shas, *family])
    if hits:
        raise ValueError('Admitted image or its pixel family is protected: ' + ', '.join(sorted(hits)))
    report = {'config': {'path': path, 'checksum': config['checksum'], 'sha256': sha256(local_path(root, path))},
              'root_admission_checksum': admission['checksum'], 'parent_gallery_sha256': PARENT_GALLERY_SHA,
              'parent_vectors_sha256': PARENT_VECTORS_SHA, 'parent_rows': PARENT_ROWS,
              'additions': [{'slug': r['slug'], 'image_sha256': r['image']['sha256'], 'size': sizes[r['slug']],
                             'views': {k: r['views'][k] for k in KINDS}} for r in records],
              'new_rows': len(records) * len(KINDS),
              'future_independent_query_excluded_sha256': family,
              'family_protection_hits': {s: hits[s] for s in family if s in hits}}
    return config, report
