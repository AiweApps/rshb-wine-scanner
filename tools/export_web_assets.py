"""Export the gateway's read-only asset pack from a recognition checkout.

Reads cards, vino-svoe.ru links and SHA-verified reference thumbnails through the checkout's own pinned
scan-web catalogue (base gallery plus the verified reference overlay of the current profile) and writes
manifest.json, cards.json and thumbnails/<sha256>.jpg into a new directory. Run it with the checkout's
Python environment; nothing in the checkout is modified.

  python tools/export_web_assets.py --source-root ../rshb-vine --out /path/to/assets/<name>
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

KIND = 'wine-scanner-web-assets-v1'
CARDS_KIND = 'wine-scanner-cards-v1'
CARD_FIELDS = ('slug', 'title', 'in_catalog', 'name', 'producer', 'category', 'region', 'grapes',
               'page_url', 'page_source', 'page_in_site_snapshot')
SLUG = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,200}$')
CHECKSUM = re.compile(r'^[0-9a-f]{64}$')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1).encode() + b'\n'


def export(source_root, out, expected_profile):
    sys.path.insert(0, str(source_root))
    from rshb_vine.scan_web_v1.catalog import Catalog

    catalog = Catalog(source_root)
    references = catalog.references
    before = catalog.summary()
    if before['reference_overlay_error']:
        raise SystemExit('Reference overlay of the current profile failed verification: '
                         + before['reference_overlay_error'])
    profile = before['pointer_profile_checksum']
    if not CHECKSUM.match(profile or ''):
        raise SystemExit('Current recognition pointer has no profile checksum')
    if expected_profile and profile != expected_profile:
        raise SystemExit(f'Current pointer profile {profile} is not the expected {expected_profile}')

    staging = out.with_name(out.name + '.partial')
    if out.exists() or staging.exists():
        raise SystemExit(f'Refusing to overwrite {out} (or its .partial staging directory)')
    (staging / 'thumbnails').mkdir(parents=True)
    files, cards, skipped, missing = {}, {}, [], []
    try:
        for slug in sorted(set(catalog.cards) | set(references)):
            if not SLUG.match(slug):
                skipped.append(slug)
                continue
            card = catalog.card(slug)
            entry = {key: card.get(key) for key in CARD_FIELDS}
            entry['reference_sha256'] = None
            if card.get('reference'):
                data = catalog.thumbnail(slug)
                if data is None:
                    missing.append(slug)
                else:
                    digest = sha256(data)
                    name = f'thumbnails/{digest}.jpg'
                    if name not in files:
                        (staging / name).write_bytes(data)
                        files[name] = {'sha256': digest, 'bytes': len(data)}
                    entry['reference_sha256'] = digest
            cards[slug] = entry
        after = catalog.summary()
        if after['pointer_profile_checksum'] != profile or after['reference_overlay'] != before['reference_overlay']:
            raise SystemExit('Recognition pointer changed during export; nothing was published')
        cards_bytes = dump({'kind': CARDS_KIND, 'profile_checksum': profile, 'cards': cards})
        (staging / 'cards.json').write_bytes(cards_bytes)
        files['cards.json'] = {'sha256': sha256(cards_bytes), 'bytes': len(cards_bytes)}
        overlay = before['reference_overlay']
        manifest = {
            'kind': KIND,
            'profile_checksum': profile,
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'exporter': {'name': 'tools/export_web_assets.py', 'sha256': sha256(Path(__file__).read_bytes())},
            'source': {
                'catalog_pins_sha256': before['pins'],
                'active_gallery': {'sha256': before['active_gallery']['sha256'],
                                   'rows': before['active_gallery']['rows']},
                'reference_overlay': None if overlay is None else {
                    'profile_checksum': overlay['profile_checksum'],
                    'descriptor_checksum': overlay['descriptor']['checksum'],
                    'gallery_sha256': overlay['gallery']['sha256'],
                    'added_slugs': sorted(overlay['added_slugs']),
                    'added_context_rows': overlay['added_context_rows']}},
            'counts': {'cards': len(cards), 'catalog_rows': before['cards'],
                       'cards_with_page_url': sum(bool(c['page_url']) for c in cards.values()),
                       'cards_with_reference': sum(bool(c['reference_sha256']) for c in cards.values()),
                       'thumbnails': len(files) - 1, 'gallery_slugs_with_reference': len(references),
                       'references_unavailable': len(missing), 'skipped_invalid_slugs': len(skipped)},
            'references_unavailable': missing,
            'files': dict(sorted(files.items())),
        }
        manifest_bytes = dump(manifest)
        (staging / 'manifest.json').write_bytes(manifest_bytes)
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return sha256(manifest_bytes), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-root', type=Path, required=True, help='Recognition checkout (rshb-vine)')
    parser.add_argument('--out', type=Path, required=True, help='New directory for the asset pack')
    parser.add_argument('--expected-profile', help='Refuse unless the current pointer has this profile checksum')
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('--out must be outside this repository: asset packs are not committed or baked into images')
    digest, manifest = export(source_root, out, args.expected_profile)
    print(json.dumps({'out': str(out), 'manifest_sha256': digest, 'profile_checksum': manifest['profile_checksum'],
                      'counts': manifest['counts']}, ensure_ascii=False, indent=1))
    print(f"\nWINE_SCANNER_ASSETS_DIR={out}\nWINE_SCANNER_ASSETS_SHA256={digest}\n"
          f"WINE_SCANNER_EXPECTED_PROFILE={manifest['profile_checksum']}")


if __name__ == '__main__':
    main()
