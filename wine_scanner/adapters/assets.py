"""Read-only web asset pack: catalogue cards and reference thumbnails exported for one recognition profile.

The manifest is accepted only when its bytes hash to the configured SHA-256 and it names the expected profile;
every file it lists is SHA-checked at startup and again when served.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading

from wine_scanner.contracts import slug as valid_slug

KIND = 'wine-scanner-web-assets-v1'
CARDS_KIND = 'wine-scanner-cards-v1'
CARD_FIELDS = ('slug', 'title', 'in_catalog', 'name', 'producer', 'category', 'region', 'grapes',
               'page_url', 'page_source', 'page_in_site_snapshot')
SITE = 'https://vino-svoe.ru/wines/'
THUMB_CACHE = 256


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


class AssetPack:
    def __init__(self, root, manifest_sha256, expected_profile):
        self.root = Path(root).resolve(strict=True)
        raw = (self.root / 'manifest.json').read_bytes()
        if _sha256(raw) != manifest_sha256:
            raise ValueError('Asset manifest does not match WINE_SCANNER_ASSETS_SHA256')
        manifest = json.loads(raw)
        if manifest.get('kind') != KIND:
            raise ValueError('Unsupported asset manifest kind')
        if manifest.get('profile_checksum') != expected_profile:
            raise ValueError('Asset pack was exported for another recognition profile')
        self.files = {}
        for name, entry in manifest['files'].items():
            path = self._path(name)
            if _sha256(path.read_bytes()) != entry['sha256']:
                raise ValueError('Asset file changed: ' + name)
            self.files[name] = entry['sha256']
        cards = json.loads(self._read('cards.json'))
        if cards.get('kind') != CARDS_KIND:
            raise ValueError('Unsupported cards kind')
        self.cards = {}
        for key, card in cards['cards'].items():
            if valid_slug(key) != key or card.get('slug') != key:
                raise ValueError('Invalid card slug')
            if card.get('page_url') and not card['page_url'].startswith(SITE):
                raise ValueError('Card page is outside vino-svoe.ru/wines/: ' + key)
            thumb = card.get('reference_sha256')
            if thumb is not None and f'thumbnails/{thumb}.jpg' not in self.files:
                raise ValueError('Card reference is not in the manifest: ' + key)
            self.cards[key] = card
        self.pack_id = manifest_sha256[:12]
        self._thumbs = OrderedDict()
        self._lock = threading.Lock()

    def _path(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Asset path escapes the pack: ' + name)
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('Asset path escapes the pack: ' + name)
        return path

    def _read(self, name):
        data = self._path(name).read_bytes()
        if _sha256(data) != self.files[name]:
            raise ValueError('Asset file changed: ' + name)
        return data

    def card(self, slug):
        """Public card of a slug; an unknown but well-formed slug gets a bare card without page or reference."""
        slug = valid_slug(slug)
        if slug is None:
            return None
        known = self.cards.get(slug)
        if known is None:
            return {'slug': slug, 'title': slug, 'in_catalog': False, 'name': None, 'producer': None,
                    'category': None, 'region': None, 'grapes': None, 'page_url': None, 'page_source': None,
                    'page_in_site_snapshot': None, 'reference': None}
        card = {key: known.get(key) for key in CARD_FIELDS}
        card['reference'] = f'/api/reference/{slug}.jpg' if known.get('reference_sha256') else None
        return card

    def thumbnail(self, slug):
        known = self.cards.get(slug)
        if not known or not known.get('reference_sha256'):
            return None
        with self._lock:
            if slug in self._thumbs:
                self._thumbs.move_to_end(slug)
                return self._thumbs[slug]
        data = self._read(f"thumbnails/{known['reference_sha256']}.jpg")
        with self._lock:
            self._thumbs[slug] = data
            while len(self._thumbs) > THUMB_CACHE:
                self._thumbs.popitem(last=False)
        return data

    def summary(self):
        return {'pack': self.pack_id, 'cards': len(self.cards),
                'cards_with_page_url': sum(bool(c.get('page_url')) for c in self.cards.values()),
                'cards_with_reference': sum(bool(c.get('reference_sha256')) for c in self.cards.values())}
