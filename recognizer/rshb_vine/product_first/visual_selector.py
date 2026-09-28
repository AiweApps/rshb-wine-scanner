"""V2-A: untruncated four-channel visual product proposals, without text or fit."""
from fractions import Fraction

CHANNELS = ('B0_label', 'B3_label', 'B0_context', 'B3_context')


def select_visual_product(channels, cards, allowed_products=None):
    """Preserve raw channel evidence; allow a separate common-pool diagnostic."""
    if set(channels) - set(CHANNELS):
        raise ValueError('Nonvisual or unknown channel')
    candidates = {}
    for channel in CHANNELS:
        rows = channels.get(channel, [])
        if len(rows) > 20:
            raise ValueError('Frozen channel exceeds Top20')
        for rank, row in enumerate(rows, 1):
            card = cards[row['slug']]
            if not card['active_in_gallery']:
                raise ValueError('Inactive visual reference')
            # Unresolved identities remain card-local, never inferred aliases.
            key = card['product_id'] if card['binding_status'] == 'source_admitted_product' else 'unresolved-card:' + row['slug']
            if allowed_products is not None and card['product_id'] not in allowed_products:
                continue
            c = candidates.setdefault(key, {'candidate_id': key, 'product_id': card['product_id'],
                'identity_status': card['binding_status'], 'card_slugs': [], 'channels': {}})
            if row['slug'] not in c['card_slugs']:
                c['card_slugs'].append(row['slug'])
            c['channels'].setdefault(channel, {'rank': rank, 'raw': dict(row)})
    ranked = list(candidates.values())
    for c in ranked:
        c['_score'] = sum((Fraction(1, 60 + e['rank']) for e in c['channels'].values()), Fraction())
        c['representative_slug'] = min(c['channels'].items(), key=lambda x: (x[1]['rank'], CHANNELS.index(x[0]), x[1]['raw']['slug']))[1]['raw']['slug']
    ranked.sort(key=lambda c: (-c['_score'], c['candidate_id']))
    margin = float(ranked[0]['_score'] - ranked[1]['_score']) if len(ranked) > 1 else None
    for c in ranked:
        c['score'] = float(c['_score'])
        c['exact_rrf'] = str(c.pop('_score'))
    return {'product_id': ranked[0]['product_id'] if ranked else None,
            'representative_slug': ranked[0]['representative_slug'] if ranked else None,
            'exact_slug': None, 'vintage': {'value': None, 'status': 'unknown'},
            'confidence': {'status': 'uncalibrated_proposal', 'probability': None},
            'rrf_margin': margin, 'ranked_products': ranked,
            'available_channels': [k for k in CHANNELS if channels.get(k)],
            'policy': 'V2-A-equal-RRF60-visual-only-untruncated'}
