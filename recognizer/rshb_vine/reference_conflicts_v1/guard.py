import json
from pathlib import Path

from rshb_vine.evaluation.reset_access import check_read
from rshb_vine.io import local_path, read_json, read_jsonl, sha256, verify

CONFIG = 'config/reference-conflicts-v1.json'
KIND = 'reference-conflicts-v1'


def source_checks(root, config):
    """Compare every pinned source fact with the current bytes/records; no effect is derived here."""
    checks = []
    for c in config['conflicts']:
        for key, expected in c['expected_old_values'].items():
            path, _, field = key.partition('#')
            actual = _actual(Path(root), c, path, field)
            checks.append({'conflict': c['id'], 'key': key, 'expected': expected, 'actual': actual,
                           'ok': actual == expected})
    return checks


def _actual(root, conflict, path, field):
    full = local_path(root, path)
    if not full.exists():
        return None
    if not field:
        return sha256(full)
    if field.endswith('.selected_sha256'):
        slug = field.removesuffix('.selected_sha256')
        rows = [r for r in read_jsonl(full) if r['slug'] == slug]
        return rows[0]['selected_sha256'] if len(rows) == 1 else None
    if field == 'pair.decision':
        pair = conflict['effects']['blocked_negative_pairs'][0]
        rows = [r for r in read_jsonl(full) if sorted((r['product_a03'], r['product_b03'])) == sorted(pair)]
        return rows[0]['decision'] if len(rows) == 1 else None
    raise ValueError(f'Unknown expected-value field {field}')


class ReferenceConflicts:
    """Fail-closed: construction verifies the seal and every source pin before any effect is available."""

    def __init__(self, root, path=CONFIG):
        self.root = Path(root)
        self.config = verify(read_json(local_path(root, path)))
        if self.config.get('kind') != KIND or self.config.get('schema_version') != 2:
            raise ValueError('Unsupported reference-conflicts config')
        self.checks = source_checks(root, self.config)
        stale = [c['key'] for c in self.checks if not c['ok']]
        if stale:
            raise ValueError('Reference-conflict source pins changed; record is stale: ' + ', '.join(stale))
        self.validated = True
        self.blocked, self.photos, self.pairs_of, self.card_values = {}, {}, {}, {}
        for c in self.config['conflicts']:
            self.photos[c['photo']['sha256']] = c
            self.card_values[c['card']['product03']] = c['card']['admitted_value']
            self.card_values[c['sibling']['product03']] = c['sibling']['admitted_value']
            for a, b in c['effects']['blocked_negative_pairs']:
                self.blocked[frozenset((a, b))] = c['id']
                self.pairs_of.setdefault(a, set()).add(b)
                self.pairs_of.setdefault(b, set()).add(a)

    @property
    def checksum(self):
        return self.config['checksum']

    def negative_pair_admissible(self, product_a, product_b):
        conflict = self.blocked.get(frozenset((product_a, product_b)))
        return (conflict is None, conflict)

    def reference_status(self, image_sha256):
        c = self.photos.get(image_sha256)
        if c is None:
            return None
        return {'conflict': c['id'], 'attribute': c['attribute'], 'card': c['card']['product03'],
                'printed_value': c['photo']['printed_value'], 'card_value': c['card']['admitted_value'],
                'resolution_status': c['resolution_status'], 'use': c['effects']['reference_attribute_evidence']}

    def injection_conflict(self, slug):
        """OCR-injection hook: informative only, so independent text evidence for the disputed card is never blocked."""
        for c in self.photos.values():
            if c['card']['slug'] == slug:
                return {'conflict': c['id'], 'blocks_injection': False,
                        'visual_reference_excluded': c['photo']['sha256'], 'card_value': c['card']['admitted_value']}
        return None

    def excluded_references(self, references):
        """Gallery rows whose photo contradicts the card they are bound to; sibling rows are kept."""
        out = []
        for i, r in enumerate(references):
            c = self.photos.get(r.get('image_sha256'))
            if c and r.get('slug') == c['card']['slug']:
                out.append({'reference_index': i, 'kind': r['kind'], 'slug': r['slug'], 'image_sha256': r['image_sha256'],
                            'conflict': c['id'], 'reason': f"printed {c['attribute']} {c['photo']['printed_value']} "
                                                          f"contradicts card {c['card']['admitted_value']}"})
        return out

    def annotate(self, candidates, top_k=5, observed=()):
        """Order-preserving note for top1 when a conflict partner is also in top_k.

        observed: high-confidence query readings of the conflicted attribute. A unique match only allows the
        selector to prefer that candidate within the pair; it never confirms the SKU or sets slug/probability.
        """
        products = [c for c in candidates[:top_k] if c]
        if not products:
            return None
        partners = self.pairs_of.get(products[0], set()) & set(products[1:])
        if not partners:
            return None
        group = sorted({products[0], *partners})
        matching = [p for p in group if self.card_values.get(p) in set(observed)]
        selectable = matching[0] if len(matching) == 1 else None
        return {'status': 'conflict_resolved_by_text' if selectable else 'ambiguous_reference_conflict',
                'top1': products[0], 'group': group, 'conflict_resolved_by_text': selectable is not None,
                'allows_candidate_selection': selectable, 'observed': sorted(set(observed)),
                'card_values': {p: self.card_values.get(p) for p in group},
                'conflicts': sorted({self.blocked[frozenset((products[0], p))] for p in partners}),
                'reason': 'reference photo prints an attribute contradicting its card; visual similarity cannot '
                          'separate these products'}

    def annotate_response(self, body, top_k=5):
        out = []
        for i, target in enumerate(body.get('targets', [])):
            products = [c.get('product_id') for c in target.get('product_candidates', [])]
            note = self.annotate(products, top_k)
            if note:
                out.append(dict(note, target_index=i))
        return out

    def audit_edges(self, edges_path):
        bad = [r for r in read_jsonl(local_path(self.root, edges_path))
               if r['decision'].startswith('negative') and not self.negative_pair_admissible(r['product_a03'], r['product_b03'])[0]]
        return {'path': edges_path, 'blocked_negative_edges': len(bad),
                'pairs': [[r['product_a03'], r['product_b03'], r['decision']] for r in bad]}

    def audit_selection(self, selection_path):
        bad = [r for r in read_jsonl(local_path(self.root, selection_path))
               if r.get('selected') and not self.negative_pair_admissible(*r['pair_key03'])[0]]
        return {'path': selection_path, 'blocked_selected_pairs': len(bad), 'pairs': [r['pair_key03'] for r in bad]}

    def audit_schedule(self, steps_path, max_global_step=None):
        """Blocked-pair presentations in a stage5 steps.jsonl replayed cyclically up to max_global_step (inclusive).

        Checked loss inputs: canonical negative, real mined negatives, and active off-diagonal real in-batch
        cells (real_excluded_base false). real_local_columns only index those same rows and are not counted twice.
        """
        full = local_path(self.root, steps_path)
        check_read(full)
        with full.open() as f:
            lines = f.readlines()
        cycle = len(lines)
        last = cycle - 1 if max_global_step is None else max_global_step
        repeats = [len(range(i, last + 1, cycle)) for i in range(cycle)]
        counts = {'canonical': 0, 'real_mined': 0, 'real_inbatch_active': 0, 'real_inbatch_masked': 0}
        examples = []
        for i, line in enumerate(lines):
            if not repeats[i] or not any(p in line for pair in self.blocked for p in pair):
                continue
            row = json.loads(line)
            for c in row['canonical']:
                if not self.negative_pair_admissible(*c['negative']['pair_key03'])[0]:
                    counts['canonical'] += repeats[i]
                    examples.append({'sealed_step': i, 'presentations': repeats[i], 'anchor': c['anchor']['view_id'],
                                     'negative': c['negative']['view_id']})
            real = row.get('real', [])
            for r in real:
                for m in r.get('mined', []):
                    if any(not self.negative_pair_admissible(a, m['product03'])[0] for a in r.get('acceptable_products03', [])):
                        counts['real_mined'] += repeats[i]
            excluded = row.get('real_excluded_base', [])
            for a_i, ra in enumerate(real):
                for b_i, rb in enumerate(real):
                    if a_i == b_i:
                        continue
                    if any(not self.negative_pair_admissible(x, y)[0]
                           for x in ra.get('acceptable_products03', []) for y in rb.get('acceptable_products03', [])):
                        counts['real_inbatch_masked' if excluded[a_i][b_i] else 'real_inbatch_active'] += repeats[i]
        return {'path': steps_path, 'steps_per_cycle': cycle, 'max_global_step': last,
                'blocked_canonical_negative_presentations': counts['canonical'],
                'blocked_real_mined_presentations': counts['real_mined'],
                'blocked_real_inbatch_active_cells': counts['real_inbatch_active'],
                'blocked_real_inbatch_masked_cells': counts['real_inbatch_masked'],
                'violations': counts['canonical'] + counts['real_mined'] + counts['real_inbatch_active'],
                'examples': examples}
