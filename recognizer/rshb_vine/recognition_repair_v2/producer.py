"""Pre-guard stage over producer_role_v2.selection.apply: producer features replaced in place, one frozen re-rank."""
from rshb_vine.io import sha256
from rshb_vine.learned_selection_features import _candidate_key
from rshb_vine.ocr_candidate_repair_v1.selection import _relabel

STAGE = 'producer-role-v2-pre-guard-stage'
PACKAGE = 'rshb_vine/producer_role_v2'
SOURCES = tuple(f'{PACKAGE}/{n}' for n in ('__init__.py', 'roles.py', 'selection.py'))
PIN = 'runs/producer-role-v2-20260926/pin.json'
PIN_CHECKSUM = '3551068124b0bfb7cacb527262720dd1847c6edf5574c1862b03c6eb471a6e35'


def check_pin(root):
    """The admitted producer sources: every package file equals the author's sealed pin 3551068c."""
    from rshb_vine.io import read_json, verify
    pin = verify(read_json(root / PIN))
    if pin['checksum'] != PIN_CHECKSUM or any(pin['sources_sha256'][p] != sha256(root / p) for p in SOURCES):
        raise ValueError('producer_role_v2 sources differ from the admitted pin 3551068c')
    return pin


class ProducerRoleStage:
    trace_key = 'producer_role'
    position = 'pre_guard'

    def __init__(self, root, inner):
        check_pin(root)
        from rshb_vine.producer_role_v2 import roles as R
        from rshb_vine.producer_role_v2 import selection as P
        self.apply_fn, self.flags = P.apply, dict(R.FULL)
        self.roles = R.ProducerRoles(inner.context, inner.stats)
        self.identity = {'stage': STAGE, 'rule': R.RULE['version'], 'flags': self.flags, 'pin_checksum': PIN_CHECKSUM,
                         'sources_sha256': {p: sha256(root / p) for p in SOURCES}}

    def apply(self, selection, out, call, request):
        control_slug = call['control_slug']
        injected = {_candidate_key(s, selection.registry.cards) for s in out['ocr_candidate_injection']['injected']}

        def rescore(rows):
            raw = selection.inner._propose(out['base'], out['v4'], rows, control_slug)
            _relabel(raw, injected)
            return raw, selection.legacy.resolver.resolve(out['base'], raw)

        out[self.trace_key] = self.apply_fn(out, self.roles, call['provenance'], call.get('other_line_keys', ()),
                                            rescore, self.flags)
        return out
