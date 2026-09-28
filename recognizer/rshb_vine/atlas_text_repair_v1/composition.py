"""Component T at the actual consumer: the composed selector's injection index gets the composite-identity proposer.

``attach(selection, root)`` works on the single composed selector object (A+B+D+M, optionally I/G from
decision_consistency_v1): its B index ``ProducerNameIndex(frozen, ShortProducerIndex)`` becomes
``ProducerNameIndex(frozen, CompositeIdentityIndex(ShortProducerIndex))``, so the guard still reads the frozen
profiles, the text-evidence trace keeps timing, the served ``index_checksum`` names T and ``frozen_index_checksum``
stays the frozen one. The object is replaced in place, not copied: component I binds ``evaluate`` to this very object.
Live ``select`` adds the T trace (original/proposed/admitted) to the published record. Stages, components, ranker,
resolver, pool rules and sources stay unchanged. ``install(release, root)`` checks the five owners first.
"""
from copy import deepcopy

from rshb_vine.atlas_text_repair_v1 import index as T
from rshb_vine.text_evidence_repair_v1.composition import ProducerNameIndex
from rshb_vine.text_evidence_repair_v2.composition import TextEvidenceSelectionV2

ADAPTER_VERSION = 'atlas-text-repair-v1-selection'
PACKAGE = 'rshb_vine/atlas_text_repair_v1'
SOURCES = tuple(f'{PACKAGE}/{n}.py' for n in ('__init__', 'index', 'composition'))
TRACE_KEY = 'atlas_text_repair_v1'


def attach(selection, root):
    if not isinstance(selection, TextEvidenceSelectionV2) or 'B' not in selection.components:
        raise ValueError('T needs the composed text-evidence selector with component B')
    if getattr(selection, 'composite_identity', None) is not None:
        raise ValueError('T is already attached to this selector')
    current = selection.index
    if type(current) is not ProducerNameIndex:
        raise ValueError('Selector index is not the component-B ProducerNameIndex')
    proposer = T.CompositeIdentityIndex(current.proposer, root)
    selection.index = ProducerNameIndex(current.frozen, proposer, current.rule)
    selection.composite_identity = proposer
    identity = {'component': 'T', 'module': T.__name__, 'rule': T.RULE['version'], 'adapter': ADAPTER_VERSION,
                'composite_checksum': proposer.checksum, 'parent_index_checksum': current.checksum,
                'index_checksum': selection.index.checksum, 'frozen_index_checksum': current.frozen.checksum,
                'composite_entries': len(proposer.entries)}
    selection.identity = dict(selection.identity, identities=[*selection.identity.get('identities', []), identity],
                              atlas_text_repair=identity)
    original = selection.select

    def select(**kwargs):
        proposer.last_trace = None
        result = original(**kwargs)
        if proposer.last_trace is None:
            raise RuntimeError('Live select did not run the T proposer')
        result['systemic_ranking_v2'][TRACE_KEY] = dict(deepcopy(proposer.last_trace), identity=identity)
        return result

    selection.select = select
    return identity


def install(release, root):
    """Attach T to the single composed selector of a loaded text-evidence release graph (F10 after I/G)."""
    inner, base = release.inner, release.base
    owners = (base.runtime.selection, base.target.inner.selection, inner.repair.selection, inner.selection,
              release.selection)
    selection = release.selection
    if any(o is not selection for o in owners):
        raise ValueError('Selector ownership differs from one composed selector')
    identity = attach(selection, root)
    if any(o is not selection or o.index is not selection.index for o in owners):
        raise ValueError('T index is not the single live consumer index')
    return {'atlas_text_repair': deepcopy(identity),
            'owners': ['base.runtime.selection', 'base.target.inner.selection', 'inner.repair.selection',
                       'inner.selection', 'release.selection']}
