"""Owner export receipt in place of the private training and evaluation records the vendored guards read.

The vendored release checks, at load time, that its encoder came from an admitted training chain, that no reference
addition is a protected evaluation photo and that reference-conflict sources are unchanged. Those checks read the
training corpus, protected split memberships and evaluation ledgers, which are not shipped. The exporter ran the
original guards against the private records and recorded, per guard, the exact normalized arguments and result.
Here only these named functions are replaced, and only for the recorded arguments; any other call fails closed:

* ``calls``: guard functions whose recorded result is returned for identical arguments;
* ``omitted_sha256``: files the vendored code only hashes, answered with the recorded digest when (and only when)
  one of the recorded caller functions asks for exactly that path and the file is absent from the tree.

Every other ``sha256``/``read_json``/``verify`` keeps its normal behaviour.
"""
from copy import deepcopy
import functools
import importlib
from pathlib import Path
import sys

from wine_scanner_core.pack import canonical, sha256_file

KIND = 'wine-scanner-core-provenance-receipt-v1'


def normalize(value, root):
    if isinstance(value, Path) or (isinstance(value, str) and value.startswith(root + '/')):
        text = str(Path(value).resolve()) if isinstance(value, Path) else value
        if text == root:
            return '<root>'
        if text.startswith(root + '/'):
            return '<root>/' + text[len(root) + 1:]
        return text
    if isinstance(value, dict):
        return {k: normalize(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v, root) for v in value]
    return value


def args_digest(args, kwargs, root):
    return canonical({'args': normalize(list(args), root), 'kwargs': normalize(kwargs, root)})


class Provenance:
    def __init__(self, root, receipt, ledger):
        self.root = Path(root).resolve()
        if receipt.get('kind') != KIND or receipt.get('checksum') != canonical(
                {k: v for k, v in receipt.items() if k != 'checksum'}):
            raise ValueError('Unsupported or unsealed provenance receipt')
        self.receipt, self.ledger = receipt, ledger
        self.expected = receipt['expected']
        self.used = {c['module'] + '.' + c['function']: 0 for c in receipt['calls']}
        self.answered = {}
        self.installed = False

    def verify(self):
        """Every receipt guard binds the vendored module it replaces; no omitted record is present in the tree."""
        for call in self.receipt['calls']:
            entry = self.ledger.files.get(call['source'])
            if entry is None or entry['original_sha256'] != call['source_sha256']:
                raise ValueError('Receipt guard source is not the vendored one: ' + call['source'])
        present = [p for p in self.receipt['omitted_sha256'] if (self.root / p).exists()]
        if present:
            raise ValueError('Omitted private records are present in the tree: %s' % present[:3])
        return {'calls': len(self.receipt['calls']), 'omitted_files': len(self.receipt['omitted_sha256'])}

    def install(self):
        """Before any vendored module other than rshb_vine.io is imported."""
        if self.installed:
            raise RuntimeError('Provenance receipt installed twice')
        for name in list(sys.modules):
            if name.startswith('rshb_vine.') and name != 'rshb_vine.io':
                raise RuntimeError('Vendored module imported before the provenance receipt: ' + name)
        self.verify()
        import rshb_vine.io as io
        io.sha256 = self._sha256(io.sha256)
        for call in self.receipt['calls']:
            module = importlib.import_module(call['module'])
            setattr(module, call['function'], self._call(call, getattr(module, call['function'])))
        self.installed = True

    def _sha256(self, original):
        omitted, root = self.receipt['omitted_sha256'], str(self.root)

        @functools.wraps(original)
        def sha256(path):
            resolved = Path(path).resolve()
            try:
                rel = resolved.relative_to(root).as_posix()
            except ValueError:
                return original(path)
            entry = omitted.get(rel)
            if entry is None:
                return original(path)
            frame = sys._getframe(1)
            caller = Path(frame.f_code.co_filename).resolve()
            site = '%s:%s' % (caller.relative_to(root).as_posix() if caller.is_relative_to(root) else caller,
                              frame.f_code.co_name)
            if site not in entry['callers'] or resolved.exists():
                raise ValueError('Omitted private record %s requested by an unrecorded caller %s' % (rel, site))
            self.answered[rel] = self.answered.get(rel, 0) + 1
            return entry['sha256']
        return sha256

    def _call(self, call, original):
        results, root, key = {c['args_digest']: c for c in call['results']}, str(self.root), \
            call['module'] + '.' + call['function']
        restore, form = RESTORE.get(call['function'], (None, None))
        if restore is None or call['restore'] != form:
            raise ValueError('Receipt names a guard this core does not restore: ' + key)

        @functools.wraps(original)
        def guard(*args, **kwargs):
            digest = args_digest(args, kwargs, root)
            recorded = results.get(digest)
            if recorded is None:
                raise ValueError('%s called with arguments the export receipt did not record' % key)
            self.used[key] += 1
            return restore(self, deepcopy(recorded['result']), args, kwargs)
        return guard

    def check_used(self):
        unused = [k for k, n in self.used.items() if not n]
        if unused:
            raise RuntimeError('Receipt guards were never reached; the vendored graph differs: %s' % unused)
        return {'calls': dict(self.used), 'omitted_answers': len(self.answered)}


def _encoder_arm(provenance, result, args, kwargs):
    """(arm, model path): the recorded B4343 arm, re-bound to the exported encoder files and the vendored models.py."""
    from rshb_vine.catalog_training_evaluation_v2 import pins
    from rshb_vine.io import digest
    arm, folder = result['arm'], provenance.root / result['model_path']
    if arm['arm'] != 'B4343' or folder.name != 'encoder' or arm['encoder_id'] == pins.PARENT_EID:
        raise ValueError('Recorded arm is not the exported stage5 B4343 encoder')
    names = sorted(p.name for p in folder.iterdir() if p.is_file())
    if names != sorted(k.split('/', 1)[1] for k in arm['files']):
        raise ValueError('Exported encoder directory differs from the recorded arm')
    for name, expected in arm['files'].items():
        if sha256_file(folder / name.split('/', 1)[1]) != expected:
            raise ValueError('Exported encoder file differs from the recorded arm: ' + name)
    identity = digest({'files': arm['files'], 'source': sha256_file(provenance.root / 'rshb_vine/models.py'),
                       'preprocessing': pins.PREPROCESSING})
    if identity != arm['encoder_id']:
        raise ValueError('Encoder identity of the exported files and vendored models.py differs from the recorded arm')
    return arm, folder


def _protected(provenance, result, args, kwargs):
    if result != {}:
        raise ValueError('Receipt records protected evaluation photos among reference additions')
    return result


def _source_checks(provenance, result, args, kwargs):
    """Every pinned conflict source matched at export; the checks cover exactly the config's expected values."""
    config = args[1] if len(args) > 1 else kwargs['config']
    expected = sorted((c['id'], key, value) for c in config['conflicts'] for key, value in c['expected_old_values'].items())
    if (not all(check['ok'] for check in result)
            or sorted((c['conflict'], c['key'], c['expected']) for c in result) != expected):
        raise ValueError('Recorded reference-conflict source checks do not cover this config or did not pass')
    return result


RESTORE = {'encoder_arm': (_encoder_arm, 'encoder_arm'), 'protected': (_protected, 'json'),
           'source_checks': (_source_checks, 'json')}
