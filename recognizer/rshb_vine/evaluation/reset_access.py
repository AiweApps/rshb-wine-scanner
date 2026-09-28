"""Prevent accidental reuse of historically mixed evaluation after RESET activation.

This is a workflow guard, not an OS security boundary against deliberate bypass.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_FINAL = ContextVar('reset_final_read', default=False)
ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ('runs/kultovo-holdout-v1/', 'runs/text-disambiguation-v1/',
              'runs/text-disambiguation-v2/', 'runs/targeted-ocr-v1/', 'runs/verifier-v1/',
              'runs/ranking-error-audit-v1.json', 'runs/kultovo-current-errors.json',
              'runs/runtime-ranking-cache-parity.json')


def check_read(path):
    path = Path(path).resolve()
    root = ROOT.resolve()
    if not path.is_relative_to(root) or not (root/'data/evaluation/reset-active.json').exists():
        return
    rel = path.relative_to(root).as_posix()
    if any(rel.startswith(prefix) for prefix in HISTORICAL):
        raise PermissionError('RESET active: mixed489 history is archived; use reset split-scoped inputs')
    if rel.startswith('data/evaluation/reset-v1/locked_test/records/') and not _FINAL.get():
        raise PermissionError('Locked test traces can be read only by the final-once comparison')
    if rel in ('runs/ranking-linear-v1/report.json','runs/ranking-linear-v1/release.json'):
        raise PermissionError('RESET active: old evaluation-selected report is not a selection input')


@contextmanager
def final_access():
    token = _FINAL.set(True)
    try:
        yield
    finally:
        _FINAL.reset(token)
