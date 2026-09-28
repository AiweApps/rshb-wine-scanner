"""OCR candidate injection plus the shared-name visual-agreement guard, on the same replay/live path."""
from rshb_vine.ocr_candidate_repair_v1 import discriminator as D
from rshb_vine.ocr_candidate_repair_v1.selection import OcrCandidateSelection

ADAPTER_VERSION = 'ocr-candidate-repair-v1-guarded-selection'


class GuardedOcrCandidateSelection(OcrCandidateSelection):
    def evaluate(self, **kwargs):
        return D.apply(super().evaluate(**kwargs), self.legacy.resolver, self.index.profile)


def install(recognition, conflicts=None):
    from rshb_vine.systemic_ranking_v2.selection import attach
    runtime, inner = recognition.runtime, recognition.target.inner.selection
    if runtime.selection is not inner or type(inner).__name__ != 'SystemicSelection':
        raise ValueError('Systemic selection ownership changed')
    wrapper = GuardedOcrCandidateSelection(inner.root, inner, conflicts)
    attach(runtime, wrapper)
    recognition.target.inner.selection = wrapper
    return wrapper
