"""One candidate-selection core for saved evidence and the HTTP challenger."""
from rshb_vine.competitive_product_evidence import CompetitiveProductEvidence
from rshb_vine.product_selection import ProductSelection


class CompetitiveSelection(ProductSelection):
    def __init__(self, root, bundle, selector_protocol, model_path):
        super().__init__(root, bundle, selector_protocol, model_path)
        self.competition = CompetitiveProductEvidence()

    def select(self, *, control_slug, raw_visual, observations, control_candidates):
        original = super().select(control_slug=control_slug, raw_visual=raw_visual,
                                  observations=observations, control_candidates=control_candidates)
        selected = self.competition.resolve(original['features'], original['proposal'])
        selected['product_resolution'] = self.describe(selected.get('representative_slug'), observations,
            {'kind': 'same_instance_ocr', 'instance_id': raw_visual['instance_id']})
        return {'features': original['features'], 'previous_proposal': original['proposal'],
                'proposal': selected}
