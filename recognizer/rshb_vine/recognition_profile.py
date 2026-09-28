"""Compose the verified ROI selection and text stages without duplicating either."""
from rshb_vine.variant_text_pipeline import VariantTextPipeline
from rshb_vine.roi_selection import RoiSelectionPipeline


class RecognitionProfile(VariantTextPipeline, RoiSelectionPipeline):
    """Cooperative order: ROI parent selection -> visual retrieval -> text rerank.

    VariantTextPipeline calls super() for visual recognition; the ROI stage
    selects a parent before reaching OrientedGeometryPipeline. With no ROI,
    SelectingDetector passes through all detections unchanged.
    """
