"""Reading recovery v1: two independent bounded route/OCR components over the current fa317ef2 graph.

R (``route``): one bottle-context label view for exactly one surviving label-less wine bottle of a zero-target
no-ROI request. O (``rotated_ocr``): one fixed 90/270 rotated Vision reread of the single target's own label
view when its Vision text is sparse and carries no producer form. Neither changes weights, thresholds or gates.
"""
