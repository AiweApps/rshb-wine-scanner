"""Layout-consistency evidence v2: deduplicated SIFT/RANSAC support between a query label and gallery labels."""
import cv2
import numpy as np
from rshb_vine.geometric_reference_probe_v1 import POLICY

CRITERIA = {'id': 'layout-consistency-v2', 'dedup': 'ratio matches unique by query keypoint coordinate, then by reference coordinate (lowest distance kept), before RANSAC',
            'min_unique_inliers': 8, 'min_hull_coverage_query': .2, 'min_hull_coverage_reference': .2,
            'min_span': 1 / 3, 'span_images': 'horizontal and vertical span in both query and reference',
            'homography': 'det(H[:2,:2]/H[2,2]) > 0, projected query corners convex, projected area / reference area in [0.05, 20]',
            'min_winner_to_next': 1.5,
            'assessable_reference': 'reference short side >= .5 x query short side (after 1000px thumbnail) and reference SIFT features >= .5 x query features'}


def _hull(points, size):
    return float(cv2.contourArea(cv2.convexHull(points))) / (size[0] * size[1]) if len(points) >= 3 else 0.0


def _span(points, size):
    if not len(points):
        return 0.0, 0.0
    return float(np.ptp(points[:, 0])) / size[0], float(np.ptp(points[:, 1])) / size[1]


def _plausible(h, qsize, rsize):
    h = np.asarray(h, float) / h[2, 2]
    if np.linalg.det(h[:2, :2]) <= 0:
        return False
    w, t = qsize
    corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, t]], [[0, t]]]), h).reshape(-1, 2)
    if not cv2.isContourConvex(corners.astype(np.float32)):
        return False
    ratio = abs(cv2.contourArea(corners)) / (rsize[0] * rsize[1])
    return .05 <= ratio <= 20


def correspond(query, reference):
    out = {'query_features': len(query['keypoints']), 'reference_features': len(reference['keypoints']),
           'reference_size': list(reference['size']), 'ratio_matches': 0, 'unique_matches': 0, 'unique_inliers': 0,
           'hull_query': 0.0, 'hull_reference': 0.0, 'span_query': [0.0, 0.0], 'span_reference': [0.0, 0.0],
           'homography_plausible': False}
    if query['descriptors'] is None or reference['descriptors'] is None or len(reference['descriptors']) < 2:
        return out
    pairs = cv2.BFMatcher().knnMatch(query['descriptors'], reference['descriptors'], k=2)
    matches = sorted((p[0] for p in pairs if len(p) == 2 and p[0].distance < POLICY['ratio_test'] * p[1].distance), key=lambda m: m.distance)
    out['ratio_matches'] = len(matches)
    seen_q, seen_r, unique = set(), set(), []
    for m in matches:
        q = tuple(np.round(query['keypoints'][m.queryIdx].pt, 1)); r = tuple(np.round(reference['keypoints'][m.trainIdx].pt, 1))
        if q in seen_q or r in seen_r:
            continue
        seen_q.add(q); seen_r.add(r); unique.append((q, r))
    out['unique_matches'] = len(unique)
    if len(unique) < POLICY['min_ratio_matches']:
        return out
    src = np.float32([u[0] for u in unique]); dst = np.float32([u[1] for u in unique])
    cv2.setRNGSeed(POLICY['rng_seed'])
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, POLICY['ransac_reprojection_px'])
    if h is None or mask is None:
        return out
    mask = mask.ravel().astype(bool)
    qp, rp = src[mask], dst[mask]
    out.update(unique_inliers=int(mask.sum()), hull_query=_hull(qp, query['size']), hull_reference=_hull(rp, reference['size']),
               span_query=list(_span(qp, query['size'])), span_reference=list(_span(rp, reference['size'])),
               homography_plausible=bool(_plausible(h, query['size'], reference['size'])), homography=h.tolist(),
               inlier_pairs=np.hstack([qp, rp]).round(1).tolist())
    return out


def layout_pass(row):
    c = CRITERIA
    return (row['unique_inliers'] >= c['min_unique_inliers'] and row['hull_query'] >= c['min_hull_coverage_query']
            and row['hull_reference'] >= c['min_hull_coverage_reference']
            and min(row['span_query'] + row['span_reference']) >= c['min_span'] and row['homography_plausible'])


def assessable(row, query):
    qs = min(query['size']); rs = min(row['reference_size'])
    return rs >= .5 * qs and row['reference_features'] >= .5 * len(query['keypoints'])


def rank_products(rows):
    best = {}
    for r in rows:
        key = (r['unique_inliers'], r['hull_query'])
        if r['slug'] not in best or key > (best[r['slug']]['unique_inliers'], best[r['slug']]['hull_query']):
            best[r['slug']] = r
    return sorted(best.values(), key=lambda r: (-r['unique_inliers'], -r['hull_query'], r['slug']))


def decide(ranking, incumbent, query, contradiction):
    """Return the layout decision for one target; contradiction(slug) -> reason or None."""
    winner = ranking[0]; runner = ranking[1] if len(ranking) > 1 else None
    ratio = (winner['unique_inliers'] / runner['unique_inliers']) if runner and runner['unique_inliers'] else None
    supported = layout_pass(winner) and (ratio is None or ratio >= CRITERIA['min_winner_to_next'])
    trace = {'winner': winner['slug'], 'next': runner and runner['slug'], 'winner_to_next': ratio,
             'winner_layout_pass': layout_pass(winner), 'supported': supported, 'incumbent': incumbent}
    if not supported:
        return dict(trace, action='no_change', reason='no layout support')
    if winner['slug'] == incumbent:
        return dict(trace, action='confirm', reason='layout agrees with incumbent')
    reason = contradiction(winner['slug'])
    if reason:
        return dict(trace, action='no_change', reason=f'reliable text contradiction: {reason}')
    inc = next((r for r in ranking if r['slug'] == incumbent), None)
    if inc is None:
        return dict(trace, action='ambiguous', reason='incumbent has no eligible front_label reference')
    if not assessable(inc, query):
        return dict(trace, action='ambiguous', reason='incumbent reference not assessable (resolution/features)')
    return dict(trace, action='propose_override', proposed=winner['slug'], reason='layout support for winner; incumbent assessable and unsupported')
