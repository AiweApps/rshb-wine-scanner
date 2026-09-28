"""Declared SIFT/RANSAC correspondence between one query label and existing gallery label references."""
import cv2
import numpy as np
from PIL import Image

POLICY = {'id': 'sift-ransac-label-v1', 'resize': 'thumbnail long side 1000 LANCZOS, both crops', 'gray': 'cv2 RGB2GRAY',
          'sift_nfeatures': 2500, 'ratio_test': .7, 'min_ratio_matches': 8, 'ransac_reprojection_px': 3.0,
          'rng_seed': 20260926, 'grid': [3, 3], 'source': 'parameters of scripts/audit_label_pair_provenance.py compare()'}


def prepare(crop):
    crop = crop.convert('RGB').copy()
    crop.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=POLICY['sift_nfeatures']).detectAndCompute(gray, None)
    return {'size': gray.shape[::-1], 'keypoints': keypoints, 'descriptors': descriptors}


def correspond(query, reference):
    out = {'query_features': len(query['keypoints']), 'reference_features': len(reference['keypoints']),
           'ratio_matches': 0, 'inliers': 0, 'inlier_ratio': None, 'hull_coverage': 0.0,
           'vertical_span': 0.0, 'grid_cells': 0, 'grid_rows': 0}
    if query['descriptors'] is None or reference['descriptors'] is None or len(reference['descriptors']) < 2:
        return out
    pairs = cv2.BFMatcher().knnMatch(query['descriptors'], reference['descriptors'], k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < POLICY['ratio_test'] * p[1].distance]
    out['ratio_matches'] = len(matches)
    if len(matches) < POLICY['min_ratio_matches']:
        return out
    src = np.float32([query['keypoints'][m.queryIdx].pt for m in matches])
    dst = np.float32([reference['keypoints'][m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(POLICY['rng_seed'])
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, POLICY['ransac_reprojection_px'])
    if h is None or mask is None:
        return out
    mask = mask.ravel().astype(bool)
    points = src[mask]
    width, height = query['size']
    out.update(inliers=int(mask.sum()), inlier_ratio=float(mask.mean()), homography=h.tolist(),
               inlier_query_points=points.round(1).tolist())
    if len(points) >= 3:
        out['hull_coverage'] = float(cv2.contourArea(cv2.convexHull(points))) / (width * height)
    if len(points):
        out['vertical_span'] = float(points[:, 1].max() - points[:, 1].min()) / height
        cols, rows = POLICY['grid']
        cells = {(min(cols - 1, int(x / width * cols)), min(rows - 1, int(y / height * rows))) for x, y in points}
        out['grid_cells'] = len(cells)
        out['grid_rows'] = len({r for _, r in cells})
    return out
