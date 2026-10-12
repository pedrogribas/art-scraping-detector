from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pavd.config import SIFTConfig
from pavd.core.preprocessing import image_array


@dataclass(frozen=True, slots=True)
class SIFTResult:
    candidate_gallery_id: str
    query_keypoints: int
    candidate_keypoints: int
    matches: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    homography: list[list[float]] | None
    reprojection_error: float | None
    status: str
    visualization_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SIFTVerifier:
    def __init__(self, config: SIFTConfig) -> None:
        self.config = config
        self.sift = cv2.SIFT_create(nfeatures=config.max_keypoints)
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    @staticmethod
    def _gray(path: Path) -> np.ndarray:
        return cv2.cvtColor(image_array(path), cv2.COLOR_RGB2GRAY)

    def verify(self, query_path: Path, candidate_path: Path, candidate_gallery_id: str) -> SIFTResult:
        query = self._gray(query_path)
        candidate = self._gray(candidate_path)
        points_q, desc_q = self.sift.detectAndCompute(query, None)
        points_c, desc_c = self.sift.detectAndCompute(candidate, None)
        n_q = len(points_q)
        n_c = len(points_c)
        if desc_q is None or desc_c is None or n_q < self.config.min_keypoints or n_c < self.config.min_keypoints:
            return SIFTResult(candidate_gallery_id, n_q, n_c, 0, 0, 0, 0.0, None, None, "insufficient_keypoints")
        pairs = self.matcher.knnMatch(np.asarray(desc_q, dtype=np.float32), np.asarray(desc_c, dtype=np.float32), k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < self.config.lowe_ratio * pair[1].distance]
        if len(good) < self.config.min_matches:
            return SIFTResult(candidate_gallery_id, n_q, n_c, len(pairs), len(good), 0, 0.0, None, None, "insufficient_matches")
        source = np.float32([points_q[m.queryIdx].pt for m in good])
        target = np.float32([points_c[m.trainIdx].pt for m in good])
        homography, mask = cv2.findHomography(
            source,
            target,
            cv2.RANSAC,
            self.config.ransac_reproj_threshold,
            confidence=self.config.ransac_confidence,
            maxIters=self.config.ransac_max_iters,
        )
        if homography is None or mask is None:
            return SIFTResult(candidate_gallery_id, n_q, n_c, len(pairs), len(good), 0, 0.0, None, None, "no_homography")
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - target, axis=1)
        reprojection = float(errors[inlier_mask].mean()) if inliers else None
        ratio = inliers / len(good)
        status = "verified" if inliers >= self.config.min_inliers and ratio >= self.config.min_inlier_ratio and (reprojection or float("inf")) <= self.config.max_reprojection_error else "weak_geometry"
        return SIFTResult(candidate_gallery_id, n_q, n_c, len(pairs), len(good), inliers, ratio, homography.tolist(), reprojection, status)
