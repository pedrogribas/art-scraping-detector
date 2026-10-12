from __future__ import annotations

from pathlib import Path
from typing import Any

from pavd.config import PAVDConfig
from pavd.core.clip_index import CLIPIndex
from pavd.core.open_set import OpenSetEvaluator
from pavd.core.reporting import write_query_report
from pavd.core.sift_verifier import SIFTVerifier
from pavd.utils.hashing import sha256_file


class PAVDAuditor:
    def __init__(self, clip_index: CLIPIndex, sift_verifier: SIFTVerifier, config: PAVDConfig) -> None:
        self.clip_index = clip_index
        self.sift_verifier = sift_verifier
        self.config = config
        self.open_set = OpenSetEvaluator(config.open_set)

    def audit_query(self, query_image_path: Path, true_gallery_id: str | None = None, k: int = 50) -> dict[str, Any]:
        clip_results = self.clip_index.search(query_image_path, k=k)
        scores = [item.clip_score for item in clip_results]
        clip_top1 = clip_results[0] if clip_results else None
        local_results = []
        for candidate in clip_results:
            local_results.append(
                self.sift_verifier.verify(
                    query_image_path,
                    Path(candidate.image_path),
                    candidate.gallery_id,
                ).as_dict()
            )
        verified = [item for item in local_results if item["status"] == "verified"]
        best_local = max(verified, key=lambda item: (item["inliers"], item["inlier_ratio"]), default=None)
        present = self.open_set.predict(scores)
        if not present:
            decision, reason = "rejected", "scores CLIP abaixo da regra de conjunto aberto"
        elif best_local is not None:
            decision, reason = "accepted", "geometria local compatível no candidato"
        else:
            decision, reason = "review", "CLIP alto sem verificação geométrica suficiente"
        report = {
            "query_id": query_image_path.stem,
            "query_sha256": sha256_file(query_image_path),
            "clip_top1": clip_top1.as_dict() if clip_top1 else None,
            "clip_top50": [item.as_dict() for item in clip_results],
            "sift_results": local_results,
            "sift_reranked_top1": best_local,
            "final_decision": decision,
            "decision_reason": reason,
            "true_gallery_id": true_gallery_id,
            "policy": {
                "open_set": self.config.open_set.__dict__ if hasattr(self.config.open_set, "__dict__") else {
                    "absolute_threshold": self.config.open_set.absolute_threshold,
                    "secondary_threshold": self.config.open_set.secondary_threshold,
                    "margin_threshold": self.config.open_set.margin_threshold,
                },
                "k": k,
            },
        }
        report_path = self.config.reports_dir / "per_query" / f"{query_image_path.stem}.json"
        write_query_report(report_path, report)
        report["report_path"] = str(report_path)
        return report
