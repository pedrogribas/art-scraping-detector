from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pavd.config import FAISSConfig
from pavd.core.preprocessing import load_rgb


@dataclass(frozen=True, slots=True)
class CLIPResult:
    rank: int
    gallery_id: str
    clip_score: float
    distance: float
    image_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "gallery_id": self.gallery_id,
            "clip_score": self.clip_score,
            "distance": self.distance,
            "image_path": self.image_path,
        }


class CLIPIndex:
    """Índice CLIP com backend Transformers e índices FAISS configuráveis."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto", config: FAISSConfig | None = None) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Instale faiss-cpu para usar CLIPIndex") from exc
        from transformers import CLIPModel, CLIPProcessor

        self._faiss = faiss
        self.model_name = model_name
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.config = config or FAISSConfig()
        self.index: Any | None = None
        self.embeddings: np.ndarray | None = None
        self.gallery_ids: list[str] = []
        self.image_paths: list[str] = []
        self.build_seconds = 0.0

    def embed_paths(self, paths: Sequence[Path], batch_size: int = 16) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            images = [load_rgb(path) for path in paths[start : start + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.get_image_features(**inputs)
                if hasattr(output, "image_embeds") and output.image_embeds is not None:
                    output = output.image_embeds
                elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                    output = output.pooler_output
                output = torch.nn.functional.normalize(output, dim=-1)
            vectors.append(output.cpu().numpy().astype(np.float32))
        if not vectors:
            return np.empty((0, 512), dtype=np.float32)
        return np.concatenate(vectors, axis=0)

    def _make_index(self, embeddings: np.ndarray) -> Any:
        faiss = self._faiss
        dim = embeddings.shape[1]
        if self.config.backend == "flat":
            return faiss.IndexFlatIP(dim)
        if self.config.backend == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
            return index
        nlist = min(self.config.nlist, max(1, len(embeddings) // 39))
        quantizer = faiss.IndexFlatIP(dim)
        if self.config.backend == "ivfpq":
            return faiss.IndexIVFPQ(quantizer, dim, nlist, 8, self.config.nbits, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    def build(self, items: Sequence[tuple[str, Path]], batch_size: int = 16) -> None:
        paths = [path for _, path in items]
        embeddings = self.embed_paths(paths, batch_size=batch_size)
        start = time.perf_counter()
        self.index = self._make_index(embeddings)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(self.config.nprobe, self.index.nlist)
        self.embeddings = embeddings
        self.gallery_ids = [gallery_id for gallery_id, _ in items]
        self.image_paths = [str(path) for _, path in items]
        self.build_seconds = time.perf_counter() - start

    def _search_index(self, query_embedding: np.ndarray, k: int) -> list[CLIPResult]:
        faiss = self._faiss
        if self.index is None:
            raise RuntimeError("índice CLIP não construído")
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, min(k, self.index.ntotal))
        return [
            CLIPResult(rank, self.gallery_ids[pos], float(score), float(1.0 - score), self.image_paths[pos])
            for rank, (score, pos) in enumerate(zip(scores[0], positions[0], strict=True), start=1)
            if pos >= 0
        ]

    def search(self, query_path: Path, k: int = 50) -> list[CLIPResult]:
        return self._search_index(self.embed_paths([query_path])[0], k)

    def search_exact(self, query_path: Path, k: int = 50) -> list[CLIPResult]:
        if self.embeddings is None:
            raise RuntimeError("embeddings não disponíveis")
        query = self.embed_paths([query_path])[0]
        scores = self.embeddings @ query
        positions = np.argsort(-scores)[: min(k, len(scores))]
        return [CLIPResult(rank, self.gallery_ids[pos], float(scores[pos]), float(1.0 - scores[pos]), self.image_paths[pos]) for rank, pos in enumerate(positions, start=1)]

    def compare_exact_vs_ann(self, query_path: Path, k: int = 50) -> dict[str, Any]:
        exact = self.search_exact(query_path, k)
        ann = self.search(query_path, k)
        exact_ids = [item.gallery_id for item in exact]
        ann_ids = [item.gallery_id for item in ann]
        return {"exact": [item.as_dict() for item in exact], "ann": [item.as_dict() for item in ann], "overlap_at_k": len(set(exact_ids) & set(ann_ids)) / max(k, 1)}

    def save(self, prefix: Path) -> None:
        faiss = self._faiss
        if self.index is None or self.embeddings is None:
            raise RuntimeError("índice vazio")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(prefix.with_suffix(".faiss")))
        np.save(prefix.with_name(prefix.stem + "_embeddings.npy"), self.embeddings)
        prefix.with_name(prefix.stem + "_metadata.json").write_text(
            json.dumps({"model_name": self.model_name, "gallery_ids": self.gallery_ids, "image_paths": self.image_paths, "config": self.config.__dict__ if hasattr(self.config, "__dict__") else {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, prefix: Path) -> None:
        faiss = self._faiss
        self.index = faiss.read_index(str(prefix.with_suffix(".faiss")))
        self.embeddings = np.load(prefix.with_name(prefix.stem + "_embeddings.npy"))
        metadata = json.loads(prefix.with_name(prefix.stem + "_metadata.json").read_text(encoding="utf-8"))
        self.gallery_ids = metadata["gallery_ids"]
        self.image_paths = metadata["image_paths"]
        if len(self.gallery_ids) != self.index.ntotal:
            raise RuntimeError("metadados e índice FAISS estão desalinhados")
