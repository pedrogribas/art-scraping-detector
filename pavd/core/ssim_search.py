from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from pavd.core.preprocessing import image_array


@dataclass(frozen=True, slots=True)
class SSIMResult:
    gallery_id: str
    score: float
    path: Path


class SSIMSearch:
    def __init__(self, resize: tuple[int, int] = (128, 128), color_space: str = "L", window_size: int = 7) -> None:
        self.resize = resize
        self.color_space = color_space
        self.window_size = window_size
        self._items: list[tuple[str, Path, np.ndarray]] = []

    def _prepare(self, path: Path) -> np.ndarray:
        array = image_array(path)
        if self.color_space.upper() == "L":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        array = cv2.resize(array, self.resize, interpolation=cv2.INTER_AREA)
        return array.astype(np.float32) / 255.0

    def index(self, items: Sequence[tuple[str, Path]]) -> None:
        self._items = [(gallery_id, path, self._prepare(path)) for gallery_id, path in items]

    def search(self, query_path: Path, top_k: int = 1) -> list[SSIMResult]:
        if not self._items:
            raise RuntimeError("índice SSIM vazio")
        query = self._prepare(query_path)
        results = [
            SSIMResult(gallery_id, float(structural_similarity(query, image, data_range=1.0)), path)
            for gallery_id, path, image in self._items
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def benchmark_cost(self, query_paths: Sequence[Path], comparisons: int | None = None) -> dict[str, float | int]:
        if not self._items:
            raise RuntimeError("índice SSIM vazio")
        selected = list(query_paths)[:comparisons] if comparisons else list(query_paths)
        start = time.perf_counter()
        for path in selected:
            self.search(path, top_k=1)
        elapsed = time.perf_counter() - start
        total = max(len(selected) * len(self._items), 1)
        return {"queries": len(selected), "gallery_size": len(self._items), "seconds": elapsed, "seconds_per_comparison": elapsed / total}
