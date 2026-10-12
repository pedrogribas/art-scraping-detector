from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(slots=True)
class GalleryConfig:
    source_dataset: str = "laion/laion2B-en-aesthetic"
    image_dir: Path = Path("data/gallery/images")
    metadata_path: Path = Path("data/gallery/gallery_index.parquet")
    official_dir: Path = Path("data/raw")
    official_size: int = 500
    expanded_size: int = 100_000


@dataclass(slots=True)
class DegradationConfig:
    scale_factor: float = 0.80
    crop_fraction: float = 0.05
    gaussian_sigma: float = 8.0
    jpeg_quality: int = 35
    interpolation: str = "INTER_AREA"
    min_crop_size: int = 16


@dataclass(slots=True)
class SIFTConfig:
    max_keypoints: int = 500
    lowe_ratio: float = 0.75
    ransac_reproj_threshold: float = 5.0
    ransac_confidence: float = 0.995
    ransac_max_iters: int = 2000
    min_keypoints: int = 50
    min_matches: int = 10
    min_inliers: int = 10
    min_inlier_ratio: float = 0.30
    max_reprojection_error: float = 5.0


@dataclass(slots=True)
class OpenSetConfig:
    absolute_threshold: float = 0.82
    secondary_threshold: float = 0.75
    margin_threshold: float = 0.10


@dataclass(slots=True)
class FAISSConfig:
    backend: Literal["flat", "hnsw", "ivf", "ivfpq"] = "flat"
    k: int = 50
    hnsw_m: int = 32
    ef_construction: int = 80
    ef_search: int = 64
    nlist: int = 4096
    nprobe: int = 16
    nbits: int = 8


@dataclass(slots=True)
class PAVDConfig:
    root: Path = Path(".")
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    open_set: OpenSetConfig = field(default_factory=OpenSetConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    reports_dir: Path = Path("data/reports")
    queries_dir: Path = Path("data/queries/degraded")
    embeddings_dir: Path = Path("data/embeddings")

    def resolve_paths(self) -> "PAVDConfig":
        for name in ("root", "reports_dir", "queries_dir", "embeddings_dir"):
            value = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, self.root / value)
        for name in ("image_dir", "metadata_path", "official_dir"):
            value = getattr(self.gallery, name)
            if not value.is_absolute():
                setattr(self.gallery, name, self.root / value)
        return self

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("root", "reports_dir", "queries_dir", "embeddings_dir"):
            value[key] = str(value[key])
        for key in ("image_dir", "metadata_path", "official_dir"):
            value["gallery"][key] = str(value["gallery"][key])
        return value

    @classmethod
    def from_yaml(cls, path: Path, root: Path | None = None) -> "PAVDConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        gallery_raw = raw.get("gallery", {})
        gallery_raw = {
            **gallery_raw,
            "image_dir": Path(gallery_raw.get("image_dir", GalleryConfig.image_dir)),
            "metadata_path": Path(gallery_raw.get("metadata_path", GalleryConfig.metadata_path)),
            "official_dir": Path(gallery_raw.get("official_dir", GalleryConfig.official_dir)),
        }
        config = cls(
            root=root or Path("."),
            model_name=raw.get("model_name", cls.model_name),
            device=raw.get("device", cls.device),
            gallery=GalleryConfig(**gallery_raw),
            degradation=DegradationConfig(**raw.get("degradation", {})),
            sift=SIFTConfig(**raw.get("sift", {})),
            open_set=OpenSetConfig(**raw.get("open_set", {})),
            faiss=FAISSConfig(**raw.get("faiss", {})),
            reports_dir=Path(raw.get("reports_dir", "data/reports")),
            queries_dir=Path(raw.get("queries_dir", "data/queries/degraded")),
            embeddings_dir=Path(raw.get("embeddings_dir", "data/embeddings")),
        )
        return config.resolve_paths()
