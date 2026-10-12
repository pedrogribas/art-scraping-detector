from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from pavd.config import DegradationConfig
from pavd.utils.hashing import sha256_file
from pavd.utils.seeds import seed_from_gallery_id


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: str
    gallery_id: str
    source_id: str
    source_url: str
    sha256_original: str
    sha256_query: str
    seed: int
    scale_factor: float
    crop_fraction: float
    gaussian_sigma: float
    jpeg_quality: int
    width_original: int
    height_original: int
    width_query: int
    height_query: int
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DegradationPipeline:
    """Gera consultas determinísticas a partir de imagens locais."""

    def __init__(self, config: DegradationConfig) -> None:
        self.config = config

    def apply(self, source: Path, gallery_id: str) -> tuple[np.ndarray, int, tuple[int, int]]:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        scaled_width = max(int(width * self.config.scale_factor), 1)
        scaled_height = max(int(height * self.config.scale_factor), 1)
        scaled = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        margin_y = int(scaled_height * self.config.crop_fraction)
        margin_x = int(scaled_width * self.config.crop_fraction)
        if (
            scaled_height - 2 * margin_y >= self.config.min_crop_size
            and scaled_width - 2 * margin_x >= self.config.min_crop_size
        ):
            scaled = scaled[margin_y : scaled_height - margin_y, margin_x : scaled_width - margin_x]
        seed = seed_from_gallery_id(gallery_id)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.config.gaussian_sigma, scaled.shape)
        noisy = np.clip(scaled.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )[1]
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"falha ao decodificar consulta gerada para {gallery_id}")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), seed, (width, height)

    def generate(
        self,
        source: Path,
        destination: Path,
        *,
        gallery_id: str,
        source_id: str = "",
        source_url: str = "",
    ) -> QueryRecord:
        result, seed, original_size = self.apply(source, gallery_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result, mode="RGB").save(
            destination, format="JPEG", quality=self.config.jpeg_quality
        )
        return QueryRecord(
            query_id=f"{gallery_id}_degraded",
            gallery_id=gallery_id,
            source_id=source_id,
            source_url=source_url,
            sha256_original=sha256_file(source),
            sha256_query=sha256_file(destination),
            seed=seed,
            scale_factor=self.config.scale_factor,
            crop_fraction=self.config.crop_fraction,
            gaussian_sigma=self.config.gaussian_sigma,
            jpeg_quality=self.config.jpeg_quality,
            width_original=original_size[0],
            height_original=original_size[1],
            width_query=result.shape[1],
            height_query=result.shape[0],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
