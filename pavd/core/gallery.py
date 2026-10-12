from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from pavd.utils.hashing import sha256_file


@dataclass(slots=True)
class GalleryItem:
    gallery_id: str
    source_id: str
    source_url: str
    local_path: str
    download_timestamp: str
    sha256: str | None
    mime_type: str | None
    width: int | None
    height: int | None
    download_status: str
    visual_category: str | None = None
    visual_category_confidence: float | None = None
    excluded_reason: str | None = None
    redirect_history: list[str] | None = None
    duplicate_of: str | None = None
    caption: str | None = None
    aesthetic_score: float | None = None
    metadata_only: bool = False
    manual_review_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gallery_id_for(source_id: str, source_url: str, path: Path) -> str:
    value = source_id or source_url or path.name
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def scan_directory(directory: Path, source_prefix: str = "local") -> list[GalleryItem]:
    items: list[GalleryItem] = []
    seen_hashes: dict[str, str] = {}
    for path in sorted(directory.iterdir() if directory.exists() else []):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            digest = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mime = Image.MIME.get(image.format) or mimetypes.guess_type(path.name)[0]
            duplicate_of = seen_hashes.get(digest)
            seen_hashes[digest] = path.name
            items.append(GalleryItem(
                gallery_id=gallery_id_for(f"{source_prefix}:{path.name}", "", path),
                source_id=f"{source_prefix}:{path.name}",
                source_url="",
                local_path=str(path),
                download_timestamp=timestamp,
                sha256=digest,
                mime_type=mime,
                width=width,
                height=height,
                download_status="duplicate" if duplicate_of else "ok",
                duplicate_of=duplicate_of,
            ))
        except (OSError, UnidentifiedImageError) as exc:
            items.append(GalleryItem(
                gallery_id=gallery_id_for(f"{source_prefix}:{path.name}", "", path),
                source_id=f"{source_prefix}:{path.name}",
                source_url="",
                local_path=str(path),
                download_timestamp=timestamp,
                sha256=None,
                mime_type=None,
                width=None,
                height=None,
                download_status="invalid",
                excluded_reason=str(exc),
            ))
    return items


def write_manifest(items: list[GalleryItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.as_dict() for item in items]).to_parquet(path, index=False)


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
