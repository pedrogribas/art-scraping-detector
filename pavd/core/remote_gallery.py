from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pavd.core.gallery import GalleryItem, gallery_id_for

URL_COLUMNS = ("URL", "url", "image_url", "IMAGE_URL")
ID_COLUMNS = ("SAMPLE_ID", "sample_id", "key", "hash", "id")
CAPTION_COLUMNS = ("TEXT", "text", "caption", "caption_en")
SCORE_COLUMNS = ("AESTHETIC_SCORE", "aesthetic_score", " aesthetic_score")


def _value(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def remote_items(rows: Iterable[dict[str, Any]], limit: int) -> list[GalleryItem]:
    """Build a URL-only catalog; it never fetches or decodes image bytes."""
    items: list[GalleryItem] = []
    seen: set[str] = set()
    for row in rows:
        source_id = _value(row, ID_COLUMNS)
        source_url = _value(row, URL_COLUMNS)
        if not source_url.startswith(("http://", "https://")):
            continue
        key = source_id or source_url
        if key in seen:
            continue
        seen.add(key)
        score_text = _value(row, SCORE_COLUMNS)
        try:
            score = float(score_text) if score_text else None
        except ValueError:
            score = None
        gallery_id = gallery_id_for(source_id, source_url, Path("remote"))
        items.append(GalleryItem(
            gallery_id=gallery_id,
            source_id=source_id,
            source_url=source_url,
            local_path="",
            download_timestamp="",
            sha256=None,
            mime_type=None,
            width=None,
            height=None,
            download_status="metadata_only",
            caption=_value(row, CAPTION_COLUMNS) or None,
            aesthetic_score=score,
            metadata_only=True,
            manual_review_status="not_applicable",
        ))
        if len(items) >= limit:
            break
    return items
