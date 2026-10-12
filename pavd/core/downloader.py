from __future__ import annotations

import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DownloadResult:
    gallery_id: str
    source_id: str
    source_url: str
    local_path: str
    download_status: str
    sha256: str | None
    mime_type: str | None
    redirect_history: list[str]
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class GalleryDownloader:
    """Downloader concorrente, limitado e retomável para URLs de metadados."""

    def __init__(self, timeout: float = 3.0, max_bytes: int = 20 * 1024 * 1024, retries: int = 2, workers: int = 8) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.retries = retries
        self.workers = workers

    def _one(self, gallery_id: str, source_id: str, url: str, destination: Path) -> DownloadResult:
        if destination.is_file() and destination.stat().st_size:
            return DownloadResult(gallery_id, source_id, url, str(destination), "existing", None, mimetypes.guess_type(destination.name)[0], [])
        last_error = "download failed"
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True, allow_redirects=True, headers={"User-Agent": "pavd-academic-audit/1.0"})
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    if chunk:
                        body.extend(chunk)
                        if len(body) > self.max_bytes:
                            raise ValueError("arquivo excede o limite de bytes")
                with Image.open(BytesIO(body)) as image:
                    image.verify()
                    mime_type = Image.MIME.get(image.format) or response.headers.get("Content-Type")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                digest = sha256(body).hexdigest()
                return DownloadResult(gallery_id, source_id, url, str(destination), "downloaded", digest, mime_type, [str(item.url) for item in response.history])
            except (requests.RequestException, OSError, ValueError, UnidentifiedImageError) as exc:
                last_error = str(exc)
                if attempt < self.retries:
                    time.sleep(0.25 * (2**attempt))
        return DownloadResult(gallery_id, source_id, url, str(destination), "failed", None, None, [], last_error)

    def download(self, rows: Iterable[tuple[str, str, str]], destination: Path) -> list[DownloadResult]:
        jobs = list(rows)
        results: list[DownloadResult] = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._one, gallery_id, source_id, url, destination / f"{gallery_id}.jpg"): (gallery_id, source_id, url)
                for gallery_id, source_id, url in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda item: item.gallery_id)
