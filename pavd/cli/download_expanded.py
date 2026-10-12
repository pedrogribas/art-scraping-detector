from __future__ import annotations

import argparse
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from pavd.core.gallery import GalleryItem, gallery_id_for, scan_directory, write_manifest
from pavd.utils.hashing import sha256_file

logger = logging.getLogger("pavd.download_expanded")

DATASET_ID = "laion/laion2B-en-aesthetic"
DATASET_SPLIT = "train"
URL_COLUMNS = ("URL", "url", "image_url", "IMAGE_URL")
ID_COLUMNS = ("SAMPLE_ID", "sample_id", "key", "hash", "id")


def _value(row: dict, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _download(row: dict, destination: Path, timeout: float, max_bytes: int) -> GalleryItem | None:
    source_id = _value(row, ID_COLUMNS)
    source_url = _value(row, URL_COLUMNS)
    if not source_url.startswith(("http://", "https://")):
        return None
    gallery_id = gallery_id_for(source_id, source_url, destination)
    path = destination / f"{gallery_id}.jpg"
    if path.is_file() and path.stat().st_size:
        try:
            digest = sha256_file(path)
            with Image.open(path) as image:
                return GalleryItem(gallery_id, source_id, source_url, str(path), "", digest, Image.MIME.get(image.format), image.width, image.height, "existing")
        except (OSError, UnidentifiedImageError):
            path.unlink(missing_ok=True)
    try:
        response = requests.get(source_url, timeout=timeout, stream=True, allow_redirects=True, headers={"User-Agent": "pavd-academic-audit/1.0"})
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(64 * 1024):
            if chunk:
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ValueError("arquivo acima do limite")
        with Image.open(BytesIO(body)) as image:
            image.load()
            width, height = image.size
            mime_type = Image.MIME.get(image.format) or response.headers.get("Content-Type")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return GalleryItem(
            gallery_id, source_id, source_url, str(path), pd.Timestamp.utcnow().isoformat(),
            sha256_file(path), mime_type, width, height, "downloaded",
            redirect_history=[item.url for item in response.history],
        )
    except (requests.RequestException, OSError, ValueError, UnidentifiedImageError) as exc:
        logger.debug("falha em %s: %s", source_url, exc)
        return GalleryItem(gallery_id, source_id, source_url, str(path), pd.Timestamp.utcnow().isoformat(), None, None, None, None, "failed", excluded_reason=str(exc), redirect_history=[])


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    image_dir = root / args.image_dir
    manifest_path = root / args.manifest
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = scan_directory(image_dir)
    official_dir = root / args.official_dir
    if official_dir.is_dir():
        official_items = scan_directory(official_dir, source_prefix="official")
        for item in official_items:
            destination = image_dir / Path(item.local_path).name
            if not destination.exists():
                destination.write_bytes(Path(item.local_path).read_bytes())
        existing = scan_directory(image_dir)
    target = args.target
    if args.dry_run:
        logger.info("dry-run: galeria atual=%d, alvo=%d, adicionais=%d", len(existing), target, max(target - len(existing), 0))
        return 0
    if not args.allow_download:
        raise SystemExit(
            "Download desativado por padrão. Para a metodologia URL-only, use "
            "python -m pavd.cli.build_remote_manifest. Para baixar explicitamente, "
            "informe --allow-download."
        )
    if len(existing) >= target:
        write_manifest(existing, manifest_path)
        logger.info("galeria já possui %d itens", len(existing))
        return 0
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    session = requests.Session()
    results = list(existing)
    seen_sources = {item.source_id for item in existing if item.source_id}
    progress = tqdm(total=target, initial=len(existing), desc="Galeria expandida", unit="img")
    try:
        for row in dataset:
            if len(results) >= target:
                break
            source_id = _value(row, ID_COLUMNS)
            if source_id and source_id in seen_sources:
                continue
            item = _download(row, image_dir, args.timeout, args.max_bytes)
            if item is None:
                continue
            if item.download_status in {"downloaded", "existing"}:
                results.append(item)
                if item.source_id:
                    seen_sources.add(item.source_id)
                progress.update(1)
            if args.limit and len(results) >= args.limit:
                break
    finally:
        session.close()
        progress.close()
    write_manifest(results, manifest_path)
    logger.info("galeria final: %d itens; manifesto: %s", len(results), manifest_path)
    return 0 if len(results) >= min(target, args.limit or target) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa explicitamente a galeria expandida do LAION-Aesthetics")
    parser.add_argument("--root", default=".")
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-dir", default="data/gallery/images")
    parser.add_argument("--official-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/gallery/gallery_index.parquet")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
