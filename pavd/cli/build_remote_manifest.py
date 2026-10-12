from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

from pavd.core.gallery import write_manifest
from pavd.core.remote_gallery import remote_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Cria catálogo remoto LAION sem baixar imagens")
    parser.add_argument("--dataset", default="laion/laion2B-en-aesthetic")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="data/gallery/remote_manifest.parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    items = remote_items(dataset, args.limit)
    write_manifest(items, Path(args.output))
    logging.info("Catálogo remoto salvo com %d itens em %s", len(items), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
