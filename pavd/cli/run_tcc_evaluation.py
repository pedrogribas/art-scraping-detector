from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.adulterator import ADULTERATED_DIR, RAW_DIR, list_images
from src.similarity_evaluator import ComparadorCLIP, ComparadorSIFT, ComparadorSSIM

METHODS = (ComparadorSSIM, ComparadorSIFT, ComparadorCLIP)


def evaluate_method(comparator, gallery: list[Path], queries: list[Path], categories: dict[str, str]) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    comparator.indexar(gallery)
    indexing_seconds = time.perf_counter() - started
    rows = []
    for query in queries:
        started = time.perf_counter()
        result = comparator.buscar(query)
        search_seconds = time.perf_counter() - started
        expected = query.name
        rows.append({
            "method": comparator.nome,
            "query_id": query.stem,
            "expected_filename": expected,
            "predicted_filename": result.filename,
            "correct": result.filename == expected,
            "score": result.score,
            "search_seconds": search_seconds,
            "category": categories.get(query.stem, "unknown"),
        })
    summary = {
        "method": comparator.nome,
        "gallery_size": len(gallery),
        "query_count": len(queries),
        "correct": sum(row["correct"] for row in rows),
        "top1": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
        "indexing_seconds": indexing_seconds,
        "mean_search_seconds": sum(row["search_seconds"] for row in rows) / len(rows) if rows else 0.0,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa a avaliação fechada de 500 itens e salva dados para o TCC")
    parser.add_argument("--gallery", type=Path, default=RAW_DIR)
    parser.add_argument("--queries", type=Path, default=ADULTERATED_DIR)
    parser.add_argument("--categories", type=Path, default=Path("data/gallery/official_categories.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/pavd"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    gallery = list_images(args.gallery)
    queries = list_images(args.queries)
    category_map: dict[str, str] = {}
    if args.categories.is_file():
        categories = pd.read_parquet(args.categories)
        category_map = dict(zip(categories["gallery_id"], categories["visual_category"], strict=False))
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for method_type in METHODS:
        logging.info("Avaliando %s", method_type.nome)
        rows, summary = evaluate_method(method_type(), gallery, queries, category_map)
        all_rows.extend(rows)
        summaries.append(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output / "block_a_per_query.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "block_a_summary.csv", index=False)
    (args.output / "block_a_summary.json").write_text(
        json.dumps({"summaries": summaries, "methodology": {
            "gallery": str(args.gallery),
            "queries": str(args.queries),
            "closed_set": True,
            "query_generation": "legacy fixed degradation preserved for official results",
        }, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
