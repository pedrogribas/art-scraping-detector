from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pavd.config import PAVDConfig
from pavd.core.gallery import scan_directory, write_manifest
from pavd.core.reporting import RunManifest, write_rows_csv

logger = logging.getLogger("pavd")


def config_from_args(args: argparse.Namespace) -> PAVDConfig:
    root = Path(args.root).resolve()
    config_path = Path(args.config) if args.config else root / "config" / "default.yaml"
    return PAVDConfig.from_yaml(config_path, root=root) if config_path.is_file() else PAVDConfig(root=root).resolve_paths()


def build_gallery(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    source = Path(args.source) if args.source else config.gallery.official_dir
    items = scan_directory(source)
    if args.dry_run:
        logger.info("dry-run: %d arquivos seriam indexados em %s", len(items), config.gallery.metadata_path)
        return 0
    write_manifest(items, config.gallery.metadata_path)
    logger.info("Manifesto salvo com %d itens em %s", len(items), config.gallery.metadata_path)
    return 0


def generate_queries(args: argparse.Namespace) -> int:
    from pavd.core.degradation import DegradationPipeline

    config = config_from_args(args)
    source = Path(args.source) if args.source else config.gallery.official_dir
    destination = config.queries_dir
    pipeline = DegradationPipeline(config.degradation)
    records = []
    for path in sorted(source.iterdir()):
        if args.limit is not None and len(records) >= args.limit:
            break
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        gallery_id = path.stem
        output = destination / f"{gallery_id}_degraded.jpg"
        if output.exists() and args.resume:
            continue
        if args.dry_run:
            records.append({"gallery_id": gallery_id, "source": str(path), "destination": str(output)})
            continue
        record = pipeline.generate(path, output, gallery_id=gallery_id, source_id=path.name)
        records.append(record.as_dict())
    if not args.dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(destination.parent / "query_manifest.parquet", index=False)
    logger.info("Consultas geradas: %d", len(records))
    return 0


def build_clip_index(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    source = Path(args.source) if args.source else config.gallery.official_dir
    items = [(path.stem, path) for path in sorted(source.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    items = items[: args.limit] if args.limit else items
    if args.dry_run:
        logger.info("dry-run: índice CLIP/FAISS para %d itens", len(items))
        return 0
    from pavd.core.clip_index import CLIPIndex

    index = CLIPIndex(config.model_name, config.device, config.faiss)
    index.build(items)
    index.save(config.embeddings_dir / "gallery_clip")
    manifest = RunManifest(config)
    manifest.add(index={"total": len(items), "build_seconds": index.build_seconds, "backend": config.faiss.backend})
    manifest.save(config.reports_dir / "summary" / "run_manifest.json")
    return 0


def run_block_a(args: argparse.Namespace) -> int:
    """Executa uma auditoria híbrida sem reescrever os resultados oficiais."""
    config = config_from_args(args)
    source = Path(args.source) if args.source else config.gallery.official_dir
    queries = Path(args.queries) if args.queries else config.queries_dir
    items = [(path.stem, path) for path in sorted(source.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    items = items[: args.limit] if args.limit else items
    if args.dry_run:
        logger.info("dry-run: Bloco A com %d itens", len(items))
        return 0
    from pavd.core.auditor import PAVDAuditor
    from pavd.core.clip_index import CLIPIndex
    from pavd.core.sift_verifier import SIFTVerifier

    index = CLIPIndex(config.model_name, config.device, config.faiss)
    index.build(items)
    auditor = PAVDAuditor(index, SIFTVerifier(config.sift), config)
    rows = []
    for query in sorted(queries.glob("*_degraded.jpg"))[: args.limit or None]:
        result = auditor.audit_query(query, true_gallery_id=query.name.removesuffix("_degraded.jpg"), k=config.faiss.k)
        rows.append({
            "query_id": result["query_id"],
            "clip_top1": result["clip_top1"]["gallery_id"] if result["clip_top1"] else None,
            "decision": result["final_decision"],
            "report_path": result["report_path"],
        })
    write_rows_csv(config.reports_dir / "summary" / "block_a.csv", rows)
    return 0


def run_block_b(args: argparse.Namespace) -> int:
    """Mede ranking CLIP/FAISS e compara uma amostra com busca exata."""
    config = config_from_args(args)
    source = Path(args.source) if args.source else config.gallery.official_dir
    queries = Path(args.queries) if args.queries else config.queries_dir
    items = [(path.stem, path) for path in sorted(source.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        logger.info("dry-run: Bloco B com %d itens", len(items))
        return 0
    from pavd.core.clip_index import CLIPIndex

    index = CLIPIndex(config.model_name, config.device, config.faiss)
    index.build(items)
    rows = []
    for query in sorted(queries.glob("*_degraded.jpg"))[: args.limit or None]:
        true_id = query.name.removesuffix("_degraded.jpg")
        results = index.search(query, k=max(config.faiss.k, 50))
        ranks = [item.rank for item in results if item.gallery_id == true_id]
        rank = ranks[0] if ranks else None
        rows.append({
            "query_id": query.stem,
            "true_gallery_id": true_id,
            "rank": rank,
            "top1": rank == 1,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_12": rank is not None and rank <= 12,
            "recall_at_50": rank is not None and rank <= 50,
        })
    write_rows_csv(config.reports_dir / "summary" / "block_b.csv", rows)
    RunManifest(config).save(config.reports_dir / "summary" / "run_manifest_block_b.json")
    return 0


def run_block_c(args: argparse.Namespace) -> int:
    """Aplica SIFT apenas ao Top-50 recuperado globalmente pelo CLIP."""
    config = config_from_args(args)
    source = Path(args.source) if args.source else config.gallery.official_dir
    queries = Path(args.queries) if args.queries else config.queries_dir
    items = [(path.stem, path) for path in sorted(source.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        logger.info("dry-run: Bloco C com %d itens", len(items))
        return 0
    from pavd.core.clip_index import CLIPIndex
    from pavd.core.sift_verifier import SIFTVerifier

    index = CLIPIndex(config.model_name, config.device, config.faiss)
    index.build(items)
    verifier = SIFTVerifier(config.sift)
    rows = []
    by_id = {gallery_id: path for gallery_id, path in items}
    for query in sorted(queries.glob("*_degraded.jpg"))[: args.limit or None]:
        candidates = index.search(query, k=50)
        local = [verifier.verify(query, by_id[item.gallery_id], item.gallery_id).as_dict() for item in candidates if item.gallery_id in by_id]
        best = max(local, key=lambda item: (item["inliers"], item["inlier_ratio"]), default=None)
        rows.append({
            "query_id": query.stem,
            "clip_top1": candidates[0].gallery_id if candidates else None,
            "sift_top1": best["candidate_gallery_id"] if best else None,
            "sift_verified": bool(best and best["status"] == "verified"),
            "sift_candidates_evaluated": len(local),
        })
    write_rows_csv(config.reports_dir / "summary" / "block_c.csv", rows)
    return 0


def run_block_d(args: argparse.Namespace) -> int:
    """Aplica a regra preliminar de conjunto aberto a scores fornecidos."""
    config = config_from_args(args)
    if args.dry_run:
        logger.info("dry-run: Bloco D; entrada %s", args.input)
        return 0
    input_path = Path(args.input)
    frame = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    from pavd.core.open_set import OpenSetEvaluator
    evaluator = OpenSetEvaluator(config.open_set)
    score_columns = [column for column in ("s1", "s2") if column in frame.columns]
    if "s1" not in score_columns:
        raise ValueError("o arquivo do Bloco D precisa conter a coluna s1")
    predictions = [
        evaluator.predict([row["s1"], row["s2"] if "s2" in frame.columns else 0.0])
        for _, row in frame.iterrows()
    ]
    frame["predicted_present"] = predictions
    output = config.reports_dir / "summary" / "block_d_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return 0


def run_all(args: argparse.Namespace) -> int:
    """Executa as etapas disponíveis em sequência; não baixa dados."""
    for function in (build_gallery, generate_queries, build_clip_index, run_block_a, run_block_b, run_block_c):
        status = function(args)
        if status:
            return status
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PAVD - Protocolo de Auditoria Visual de Datasets")
    parser.add_argument("--root", default=".", help="raiz do projeto")
    parser.add_argument("--config", help="arquivo YAML")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("build_gallery", build_gallery),
        ("generate_queries", generate_queries),
        ("build_clip_index", build_clip_index),
        ("run_block_a", run_block_a),
        ("run_block_b", run_block_b),
        ("run_block_c", run_block_c),
        ("run_block_d", run_block_d),
        ("run_all", run_all),
    ):
        command = sub.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--source")
        command.add_argument("--queries")
        command.add_argument("--limit", type=int)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--dry-run", action="store_true")
        if name == "run_block_d":
            command.add_argument("--input", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
