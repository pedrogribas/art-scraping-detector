"""Orquestra a busca em duas etapas: FAISS grosso e SIFT fino se couber.

Uso:
    python main_pipeline.py --consulta foto.jpg --indice indices/laion
    python main_pipeline.py --construir --metadados data/parquet --saida indices/laion
    python main_pipeline.py --consulta foto.jpg --mock 2000

O mock existe só para exercitar o fluxo sem o dump do LAION.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.category_router import VisualRouter
from src.data_ingestion import gerar_embeddings_mock
from src.fine_search import lupa
from src.vector_index import FaissIndexer, construir_indice


def _indice_mock(n: int, dim: int = 512) -> FaissIndexer:
    """Índice sintético para validar a orquestração em máquina sem parquet."""
    vetores = gerar_embeddings_mock(n, dim=dim, semente=7)
    urls = [f"https://exemplo.invalid/mock/{i:05d}.jpg" for i in range(n)]
    ids = [f"mock-{i:05d}" for i in range(n)]
    indexador = FaissIndexer(dim=dim, backend="auto")
    indexador.adicionar(vetores, urls, ids)
    return indexador


def _imprimir_clip(rota, candidatos) -> None:
    print(f"categoria: {rota.tag} ({rota.categoria})")
    print(f"confiança zero-shot: {rota.confianca:.3f}")
    print("requires_sift: False — laudo CLIP/FAISS (Recall@50)")
    print()
    for posicao, item in enumerate(candidatos, start=1):
        print(f"{posicao:02d}. cosseno={item.score:.3f}  {item.url}")


def _imprimir_sift(rota, laudos) -> None:
    print(f"categoria: {rota.tag} ({rota.categoria})")
    print(f"confiança zero-shot: {rota.confianca:.3f}")
    print("requires_sift: True — ranking reordenado por inliers RANSAC")
    print()
    for posicao, item in enumerate(laudos, start=1):
        extra = f"  [{item.erro}]" if item.erro else ""
        print(
            f"{posicao:02d}. inliers={item.inliers:3d}  "
            f"lowe={item.matches_lowe:3d}  clip={item.score_clip:.3f}  "
            f"{item.url}{extra}"
        )


def executar(
    consulta: Path,
    indice: FaissIndexer,
    k: int = 50,
) -> None:
    """Liga roteador → FAISS → (opcional) SIFT."""
    if not consulta.is_file():
        raise FileNotFoundError(consulta)

    roteador = VisualRouter()
    rota = roteador.classificar(consulta)
    candidatos = indice.search(rota.embedding, k=k)

    if not rota.requires_sift:
        _imprimir_clip(rota, candidatos)
        return

    laudos = lupa(consulta, candidatos)
    _imprimir_sift(rota, laudos)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca em duas etapas para auditoria do LAION-Aesthetics.",
    )
    parser.add_argument("--consulta", type=Path, help="imagem local do artista")
    parser.add_argument("--indice", type=Path, help="prefixo do .faiss + .mapeamento.json")
    parser.add_argument("--k", type=int, default=50, help="Recall@K da etapa grossa")
    parser.add_argument("--construir", action="store_true", help="indexa parquets e sai")
    parser.add_argument("--metadados", type=Path, help="arquivo ou pasta de .parquet")
    parser.add_argument("--embeddings", type=Path, help=".npy/.npz alinhado às linhas")
    parser.add_argument("--saida", type=Path, default=Path("indices/laion"))
    parser.add_argument("--mock", type=int, metavar="N", help="índice sintético com N vetores")
    return parser.parse_args()


def main() -> int:
    args = _parse()
    try:
        if args.construir:
            if args.metadados is None:
                raise SystemExit("--construir exige --metadados")
            indexador = construir_indice(args.metadados, args.saida, args.embeddings)
            print(f"índice gravado: {args.saida}.faiss  ({indexador.total} vetores)")
            return 0

        if args.consulta is None:
            raise SystemExit("informe --consulta ou --construir")

        if args.mock:
            indice = _indice_mock(args.mock)
        elif args.indice is not None:
            indice = FaissIndexer()
            indice.carregar(args.indice)
        else:
            raise SystemExit("informe --indice ou --mock N")

        executar(args.consulta, indice, k=args.k)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
