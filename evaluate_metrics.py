"""Gera o CSV de desempenho da arquitetura em duas etapas.

Uso:
    python evaluate_metrics.py
    python evaluate_metrics.py --indice indices/laion --k 50

As consultas padrão são as sete de ``data/exemplos/``. O gabarito é o
mesmo nome em ``data/raw/``. ``fora.jpg`` não tem original: Recall@50 e
Top-1 geométrico devem falhar.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from src.adulterator import RAW_DIR
from src.category_router import VisualRouter
from src.fine_search import SiftVerifier, baixar_candidatos
from src.vector_index import FaissIndexer, indice_da_amostra

RAIZ = Path(__file__).resolve().parent
EXEMPLOS_DIR = RAIZ / "data" / "exemplos"
INDICE_NOMES = RAIZ / "web" / "public" / "indice" / "clip_nomes.json"
INDICE_VETORES = RAIZ / "web" / "public" / "indice" / "clip_vetores.bin"
SAIDA_PADRAO = RAIZ / "data" / "processed" / "tcc_resultados_finais.csv"

CONSULTAS_PADRAO: tuple[tuple[str, str | None], ...] = (
    ("laion_001.jpg", "laion_001.jpg"),
    ("laion_050.jpg", "laion_050.jpg"),
    ("laion_080.jpg", "laion_080.jpg"),
    ("laion_160.jpg", "laion_160.jpg"),
    ("laion_330.jpg", "laion_330.jpg"),
    ("laion_400.jpg", "laion_400.jpg"),
    ("fora.jpg", None),
)


@dataclass(frozen=True, slots=True)
class LinhaMetrica:
    consulta: str
    gabarito: str
    tag: str
    requires_sift: bool
    confianca: float
    t_clip_s: float
    t_faiss_s: float
    t_download_s: float
    t_sift_s: float
    t_total_s: float
    recall_50: bool
    top1_geometrico: bool | None
    top1_id: str
    top1_inliers: int
    ram_pico_mb: float


def _carregar_indice(prefixo: Path | None) -> FaissIndexer:
    if prefixo is not None:
        indexador = FaissIndexer()
        indexador.carregar(prefixo)
        return indexador
    return indice_da_amostra(RAW_DIR, INDICE_NOMES, INDICE_VETORES)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def avaliar_consulta(
    consulta: Path,
    gabarito: str | None,
    roteador: VisualRouter,
    indice: FaissIndexer,
    k: int,
) -> LinhaMetrica:
    pico = _rss_mb()
    inicio = time.perf_counter()

    t0 = time.perf_counter()
    rota = roteador.classificar(consulta)
    t_clip = time.perf_counter() - t0
    pico = max(pico, _rss_mb())

    t0 = time.perf_counter()
    candidatos = indice.search(rota.embedding, k=k)
    t_faiss = time.perf_counter() - t0
    pico = max(pico, _rss_mb())

    ids_grosso = {item.sample_id for item in candidatos}
    recall = gabarito is not None and gabarito in ids_grosso

    t_download = 0.0
    t_sift = 0.0
    top1_id = candidatos[0].sample_id if candidatos else ""
    top1_inliers = 0
    top1_geo: bool | None = None

    if rota.requires_sift and candidatos:
        pasta = RAIZ / "data" / "processed" / "_tmp_eval"
        t0 = time.perf_counter()
        baixados = baixar_candidatos(candidatos, pasta)
        t_download = time.perf_counter() - t0
        pico = max(pico, _rss_mb())

        t0 = time.perf_counter()
        laudos = SiftVerifier().ranquear(consulta, candidatos, baixados)
        t_sift = time.perf_counter() - t0
        pico = max(pico, _rss_mb())
        if laudos:
            top1_id = laudos[0].sample_id
            top1_inliers = laudos[0].inliers
            top1_geo = gabarito is not None and laudos[0].sample_id == gabarito
        else:
            top1_geo = False
    elif gabarito is not None:
        top1_geo = candidatos[0].sample_id == gabarito if candidatos else False

    return LinhaMetrica(
        consulta=consulta.name,
        gabarito=gabarito or "",
        tag=rota.tag,
        requires_sift=rota.requires_sift,
        confianca=round(rota.confianca, 4),
        t_clip_s=round(t_clip, 4),
        t_faiss_s=round(t_faiss, 4),
        t_download_s=round(t_download, 4),
        t_sift_s=round(t_sift, 4),
        t_total_s=round(time.perf_counter() - inicio, 4),
        recall_50=recall,
        top1_geometrico=top1_geo,
        top1_id=top1_id,
        top1_inliers=top1_inliers,
        ram_pico_mb=round(pico, 1),
    )


def gravar_csv(linhas: list[LinhaMetrica], saida: Path) -> None:
    saida.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "consulta",
        "gabarito",
        "tag",
        "requires_sift",
        "confianca",
        "t_clip_s",
        "t_faiss_s",
        "t_download_s",
        "t_sift_s",
        "t_total_s",
        "recall_50",
        "top1_geometrico",
        "top1_id",
        "top1_inliers",
        "ram_pico_mb",
    ]
    with saida.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow(linha.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Métricas da busca em duas etapas.")
    parser.add_argument("--indice", type=Path, help="prefixo .faiss; sem isso usa as 500 locais")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()

    indice = _carregar_indice(args.indice)
    roteador = VisualRouter()
    linhas: list[LinhaMetrica] = []
    for arquivo, gabarito in CONSULTAS_PADRAO:
        caminho = EXEMPLOS_DIR / arquivo
        if not caminho.is_file():
            print(f"pula {arquivo}: arquivo ausente")
            continue
        print(f"avaliando {arquivo}…")
        linhas.append(avaliar_consulta(caminho, gabarito, roteador, indice, args.k))

    gravar_csv(linhas, args.saida)
    print(f"CSV: {args.saida}  ({len(linhas)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
