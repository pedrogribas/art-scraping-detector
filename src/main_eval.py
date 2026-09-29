"""Avaliação da prova de conceito: busca reversa após adulteração.

Para cada imagem em ``data/adulterated/``, SSIM, SIFT e CLIP procuram a
original correspondente em ``data/raw``. Um acerto é o Top 1 com o mesmo nome
de arquivo. O relatório traz a acurácia e o tempo médio de busca por imagem.

A indexação da galeria é feita uma vez por método e não entra na média de busca.

Uso:
    python -m src.main_eval

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.adulterator import ADULTERATED_DIR, RAW_DIR, adulterate_all, list_images
from src.similarity_evaluator import (
    ComparadorCLIP,
    ComparadorSIFT,
    ComparadorSSIM,
    ResultadoBusca,
)

logger = logging.getLogger("main_eval")


@dataclass(slots=True)
class RelatorioMetodo:
    """Resumo de um algoritmo na avaliação."""

    nome: str
    acertos: int
    total: int
    tempo_indexacao: float
    tempo_medio_busca: float

    @property
    def acuracia(self) -> float:
        """Taxa de acerto Top 1."""
        if self.total == 0:
            return 0.0
        return self.acertos / self.total


def setup_logging() -> None:
    """Configura o log no terminal, com UTF-8 no console do Windows."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def avaliar(comparador, consultas: list[Path]) -> RelatorioMetodo:
    """Indexa a galeria e mede acerto e tempo de busca de um comparador."""
    galeria = list_images(RAW_DIR)
    logger.info("Indexando %s em %d imagem(ns)...", comparador.nome, len(galeria))
    inicio = time.perf_counter()
    comparador.indexar(galeria)
    tempo_indexacao = time.perf_counter() - inicio
    logger.info("Índice %s pronto em %.1f s.", comparador.nome, tempo_indexacao)

    acertos = 0
    tempos: list[float] = []
    for consulta in tqdm(consultas, desc=comparador.nome, unit="busca", file=sys.stdout):
        t0 = time.perf_counter()
        resultado: ResultadoBusca = comparador.buscar(consulta)
        tempos.append(time.perf_counter() - t0)
        if resultado.filename == consulta.name:
            acertos += 1

    media = sum(tempos) / len(tempos) if tempos else 0.0
    return RelatorioMetodo(
        nome=comparador.nome,
        acertos=acertos,
        total=len(consultas),
        tempo_indexacao=tempo_indexacao,
        tempo_medio_busca=media,
    )


def imprimir_relatorio(relatorios: list[RelatorioMetodo]) -> None:
    """Imprime a tabela final no terminal."""
    logger.info("=" * 78)
    logger.info("RELATÓRIO DA PROVA DE CONCEITO")
    logger.info("=" * 78)
    logger.info(
        "%-8s %10s %12s %18s %22s",
        "Método",
        "Acurácia",
        "Acertos",
        "Indexação (s)",
        "Busca média (s/img)",
    )
    logger.info("-" * 78)
    for relatorio in relatorios:
        logger.info(
            "%-8s %9.1f%% %6d/%-5d %18.2f %22.4f",
            relatorio.nome,
            relatorio.acuracia * 100,
            relatorio.acertos,
            relatorio.total,
            relatorio.tempo_indexacao,
            relatorio.tempo_medio_busca,
        )
    logger.info("=" * 78)


def main() -> int:
    """Gera as adulterações, se faltarem, e avalia os três algoritmos."""
    setup_logging()
    originais = list_images(RAW_DIR)
    if not originais:
        logger.error("Pasta vazia: %s", RAW_DIR)
        return 1

    adulteradas = list_images(ADULTERATED_DIR)
    if len(adulteradas) < len(originais):
        logger.info("Gerando versões adulteradas antes da avaliação.")
        adulterate_all()
        adulteradas = list_images(ADULTERATED_DIR)

    if not adulteradas:
        logger.error("Nenhuma imagem adulterada em %s", ADULTERATED_DIR)
        return 1

    logger.info("Consultas: %d | Galeria: %d", len(adulteradas), len(originais))
    relatorios = [
        avaliar(ComparadorSSIM(), adulteradas),
        avaliar(ComparadorSIFT(), adulteradas),
        avaliar(ComparadorCLIP(), adulteradas),
    ]
    imprimir_relatorio(relatorios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
