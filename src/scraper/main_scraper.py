"""Orquestrador do pipeline de coleta da Amostra de Controle.

Consome os módulos de fonte (Reddit, DeviantArt e web genérico/Unsplash),
unifica o download em um único fluxo com barra de progresso e encerra
automaticamente ao atingir a meta de imagens salvas em ``data/raw``.

Uso:
    python -m src.scraper.main_scraper
    python -m src.scraper.main_scraper --target 1500 --sources reddit deviantart
    python -m src.scraper.main_scraper --dry-run --target 20

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Iterator

from tqdm import tqdm

from . import config, deviantart_scraper, reddit_scraper, web_scraper
from .downloader import ImageDownloader
from .models import ImageCandidate

logger = config.get_logger("scraper.main")

# Ordem de prioridade das fontes: da mais estável para a mais suscetível a bloqueio.
# A fonte "web" cobre Unsplash e Openverse (ver web_scraper.collect).
SOURCES: dict[str, Callable[[int | None], Iterator[ImageCandidate]]] = {
    "reddit": reddit_scraper.collect,
    "deviantart": deviantart_scraper.collect,
    "web": web_scraper.collect,
}

# Candidatos avaliados por fonte para cada imagem ainda faltante. Compensa
# duplicatas, links quebrados e arquivos reprovados na validação.
OVERSAMPLING_FACTOR = 4


def iter_candidates(fontes: list[str], faltantes: int) -> Iterator[ImageCandidate]:
    """Encadeia os geradores das fontes selecionadas em um único fluxo.

    Uma fonte que falhe por completo (credenciais ausentes, site fora do ar)
    apenas registra o erro; as demais seguem normalmente.
    """
    for nome in fontes:
        coletor = SOURCES.get(nome)
        if coletor is None:
            logger.warning("Fonte desconhecida ignorada: '%s'.", nome)
            continue

        logger.info("=" * 70)
        logger.info("Iniciando coleta na fonte: %s", nome.upper())
        logger.info("=" * 70)

        try:
            yield from coletor(faltantes * OVERSAMPLING_FACTOR)
        except KeyboardInterrupt:
            raise
        except Exception as erro:  # noqa: BLE001 - isola a falha de uma fonte
            logger.error("Fonte '%s' interrompida por erro: %s", nome, erro)
            continue


def run(
    target: int = config.TARGET_IMAGE_COUNT,
    fontes: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Executa a coleta até atingir a meta de imagens salvas.

    Args:
        target: total de imagens desejado em ``data/raw`` (inclui as já existentes).
        fontes: fontes a utilizar. Padrão: todas, na ordem de prioridade.
        dry_run: se ``True``, apenas lista os candidatos, sem baixar nada.

    Returns:
        Código de saída: ``0`` se a meta foi atingida, ``1`` caso contrário.
    """
    config.ensure_directories()
    fontes = fontes or list(SOURCES.keys())
    inicio = time.time()

    if dry_run:
        return _run_dry(fontes, target)

    downloader = ImageDownloader()
    ja_salvas = downloader.manifest.total_saved
    faltantes = max(target - ja_salvas, 0)

    logger.info("Meta: %d imagens | Já coletadas: %d | Faltam: %d", target, ja_salvas, faltantes)
    logger.info("Destino: %s", config.RAW_DIR)

    if faltantes == 0:
        logger.info("Meta já atingida. Nada a fazer.")
        downloader.close()
        return 0

    estatisticas: dict[str, int] = {}
    avaliados = 0

    barra = tqdm(
        total=target,
        initial=ja_salvas,
        unit="img",
        desc="Coletando",
        ncols=100,
        file=sys.stdout,
    )

    try:
        for candidato in iter_candidates(fontes, faltantes):
            if downloader.manifest.total_saved >= target:
                break

            avaliados += 1
            resultado = downloader.download(candidato)

            if resultado.success:
                barra.update(1)
                barra.set_postfix_str(f"{candidato.source}: {resultado.filename}")
                # Salvamento periódico: uma interrupção não perde o progresso.
                if downloader.manifest.total_saved % 50 == 0:
                    downloader.manifest.save()
                config.polite_sleep(0.3, 1.0)
            else:
                estatisticas[resultado.reason] = estatisticas.get(resultado.reason, 0) + 1
                if resultado.reason == "falha_de_rede":
                    config.polite_sleep(0.2, 0.6)

    except KeyboardInterrupt:
        logger.warning("Coleta interrompida pelo usuário (Ctrl+C).")
    except Exception as erro:  # noqa: BLE001 - garante o relatório final
        logger.exception("Erro inesperado no pipeline: %s", erro)
    finally:
        barra.close()
        downloader.close()

    total = downloader.manifest.total_saved
    _relatorio_final(
        total=total,
        meta=target,
        avaliados=avaliados,
        por_fonte=downloader.manifest.counts_by_source(),
        descartes=estatisticas,
        duracao=time.time() - inicio,
    )
    return 0 if total >= target else 1


def _run_dry(fontes: list[str], target: int) -> int:
    """Lista candidatos sem baixá-los — útil para validar uma fonte nova."""
    logger.info("MODO DRY-RUN: nenhum arquivo será salvo.")
    contagem: dict[str, int] = {}

    try:
        for indice, candidato in enumerate(iter_candidates(fontes, target), start=1):
            contagem[candidato.source] = contagem.get(candidato.source, 0) + 1
            logger.info("[%04d] %s | %s", indice, candidato.source, candidato.url[:110])
            if indice >= target:
                break
    except KeyboardInterrupt:
        logger.warning("Dry-run interrompido pelo usuário.")

    logger.info("Candidatos por fonte: %s", contagem or "nenhum")
    return 0


def _relatorio_final(
    total: int,
    meta: int,
    avaliados: int,
    por_fonte: dict[str, int],
    descartes: dict[str, int],
    duracao: float,
) -> None:
    """Imprime o resumo da execução no terminal e no log."""
    logger.info("=" * 70)
    logger.info("RELATÓRIO FINAL DA COLETA")
    logger.info("=" * 70)
    logger.info("Imagens salvas .......: %d / %d", total, meta)
    logger.info("Candidatos avaliados .: %d", avaliados)
    logger.info("Tempo de execução ....: %.1f min", duracao / 60)
    logger.info("Diretório ............: %s", config.RAW_DIR)
    logger.info("Manifesto ............: %s", config.MANIFEST_PATH)

    logger.info("-" * 70)
    logger.info("Distribuição por fonte:")
    for fonte, quantidade in sorted(por_fonte.items(), key=lambda par: -par[1]):
        proporcao = (quantidade / total * 100) if total else 0
        logger.info("  %-14s %5d (%.1f%%)", fonte, quantidade, proporcao)

    if descartes:
        logger.info("-" * 70)
        logger.info("Motivos de descarte:")
        for motivo, quantidade in sorted(descartes.items(), key=lambda par: -par[1]):
            logger.info("  %-22s %5d", motivo, quantidade)

    logger.info("=" * 70)
    if total >= meta:
        logger.info("Meta atingida. Amostra de Controle pronta para a próxima etapa.")
    else:
        logger.warning(
            "Meta não atingida (%d/%d). Execute novamente: o manifesto retoma "
            "de onde parou, sem duplicar imagens.",
            total,
            meta,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        prog="main_scraper",
        description=(
            "Pipeline de coleta da Amostra de Controle — TCC de Sistemas de "
            "Informação (IFMG Sabará), por Pedro Garcia Ribas."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=config.TARGET_IMAGE_COUNT,
        help=f"Total de imagens desejado (padrão: {config.TARGET_IMAGE_COUNT}).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCES.keys()),
        default=list(SOURCES.keys()),
        help="Fontes a utilizar, na ordem informada.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista os candidatos encontrados sem baixar nenhum arquivo.",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosidade dos logs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    args = parse_args(argv)
    config.setup_logging(args.log_level)
    logger.info("Art Scraping Detector — coleta da Amostra de Controle")
    return run(target=args.target, fontes=args.sources, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
