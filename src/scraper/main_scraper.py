"""Orquestrador do pipeline de coleta da Amostra de Controle.

Executa os módulos de coleta de forma balanceada: cada um recebe uma cota
proporcional da meta (``config.SOURCE_WEIGHTS``) e é acompanhado por sua
própria barra de progresso.

A execução tem duas rodadas:

1. **Rodada balanceada** — cada módulo coleta até a sua cota. Módulos
   indisponíveis (sem credencial, fora do ar) simplesmente não produzem.
2. **Rodada de compensação** — se a meta não foi atingida, o que faltou é
   redistribuído entre os módulos que ainda têm material, incluindo as fontes
   de reserva. Como os geradores são preguiçosos e preservados entre as
   rodadas, a coleta continua da página onde havia parado, sem repetir
   requisições já feitas.

Uso:
    python -m src.scraper.main_scraper
    python -m src.scraper.main_scraper --target 5000
    python -m src.scraper.main_scraper --sources museum booru --target 1000
    python -m src.scraper.main_scraper --dry-run --target 20

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Iterator

from tqdm import tqdm

from . import (
    booru_scraper,
    config,
    museum_api_scraper,
    pexels_scraper,
    rss_scraper,
    web_scraper,
)
from .downloader import ImageDownloader
from .models import ImageCandidate

logger = config.get_logger("scraper.main")

Coletor = Callable[[int | None], Iterator[ImageCandidate]]

# Módulos de coleta disponíveis. A fonte "web" (Openverse/Unsplash) é reserva:
# só entra na rodada de compensação, conforme config.RESERVE_SOURCES.
SOURCES: dict[str, Coletor] = {
    "museum": museum_api_scraper.collect,
    "booru": booru_scraper.collect,
    "pexels": pexels_scraper.collect,
    "rss": rss_scraper.collect,
    "web": web_scraper.collect,
}

DESCRICOES: dict[str, str] = {
    "museum": "Met + Art Institute of Chicago",
    "booru": "Safebooru",
    "pexels": "Pexels",
    "rss": "DeviantArt + Flickr (RSS)",
    "web": "Openverse / Unsplash (reserva)",
}

# Candidatos consecutivos rejeitados que fazem um módulo ser considerado
# improdutivo (links quebrados, duplicatas ou arquivos fora dos critérios).
MAX_FALHAS_CONSECUTIVAS = 150


class SourceRunner:
    """Mantém o estado de um módulo de coleta ao longo das rodadas.

    O gerador é criado uma única vez e preservado entre as rodadas: retomá-lo
    continua da página em que parou, em vez de refazer as requisições iniciais.
    """

    def __init__(self, nome: str, coletor: Coletor) -> None:
        self.nome = nome
        self._coletor = coletor
        self._gerador: Iterator[ImageCandidate] | None = None
        self.exhausted = False
        self.saved = 0
        self.evaluated = 0

    @property
    def gerador(self) -> Iterator[ImageCandidate]:
        """Instancia o gerador na primeira chamada e o reaproveita depois."""
        if self._gerador is None:
            self._gerador = self._coletor(None)
        return self._gerador

    def next_candidate(self) -> ImageCandidate | None:
        """Obtém o próximo candidato, ou ``None`` se a fonte se esgotou."""
        if self.exhausted:
            return None
        try:
            return next(self.gerador)
        except StopIteration:
            self.exhausted = True
            return None
        except KeyboardInterrupt:
            raise
        except Exception as erro:  # noqa: BLE001 - isola a falha do módulo
            logger.error("Módulo '%s' interrompido por erro: %s", self.nome, erro)
            self.exhausted = True
            return None


def coletar_modulo(
    runner: SourceRunner,
    downloader: ImageDownloader,
    cota: int,
    meta_global: int,
    descartes: dict[str, int],
) -> int:
    """Coleta até ``cota`` imagens de um módulo, com barra de progresso própria.

    Returns:
        Quantidade de imagens efetivamente salvas nesta chamada.
    """
    if cota <= 0 or runner.exhausted:
        return 0

    logger.info("-" * 70)
    logger.info(
        "Módulo '%s' (%s) — cota desta rodada: %d imagem(ns).",
        runner.nome,
        DESCRICOES.get(runner.nome, runner.nome),
        cota,
    )

    salvas = 0
    falhas_seguidas = 0
    barra = tqdm(
        total=cota,
        unit="img",
        desc=f"{runner.nome:>8}",
        ncols=100,
        leave=True,
        file=sys.stdout,
    )

    try:
        while salvas < cota:
            if downloader.manifest.total_saved >= meta_global:
                break

            candidato = runner.next_candidate()
            if candidato is None:
                logger.info("Módulo '%s' não tem mais candidatos.", runner.nome)
                break

            runner.evaluated += 1
            resultado = downloader.download(candidato)

            if resultado.success:
                salvas += 1
                runner.saved += 1
                falhas_seguidas = 0
                barra.update(1)
                barra.set_postfix_str(resultado.filename)

                # Salvamento periódico: uma interrupção não perde o progresso.
                if downloader.manifest.total_saved % 50 == 0:
                    downloader.manifest.save()

                config.polite_sleep(*config.get_download_delay(candidato.source))
            else:
                descartes[resultado.reason] = descartes.get(resultado.reason, 0) + 1
                falhas_seguidas += 1

                if resultado.reason == "falha_de_rede":
                    config.polite_sleep(0.2, 0.6)

                if falhas_seguidas >= MAX_FALHAS_CONSECUTIVAS:
                    logger.warning(
                        "Módulo '%s': %d candidatos seguidos descartados. "
                        "Passando para o próximo módulo.",
                        runner.nome,
                        falhas_seguidas,
                    )
                    break
    finally:
        barra.close()

    logger.info("Módulo '%s': %d imagem(ns) salva(s) nesta rodada.", runner.nome, salvas)
    return salvas


def run(
    target: int = config.TARGET_IMAGE_COUNT,
    fontes: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Executa a coleta balanceada até atingir a meta.

    Args:
        target: total de imagens desejado em ``data/raw`` (inclui as já existentes).
        fontes: módulos a utilizar. Padrão: todos.
        dry_run: se ``True``, apenas lista os candidatos, sem baixar nada.

    Returns:
        Código de saída: ``0`` se a meta foi atingida, ``1`` caso contrário.
    """
    config.ensure_directories()
    modulos = fontes or list(SOURCES.keys())
    inicio = time.time()

    if dry_run:
        return _run_dry(modulos, target)

    downloader = ImageDownloader()
    ja_salvas = downloader.manifest.total_saved
    faltantes = max(target - ja_salvas, 0)

    logger.info("Meta: %d imagens | Já coletadas: %d | Faltam: %d", target, ja_salvas, faltantes)
    logger.info("Destino: %s", config.RAW_DIR)

    if faltantes == 0:
        logger.info("Meta já atingida. Nada a fazer.")
        downloader.close()
        return 0

    runners = {nome: SourceRunner(nome, SOURCES[nome]) for nome in modulos if nome in SOURCES}
    for nome in modulos:
        if nome not in SOURCES:
            logger.warning("Módulo desconhecido ignorado: '%s'.", nome)

    cotas = config.build_quotas(faltantes, list(runners.keys()))
    logger.info("=" * 70)
    logger.info("DISTRIBUIÇÃO DAS COTAS")
    for nome, cota in cotas.items():
        if cota:
            logger.info("  %-8s %5d  (%s)", nome, cota, DESCRICOES.get(nome, nome))
    reservas = [n for n in runners if n in config.RESERVE_SOURCES]
    if reservas:
        logger.info("  reserva: %s (acionada só se faltar imagem)", ", ".join(reservas))
    logger.info("=" * 70)

    descartes: dict[str, int] = {}

    try:
        # Rodada 1: cada módulo até a sua cota.
        for nome, runner in runners.items():
            if downloader.manifest.total_saved >= target:
                break
            coletar_modulo(runner, downloader, cotas.get(nome, 0), target, descartes)
            downloader.manifest.save()

        # Rodada 2: redistribui o que faltou entre os módulos ainda produtivos.
        rodada = 0
        while downloader.manifest.total_saved < target:
            disponiveis = [r for r in runners.values() if not r.exhausted]
            if not disponiveis:
                logger.warning("Todas as fontes se esgotaram antes de atingir a meta.")
                break

            rodada += 1
            restante = target - downloader.manifest.total_saved
            logger.info("=" * 70)
            logger.info(
                "RODADA DE COMPENSAÇÃO %d — faltam %d imagem(ns) para a meta.",
                rodada,
                restante,
            )
            logger.info("=" * 70)

            por_modulo = max(restante // len(disponiveis), 1)
            progresso_rodada = 0

            for runner in disponiveis:
                if downloader.manifest.total_saved >= target:
                    break
                cota = min(por_modulo, target - downloader.manifest.total_saved)
                progresso_rodada += coletar_modulo(
                    runner, downloader, cota, target, descartes
                )
                downloader.manifest.save()

            if progresso_rodada == 0:
                logger.warning("Nenhuma imagem nova nesta rodada. Encerrando.")
                break

    except KeyboardInterrupt:
        logger.warning("Coleta interrompida pelo usuário (Ctrl+C).")
    except Exception as erro:  # noqa: BLE001 - garante o relatório final
        logger.exception("Erro inesperado no pipeline: %s", erro)
    finally:
        downloader.close()

    total = downloader.manifest.total_saved
    _relatorio_final(
        total=total,
        meta=target,
        runners=runners,
        por_fonte=downloader.manifest.counts_by_source(),
        descartes=descartes,
        duracao=time.time() - inicio,
    )
    return 0 if total >= target else 1


def _run_dry(modulos: list[str], target: int) -> int:
    """Lista candidatos sem baixá-los — útil para validar uma fonte nova."""
    logger.info("MODO DRY-RUN: nenhum arquivo será salvo.")
    contagem: dict[str, int] = {}
    ativos = [nome for nome in modulos if nome in SOURCES]
    if not ativos:
        logger.error("Nenhum módulo válido informado.")
        return 1

    por_modulo = max(target // len(ativos), 1)

    try:
        for nome in ativos:
            logger.info("-" * 70)
            logger.info("Módulo '%s' (%s)", nome, DESCRICOES.get(nome, nome))
            runner = SourceRunner(nome, SOURCES[nome])

            for indice in range(1, por_modulo + 1):
                candidato = runner.next_candidate()
                if candidato is None:
                    break
                contagem[candidato.source] = contagem.get(candidato.source, 0) + 1
                logger.info(
                    "[%s %03d] %-11s %s",
                    nome,
                    indice,
                    candidato.source,
                    candidato.url[:95],
                )
    except KeyboardInterrupt:
        logger.warning("Dry-run interrompido pelo usuário.")

    logger.info("-" * 70)
    logger.info("Candidatos por fonte: %s", contagem or "nenhum")
    return 0


def _relatorio_final(
    total: int,
    meta: int,
    runners: dict[str, SourceRunner],
    por_fonte: dict[str, int],
    descartes: dict[str, int],
    duracao: float,
) -> None:
    """Imprime o resumo da execução no terminal e no log."""
    avaliados = sum(r.evaluated for r in runners.values())

    logger.info("=" * 70)
    logger.info("RELATÓRIO FINAL DA COLETA")
    logger.info("=" * 70)
    logger.info("Imagens salvas .......: %d / %d", total, meta)
    logger.info("Candidatos avaliados .: %d", avaliados)
    logger.info("Tempo de execução ....: %.1f min", duracao / 60)
    logger.info("Diretório ............: %s", config.RAW_DIR)
    logger.info("Manifesto ............: %s", config.MANIFEST_PATH)

    logger.info("-" * 70)
    logger.info("Distribuição por fonte (acervo completo em data/raw):")
    for fonte, quantidade in sorted(por_fonte.items(), key=lambda par: -par[1]):
        proporcao = (quantidade / total * 100) if total else 0
        logger.info("  %-12s %5d (%.1f%%)", fonte, quantidade, proporcao)

    logger.info("-" * 70)
    logger.info("Rendimento por módulo nesta execução:")
    for nome, runner in runners.items():
        estado = "esgotado" if runner.exhausted else "disponível"
        logger.info(
            "  %-8s salvas=%-5d avaliados=%-6d (%s)",
            nome,
            runner.saved,
            runner.evaluated,
            estado,
        )

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
        help="Módulos a utilizar, na ordem informada.",
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
