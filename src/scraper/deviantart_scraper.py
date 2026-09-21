"""Coleta de obras no DeviantArt através do feed RSS público.

O DeviantArt protege suas páginas HTML com Cloudflare, mas mantém um endpoint
RSS que aceita a mesma sintaxe de busca do site (``in:digitalart``,
``boost:popular``) e responde em XML — muito mais estável para scraping
acadêmico do que raspar o HTML renderizado por JavaScript.

Endpoint: ``https://backend.deviantart.com/rss.xml?q=<consulta>&offset=<n>``

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Any, Iterator
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.deviantart")

SOURCE_NAME = "deviantart"
_ACCEPT_XML = "application/rss+xml,application/xml;q=0.9,*/*;q=0.8"


def _build_soup(xml: str) -> BeautifulSoup | None:
    """Cria o parser do feed, com fallback caso ``lxml`` não esteja instalado."""
    for parser in ("xml", "lxml-xml", "html.parser"):
        try:
            return BeautifulSoup(xml, parser)
        except Exception:  # noqa: BLE001 - parser indisponível, tenta o próximo
            continue
    logger.error("Nenhum parser XML disponível para o feed do DeviantArt.")
    return None


def _tag_ending_with(item: Any, sufixo: str) -> list[Any]:
    """Busca tags pelo final do nome, ignorando o prefixo de namespace.

    Necessário porque o parser XML expõe ``<media:content>`` como ``content``,
    enquanto o ``html.parser`` mantém o nome completo.
    """
    return item.find_all(lambda tag: tag.name and tag.name.lower().endswith(sufixo))


def _extract_image_url(item: Any) -> str:
    """Retorna a maior imagem disponível no item do feed."""
    melhor_url = ""
    maior_area = -1

    for tag in _tag_ending_with(item, "content") + _tag_ending_with(item, "thumbnail"):
        url = tag.get("url")
        if not url:
            continue
        medium = (tag.get("medium") or "").lower()
        if medium and medium != "image":
            continue
        try:
            area = int(tag.get("width", 0)) * int(tag.get("height", 0))
        except (TypeError, ValueError):
            area = 0
        if area > maior_area:
            maior_area, melhor_url = area, url

    return melhor_url


def _extract_author(item: Any) -> str:
    """Extrai o autor creditado no item, quando presente."""
    for tag in _tag_ending_with(item, "credit"):
        if (tag.get("role") or "").lower() == "author" and tag.text:
            return tag.text.strip()
    return "desconhecido"


def _fetch_feed(session: requests.Session, consulta: str, offset: int) -> str | None:
    """Faz o GET de uma página do feed RSS, com retentativas."""
    url = f"{config.DEVIANTART_RSS_BASE}?" + urlencode(
        {"type": "deviation", "q": consulta, "offset": offset}
    )

    for tentativa in range(1, config.MAX_RETRIES + 1):
        try:
            resposta = session.get(
                url,
                headers=config.get_headers(
                    referer="https://www.deviantart.com/", accept=_ACCEPT_XML
                ),
                timeout=config.REQUEST_TIMEOUT,
            )
            if resposta.status_code == 429:
                logger.warning("DeviantArt aplicou rate limit. Aguardando 30s.")
                config.polite_sleep(30, 45)
                continue
            resposta.raise_for_status()
            return resposta.text
        except requests.exceptions.RequestException as erro:
            logger.debug(
                "Falha no feed (%d/%d) offset=%d: %s",
                tentativa,
                config.MAX_RETRIES,
                offset,
                erro,
            )
            config.polite_sleep(2.0, 5.0)

    logger.warning("Feed inacessível para '%s' (offset %d).", consulta, offset)
    return None


def collect(
    limit: int | None = None,
    queries: tuple[str, ...] | None = None,
) -> Iterator[ImageCandidate]:
    """Gera candidatos a imagem a partir das consultas configuradas.

    Args:
        limit: número máximo de candidatos a produzir. ``None`` = sem limite.
        queries: consultas no formato de busca do DeviantArt.

    Yields:
        ``ImageCandidate`` pronto para download.
    """
    consultas = queries or config.DEVIANTART_QUERIES
    produzidos = 0

    with requests.Session() as session:
        for consulta in consultas:
            if limit is not None and produzidos >= limit:
                break

            logger.info("Consultando DeviantArt: '%s'", consulta)

            for pagina in range(config.DEVIANTART_MAX_PAGES):
                if limit is not None and produzidos >= limit:
                    break

                offset = pagina * config.DEVIANTART_PAGE_SIZE
                xml = _fetch_feed(session, consulta, offset)
                if xml is None:
                    break

                soup = _build_soup(xml)
                if soup is None:
                    return

                itens = soup.find_all("item")
                if not itens:
                    logger.info("Fim dos resultados para '%s' (offset %d).", consulta, offset)
                    break

                encontrados = 0
                for item in itens:
                    try:
                        url_imagem = _extract_image_url(item)
                        if not url_imagem:
                            continue

                        titulo_tag = item.find("title")
                        link_tag = item.find("link")
                        guid_tag = item.find("guid")

                        yield ImageCandidate(
                            url=url_imagem,
                            source=SOURCE_NAME,
                            title=titulo_tag.text.strip() if titulo_tag else "",
                            author=_extract_author(item),
                            page_url=(link_tag.text.strip() if link_tag else ""),
                            external_id=(guid_tag.text.strip() if guid_tag else ""),
                            extra={"query": consulta},
                        )
                        produzidos += 1
                        encontrados += 1

                        if limit is not None and produzidos >= limit:
                            logger.info("Limite de candidatos do DeviantArt atingido.")
                            return
                    except Exception as erro:  # noqa: BLE001 - item malformado
                        logger.debug("Item do feed ignorado: %s", erro)
                        continue

                logger.info(
                    "DeviantArt '%s' offset=%d: %d candidato(s).",
                    consulta,
                    offset,
                    encontrados,
                )
                config.polite_sleep()

    logger.info("DeviantArt finalizado: %d candidato(s) no total.", produzidos)
