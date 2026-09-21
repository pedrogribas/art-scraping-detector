"""Coleta por feeds RSS públicos (DeviantArt e Flickr).

Sucessor do antigo ``deviantart_scraper``, generalizado para qualquer feed no
padrão RSS 2.0 com extensão Media RSS.

* **DeviantArt** — ``backend.deviantart.com/rss.xml`` aceita a mesma sintaxe de
  busca do site (``in:digitalart``, ``boost:popular``) e pagina por ``offset``,
  o que o torna a fonte mais produtiva deste módulo. O HTML do site é protegido
  por Cloudflare; o feed, não.
* **Flickr** — ``photos_public.gne`` filtra por tags e devolve cerca de vinte
  itens por requisição, sem paginação. Entra como complemento de diversidade.

Feeds são servidos por infraestrutura compartilhada e não têm contrato de uso
como uma API. Por isso este módulo aplica o intervalo mais conservador do
pipeline: de 1 a 3 segundos entre requisições e entre downloads
(``config.RSS_DOWNLOAD_DELAY``).

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Any, Iterator
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.rss")

DEVIANTART_SOURCE = "deviantart"
FLICKR_SOURCE = "flickr"

_ACCEPT_XML = "application/rss+xml,application/xml;q=0.9,*/*;q=0.8"


# ---------------------------------------------------------------------------
# Utilidades de parsing de feeds
# ---------------------------------------------------------------------------


def _build_soup(xml: str) -> BeautifulSoup | None:
    """Cria o parser do feed, com fallback caso ``lxml`` não esteja instalado."""
    for parser in ("xml", "lxml-xml", "html.parser"):
        try:
            return BeautifulSoup(xml, parser)
        except Exception:  # noqa: BLE001 - parser indisponível, tenta o próximo
            continue
    logger.error("Nenhum parser XML disponível para leitura dos feeds.")
    return None


def _tag_ending_with(item: Any, sufixo: str) -> list[Any]:
    """Busca tags pelo final do nome, ignorando o prefixo de namespace.

    Necessário porque o parser XML expõe ``<media:content>`` como ``content``,
    enquanto o ``html.parser`` mantém o nome completo.
    """
    return item.find_all(lambda tag: tag.name and tag.name.lower().endswith(sufixo))


def _extract_image_url(item: Any) -> str:
    """Retorna a maior imagem declarada no item do feed."""
    melhor_url = ""
    maior_area = -1

    candidatas = (
        _tag_ending_with(item, "content")
        + _tag_ending_with(item, "thumbnail")
        + item.find_all("enclosure")
    )

    for tag in candidatas:
        url = tag.get("url")
        if not url:
            continue

        medium = (tag.get("medium") or "").lower()
        tipo = (tag.get("type") or "").lower()
        if medium and medium != "image":
            continue
        if tipo and not tipo.startswith("image/"):
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

    for nome in ("creator", "author"):
        tag = item.find(lambda t: t.name and t.name.lower().endswith(nome))
        if tag and tag.text:
            return tag.text.strip()
    return "desconhecido"


def _texto(item: Any, nome_tag: str) -> str:
    """Devolve o texto de uma tag do item, ou string vazia."""
    tag = item.find(nome_tag)
    return tag.text.strip() if tag and tag.text else ""


def fetch_feed(
    session: requests.Session, url: str, referer: str = ""
) -> str | None:
    """Baixa um feed RSS com retentativas e ritmo conservador."""
    for tentativa in range(1, config.MAX_RETRIES + 1):
        try:
            resposta = session.get(
                url,
                headers=config.get_headers(referer=referer or None, accept=_ACCEPT_XML),
                timeout=config.REQUEST_TIMEOUT,
            )
            if resposta.status_code == 429:
                logger.warning("Feed aplicou rate limit (%s). Aguardando.", url[:60])
                config.polite_sleep(30, 45)
                continue
            if resposta.status_code in (403, 503):
                logger.warning("HTTP %s no feed %s.", resposta.status_code, url[:80])
                return None
            resposta.raise_for_status()
            return resposta.text
        except requests.exceptions.RequestException as erro:
            logger.debug(
                "Falha no feed (%d/%d) %s: %s",
                tentativa,
                config.MAX_RETRIES,
                url[:70],
                erro,
            )
            config.polite_sleep(2.0, 5.0)

    logger.warning("Feed inacessível: %s", url[:90])
    return None


def parse_feed(
    xml: str, source: str, extra: dict[str, Any] | None = None
) -> Iterator[ImageCandidate]:
    """Converte o XML de um feed em candidatos a download.

    Args:
        xml: corpo do feed.
        source: identificador da fonte, usado no nome dos arquivos.
        extra: metadados adicionais anexados a cada candidato.
    """
    soup = _build_soup(xml)
    if soup is None:
        return

    for item in soup.find_all("item"):
        try:
            url_imagem = _extract_image_url(item)
            if not url_imagem:
                continue

            yield ImageCandidate(
                url=url_imagem,
                source=source,
                title=_texto(item, "title"),
                author=_extract_author(item),
                page_url=_texto(item, "link"),
                external_id=_texto(item, "guid") or url_imagem,
                extra=dict(extra or {}),
            )
        except Exception as erro:  # noqa: BLE001 - item malformado não derruba o feed
            logger.debug("Item de feed ignorado: %s", erro)
            continue


# ---------------------------------------------------------------------------
# DeviantArt
# ---------------------------------------------------------------------------


def collect_deviantart(
    limit: int | None = None,
    queries: tuple[str, ...] | None = None,
) -> Iterator[ImageCandidate]:
    """Gera candidatos do feed de busca do DeviantArt."""
    consultas = queries or config.DEVIANTART_QUERIES
    produzidos = 0
    vistos: set[str] = set()

    with requests.Session() as session:
        for consulta in consultas:
            if limit is not None and produzidos >= limit:
                return

            logger.info("DeviantArt: consultando '%s'...", consulta)
            total_consulta = 0

            for pagina in range(config.DEVIANTART_MAX_PAGES):
                if limit is not None and produzidos >= limit:
                    return

                offset = pagina * config.DEVIANTART_PAGE_SIZE
                url = f"{config.DEVIANTART_RSS_BASE}?" + urlencode(
                    {"type": "deviation", "q": consulta, "offset": offset}
                )

                xml = fetch_feed(session, url, referer="https://www.deviantart.com/")
                if xml is None:
                    break

                encontrados = 0
                for candidato in parse_feed(
                    xml, DEVIANTART_SOURCE, {"query": consulta, "feed": "deviantart"}
                ):
                    if candidato.external_id in vistos:
                        continue
                    vistos.add(candidato.external_id)

                    yield candidato
                    produzidos += 1
                    encontrados += 1
                    total_consulta += 1

                    if limit is not None and produzidos >= limit:
                        return

                if encontrados == 0:
                    logger.info(
                        "DeviantArt: fim dos resultados de '%s' (offset %d).",
                        consulta,
                        offset,
                    )
                    break

                config.polite_sleep(*config.RSS_DOWNLOAD_DELAY)

            logger.info("DeviantArt '%s': %d candidato(s).", consulta, total_consulta)

    logger.info("DeviantArt finalizado: %d candidato(s).", produzidos)


# ---------------------------------------------------------------------------
# Flickr
# ---------------------------------------------------------------------------


def _flickr_maior_resolucao(url: str) -> str:
    """Troca o sufixo de tamanho da URL do Flickr pela versão grande (``_b``).

    As URLs seguem o padrão ``..._<id>_<sufixo>.jpg``, em que ``m`` (500 px) e
    ``c`` (800 px) são miniaturas e ``b`` corresponde a 1.024 px.
    """
    for sufixo in ("_m.jpg", "_n.jpg", "_c.jpg", "_z.jpg", "_q.jpg", "_s.jpg", "_t.jpg"):
        if url.endswith(sufixo):
            return url[: -len(sufixo)] + "_b.jpg"
    return url


def collect_flickr(
    limit: int | None = None,
    tags: tuple[str, ...] | None = None,
) -> Iterator[ImageCandidate]:
    """Gera candidatos dos feeds públicos do Flickr, por tag."""
    etiquetas = tags or config.FLICKR_TAGS
    produzidos = 0
    vistos: set[str] = set()

    with requests.Session() as session:
        for tag in etiquetas:
            if limit is not None and produzidos >= limit:
                return

            # tagmode=all exige todas as tags; com uma só, mantém a busca ampla.
            url = f"{config.FLICKR_RSS_URL}?" + urlencode(
                {"tags": tag, "tagmode": "all", "format": "rss_200"}
            )

            xml = fetch_feed(session, url, referer="https://www.flickr.com/")
            if xml is None:
                continue

            encontrados = 0
            for candidato in parse_feed(xml, FLICKR_SOURCE, {"tag": tag, "feed": "flickr"}):
                if candidato.external_id in vistos:
                    continue
                vistos.add(candidato.external_id)

                candidato.url = _flickr_maior_resolucao(candidato.url)
                yield candidato
                produzidos += 1
                encontrados += 1

                if limit is not None and produzidos >= limit:
                    return

            logger.info("Flickr '%s': %d candidato(s).", tag, encontrados)
            config.polite_sleep(*config.RSS_DOWNLOAD_DELAY)

    logger.info("Flickr finalizado: %d candidato(s).", produzidos)


# ---------------------------------------------------------------------------
# Ponto de entrada do módulo
# ---------------------------------------------------------------------------


def collect(limit: int | None = None) -> Iterator[ImageCandidate]:
    """Gera candidatos dos feeds RSS, alternando entre DeviantArt e Flickr.

    O Flickr entrega poucos itens por tag e se esgota rapidamente; a partir daí
    o DeviantArt assume sozinho o restante da cota.
    """
    geradores = {
        DEVIANTART_SOURCE: collect_deviantart(None),
        FLICKR_SOURCE: collect_flickr(None),
    }
    produzidos = 0

    while geradores:
        for nome in list(geradores):
            if limit is not None and produzidos >= limit:
                return
            try:
                candidato = next(geradores[nome])
            except StopIteration:
                logger.info("Feed '%s' esgotado.", nome)
                del geradores[nome]
                continue
            except Exception as erro:  # noqa: BLE001 - isola a falha de um feed
                logger.error("Erro na leitura do feed '%s': %s", nome, erro)
                del geradores[nome]
                continue

            yield candidato
            produzidos += 1
