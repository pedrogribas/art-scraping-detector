"""Scraper HTML genérico aplicado a plataformas abertas.

ArtStation e Pinterest foram descartados por exigirem contornar o Cloudflare e
renderizarem as galerias via JavaScript. Restaram duas rotas viáveis, tentadas
em cascata:

    1. **Unsplash** — pela API oficial, quando ``UNSPLASH_ACCESS_KEY`` está no
       ``.env``. Sem a chave, o HTML público responde HTTP 403 (Cloudflare), e
       a tentativa é apenas registrada no log.
    2. **Openverse** — agregador de imagens Creative Commons mantido pela
       WordPress Foundation. API aberta, sem cadastro, e indexa Flickr, museus
       e acervos digitais. É o caminho usado por padrão.

``fetch_html()``, ``extract_image_urls()`` e ``scrape_page()`` são genéricas e
podem ser apontadas para qualquer site aberto que sirva HTML estático.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Iterator
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.web")

UNSPLASH_SOURCE = "unsplash"
OPENVERSE_SOURCE = "openverse"

UNSPLASH_SEARCH_URL = "https://unsplash.com/s/photos/{consulta}?page={pagina}"
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"

# Atributos usados por lazy loading em galerias modernas.
_IMG_ATTRS = ("src", "data-src", "data-lazy-src", "data-original")


# ---------------------------------------------------------------------------
# Utilidades genéricas de HTML
# ---------------------------------------------------------------------------


def fetch_html(session: requests.Session, url: str, referer: str = "") -> str | None:
    """Baixa o HTML de uma página com retentativas e headers de navegador."""
    for tentativa in range(1, config.MAX_RETRIES + 1):
        try:
            resposta = session.get(
                url,
                headers=config.get_headers(referer=referer or None),
                timeout=config.REQUEST_TIMEOUT,
            )
            if resposta.status_code in (403, 503):
                logger.warning(
                    "HTTP %s em %s — provável proteção anti-bot (Cloudflare).",
                    resposta.status_code,
                    url,
                )
                return None
            if resposta.status_code == 429:
                logger.warning("Rate limit em %s. Aguardando.", url)
                config.polite_sleep(20, 40)
                continue
            resposta.raise_for_status()
            return resposta.text
        except requests.exceptions.RequestException as erro:
            logger.debug(
                "Falha ao carregar (%d/%d) %s: %s",
                tentativa,
                config.MAX_RETRIES,
                url,
                erro,
            )
            config.polite_sleep(2.0, 4.0)
    return None


def _best_from_srcset(srcset: str) -> str:
    """Escolhe a maior variante declarada em um atributo ``srcset``."""
    melhor_url, melhor_largura = "", -1
    for parte in srcset.split(","):
        pedacos = parte.strip().split()
        if not pedacos:
            continue
        url = pedacos[0]
        largura = 0
        if len(pedacos) > 1 and pedacos[1].endswith("w"):
            try:
                largura = int(pedacos[1][:-1])
            except ValueError:
                largura = 0
        if largura > melhor_largura:
            melhor_url, melhor_largura = url, largura
    return melhor_url


def extract_image_urls(html: str, base_url: str) -> list[str]:
    """Extrai URLs de imagem de uma página HTML.

    Percorre tags ``<img>`` (incluindo ``srcset`` e atributos de lazy loading)
    e as meta tags Open Graph, devolvendo URLs absolutas e sem repetição.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as erro:  # noqa: BLE001
        logger.warning("Não foi possível interpretar o HTML: %s", erro)
        return []

    urls: list[str] = []
    vistas: set[str] = set()

    def _adicionar(bruta: str | None) -> None:
        if not bruta:
            return
        absoluta = urljoin(base_url, bruta.strip())
        if absoluta.startswith(("http://", "https://")) and absoluta not in vistas:
            vistas.add(absoluta)
            urls.append(absoluta)

    for img in soup.find_all("img"):
        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            _adicionar(_best_from_srcset(srcset))
        for atributo in _IMG_ATTRS:
            _adicionar(img.get(atributo))

    for meta in soup.find_all("meta", property="og:image"):
        _adicionar(meta.get("content"))

    return urls


def scrape_page(
    url: str,
    source_name: str,
    session: requests.Session | None = None,
    filtro_dominio: str = "",
) -> Iterator[ImageCandidate]:
    """Raspa uma página arbitrária e gera candidatos a imagem.

    Args:
        url: página a ser raspada.
        source_name: identificador da fonte, usado no nome dos arquivos.
        session: sessão HTTP reaproveitada (opcional).
        filtro_dominio: se informado, mantém apenas URLs cujo host contenha o valor.
    """
    proprietaria = session is None
    session = session or requests.Session()
    try:
        html = fetch_html(session, url)
        if not html:
            return

        for url_imagem in extract_image_urls(html, url):
            if filtro_dominio:
                host = urlparse(url_imagem).hostname or ""
                if filtro_dominio not in host:
                    continue
            yield ImageCandidate(
                url=url_imagem,
                source=source_name,
                page_url=url,
                extra={"strategy": "html"},
            )
    finally:
        if proprietaria:
            session.close()


# ---------------------------------------------------------------------------
# Unsplash
# ---------------------------------------------------------------------------


def _normalizar_unsplash(url: str) -> str:
    """Remove os parâmetros de redimensionamento, buscando a versão maior."""
    base = url.split("?")[0]
    return f"{base}?auto=format&fit=max&w=1600&q=85"


def _is_foto_unsplash(url: str) -> bool:
    """Descarta avatares e assets de interface do Unsplash."""
    host = urlparse(url).hostname or ""
    if "images.unsplash.com" not in host:
        return False
    return "/photo-" in url and "profile" not in url


def _collect_unsplash_api(limit: int | None) -> Iterator[ImageCandidate]:
    """Coleta via API oficial do Unsplash (requer ``UNSPLASH_ACCESS_KEY``)."""
    cabecalhos = {
        "Authorization": f"Client-ID {config.UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1",
        "User-Agent": config.get_random_user_agent(),
    }
    produzidos = 0

    with requests.Session() as session:
        for consulta in config.UNSPLASH_QUERIES:
            if limit is not None and produzidos >= limit:
                return

            for pagina in range(1, config.UNSPLASH_MAX_PAGES + 1):
                if limit is not None and produzidos >= limit:
                    return

                try:
                    resposta = session.get(
                        UNSPLASH_API_URL,
                        headers=cabecalhos,
                        params={
                            "query": consulta,
                            "page": pagina,
                            "per_page": config.UNSPLASH_PER_PAGE,
                            "orientation": "landscape",
                        },
                        timeout=config.REQUEST_TIMEOUT,
                    )
                    resposta.raise_for_status()
                    resultados = resposta.json().get("results", [])
                except requests.exceptions.RequestException as erro:
                    logger.warning("API do Unsplash indisponível: %s", erro)
                    return
                except ValueError as erro:
                    logger.warning("Resposta inválida da API do Unsplash: %s", erro)
                    return

                if not resultados:
                    break

                for foto in resultados:
                    try:
                        url_imagem = (foto.get("urls") or {}).get("regular")
                        if not url_imagem:
                            continue
                        autor = ((foto.get("user") or {}).get("name")) or "desconhecido"
                        yield ImageCandidate(
                            url=url_imagem,
                            source=UNSPLASH_SOURCE,
                            title=(foto.get("description") or foto.get("alt_description") or ""),
                            author=autor,
                            page_url=(foto.get("links") or {}).get("html", ""),
                            external_id=foto.get("id", ""),
                            extra={"query": consulta, "strategy": "api"},
                        )
                        produzidos += 1
                        if limit is not None and produzidos >= limit:
                            return
                    except (AttributeError, TypeError) as erro:
                        logger.debug("Foto ignorada na API: %s", erro)
                        continue

                logger.info("Unsplash API '%s' página %d processada.", consulta, pagina)
                config.polite_sleep(0.8, 1.6)

    logger.info("Unsplash (API) finalizado: %d candidato(s).", produzidos)


def _collect_unsplash_html(limit: int | None) -> Iterator[ImageCandidate]:
    """Coleta via HTML público do Unsplash (sem necessidade de chave)."""
    produzidos = 0

    with requests.Session() as session:
        for consulta in config.UNSPLASH_QUERIES:
            if limit is not None and produzidos >= limit:
                return

            slug = quote_plus(consulta.replace(" ", "-"))

            for pagina in range(1, config.UNSPLASH_MAX_PAGES + 1):
                if limit is not None and produzidos >= limit:
                    return

                url = UNSPLASH_SEARCH_URL.format(consulta=slug, pagina=pagina)
                html = fetch_html(session, url, referer="https://unsplash.com/")
                if not html:
                    # Bloqueio já na primeira página da primeira consulta indica
                    # proteção ativa no site inteiro: insistir só gasta tempo.
                    if produzidos == 0 and pagina == 1:
                        logger.info(
                            "Unsplash bloqueia o acesso sem credencial. "
                            "Abandonando o scraping de HTML."
                        )
                        return
                    logger.info("Interrompendo '%s': página %d inacessível.", consulta, pagina)
                    break

                encontrados = 0
                for url_imagem in extract_image_urls(html, url):
                    if not _is_foto_unsplash(url_imagem):
                        continue
                    yield ImageCandidate(
                        url=_normalizar_unsplash(url_imagem),
                        source=UNSPLASH_SOURCE,
                        page_url=url,
                        extra={"query": consulta, "strategy": "html"},
                    )
                    produzidos += 1
                    encontrados += 1
                    if limit is not None and produzidos >= limit:
                        return

                logger.info(
                    "Unsplash HTML '%s' página %d: %d candidato(s).",
                    consulta,
                    pagina,
                    encontrados,
                )
                if encontrados == 0:
                    break
                config.polite_sleep()

    logger.info("Unsplash (HTML) finalizado: %d candidato(s).", produzidos)


# ---------------------------------------------------------------------------
# Openverse (API aberta, sem cadastro)
# ---------------------------------------------------------------------------


def _collect_openverse(limit: int | None) -> Iterator[ImageCandidate]:
    """Coleta imagens Creative Commons pela API pública do Openverse."""
    cabecalhos = {
        "User-Agent": "art-scraping-detector/1.0 (pesquisa academica IFMG Sabara)",
        "Accept": "application/json",
    }
    produzidos = 0

    with requests.Session() as session:
        for consulta in config.OPENVERSE_QUERIES:
            if limit is not None and produzidos >= limit:
                return

            for pagina in range(1, config.OPENVERSE_MAX_PAGES + 1):
                if limit is not None and produzidos >= limit:
                    return

                try:
                    resposta = session.get(
                        config.OPENVERSE_API_URL,
                        headers=cabecalhos,
                        params={
                            "q": consulta,
                            "page": pagina,
                            "page_size": config.OPENVERSE_PAGE_SIZE,
                            "mature": "false",
                        },
                        timeout=config.REQUEST_TIMEOUT,
                    )
                except requests.exceptions.RequestException as erro:
                    logger.warning("Openverse inacessível: %s", erro)
                    return

                # O acesso anônimo limita a profundidade da paginação; ao ser
                # recusado, seguimos para a próxima consulta em vez de abortar.
                if resposta.status_code in (401, 400):
                    logger.info(
                        "Openverse limitou a paginação anônima em '%s' (página %d).",
                        consulta,
                        pagina,
                    )
                    break
                if resposta.status_code == 429:
                    logger.warning("Openverse aplicou rate limit. Aguardando 60s.")
                    config.polite_sleep(60, 75)
                    continue

                try:
                    resposta.raise_for_status()
                    resultados = resposta.json().get("results", [])
                except (requests.exceptions.RequestException, ValueError) as erro:
                    logger.warning("Resposta inválida do Openverse: %s", erro)
                    break

                if not resultados:
                    break

                encontrados = 0
                for item in resultados:
                    try:
                        url_imagem = item.get("url")
                        if not url_imagem:
                            continue
                        yield ImageCandidate(
                            url=url_imagem,
                            source=OPENVERSE_SOURCE,
                            title=item.get("title") or "",
                            author=item.get("creator") or "desconhecido",
                            page_url=item.get("foreign_landing_url") or "",
                            external_id=item.get("id") or "",
                            extra={
                                "query": consulta,
                                "provider": item.get("provider") or "",
                                "license": item.get("license") or "",
                                "strategy": "api",
                            },
                        )
                        produzidos += 1
                        encontrados += 1
                        if limit is not None and produzidos >= limit:
                            return
                    except (AttributeError, TypeError) as erro:
                        logger.debug("Item do Openverse ignorado: %s", erro)
                        continue

                logger.info(
                    "Openverse '%s' página %d: %d candidato(s).",
                    consulta,
                    pagina,
                    encontrados,
                )
                # Respeita o limite anônimo de 20 requisições por minuto.
                config.polite_sleep(3.5, 5.0)

    logger.info("Openverse finalizado: %d candidato(s).", produzidos)


# ---------------------------------------------------------------------------
# Ponto de entrada do módulo
# ---------------------------------------------------------------------------


def collect(limit: int | None = None) -> Iterator[ImageCandidate]:
    """Gera candidatos das plataformas abertas, em cascata.

    Tenta primeiro o Unsplash (API se houver chave, HTML caso contrário) e
    complementa com o Openverse até alcançar o limite solicitado.
    """
    produzidos = 0

    if config.UNSPLASH_ACCESS_KEY:
        logger.info("UNSPLASH_ACCESS_KEY encontrada: usando a API oficial do Unsplash.")
        gerador = _collect_unsplash_api(limit)
    else:
        logger.info("Sem UNSPLASH_ACCESS_KEY: tentando o HTML público do Unsplash.")
        gerador = _collect_unsplash_html(limit)

    for candidato in gerador:
        yield candidato
        produzidos += 1
        if limit is not None and produzidos >= limit:
            return

    if produzidos == 0:
        logger.info("Unsplash não retornou resultados. Alternando para o Openverse.")
    else:
        logger.info("Complementando com o Openverse (%d obtidos no Unsplash).", produzidos)

    restante = None if limit is None else max(limit - produzidos, 0)
    if restante == 0:
        return

    yield from _collect_openverse(restante)
