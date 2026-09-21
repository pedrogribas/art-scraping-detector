"""Coleta de ilustrações 2D no Safebooru (API pública, sem token).

O Safebooru expõe a API no padrão Gelbooru:

    ``index.php?page=dapi&s=post&q=index&json=1&tags=<tags>&limit=<n>&pid=<pagina>``

As tags são combinadas com AND. ``highres`` (imagens acima de 1.600 px) e
``digital_media`` delimitam justamente o material de interesse do TCC:
ilustração digital em boa resolução. O acervo é integralmente classificado como
*safe*, o que dispensa filtragem adicional de conteúdo adulto.

Cada conjunto de tags tem profundidade própria — ``highres digital_media``, por
exemplo, esgota por volta da página 30. Por isso o módulo percorre vários
conjuntos, o que também diversifica os estilos presentes na amostra.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.booru")

SOURCE_NAME = "safebooru"

# O site recusa hotlink sem Referer da própria origem.
_REFERER = "https://safebooru.org/"


def _fetch_page(
    session: requests.Session, tags: str, pid: int
) -> list[dict[str, Any]] | None:
    """Busca uma página de posts, com retentativas.

    Returns:
        A lista de posts, ``[]`` quando a página está além do fim do acervo, ou
        ``None`` se todas as tentativas falharam.
    """
    parametros = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "json": 1,
        "tags": tags,
        "limit": config.SAFEBOORU_PAGE_SIZE,
        "pid": pid,
    }

    for tentativa in range(1, config.MAX_RETRIES + 1):
        try:
            resposta = session.get(
                config.SAFEBOORU_API_URL,
                headers=config.get_api_headers(),
                params=parametros,
                timeout=config.REQUEST_TIMEOUT,
            )
            if resposta.status_code == 429:
                logger.warning("Safebooru aplicou rate limit. Aguardando.")
                config.polite_sleep(20, 35)
                continue
            resposta.raise_for_status()

            # Sem resultados, a API devolve corpo vazio em vez de lista JSON.
            corpo = resposta.text.strip()
            if not corpo:
                return []

            dados = resposta.json()
            return dados if isinstance(dados, list) else []

        except ValueError:
            # Resposta não-JSON costuma indicar fim da paginação.
            return []
        except requests.exceptions.RequestException as erro:
            logger.debug(
                "Falha no Safebooru (%d/%d) tags=%r pid=%d: %s",
                tentativa,
                config.MAX_RETRIES,
                tags,
                pid,
                erro,
            )
            config.polite_sleep(2.0, 4.0)

    return None


def _build_url(post: dict[str, Any]) -> str:
    """Resolve a URL do arquivo, com fallback para a montagem manual."""
    url = post.get("file_url")
    if url:
        return str(url)

    diretorio = post.get("directory")
    arquivo = post.get("image")
    if diretorio is not None and arquivo:
        return f"https://safebooru.org/images/{diretorio}/{arquivo}"
    return ""


def _to_candidate(post: dict[str, Any], tags: str) -> ImageCandidate | None:
    """Converte um post da API em candidato a download."""
    url = _build_url(post)
    if not url:
        return None

    post_id = post.get("id", "")
    return ImageCandidate(
        url=url,
        source=SOURCE_NAME,
        title=(post.get("tags") or "")[:120],
        author=post.get("owner") or "desconhecido",
        page_url=f"https://safebooru.org/index.php?page=post&s=view&id={post_id}",
        external_id=str(post_id),
        headers={"Referer": _REFERER},
        extra={
            "tag_set": tags,
            "width": post.get("width"),
            "height": post.get("height"),
            "rating": post.get("rating") or "",
        },
    )


def collect(
    limit: int | None = None,
    tag_sets: tuple[str, ...] | None = None,
) -> Iterator[ImageCandidate]:
    """Gera candidatos do Safebooru para os conjuntos de tags configurados.

    Args:
        limit: número máximo de candidatos. ``None`` = sem limite.
        tag_sets: conjuntos de tags. Usa ``config.SAFEBOORU_TAG_SETS`` se omitido.

    Yields:
        ``ImageCandidate`` pronto para download.
    """
    conjuntos = tag_sets or config.SAFEBOORU_TAG_SETS
    produzidos = 0
    vistos: set[str] = set()

    with requests.Session() as session:
        for tags in conjuntos:
            if limit is not None and produzidos >= limit:
                return

            logger.info("Safebooru: buscando tags '%s'...", tags)
            total_tags = 0

            for pid in range(config.SAFEBOORU_MAX_PAGES):
                if limit is not None and produzidos >= limit:
                    return

                posts = _fetch_page(session, tags, pid)
                if posts is None:
                    logger.warning("Safebooru inacessível para '%s'. Seguindo adiante.", tags)
                    break
                if not posts:
                    logger.info("Safebooru: fim dos resultados de '%s' na página %d.", tags, pid)
                    break

                for post in posts:
                    try:
                        candidato = _to_candidate(post, tags)
                        if candidato is None or candidato.external_id in vistos:
                            continue
                        vistos.add(candidato.external_id)

                        yield candidato
                        produzidos += 1
                        total_tags += 1

                        if limit is not None and produzidos >= limit:
                            return
                    except (AttributeError, TypeError) as erro:
                        logger.debug("Post ignorado no Safebooru: %s", erro)
                        continue

                config.polite_sleep(1.0, 2.0)

            logger.info("Safebooru '%s': %d candidato(s).", tags, total_tags)

    logger.info("Safebooru finalizado: %d candidato(s).", produzidos)
