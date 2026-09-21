"""Coleta de arte digital e renders 3D na API oficial do Pexels.

O Pexels exige uma chave de API gratuita, lida de ``PEXELS_API_KEY`` no
``.env``. Sem a chave, o módulo apenas registra um aviso e não produz nada,
deixando que os demais módulos assumam a cota (ver ``main_scraper``).

Documentação: https://www.pexels.com/api/documentation/
Limites do plano gratuito: 200 requisições/hora e 20.000/mês.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.pexels")

SOURCE_NAME = "pexels"

# Ordem de preferência das versões servidas pela API, da maior para a menor.
_TAMANHOS = ("original", "large2x", "large", "medium")


def credentials_ok() -> bool:
    """Indica se há chave de API configurada."""
    return bool(config.PEXELS_API_KEY)


def _pick_url(foto: dict[str, Any]) -> str:
    """Escolhe a melhor URL disponível, partindo do tamanho configurado."""
    fontes = foto.get("src") or {}
    preferido = fontes.get(config.PEXELS_IMAGE_SIZE)
    if preferido:
        return str(preferido)

    for tamanho in _TAMANHOS:
        url = fontes.get(tamanho)
        if url:
            return str(url)
    return ""


def _fetch_page(
    session: requests.Session, consulta: str, pagina: int
) -> dict[str, Any] | None:
    """Busca uma página de resultados na API.

    Returns:
        O JSON da resposta ou ``None`` quando a consulta deve ser abandonada
        (chave inválida, limite de requisições ou falha persistente de rede).
    """
    cabecalhos = {
        "Authorization": config.PEXELS_API_KEY,
        "User-Agent": config.PROJECT_USER_AGENT,
        "Accept": "application/json",
    }
    parametros = {
        "query": consulta,
        "page": pagina,
        "per_page": config.PEXELS_PER_PAGE,
    }

    for tentativa in range(1, config.MAX_RETRIES + 1):
        try:
            resposta = session.get(
                config.PEXELS_API_URL,
                headers=cabecalhos,
                params=parametros,
                timeout=config.REQUEST_TIMEOUT,
            )

            if resposta.status_code == 401:
                logger.error("PEXELS_API_KEY inválida ou expirada. Fonte desativada.")
                return None
            if resposta.status_code == 429:
                logger.warning(
                    "Cota horária do Pexels esgotada (200 req/h). Encerrando a fonte."
                )
                return None

            resposta.raise_for_status()
            return resposta.json()

        except ValueError as erro:
            logger.warning("Resposta inválida do Pexels: %s", erro)
            return None
        except requests.exceptions.RequestException as erro:
            logger.debug(
                "Falha no Pexels (%d/%d) consulta=%r página=%d: %s",
                tentativa,
                config.MAX_RETRIES,
                consulta,
                pagina,
                erro,
            )
            config.polite_sleep(2.0, 4.0)

    return None


def _to_candidate(foto: dict[str, Any], consulta: str) -> ImageCandidate | None:
    """Converte uma foto da API em candidato a download."""
    url = _pick_url(foto)
    if not url:
        return None

    return ImageCandidate(
        url=url,
        source=SOURCE_NAME,
        title=foto.get("alt") or "",
        author=foto.get("photographer") or "desconhecido",
        page_url=foto.get("url") or "",
        external_id=str(foto.get("id", "")),
        extra={
            "query": consulta,
            "width": foto.get("width"),
            "height": foto.get("height"),
            "photographer_url": foto.get("photographer_url") or "",
        },
    )


def collect(
    limit: int | None = None,
    queries: tuple[str, ...] | None = None,
) -> Iterator[ImageCandidate]:
    """Gera candidatos do Pexels para as consultas configuradas.

    Args:
        limit: número máximo de candidatos. ``None`` = sem limite.
        queries: termos de busca. Usa ``config.PEXELS_QUERIES`` se omitido.

    Yields:
        ``ImageCandidate`` pronto para download.
    """
    if not credentials_ok():
        logger.warning(
            "PEXELS_API_KEY ausente no .env. Fonte 'pexels' ignorada — a cota "
            "dela será redistribuída entre as demais fontes."
        )
        return

    consultas = queries or config.PEXELS_QUERIES
    produzidos = 0
    vistos: set[str] = set()

    with requests.Session() as session:
        for consulta in consultas:
            if limit is not None and produzidos >= limit:
                return

            logger.info("Pexels: buscando '%s'...", consulta)

            for pagina in range(1, config.PEXELS_MAX_PAGES + 1):
                if limit is not None and produzidos >= limit:
                    return

                dados = _fetch_page(session, consulta, pagina)
                if dados is None:
                    return

                fotos = dados.get("photos") or []
                if not fotos:
                    logger.info("Pexels: fim dos resultados de '%s'.", consulta)
                    break

                encontrados = 0
                for foto in fotos:
                    try:
                        candidato = _to_candidate(foto, consulta)
                        if candidato is None or candidato.external_id in vistos:
                            continue
                        vistos.add(candidato.external_id)

                        yield candidato
                        produzidos += 1
                        encontrados += 1

                        if limit is not None and produzidos >= limit:
                            return
                    except (AttributeError, TypeError) as erro:
                        logger.debug("Foto ignorada no Pexels: %s", erro)
                        continue

                logger.info(
                    "Pexels '%s' página %d: %d candidato(s).", consulta, pagina, encontrados
                )

                if not dados.get("next_page"):
                    break
                config.polite_sleep(0.8, 1.8)

    logger.info("Pexels finalizado: %d candidato(s).", produzidos)
