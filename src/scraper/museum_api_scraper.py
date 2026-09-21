"""Coleta de obras clássicas em APIs públicas de museus (sem token).

Duas instituições disponibilizam seus acervos digitalizados via REST/JSON e
liberam as reproduções em domínio público:

* **Art Institute of Chicago** — ``api.artic.edu``. As imagens são servidas por
  um endpoint IIIF que permite escolher a resolução na própria URL. O servidor
  exige o header ``AIC-User-Agent`` com identificação e contato; sem ele,
  responde HTTP 403.
* **Metropolitan Museum of Art** — ``collectionapi.metmuseum.org``. A busca
  devolve apenas identificadores, exigindo uma segunda requisição por obra para
  obter a URL da imagem e os metadados.

Particularidades de paginação descobertas em testes e tratadas aqui:
    * ``/artworks/search`` do AIC recusa páginas acima da décima (HTTP 403),
      limitando cada consulta a mil resultados — por isso a coleta alterna entre
      várias consultas e, ao esgotá-las, migra para a listagem ``/artworks``,
      que não tem esse teto;
    * o Met não pagina: a busca devolve a lista completa de identificadores de
      uma vez, e o consumo é feito sob demanda.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.museum")

AIC_SOURCE = "artic"
MET_SOURCE = "metmuseum"


# ---------------------------------------------------------------------------
# Art Institute of Chicago
# ---------------------------------------------------------------------------

_AIC_FIELDS = ",".join(
    (
        "id",
        "title",
        "image_id",
        "artist_title",
        "date_display",
        "classification_title",
        "is_public_domain",
    )
)


def _aic_headers() -> dict[str, str]:
    """Headers de API do AIC, com a identificação exigida pela instituição."""
    cabecalhos = config.get_api_headers()
    cabecalhos["AIC-User-Agent"] = config.AIC_CONTACT_HEADER
    return cabecalhos


def _aic_request(
    session: requests.Session, url: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    """Executa uma requisição à API do AIC, tolerando falhas e bloqueios."""
    try:
        resposta = session.get(
            url, headers=_aic_headers(), params=params, timeout=config.REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException as erro:
        logger.debug("Falha de rede no AIC (%s): %s", url, erro)
        return None

    if resposta.status_code == 403:
        # Sinaliza o teto de paginação do endpoint de busca.
        logger.debug("AIC recusou a página solicitada (HTTP 403).")
        return None
    if resposta.status_code == 429:
        logger.warning("AIC aplicou rate limit. Aguardando.")
        config.polite_sleep(20, 35)
        return None

    try:
        resposta.raise_for_status()
        return resposta.json()
    except (requests.exceptions.RequestException, ValueError) as erro:
        logger.debug("Resposta inválida do AIC: %s", erro)
        return None


def _aic_candidate(obra: dict[str, Any]) -> ImageCandidate | None:
    """Converte um registro do AIC em candidato, montando a URL IIIF."""
    image_id = obra.get("image_id")
    if not image_id:
        return None

    url = (
        f"{config.AIC_IIIF_BASE}/{image_id}/full/"
        f"{config.AIC_IMAGE_WIDTH},/0/default.jpg"
    )
    obra_id = obra.get("id", "")

    return ImageCandidate(
        url=url,
        source=AIC_SOURCE,
        title=obra.get("title") or "",
        author=obra.get("artist_title") or "desconhecido",
        page_url=f"https://www.artic.edu/artworks/{obra_id}",
        external_id=str(obra_id),
        headers={"AIC-User-Agent": config.AIC_CONTACT_HEADER},
        extra={
            "museum": "Art Institute of Chicago",
            "date": obra.get("date_display") or "",
            "classification": obra.get("classification_title") or "",
            "public_domain": bool(obra.get("is_public_domain")),
        },
    )


def collect_aic(limit: int | None = None, apenas_dominio_publico: bool = True) -> Iterator[ImageCandidate]:
    """Gera candidatos do Art Institute of Chicago.

    Percorre primeiro as consultas temáticas e, quando elas se esgotam, a
    listagem geral do acervo.

    Args:
        limit: número máximo de candidatos. ``None`` = sem limite.
        apenas_dominio_publico: descarta obras ainda sob direitos autorais.
    """
    produzidos = 0
    vistos: set[str] = set()

    def _emitir(obras: list[dict[str, Any]]) -> Iterator[ImageCandidate]:
        nonlocal produzidos
        for obra in obras:
            if apenas_dominio_publico and not obra.get("is_public_domain"):
                continue
            candidato = _aic_candidate(obra)
            if candidato is None or candidato.external_id in vistos:
                continue
            vistos.add(candidato.external_id)
            yield candidato
            produzidos += 1

    with requests.Session() as session:
        # Etapa 1: consultas temáticas (até 10 páginas cada).
        for consulta in config.AIC_QUERIES:
            if limit is not None and produzidos >= limit:
                return

            logger.info("AIC: consultando '%s'...", consulta)

            for pagina in range(1, config.AIC_SEARCH_MAX_PAGES + 1):
                if limit is not None and produzidos >= limit:
                    return

                dados = _aic_request(
                    session,
                    f"{config.AIC_API_BASE}/search",
                    {
                        "q": consulta,
                        "limit": config.AIC_PAGE_SIZE,
                        "page": pagina,
                        "fields": _AIC_FIELDS,
                    },
                )
                if dados is None:
                    break

                obras = dados.get("data", [])
                if not obras:
                    break

                for candidato in _emitir(obras):
                    yield candidato
                    if limit is not None and produzidos >= limit:
                        return

                config.polite_sleep(0.6, 1.4)

        # Etapa 2: listagem geral, sem o teto de paginação da busca.
        logger.info("AIC: consultas esgotadas, seguindo pela listagem do acervo.")
        pagina = config.AIC_LISTING_START_PAGE
        ultima = config.AIC_LISTING_START_PAGE + config.AIC_LISTING_MAX_PAGES

        while pagina < ultima:
            if limit is not None and produzidos >= limit:
                return

            dados = _aic_request(
                session,
                config.AIC_API_BASE,
                {
                    "limit": config.AIC_PAGE_SIZE,
                    "page": pagina,
                    "fields": _AIC_FIELDS,
                },
            )
            if dados is None:
                break

            obras = dados.get("data", [])
            if not obras:
                break

            for candidato in _emitir(obras):
                yield candidato
                if limit is not None and produzidos >= limit:
                    return

            pagina += 1
            config.polite_sleep(0.6, 1.4)

    logger.info("Art Institute of Chicago finalizado: %d candidato(s).", produzidos)


# ---------------------------------------------------------------------------
# Metropolitan Museum of Art
# ---------------------------------------------------------------------------


def _met_request(
    session: requests.Session, url: str, params: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Executa uma requisição à API do Met, tolerando falhas."""
    try:
        resposta = session.get(
            url,
            headers=config.get_api_headers(),
            params=params,
            timeout=config.REQUEST_TIMEOUT,
        )
        if resposta.status_code == 429:
            logger.warning("Met aplicou rate limit. Aguardando.")
            config.polite_sleep(20, 35)
            return None
        resposta.raise_for_status()
        return resposta.json()
    except (requests.exceptions.RequestException, ValueError) as erro:
        logger.debug("Falha na API do Met (%s): %s", url, erro)
        return None


def _met_object_ids(session: requests.Session) -> Iterator[int]:
    """Gera identificadores de obras com imagem, sem repetição.

    Combina duas estratégias: buscas temáticas restritas ao domínio público e a
    listagem direta dos departamentos com maior acervo reproduzido.
    """
    vistos: set[int] = set()

    for consulta in config.MET_QUERIES:
        dados = _met_request(
            session,
            f"{config.MET_API_BASE}/search",
            {"q": consulta, "hasImages": "true", "isPublicDomain": "true"},
        )
        identificadores = (dados or {}).get("objectIDs") or []
        logger.info("Met: consulta '%s' retornou %d obra(s).", consulta, len(identificadores))

        for objeto_id in identificadores:
            if objeto_id not in vistos:
                vistos.add(objeto_id)
                yield objeto_id

        config.polite_sleep(0.5, 1.2)

    for departamento in config.MET_DEPARTMENT_IDS:
        dados = _met_request(
            session,
            f"{config.MET_API_BASE}/objects",
            {"departmentIds": departamento},
        )
        identificadores = (dados or {}).get("objectIDs") or []
        logger.info(
            "Met: departamento %d listou %d obra(s).", departamento, len(identificadores)
        )

        for objeto_id in identificadores:
            if objeto_id not in vistos:
                vistos.add(objeto_id)
                yield objeto_id

        config.polite_sleep(0.5, 1.2)


def _met_obra_bidimensional(obra: dict[str, Any]) -> bool:
    """Indica se a obra é pintura, desenho ou gravura.

    A listagem por departamento devolve também esculturas, cerâmica e
    mobiliário, que estão fora do escopo do trabalho. Obras sem classificação
    declarada são aceitas para não descartar registros por falta de metadado.
    """
    classificacao = (obra.get("classification") or "").lower()
    if not classificacao:
        return True
    return any(termo in classificacao for termo in config.MET_ACCEPTED_CLASSIFICATIONS)


def _met_candidate(obra: dict[str, Any]) -> ImageCandidate | None:
    """Converte a resposta de um objeto do Met em candidato."""
    if not _met_obra_bidimensional(obra):
        return None

    pequena = obra.get("primaryImageSmall") or ""
    original = obra.get("primaryImage") or ""

    url = (pequena or original) if config.MET_PREFER_SMALL_IMAGE else (original or pequena)
    if not url:
        return None

    return ImageCandidate(
        url=url,
        source=MET_SOURCE,
        title=obra.get("title") or "",
        author=obra.get("artistDisplayName") or "desconhecido",
        page_url=obra.get("objectURL") or "",
        external_id=str(obra.get("objectID", "")),
        extra={
            "museum": "The Metropolitan Museum of Art",
            "date": obra.get("objectDate") or "",
            "classification": obra.get("classification") or "",
            "department": obra.get("department") or "",
            "public_domain": bool(obra.get("isPublicDomain")),
        },
    )


def collect_met(limit: int | None = None) -> Iterator[ImageCandidate]:
    """Gera candidatos do Metropolitan Museum of Art.

    Cada obra exige uma requisição de metadados adicional, feita sob demanda
    para não acumular milhares de chamadas antes do primeiro download.
    """
    produzidos = 0
    consultados = 0

    with requests.Session() as session:
        for objeto_id in _met_object_ids(session):
            if limit is not None and produzidos >= limit:
                break

            obra = _met_request(session, f"{config.MET_API_BASE}/objects/{objeto_id}")
            consultados += 1

            # Ritmo educado: a documentação do Met pede no máximo 80 req/s.
            config.polite_sleep(0.15, 0.45)

            if not obra:
                continue

            candidato = _met_candidate(obra)
            if candidato is None:
                continue

            yield candidato
            produzidos += 1

            if consultados % 100 == 0:
                logger.info(
                    "Met: %d obra(s) consultada(s), %d com imagem.", consultados, produzidos
                )

    logger.info("Metropolitan Museum finalizado: %d candidato(s).", produzidos)


# ---------------------------------------------------------------------------
# Ponto de entrada do módulo
# ---------------------------------------------------------------------------


def collect(limit: int | None = None) -> Iterator[ImageCandidate]:
    """Gera candidatos dos dois museus, alternando entre eles.

    A alternância mantém o equilíbrio entre os acervos mesmo que a coleta seja
    interrompida antes de atingir a cota do módulo.
    """
    geradores = {
        MET_SOURCE: collect_met(None),
        AIC_SOURCE: collect_aic(None),
    }
    produzidos = 0

    while geradores:
        for nome in list(geradores):
            if limit is not None and produzidos >= limit:
                return
            try:
                candidato = next(geradores[nome])
            except StopIteration:
                logger.info("Museu '%s' esgotado.", nome)
                del geradores[nome]
                continue
            except Exception as erro:  # noqa: BLE001 - isola a falha de um acervo
                logger.error("Erro na coleta do museu '%s': %s", nome, erro)
                del geradores[nome]
                continue

            yield candidato
            produzidos += 1
