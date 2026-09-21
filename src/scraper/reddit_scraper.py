"""Coleta de obras digitais no Reddit via API oficial (biblioteca ``praw``).

O Reddit é a fonte primária da Amostra de Controle: possui API pública
documentada, não depende de burlar proteções anti-bot e reúne comunidades
grandes e ativas de arte digital (r/DigitalArt, r/Art, r/conceptart).

Requer credenciais de um app do tipo *script* definidas no arquivo ``.env``.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from typing import Any, Iterator
from urllib.parse import urlparse

from . import config
from .models import ImageCandidate

logger = config.get_logger("scraper.reddit")

SOURCE_NAME = "reddit"

# Domínios que servem o arquivo de imagem diretamente.
_DIRECT_IMAGE_HOSTS = ("i.redd.it", "i.imgur.com", "preview.redd.it")


def _is_direct_image(url: str) -> bool:
    """Verifica se a URL aponta para um arquivo de imagem suportado."""
    try:
        analisada = urlparse(url)
    except ValueError:
        return False
    if analisada.hostname and analisada.hostname.lower() in _DIRECT_IMAGE_HOSTS:
        return True
    caminho = analisada.path.lower()
    return caminho.endswith(config.ALLOWED_EXTENSIONS)


def _build_client() -> Any | None:
    """Instancia o cliente PRAW em modo somente leitura.

    Returns:
        A instância de ``praw.Reddit`` ou ``None`` se a biblioteca não estiver
        instalada, faltarem credenciais ou a autenticação falhar.
    """
    if not config.reddit_credentials_ok():
        logger.warning(
            "REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET ausentes no .env. "
            "Fonte 'reddit' será ignorada."
        )
        return None

    try:
        import praw  # import tardio: a ausência da lib não derruba o pipeline
    except ImportError:
        logger.error("Biblioteca 'praw' não instalada. Rode: pip install -r requirements.txt")
        return None

    try:
        cliente = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
            check_for_async=False,
        )
        cliente.read_only = True
        # Chamada barata só para validar as credenciais antes da coleta longa.
        cliente.subreddit("DigitalArt").id
        logger.info("Autenticado no Reddit em modo somente leitura.")
        return cliente
    except Exception as erro:  # noqa: BLE001 - praw levanta exceções variadas
        logger.error("Falha ao autenticar no Reddit: %s", erro)
        return None


def _extract_from_submission(submission: Any) -> Iterator[ImageCandidate]:
    """Extrai todas as imagens utilizáveis de um post.

    Cobre três formatos: link direto para imagem, galeria nativa do Reddit e
    preview gerado pela plataforma para posts de outros domínios.
    """
    titulo = getattr(submission, "title", "") or ""
    autor = str(getattr(submission, "author", "") or "desconhecido")
    permalink = f"https://www.reddit.com{getattr(submission, 'permalink', '')}"
    subreddit = str(getattr(submission, "subreddit", "") or "")
    post_id = getattr(submission, "id", "")

    def _montar(url: str, sufixo: str = "") -> ImageCandidate:
        return ImageCandidate(
            url=url,
            source=SOURCE_NAME,
            title=titulo,
            author=autor,
            page_url=permalink,
            external_id=f"{post_id}{sufixo}",
            extra={"subreddit": subreddit, "score": getattr(submission, "score", 0)},
        )

    # 1) Galeria nativa (vários itens por post).
    if getattr(submission, "is_gallery", False):
        metadados = getattr(submission, "media_metadata", None) or {}
        for item_id, item in metadados.items():
            try:
                if item.get("e") != "Image":
                    continue
                url = item["s"].get("u") or item["s"].get("gif")
                if url:
                    yield _montar(url.replace("&amp;", "&"), sufixo=f"_{item_id}")
            except (AttributeError, KeyError, TypeError):
                continue
        return

    # 2) Link direto para arquivo de imagem.
    url_post = getattr(submission, "url", "") or ""
    if _is_direct_image(url_post):
        yield _montar(url_post)
        return

    # 3) Preview em alta resolução gerado pelo Reddit.
    try:
        preview = getattr(submission, "preview", None) or {}
        fonte = preview["images"][0]["source"]["url"]
        yield _montar(fonte.replace("&amp;", "&"), sufixo="_preview")
    except (KeyError, IndexError, TypeError, AttributeError):
        return


def _iter_listing(subreddit: Any, listagem: str, limite: int) -> Iterator[Any]:
    """Percorre uma ordenação do subreddit (``hot``, ``top`` ou ``new``)."""
    if listagem == "top":
        return subreddit.top(time_filter=config.REDDIT_TOP_TIME_FILTER, limit=limite)
    if listagem == "new":
        return subreddit.new(limit=limite)
    return subreddit.hot(limit=limite)


def collect(
    limit: int | None = None,
    subreddits: tuple[str, ...] | None = None,
) -> Iterator[ImageCandidate]:
    """Gera candidatos a imagem a partir dos subreddits configurados.

    Args:
        limit: número máximo de candidatos a produzir. ``None`` = sem limite.
        subreddits: lista de subreddits. Usa ``config.REDDIT_SUBREDDITS`` se omitida.

    Yields:
        ``ImageCandidate`` pronto para download.
    """
    cliente = _build_client()
    if cliente is None:
        return

    alvos = subreddits or config.REDDIT_SUBREDDITS
    produzidos = 0

    for nome_sub in alvos:
        if limit is not None and produzidos >= limit:
            return

        try:
            subreddit = cliente.subreddit(nome_sub)
        except Exception as erro:  # noqa: BLE001
            logger.warning("Não foi possível acessar r/%s: %s", nome_sub, erro)
            continue

        for listagem in config.REDDIT_LISTINGS:
            if limit is not None and produzidos >= limit:
                return

            logger.info("Lendo r/%s [%s]...", nome_sub, listagem)
            antes = produzidos

            try:
                posts = _iter_listing(subreddit, listagem, config.REDDIT_POSTS_PER_LISTING)
                for submission in posts:
                    try:
                        if getattr(submission, "over_18", False):
                            continue
                        for candidato in _extract_from_submission(submission):
                            yield candidato
                            produzidos += 1
                            if limit is not None and produzidos >= limit:
                                logger.info("Limite de candidatos do Reddit atingido.")
                                return
                    except Exception as erro:  # noqa: BLE001 - post problemático
                        logger.debug("Post ignorado em r/%s: %s", nome_sub, erro)
                        continue
            except Exception as erro:  # noqa: BLE001 - erro de API/rede
                logger.warning("Erro ao listar r/%s [%s]: %s", nome_sub, listagem, erro)
                continue

            logger.info(
                "r/%s [%s]: %d candidato(s) encontrado(s).",
                nome_sub,
                listagem,
                produzidos - antes,
            )
            config.polite_sleep(1.0, 2.5)

    logger.info("Reddit finalizado: %d candidato(s) no total.", produzidos)
