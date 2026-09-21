"""Configuração central do pipeline de coleta.

Concentra caminhos do projeto, parâmetros de rede (User-Agents rotativos,
delays, timeouts), leitura de variáveis de ambiente e configuração de logging.
Nenhum outro módulo deve ler variáveis de ambiente diretamente.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------

# config.py -> scraper -> src -> raiz do projeto
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
MANIFEST_PATH: Final[Path] = DATA_DIR / "manifest.json"

load_dotenv(PROJECT_ROOT / ".env")


def _env_int(nome: str, padrao: int) -> int:
    """Lê uma variável de ambiente inteira, caindo no padrão se inválida."""
    try:
        return int(os.getenv(nome, padrao))
    except (TypeError, ValueError):
        return padrao


def _env_float(nome: str, padrao: float) -> float:
    """Lê uma variável de ambiente float, caindo no padrão se inválida."""
    try:
        return float(os.getenv(nome, padrao))
    except (TypeError, ValueError):
        return padrao


def ensure_directories() -> None:
    """Garante a existência dos diretórios de dados e logs."""
    for diretorio in (RAW_DIR, PROCESSED_DIR, LOGS_DIR):
        diretorio.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metas e limites da coleta
# ---------------------------------------------------------------------------

TARGET_IMAGE_COUNT: Final[int] = _env_int("TARGET_IMAGE_COUNT", 1500)

# Limites de sanidade para descartar ícones, avatares e arquivos corrompidos.
MIN_FILE_SIZE_BYTES: Final[int] = 15 * 1024          # 15 KB
MAX_FILE_SIZE_BYTES: Final[int] = 25 * 1024 * 1024   # 25 MB

ALLOWED_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".webp")
ALLOWED_CONTENT_TYPES: Final[tuple[str, ...]] = (
    "image/jpeg",
    "image/png",
    "image/webp",
)

# Prefixo aplicado aos arquivos salvos: arte_<fonte>_<sequencial>.<ext>
FILENAME_PREFIX: Final[str] = "arte"

# ---------------------------------------------------------------------------
# Parâmetros de rede
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT: Final[int] = _env_int("REQUEST_TIMEOUT", 20)
MAX_RETRIES: Final[int] = _env_int("MAX_RETRIES", 3)
BACKOFF_FACTOR: Final[float] = _env_float("BACKOFF_FACTOR", 1.8)

MIN_REQUEST_DELAY: Final[float] = _env_float("MIN_REQUEST_DELAY", 1.5)
MAX_REQUEST_DELAY: Final[float] = _env_float("MAX_REQUEST_DELAY", 4.0)

# Pool de User-Agents reais e atuais. A rotação reduz a chance de bloqueio por
# fingerprint em plataformas com proteção anti-bot.
USER_AGENTS: Final[tuple[str, ...]] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
)


def get_random_user_agent() -> str:
    """Retorna um User-Agent aleatório do pool."""
    return random.choice(USER_AGENTS)


def get_headers(referer: str | None = None, accept: str | None = None) -> dict[str, str]:
    """Monta headers HTTP realistas para uma requisição.

    Args:
        referer: valor opcional do header ``Referer``, útil para hotlink de imagens.
        accept: sobrescreve o header ``Accept`` (por exemplo, para XML/RSS).
    """
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": accept
        or "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "no-cache",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def polite_sleep(minimo: float | None = None, maximo: float | None = None) -> None:
    """Aguarda um intervalo aleatório entre requisições (scraping educado).

    O jitter aleatório evita um padrão temporal regular, que é um dos sinais
    mais simples usados por sistemas anti-bot para identificar automação.
    """
    inicio = MIN_REQUEST_DELAY if minimo is None else minimo
    fim = MAX_REQUEST_DELAY if maximo is None else maximo
    if fim < inicio:
        inicio, fim = fim, inicio
    time.sleep(random.uniform(inicio, fim))


# ---------------------------------------------------------------------------
# Credenciais (via .env)
# ---------------------------------------------------------------------------

REDDIT_CLIENT_ID: Final[str] = os.getenv("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET: Final[str] = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
REDDIT_USER_AGENT: Final[str] = os.getenv(
    "REDDIT_USER_AGENT", "python:art-scraping-detector:v1.0 (by /u/unknown)"
).strip()

UNSPLASH_ACCESS_KEY: Final[str] = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()


def reddit_credentials_ok() -> bool:
    """Indica se há credenciais suficientes para autenticar no Reddit."""
    return bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)


# ---------------------------------------------------------------------------
# Fontes de dados
# ---------------------------------------------------------------------------

REDDIT_SUBREDDITS: Final[tuple[str, ...]] = (
    "DigitalArt",
    "Art",
    "conceptart",
    "ImaginaryLandscapes",
    "ImaginaryCharacters",
    "SpecArt",
    "painting",
)

# Ordenações percorridas em cada subreddit, ampliando a diversidade temporal.
REDDIT_LISTINGS: Final[tuple[str, ...]] = ("hot", "top", "new")
REDDIT_POSTS_PER_LISTING: Final[int] = 200
REDDIT_TOP_TIME_FILTER: Final[str] = "year"

DEVIANTART_RSS_BASE: Final[str] = "https://backend.deviantart.com/rss.xml"
DEVIANTART_QUERIES: Final[tuple[str, ...]] = (
    "in:digitalart",
    "in:digitalart/paintings",
    "in:digitalart/drawings",
    "boost:popular in:digitalart",
    "concept art",
    "digital illustration",
)
DEVIANTART_PAGE_SIZE: Final[int] = 60
DEVIANTART_MAX_PAGES: Final[int] = 12

UNSPLASH_QUERIES: Final[tuple[str, ...]] = (
    "digital art",
    "illustration",
    "concept art",
    "abstract painting",
    "surreal artwork",
)
UNSPLASH_MAX_PAGES: Final[int] = 10
UNSPLASH_PER_PAGE: Final[int] = 30

# Openverse: agregador de imagens com licença aberta (Creative Commons).
# Usado como alternativa do módulo web quando o Unsplash exige credenciais.
# Limites do acesso anônimo: 20 itens por página, 20 requisições/min, 200/dia.
OPENVERSE_API_URL: Final[str] = "https://api.openverse.org/v1/images/"
OPENVERSE_QUERIES: Final[tuple[str, ...]] = (
    "digital art",
    "digital painting",
    "illustration",
    "concept art",
    "fantasy art",
    "surreal art",
    "abstract art",
    "character design",
    "matte painting",
    "watercolor painting",
)
OPENVERSE_PAGE_SIZE: Final[int] = 20
OPENVERSE_MAX_PAGES: Final[int] = 12

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
DATE_FORMAT: Final[str] = "%H:%M:%S"


def setup_logging(nivel: str | None = None, arquivo: Path | None = None) -> None:
    """Configura o logging raiz para terminal e arquivo.

    Args:
        nivel: nível de log (``DEBUG``, ``INFO``...). Usa ``LOG_LEVEL`` se omitido.
        arquivo: caminho do arquivo de log. Padrão: ``logs/scraping.log``.
    """
    ensure_directories()
    destino = arquivo or (LOGS_DIR / "scraping.log")

    # O console do Windows usa cp1252 por padrão e corromperia os acentos.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    handlers: list[logging.Handler] = [console]
    try:
        em_arquivo = logging.FileHandler(destino, encoding="utf-8")
        em_arquivo.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handlers.append(em_arquivo)
    except OSError:
        # Falha ao abrir o arquivo não deve impedir a coleta: segue só no console.
        pass

    logging.basicConfig(
        level=getattr(logging, (nivel or LOG_LEVEL), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Bibliotecas de terceiro são verbosas demais em DEBUG.
    for ruidoso in ("urllib3", "requests", "prawcore", "charset_normalizer"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_logger(nome: str) -> logging.Logger:
    """Atalho para obter um logger nomeado."""
    return logging.getLogger(nome)
