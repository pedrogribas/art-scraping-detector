"""Configuração central do pipeline de coleta.

Concentra caminhos do projeto, parâmetros de rede (User-Agents rotativos,
delays, timeouts), credenciais lidas do ``.env``, definição das fontes e a
distribuição de cotas entre elas. Nenhum outro módulo lê variáveis de
ambiente diretamente.

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

TARGET_IMAGE_COUNT: Final[int] = _env_int("TARGET_IMAGE_COUNT", 5000)

# Limites de sanidade para descartar ícones, avatares e arquivos corrompidos.
MIN_FILE_SIZE_BYTES: Final[int] = 15 * 1024          # 15 KB
MAX_FILE_SIZE_BYTES: Final[int] = 25 * 1024 * 1024   # 25 MB

ALLOWED_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".webp")

# Arquivos são salvos como <fonte>_<sequencial>.<ext>, com contador próprio por
# fonte — por exemplo, metmuseum_0001.jpg e safebooru_0001.png. São 4 dígitos
# porque uma única fonte pode ultrapassar 1.000 imagens na meta de 5.000.
FILENAME_PADDING: Final[int] = 4

# ---------------------------------------------------------------------------
# Parâmetros de rede
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT: Final[int] = _env_int("REQUEST_TIMEOUT", 20)
MAX_RETRIES: Final[int] = _env_int("MAX_RETRIES", 3)
BACKOFF_FACTOR: Final[float] = _env_float("BACKOFF_FACTOR", 1.8)

MIN_REQUEST_DELAY: Final[float] = _env_float("MIN_REQUEST_DELAY", 1.5)
MAX_REQUEST_DELAY: Final[float] = _env_float("MAX_REQUEST_DELAY", 4.0)

# Intervalo entre downloads, por fonte. APIs oficiais toleram ritmo mais alto;
# feeds RSS são servidos por infraestrutura compartilhada e recebem o intervalo
# conservador de 1 a 3 segundos.
RSS_DOWNLOAD_DELAY: Final[tuple[float, float]] = (1.0, 3.0)
DEFAULT_DOWNLOAD_DELAY: Final[tuple[float, float]] = (0.3, 1.0)
SOURCE_DOWNLOAD_DELAYS: Final[dict[str, tuple[float, float]]] = {
    "deviantart": RSS_DOWNLOAD_DELAY,
    "flickr": RSS_DOWNLOAD_DELAY,
    "metmuseum": (0.2, 0.6),
    "artic": (0.2, 0.6),
    "safebooru": (0.5, 1.2),
    "pexels": (0.2, 0.6),
    "openverse": (0.5, 1.2),
}


def get_download_delay(fonte: str) -> tuple[float, float]:
    """Retorna o intervalo (mínimo, máximo) de espera após baixar da fonte."""
    return SOURCE_DOWNLOAD_DELAYS.get(fonte, DEFAULT_DOWNLOAD_DELAY)


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

# Identificação honesta do projeto, usada nas APIs que pedem contato.
PROJECT_USER_AGENT: Final[str] = os.getenv(
    "PROJECT_USER_AGENT",
    "art-scraping-detector/2.0 (TCC IFMG Sabara; contato: pedrogarciaribas@hotmail.com)",
).strip()


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


def get_api_headers(accept: str = "application/json") -> dict[str, str]:
    """Headers para APIs REST, com identificação explícita do projeto."""
    return {
        "User-Agent": get_random_user_agent(),
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
        "Connection": "keep-alive",
    }


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

PEXELS_API_KEY: Final[str] = os.getenv("PEXELS_API_KEY", "").strip()
UNSPLASH_ACCESS_KEY: Final[str] = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()

# ---------------------------------------------------------------------------
# Fonte: Art Institute of Chicago (API pública, sem token)
# ---------------------------------------------------------------------------

AIC_API_BASE: Final[str] = "https://api.artic.edu/api/v1/artworks"
AIC_IIIF_BASE: Final[str] = "https://www.artic.edu/iiif/2"

# O servidor IIIF do museu recusa requisições (HTTP 403) sem este header de
# identificação, conforme a política de uso da própria instituição.
AIC_CONTACT_HEADER: Final[str] = os.getenv(
    "AIC_CONTACT", "art-scraping-detector (pedrogarciaribas@hotmail.com)"
).strip()

# Largura solicitada ao servidor IIIF: resolução suficiente para os
# experimentos de similaridade sem inflar o volume em disco.
AIC_IMAGE_WIDTH: Final[int] = _env_int("AIC_IMAGE_WIDTH", 843)
AIC_PAGE_SIZE: Final[int] = 100

# O endpoint /artworks/search recusa páginas acima da 10ª (HTTP 403), o que
# limita cada consulta a mil resultados.
AIC_SEARCH_MAX_PAGES: Final[int] = 10
AIC_QUERIES: Final[tuple[str, ...]] = (
    "painting",
    "portrait",
    "landscape",
    "still life",
    "watercolor",
    "impressionism",
    "modern art",
    "japanese print",
    "drawing",
    "abstract",
)

# A listagem paginada (/artworks) não tem o teto do search e é usada para
# complementar a cota quando as consultas se esgotam.
AIC_LISTING_START_PAGE: Final[int] = _env_int("AIC_LISTING_START_PAGE", 1)
AIC_LISTING_MAX_PAGES: Final[int] = _env_int("AIC_LISTING_MAX_PAGES", 400)

# ---------------------------------------------------------------------------
# Fonte: Metropolitan Museum of Art (API pública, sem token)
# ---------------------------------------------------------------------------

MET_API_BASE: Final[str] = "https://collectionapi.metmuseum.org/public/collection/v1"

# primaryImageSmall entrega cerca de 300 KB contra vários MB do original —
# resolução adequada ao experimento e viável para milhares de imagens.
MET_PREFER_SMALL_IMAGE: Final[bool] = os.getenv("MET_PREFER_SMALL_IMAGE", "1") != "0"

MET_QUERIES: Final[tuple[str, ...]] = (
    "painting",
    "portrait",
    "landscape",
    "still life",
    "watercolor",
    "drawing",
    "illustration",
    "mythology",
)

# O acervo do Met inclui esculturas, cerâmica e mobiliário, que fogem do escopo
# de obras bidimensionais do trabalho. O parâmetro 'medium' da API foi descartado
# como filtro: combinar vários valores com '|' funciona como E lógico (não OU) e
# reduz a busca a poucas dezenas de resultados. A triagem é feita localmente,
# pela classificação declarada em cada obra.
MET_ACCEPTED_CLASSIFICATIONS: Final[tuple[str, ...]] = (
    "painting",
    "drawing",
    "print",
    "watercolor",
    "illustration",
    "poster",
    "miniature",
)

# Departamentos com maior densidade de obras bidimensionais reproduzidas em
# imagem: 11 = European Paintings, 21 = Modern and Contemporary Art,
# 9 = Drawings and Prints, 6 = Asian Art (pinturas em rolo e gravuras).
MET_DEPARTMENT_IDS: Final[tuple[int, ...]] = (11, 21, 9, 6)

# ---------------------------------------------------------------------------
# Fonte: Safebooru (API pública no padrão Gelbooru, sem token)
# ---------------------------------------------------------------------------

SAFEBOORU_API_URL: Final[str] = "https://safebooru.org/index.php"
SAFEBOORU_PAGE_SIZE: Final[int] = 100
SAFEBOORU_MAX_PAGES: Final[int] = 40

# Cada entrada é um conjunto de tags combinadas com AND pela API.
# 'highres' garante imagens grandes; 'digital_media' restringe a arte digital.
SAFEBOORU_TAG_SETS: Final[tuple[str, ...]] = (
    "highres digital_media",
    "absurdres digital_media",
    "highres digital_media scenery",
    "highres digital_media original",
    "highres painting_(medium)",
    "highres watercolor_(medium)",
    "absurdres scenery",
)

# ---------------------------------------------------------------------------
# Fonte: Pexels (API oficial, requer chave gratuita no .env)
# ---------------------------------------------------------------------------

PEXELS_API_URL: Final[str] = "https://api.pexels.com/v1/search"
PEXELS_PER_PAGE: Final[int] = 80
PEXELS_MAX_PAGES: Final[int] = 25
PEXELS_QUERIES: Final[tuple[str, ...]] = (
    "digital art",
    "3d render",
    "abstract art",
    "digital painting",
    "3d abstract",
    "surreal art",
)
# Tamanho retornado pela API: large2x tem cerca de 1.880 px de largura.
PEXELS_IMAGE_SIZE: Final[str] = os.getenv("PEXELS_IMAGE_SIZE", "large2x").strip()

# ---------------------------------------------------------------------------
# Fonte: feeds RSS
# ---------------------------------------------------------------------------

DEVIANTART_RSS_BASE: Final[str] = "https://backend.deviantart.com/rss.xml"
DEVIANTART_QUERIES: Final[tuple[str, ...]] = (
    "in:digitalart",
    "in:digitalart/paintings",
    "in:digitalart/drawings",
    "in:digitalart/3dimages",
    "boost:popular in:digitalart",
    "concept art",
    "digital illustration",
    "3d render",
)
DEVIANTART_PAGE_SIZE: Final[int] = 60
DEVIANTART_MAX_PAGES: Final[int] = 25

FLICKR_RSS_URL: Final[str] = "https://www.flickr.com/services/feeds/photos_public.gne"
FLICKR_TAGS: Final[tuple[str, ...]] = (
    "digitalart",
    "digitalpainting",
    "conceptart",
    "3drender",
    "digitalillustration",
    "fractalart",
)

# ---------------------------------------------------------------------------
# Fonte complementar: Openverse / Unsplash (módulo web_scraper)
# ---------------------------------------------------------------------------

UNSPLASH_QUERIES: Final[tuple[str, ...]] = (
    "digital art",
    "illustration",
    "concept art",
    "abstract painting",
    "surreal artwork",
)
UNSPLASH_MAX_PAGES: Final[int] = 10
UNSPLASH_PER_PAGE: Final[int] = 30

# Limites do acesso anônimo do Openverse: 20 itens por página, 20 req/min, 200/dia.
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
# Distribuição das cotas entre os módulos
# ---------------------------------------------------------------------------

# Proporção de cada módulo na meta total. Os pesos garantem a mistura de estilos
# pretendida no TCC: arte clássica (museus), ilustração 2D (booru e RSS),
# render 3D (Pexels e DeviantArt) e arte digital contemporânea.
SOURCE_WEIGHTS: Final[dict[str, float]] = {
    "museum": 0.44,   # Met + Art Institute of Chicago
    "booru": 0.20,    # Safebooru
    "pexels": 0.16,   # Pexels
    "rss": 0.20,      # DeviantArt + Flickr
}

# Módulo acionado apenas na rodada de compensação, se a meta não for atingida.
RESERVE_SOURCES: Final[tuple[str, ...]] = ("web",)


def build_quotas(meta: int, modulos: list[str]) -> dict[str, int]:
    """Distribui a meta entre os módulos conforme ``SOURCE_WEIGHTS``.

    Módulos sem peso definido recebem fatia igual ao restante. A sobra da
    divisão inteira é atribuída ao primeiro módulo, garantindo que a soma das
    cotas seja exatamente a meta.
    """
    ativos = [nome for nome in modulos if nome not in RESERVE_SOURCES]
    if not ativos:
        return {nome: meta for nome in modulos[:1]}

    peso_total = sum(SOURCE_WEIGHTS.get(nome, 0.0) for nome in ativos)
    if peso_total <= 0:
        base = meta // len(ativos)
        cotas = {nome: base for nome in ativos}
    else:
        cotas = {
            nome: int(meta * SOURCE_WEIGHTS.get(nome, 0.0) / peso_total)
            for nome in ativos
        }

    sobra = meta - sum(cotas.values())
    if sobra and ativos:
        cotas[ativos[0]] += sobra

    for nome in modulos:
        cotas.setdefault(nome, 0)
    return cotas


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
    for ruidoso in ("urllib3", "requests", "charset_normalizer"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_logger(nome: str) -> logging.Logger:
    """Atalho para obter um logger nomeado."""
    return logging.getLogger(nome)
