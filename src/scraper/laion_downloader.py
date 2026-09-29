"""Download de um subconjunto estético do LAION-5B (Hugging Face).

A Amostra de Controle deixa de ser simulada por scrapers próprios e passa a
usar URLs reais do subset ``laion/laion2B-en-aesthetic``. O dataset é lido em
streaming, então apenas os metadados necessários trafegam: o arquivo completo
(vários GB de parquet) não é baixado.

O repositório no Hugging Face é restrito. Antes da primeira execução:

    huggingface-cli login
    # aceite os termos em:
    # https://huggingface.co/datasets/laion/laion2B-en-aesthetic

Uso:
    python -m src.scraper.laion_downloader

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from requests import RequestException
from tqdm import tqdm

logger = logging.getLogger("laion_downloader")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_ID = "laion/laion2B-en-aesthetic"
DATASET_SPLIT = "train"
URL_COLUMNS = ("URL", "url", "image_url", "IMAGE_URL")

TARGET_IMAGES = 500
REQUEST_TIMEOUT = 3
MAX_IMAGE_BYTES = 20 * 1024 * 1024
JPEG_QUALITY = 90
USER_AGENT = (
    "art-scraping-detector/3.0 (TCC IFMG Sabara; contato: pedrogarciaribas@hotmail.com)"
)


def setup_logging() -> None:
    """Configura logs no terminal, com UTF-8 no console do Windows."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def next_index(diretorio: Path) -> int:
    """Próximo número livre em ``laion_NNN.jpg`` já presentes na pasta."""
    maior = 0
    for arquivo in diretorio.glob("laion_*.jpg"):
        sufixo = arquivo.stem.removeprefix("laion_")
        if sufixo.isdigit():
            maior = max(maior, int(sufixo))
    return maior + 1


def saved_count(diretorio: Path) -> int:
    """Quantidade de imagens LAION já gravadas em ``data/raw``."""
    return sum(1 for _ in diretorio.glob("laion_*.jpg"))


def extract_url(row: dict) -> str | None:
    """Lê a URL da linha, aceitando as variações de nome usadas no LAION."""
    for coluna in URL_COLUMNS:
        valor = row.get(coluna)
        if isinstance(valor, str) and valor.startswith(("http://", "https://")):
            return valor.strip()
    return None


def download_bytes(session: requests.Session, url: str) -> bytes:
    """Baixa o corpo da URL com timeout curto e limite de tamanho.

    Raises:
        RequestException: falha de rede, timeout ou HTTP 4xx/5xx.
        ValueError: corpo vazio ou maior que ``MAX_IMAGE_BYTES``.
    """
    with session.get(
        url,
        timeout=REQUEST_TIMEOUT,
        stream=True,
        headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
    ) as resposta:
        resposta.raise_for_status()

        partes: list[bytes] = []
        total = 0
        for pedaco in resposta.iter_content(chunk_size=65_536):
            if not pedaco:
                continue
            total += len(pedaco)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("arquivo acima do limite de tamanho")
            partes.append(pedaco)

    conteudo = b"".join(partes)
    if not conteudo:
        raise ValueError("resposta vazia")
    return conteudo


def decode_image(conteudo: bytes) -> Image.Image:
    """Abre o binário com o Pillow e devolve uma imagem RGB pronta para JPEG.

    ``verify()`` descarta arquivos que não são imagem. O arquivo é reaberto em
    seguida porque a verificação consome o decoder.

    Raises:
        UnidentifiedImageError: conteúdo que o Pillow não reconhece.
        OSError: imagem truncada ou corrompida.
    """
    with Image.open(BytesIO(conteudo)) as candidata:
        candidata.verify()

    with Image.open(BytesIO(conteudo)) as imagem:
        imagem.load()
        if imagem.mode != "RGB":
            imagem = imagem.convert("RGB")
        else:
            imagem = imagem.copy()
    return imagem


def save_jpeg(imagem: Image.Image, caminho: Path) -> None:
    """Grava a imagem como JPEG. Remove o arquivo se a escrita falhar."""
    try:
        imagem.save(caminho, format="JPEG", quality=JPEG_QUALITY)
    except OSError:
        caminho.unlink(missing_ok=True)
        raise
    finally:
        imagem.close()


def iter_rows():
    """Percorre o subset estético em streaming, sem materializar o dataset."""
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    yield from dataset


def run(target: int = TARGET_IMAGES) -> int:
    """Baixa imagens até a pasta ``data/raw`` conter exatamente ``target`` JPEGs.

    Returns:
        ``0`` quando a meta é atingida, ``1`` se o stream acabar antes disso.
    """
    setup_logging()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ja_salvas = saved_count(RAW_DIR)
    if ja_salvas >= target:
        logger.info("A pasta já contém %d imagem(ns). Nada a baixar.", ja_salvas)
        return 0

    indice = next_index(RAW_DIR)
    logger.info("Dataset: %s (streaming)", DATASET_ID)
    logger.info("Destino: %s | já salvas: %d | meta: %d", RAW_DIR, ja_salvas, target)

    session = requests.Session()
    barra = tqdm(
        total=target,
        initial=ja_salvas,
        unit="img",
        desc="LAION",
        ncols=100,
        file=sys.stdout,
    )
    descartes = 0

    try:
        linhas = iter_rows()
        for linha in linhas:
            if barra.n >= target:
                break

            url = extract_url(linha if isinstance(linha, dict) else {})
            if not url:
                descartes += 1
                continue

            caminho = RAW_DIR / f"laion_{indice:03d}.jpg"
            try:
                conteudo = download_bytes(session, url)
                imagem = decode_image(conteudo)
                save_jpeg(imagem, caminho)
            except RequestException:
                descartes += 1
                continue
            except (UnidentifiedImageError, OSError, ValueError):
                descartes += 1
                caminho.unlink(missing_ok=True)
                continue
            except Exception as erro:  # noqa: BLE001 - uma URL ruim não pode parar a coleta
                logger.debug("Linha ignorada (%s): %s", url[:80], erro)
                descartes += 1
                caminho.unlink(missing_ok=True)
                continue

            indice += 1
            barra.update(1)
            barra.set_postfix(descartadas=descartes, refresh=False)

            if barra.n >= target:
                break
        else:
            logger.warning(
                "O stream terminou com %d/%d imagem(ns) válidas.", barra.n, target
            )
            return 1
    except KeyboardInterrupt:
        logger.warning("Download interrompido. Imagens já salvas foram mantidas.")
        return 1
    except Exception as erro:
        mensagem = str(erro).lower()
        if any(trecho in mensagem for trecho in ("gated", "401", "403", "unauthorized")):
            logger.error(
                "Não foi possível ler %s. Aceite os termos do dataset e autentique "
                "com `huggingface-cli login`. Detalhe: %s",
                DATASET_ID,
                erro,
            )
            return 1
        logger.exception("Falha inesperada na leitura do dataset: %s", erro)
        return 1
    finally:
        barra.close()
        session.close()

    logger.info("Meta atingida: %d imagens em %s", target, RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
