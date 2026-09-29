"""Degradação fixa das imagens da Amostra de Controle.

Simula o que costuma acontecer com uma obra depois de circular na internet:
redimensionamento, corte das bordas, ruído e recompressão JPEG. O nome do
arquivo é preservado para que a versão em ``data/adulterated/`` sirva de
gabarito na avaliação.

Uso:
    python -m src.adulterator

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import logging
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("adulterator")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ADULTERATED_DIR = PROJECT_ROOT / "data" / "adulterated"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Pipeline fixo. Os valores não variam entre imagens, para que a comparação
# entre SSIM, SIFT e CLIP meça o algoritmo e não a intensidade da degradação.
SCALE = 0.80
CROP_RATIO = 0.05
NOISE_SIGMA = 8.0
JPEG_QUALITY = 35


def setup_logging() -> None:
    """Configura o log no terminal, com UTF-8 no console do Windows."""
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


def read_image(caminho: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Lê uma imagem sem passar o caminho ao OpenCV.

    No Windows, ``cv2.imread`` falha quando o caminho contém acentos, como em
    ``Área de Trabalho``.
    """
    try:
        buffer = np.fromfile(caminho, dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def write_image(caminho: Path, imagem: np.ndarray) -> bool:
    """Grava uma imagem em caminho com caracteres não ASCII."""
    extensao = caminho.suffix if caminho.suffix else ".jpg"
    sucesso, buffer = cv2.imencode(extensao, imagem)
    if not sucesso:
        return False
    try:
        buffer.tofile(caminho)
    except OSError:
        return False
    return True


def list_images(diretorio: Path) -> list[Path]:
    """Lista imagens do diretório em ordem estável."""
    if not diretorio.exists():
        return []
    return sorted(
        caminho
        for caminho in diretorio.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in IMAGE_EXTENSIONS
    )


def adulterate(image: np.ndarray, nome: str) -> np.ndarray:
    """Aplica o pipeline de degradação e devolve a imagem em BGR uint8.

    A semente do ruído depende só do nome do arquivo, então a mesma original
    produz sempre a mesma versão adulterada.
    """
    if image is None or image.size == 0:
        raise ValueError("imagem vazia")

    altura, largura = image.shape[:2]
    novo_tamanho = (max(int(largura * SCALE), 1), max(int(altura * SCALE), 1))
    redimensionada = cv2.resize(image, novo_tamanho, interpolation=cv2.INTER_AREA)

    altura, largura = redimensionada.shape[:2]
    margem_y = int(altura * CROP_RATIO)
    margem_x = int(largura * CROP_RATIO)
    if altura - 2 * margem_y >= 16 and largura - 2 * margem_x >= 16:
        recortada = redimensionada[margem_y : altura - margem_y, margem_x : largura - margem_x]
    else:
        recortada = redimensionada

    semente = zlib.crc32(nome.encode("utf-8"))
    gerador = np.random.default_rng(semente)
    ruido = gerador.normal(0.0, NOISE_SIGMA, recortada.shape).astype(np.float32)
    com_ruido = np.clip(recortada.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    sucesso, buffer = cv2.imencode(
        ".jpg",
        com_ruido,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not sucesso:
        raise ValueError("falha na compressão JPEG")

    comprimida = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if comprimida is None:
        raise ValueError("JPEG adulterado ilegível")
    return comprimida


def adulterate_all(
    origem: Path = RAW_DIR,
    destino: Path = ADULTERATED_DIR,
) -> int:
    """Adultera todas as imagens de ``origem`` e grava em ``destino``.

    Returns:
        Quantidade de arquivos gravados com sucesso.
    """
    setup_logging()
    destino.mkdir(parents=True, exist_ok=True)
    imagens = list_images(origem)
    if not imagens:
        logger.error("Nenhuma imagem encontrada em %s", origem)
        return 0

    logger.info("Degradação: escala %.0f%%, corte %.0f%%, ruído σ=%.0f, JPEG %d",
                SCALE * 100, CROP_RATIO * 100, NOISE_SIGMA, JPEG_QUALITY)
    logger.info("Origem: %s", origem)
    logger.info("Destino: %s", destino)

    gravadas = 0
    for caminho in tqdm(imagens, desc="Adulterando", unit="img", file=sys.stdout):
        try:
            imagem = read_image(caminho)
            if imagem is None:
                logger.warning("Não foi possível ler %s", caminho.name)
                continue
            resultado = adulterate(imagem, caminho.name)
            saida = destino / caminho.name
            if not write_image(saida, resultado):
                logger.warning("Falha ao gravar %s", saida.name)
                continue
            gravadas += 1
        except (ValueError, cv2.error, OSError) as erro:
            logger.warning("Imagem ignorada (%s): %s", caminho.name, erro)

    logger.info("Versões adulteradas gravadas: %d / %d", gravadas, len(imagens))
    return gravadas


if __name__ == "__main__":
    raise SystemExit(0 if adulterate_all() else 1)
