"""Ingestão em lotes dos metadados e embeddings do LAION-Aesthetics.

Os parquets do subset trazem URL e texto. Os vetores CLIP costumam vir em
arquivos à parte. Nada aqui materializa o corpus inteiro na RAM: o teto
assumido é 16 GB, então cada lote cabe em poucos gigabytes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUNAS_ID: Final[tuple[str, ...]] = ("SAMPLE_ID", "sample_id", "key", "hash", "id")
COLUNAS_URL: Final[tuple[str, ...]] = ("URL", "url", "image_url")
COLUNAS_TEXTO: Final[tuple[str, ...]] = ("TEXT", "text", "caption")
COLUNAS_VETOR: Final[tuple[str, ...]] = ("embedding", "clip", "vector", "EMBEDDING")

BYTES_POR_FLOAT32: Final[int] = 4
RAM_TETO_BYTES: Final[int] = 16 * 1024**3
RAM_RESERVA_BYTES: Final[int] = 4 * 1024**3
DIM_CLIP: Final[int] = 512


@dataclass(frozen=True, slots=True)
class LoteIngestao:
    """Um pedaço alinhado de metadados e vetores."""

    sample_ids: list[str]
    urls: list[str]
    textos: list[str]
    embeddings: np.ndarray


def _escolher_coluna(colunas: list[str], candidatas: tuple[str, ...]) -> str | None:
    """Devolve o primeiro nome de coluna que existir."""
    conjunto = set(colunas)
    for nome in candidatas:
        if nome in conjunto:
            return nome
    return None


def _listar_parquets(origem: Path) -> list[Path]:
    """Aceita um arquivo ou uma pasta de ``*.parquet``."""
    if origem.is_file():
        return [origem]
    if origem.is_dir():
        arquivos = sorted(origem.glob("*.parquet"))
        if arquivos:
            return arquivos
    raise FileNotFoundError(f"nenhum parquet em {origem}")


def tamanho_lote_seguro(dim: int = DIM_CLIP, ram_livre: int | None = None) -> int:
    """Quantas linhas cabem num lote sem passar do teto de 16 GB.

    Reserva 4 GB para o SO, o CLIP e o índice FAISS. O resto divide-se
    entre a matriz float32 e uma folga de metadados (~30%).
    """
    disponivel = (ram_livre if ram_livre is not None else RAM_TETO_BYTES) - RAM_RESERVA_BYTES
    if disponivel <= 0:
        raise ValueError("RAM disponível insuficiente para ingestão")
    bytes_por_linha = int(dim * BYTES_POR_FLOAT32 * 1.3)
    return max(8_192, disponivel // bytes_por_linha)


def gerar_embeddings_mock(
    n: int,
    dim: int = DIM_CLIP,
    semente: int = 0,
) -> np.ndarray:
    """Vetores aleatórios L2-normalizados para testes sem o dump oficial."""
    rng = np.random.default_rng(semente)
    matriz = rng.standard_normal((n, dim), dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas = np.clip(normas, 1e-12, None)
    return matriz / normas


def _ler_embeddings_alinhados(
    caminho: Path | None,
    inicio: int,
    n: int,
    dim: int,
    semente_mock: int,
) -> np.ndarray:
    """Lê ``n`` vetores a partir de ``inicio``, ou fabrica um mock."""
    if caminho is None:
        return gerar_embeddings_mock(n, dim=dim, semente=semente_mock + inicio)

    if caminho.suffix == ".npy":
        matriz = np.load(caminho, mmap_mode="r")
        fatia = np.asarray(matriz[inicio : inicio + n], dtype=np.float32)
        if fatia.ndim != 2:
            raise ValueError(f"embeddings em {caminho} devem ser 2D")
        return np.ascontiguousarray(fatia)

    if caminho.suffix in {".npz"}:
        arquivo = np.load(caminho)
        chave = "embeddings" if "embeddings" in arquivo else arquivo.files[0]
        fatia = np.asarray(arquivo[chave][inicio : inicio + n], dtype=np.float32)
        return np.ascontiguousarray(fatia)

    raise ValueError(f"formato de embeddings não suportado: {caminho.suffix}")


def iterar_lotes(
    metadados: Path,
    embeddings: Path | None = None,
    *,
    linhas_por_lote: int | None = None,
    dim: int = DIM_CLIP,
    mock_se_ausente: bool = True,
) -> Iterator[LoteIngestao]:
    """Percorre parquets e vetores em pedaços alinhados.

    ``embeddings`` pode ser um ``.npy``/``.npz`` na mesma ordem das linhas
    dos parquets, ou ``None`` para mock (só desenvolvimento).
    """
    arquivos = _listar_parquets(metadados)
    lote_alvo = linhas_por_lote or tamanho_lote_seguro(dim)
    deslocamento = 0

    for parquet in arquivos:
        try:
            tabela = pq.ParquetFile(parquet)
        except (OSError, ValueError) as erro:
            raise RuntimeError(f"falha ao abrir {parquet}") from erro

        schema = [campo.name for campo in tabela.schema_arrow]
        col_id = _escolher_coluna(schema, COLUNAS_ID)
        col_url = _escolher_coluna(schema, COLUNAS_URL)
        col_texto = _escolher_coluna(schema, COLUNAS_TEXTO)
        col_vetor = _escolher_coluna(schema, COLUNAS_VETOR)
        if col_url is None:
            raise ValueError(f"{parquet} não tem coluna de URL")

        for lote_arrow in tabela.iter_batches(batch_size=lote_alvo):
            quadro = lote_arrow.to_pandas()
            urls = quadro[col_url].fillna("").astype(str).tolist()
            ids = (
                quadro[col_id].astype(str).tolist()
                if col_id is not None
                else [str(deslocamento + i) for i in range(len(urls))]
            )
            textos = (
                quadro[col_texto].fillna("").astype(str).tolist()
                if col_texto is not None
                else [""] * len(urls)
            )

            if col_vetor is not None:
                brutos = list(quadro[col_vetor])
                try:
                    vetores = np.asarray(brutos, dtype=np.float32)
                except (TypeError, ValueError) as erro:
                    raise ValueError(f"coluna {col_vetor} em {parquet} não é numérica") from erro
                if vetores.ndim != 2:
                    raise ValueError("embeddings no parquet devem ser vetores de mesma dimensão")
            elif embeddings is not None or mock_se_ausente:
                vetores = _ler_embeddings_alinhados(
                    embeddings,
                    deslocamento,
                    len(urls),
                    dim,
                    semente_mock=0,
                )
            else:
                raise FileNotFoundError("embeddings ausentes e mock desligado")

            if vetores.shape[0] != len(urls):
                raise ValueError(
                    f"lote desalinhado: {vetores.shape[0]} vetores e {len(urls)} URLs"
                )
            yield LoteIngestao(ids, urls, textos, np.ascontiguousarray(vetores, dtype=np.float32))
            deslocamento += len(urls)


def ler_metadados(origem: Path, colunas: list[str] | None = None) -> pd.DataFrame:
    """Lê um parquet (ou pasta) para inspeção. Não use no corpus cheio."""
    arquivos = _listar_parquets(origem)
    if len(arquivos) == 1:
        return pd.read_parquet(arquivos[0], columns=colunas)
    return pd.concat(
        (pd.read_parquet(arquivo, columns=colunas) for arquivo in arquivos),
        ignore_index=True,
    )
