"""Índice FAISS da busca grossa (Recall@K por cosseno).

Os vetores entram L2-normalizados. Depois disso o produto interno é o
cosseno. Em galerias pequenas o índice é exato (``IndexFlatIP``). Em
escala, IVF-Flat ou HNSW.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import faiss
except ImportError as erro:  # pragma: no cover - dependência opcional até o pip
    raise ImportError(
        "faiss-cpu não está instalado. Rode: pip install faiss-cpu pyarrow"
    ) from erro

from src.data_ingestion import LoteIngestao, iterar_lotes

BackendIndice = Literal["auto", "flat", "ivf", "hnsw"]


@dataclass(frozen=True, slots=True)
class ResultadoGrosso:
    """Uma URL candidata da busca aproximada."""

    url: str
    sample_id: str
    score: float
    indice: int


def _normalizar_l2(matriz: np.ndarray) -> np.ndarray:
    """Normaliza linhas in-place quando possível e devolve float32 C-contígua."""
    vetores = np.ascontiguousarray(matriz, dtype=np.float32)
    if vetores.ndim == 1:
        vetores = vetores.reshape(1, -1)
    faiss.normalize_L2(vetores)
    return vetores


class FaissIndexer:
    """Índice de cosseno com persistência do mapa vetor → URL."""

    def __init__(
        self,
        dim: int = 512,
        backend: BackendIndice = "auto",
        nlist: int = 4096,
        nprobe: int = 16,
        hnsw_m: int = 32,
        hnsw_ef: int = 64,
    ) -> None:
        self.dim = dim
        self.backend = backend
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef = hnsw_ef
        self.indice: faiss.Index | None = None
        self.urls: list[str] = []
        self.sample_ids: list[str] = []

    @property
    def total(self) -> int:
        """Quantos vetores já foram inseridos."""
        return 0 if self.indice is None else int(self.indice.ntotal)

    def _escolher_backend(self, n_treino: int) -> str:
        if self.backend != "auto":
            return self.backend
        if n_treino < 8_192:
            return "flat"
        if n_treino < 200_000:
            return "hnsw"
        return "ivf"

    def _construir_indice(self, n_treino: int) -> faiss.Index:
        escolhido = self._escolher_backend(n_treino)
        if escolhido == "flat":
            return faiss.IndexFlatIP(self.dim)
        if escolhido == "hnsw":
            indice = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            indice.hnsw.efSearch = self.hnsw_ef
            indice.hnsw.efConstruction = max(self.hnsw_ef, 80)
            return indice
        nlist = min(self.nlist, max(16, n_treino // 39))
        quantizador = faiss.IndexFlatIP(self.dim)
        indice = faiss.IndexIVFFlat(quantizador, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        return indice

    def treinar(self, embeddings: np.ndarray) -> None:
        """Cria o índice e, se for IVF, treina os centróides."""
        vetores = _normalizar_l2(embeddings)
        if vetores.shape[1] != self.dim:
            raise ValueError(f"dimensão {vetores.shape[1]} ≠ {self.dim}")
        self.indice = self._construir_indice(vetores.shape[0])
        if isinstance(self.indice, faiss.IndexIVFFlat):
            self.indice.train(vetores)
            self.indice.nprobe = min(self.nprobe, self.indice.nlist)

    def adicionar(self, embeddings: np.ndarray, urls: list[str], sample_ids: list[str] | None = None) -> None:
        """Insere um lote já alinhado com as URLs."""
        if self.indice is None:
            self.treinar(embeddings)
        vetores = _normalizar_l2(embeddings)
        if vetores.shape[0] != len(urls):
            raise ValueError("número de vetores e URLs não bate")
        if not self.indice.is_trained:
            self.indice.train(vetores)
        self.indice.add(vetores)
        self.urls.extend(urls)
        self.sample_ids.extend(sample_ids if sample_ids is not None else [""] * len(urls))

    def adicionar_lote(self, lote: LoteIngestao) -> None:
        """Atalho para o iterador de ingestão."""
        self.adicionar(lote.embeddings, lote.urls, lote.sample_ids)

    def search(self, query_embedding: np.ndarray, k: int = 50) -> list[ResultadoGrosso]:
        """Devolve as Top-K URLs por cosseno aproximado."""
        if self.indice is None or self.total == 0:
            raise RuntimeError("índice FAISS vazio")
        consulta = _normalizar_l2(query_embedding)
        k = min(k, self.total)
        if isinstance(self.indice, faiss.IndexIVFFlat):
            self.indice.nprobe = min(self.nprobe, self.indice.nlist)
        scores, indices = self.indice.search(consulta, k)
        resultados: list[ResultadoGrosso] = []
        for score, posicao in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
            if posicao < 0:
                continue
            resultados.append(
                ResultadoGrosso(
                    url=self.urls[posicao],
                    sample_id=self.sample_ids[posicao] if posicao < len(self.sample_ids) else "",
                    score=float(score),
                    indice=int(posicao),
                )
            )
        return resultados

    def salvar(self, destino: Path) -> None:
        """Grava ``destino.faiss`` e o mapa ``destino.mapeamento.json``."""
        if self.indice is None:
            raise RuntimeError("nada para salvar")
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.indice, str(destino.with_suffix(".faiss")))
        mapa = {
            "dim": self.dim,
            "backend": self.backend,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "total": self.total,
            "urls": self.urls,
            "sample_ids": self.sample_ids,
        }
        destino.with_name(destino.stem + ".mapeamento.json").write_text(
            json.dumps(mapa, ensure_ascii=False),
            encoding="utf-8",
        )

    def carregar(self, origem: Path) -> None:
        """Recarrega o índice e o dicionário ID → URL."""
        origem = Path(origem)
        caminho_indice = origem if origem.suffix == ".faiss" else origem.with_suffix(".faiss")
        caminho_mapa = origem.with_name(origem.stem + ".mapeamento.json")
        if not caminho_indice.is_file():
            raise FileNotFoundError(caminho_indice)
        self.indice = faiss.read_index(str(caminho_indice))
        if isinstance(self.indice, faiss.IndexIVFFlat):
            self.indice.nprobe = min(self.nprobe, self.indice.nlist)
        if caminho_mapa.is_file():
            mapa = json.loads(caminho_mapa.read_text(encoding="utf-8"))
            self.urls = list(mapa.get("urls", []))
            self.sample_ids = list(mapa.get("sample_ids", []))
            self.dim = int(mapa.get("dim", self.dim))
        if self.total != len(self.urls):
            raise RuntimeError(
                f"mapa ({len(self.urls)}) e índice ({self.total}) desalinhados"
            )


def construir_indice(
    metadados: Path,
    saida: Path,
    embeddings: Path | None = None,
    backend: BackendIndice = "auto",
) -> FaissIndexer:
    """Constrói o índice em uma passagem pelos lotes, sem juntar o corpus."""
    indexador = FaissIndexer(backend=backend)
    for lote in iterar_lotes(metadados, embeddings):
        indexador.adicionar_lote(lote)
    indexador.salvar(saida)
    return indexador


def indice_da_amostra(
    raw_dir: Path,
    nomes_json: Path,
    vetores_bin: Path,
    dim: int = 512,
) -> FaissIndexer:
    """Monta um FAISS Flat a partir do índice CLIP já medido nas 500."""
    nomes = json.loads(nomes_json.read_text(encoding="utf-8"))
    vetores = np.fromfile(vetores_bin, dtype=np.float32)
    if vetores.size != len(nomes) * dim:
        raise ValueError("clip_vetores.bin não bate com clip_nomes.json")
    matriz = vetores.reshape(len(nomes), dim)
    urls: list[str] = []
    ids: list[str] = []
    for nome in nomes:
        caminho = (raw_dir / nome).resolve()
        if not caminho.is_file():
            raise FileNotFoundError(caminho)
        urls.append(str(caminho))
        ids.append(nome)
    indexador = FaissIndexer(dim=dim, backend="flat")
    indexador.adicionar(matriz, urls, ids)
    return indexador
