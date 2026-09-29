"""Comparadores de similaridade visual usados na prova de conceito.

Cada comparador indexa a galeria (``data/raw``) uma única vez e, na busca,
devolve o arquivo da galeria mais parecido com a consulta. O tempo medido pelo
orquestrador é o de ``buscar``, não o da indexação.

- SSIM: similaridade estrutural em imagens normalizadas para o mesmo tamanho.
- SIFT: descritores locais e ``FlannBasedMatcher``, com voto do Lowe ratio test.
- CLIP: embeddings da imagem e similaridade de cosseno.

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

SSIM_SIZE = 128
SIFT_FEATURES = 500
SIFT_RATIO = 0.75
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_BATCH = 16


@dataclass(frozen=True, slots=True)
class ResultadoBusca:
    """Correspondência encontrada na galeria."""

    filename: str
    score: float
    caminho: Path | None = None


def _ler(caminho: Path, flags: int) -> np.ndarray:
    """Lê uma imagem sem passar o caminho ao OpenCV.

    ``cv2.imread`` não abre arquivos cujo caminho tem acentos no Windows.
    """
    try:
        buffer = np.fromfile(caminho, dtype=np.uint8)
    except OSError as erro:
        raise ValueError(f"não foi possível ler {caminho.name}") from erro
    imagem = cv2.imdecode(buffer, flags) if buffer.size else None
    if imagem is None:
        raise ValueError(f"não foi possível ler {caminho.name}")
    return imagem


def _ler_bgr(caminho: Path) -> np.ndarray:
    """Lê uma imagem colorida."""
    return _ler(caminho, cv2.IMREAD_COLOR)


def _cinza_normalizado(caminho: Path, tamanho: int = SSIM_SIZE) -> np.ndarray:
    """Converte a imagem para cinza float32 no intervalo [0, 1], em tamanho fixo."""
    cinza = _ler(caminho, cv2.IMREAD_GRAYSCALE)
    cinza = cv2.resize(cinza, (tamanho, tamanho), interpolation=cv2.INTER_AREA)
    return cinza.astype(np.float32) / 255.0


class ComparadorSSIM:
    """Busca pela maior similaridade estrutural (SSIM)."""

    nome = "SSIM"

    def __init__(self) -> None:
        self._nomes: list[str] = []
        self._galeria: list[np.ndarray] = []

    def indexar(self, caminhos: list[Path]) -> None:
        """Pré-processa a galeria para o tamanho usado na comparação."""
        self._nomes = [caminho.name for caminho in caminhos]
        self._galeria = [_cinza_normalizado(caminho) for caminho in caminhos]

    def buscar(self, consulta: Path) -> ResultadoBusca:
        """Devolve a imagem da galeria com maior SSIM em relação à consulta."""
        if not self._galeria:
            raise RuntimeError("galeria SSIM não indexada")

        query = _cinza_normalizado(consulta)
        melhor_indice = 0
        melhor_score = -1.0
        for indice, referencia in enumerate(self._galeria):
            score = float(structural_similarity(query, referencia, data_range=1.0))
            if score > melhor_score:
                melhor_score = score
                melhor_indice = indice
        return ResultadoBusca(self._nomes[melhor_indice], melhor_score)


class ComparadorSIFT:
    """Busca por descritores SIFT com FlannBasedMatcher.

    Os descritores da galeria entram num único índice FLANN. Cada descritor da
    consulta passa pelo teste de razão de Lowe e vota na imagem de origem do
    vizinho mais próximo. A imagem com mais votos é o Top 1.
    """

    nome = "SIFT"

    def __init__(self) -> None:
        self._sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50),
        )
        self._nomes: list[str] = []
        self._donos: np.ndarray | None = None
        self._indexado = False

    def indexar(self, caminhos: list[Path]) -> None:
        """Extrai os descritores da galeria e treina o índice FLANN."""
        self._nomes = [caminho.name for caminho in caminhos]
        blocos: list[np.ndarray] = []
        donos: list[np.ndarray] = []

        for indice, caminho in enumerate(caminhos):
            imagem = _ler_bgr(caminho)
            _pontos, descritores = self._sift.detectAndCompute(imagem, None)
            if descritores is None or len(descritores) < 2:
                continue
            blocos.append(np.asarray(descritores, dtype=np.float32))
            donos.append(np.full(len(descritores), indice, dtype=np.int32))

        if not blocos:
            raise RuntimeError("nenhuma imagem da galeria produziu descritores SIFT")

        treino = np.vstack(blocos)
        self._donos = np.concatenate(donos)
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50),
        )
        self._matcher.add([treino])
        self._matcher.train()
        self._indexado = True

    def buscar(self, consulta: Path) -> ResultadoBusca:
        """Conta os votos de correspondência e devolve a imagem mais votada."""
        if not self._indexado or self._donos is None:
            raise RuntimeError("galeria SIFT não indexada")

        imagem = _ler_bgr(consulta)
        _pontos, descritores = self._sift.detectAndCompute(imagem, None)
        if descritores is None or len(descritores) < 2:
            return ResultadoBusca(self._nomes[0], 0.0)

        pares = self._matcher.knnMatch(np.asarray(descritores, dtype=np.float32), k=2)
        votos = np.zeros(len(self._nomes), dtype=np.int32)
        for par in pares:
            if len(par) < 2:
                continue
            melhor, segundo = par
            if melhor.distance < SIFT_RATIO * segundo.distance:
                votos[self._donos[melhor.trainIdx]] += 1

        indice = int(np.argmax(votos))
        return ResultadoBusca(self._nomes[indice], float(votos[indice]))


class ComparadorCLIP:
    """Busca pela maior similaridade de cosseno entre embeddings CLIP."""

    nome = "CLIP"

    def __init__(self, modelo_id: str = CLIP_MODEL_ID) -> None:
        self._modelo_id = modelo_id
        self._dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._modelo = None
        self._processador = None
        self._nomes: list[str] = []
        self._caminhos: list[Path] = []
        self._galeria: torch.Tensor | None = None

    def _carregar(self) -> None:
        """Carrega o modelo só na indexação, para não atrasar os outros métodos."""
        if self._modelo is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processador = CLIPProcessor.from_pretrained(self._modelo_id)
        self._modelo = CLIPModel.from_pretrained(self._modelo_id)
        self._modelo.to(self._dispositivo)
        self._modelo.eval()

    def _vetorizar(self, imagens: list) -> torch.Tensor:
        """Gera embeddings L2-normalizados para imagens PIL já em RGB."""
        vetores: list[torch.Tensor] = []
        with torch.inference_mode():
            for inicio in range(0, len(imagens), CLIP_BATCH):
                lote = imagens[inicio : inicio + CLIP_BATCH]
                entradas = self._processador(images=lote, return_tensors="pt")
                entradas = {
                    chave: valor.to(self._dispositivo) for chave, valor in entradas.items()
                }
                saida = self._modelo.get_image_features(**entradas)
                if hasattr(saida, "image_embeds") and saida.image_embeds is not None:
                    vetor = saida.image_embeds
                elif hasattr(saida, "pooler_output") and saida.pooler_output is not None:
                    vetor = saida.pooler_output
                else:
                    vetor = saida
                vetor = torch.nn.functional.normalize(vetor, dim=-1)
                vetores.append(vetor.cpu())
        return torch.cat(vetores, dim=0)

    def _embeddings(self, caminhos: list[Path]) -> torch.Tensor:
        """Gera embeddings L2-normalizados para uma lista de arquivos."""
        from PIL import Image

        imagens = [Image.open(caminho).convert("RGB") for caminho in caminhos]
        try:
            return self._vetorizar(imagens)
        finally:
            for imagem in imagens:
                imagem.close()

    def indexar(self, caminhos: list[Path]) -> None:
        """Calcula e guarda os embeddings normalizados da galeria."""
        self._carregar()
        self._caminhos = list(caminhos)
        self._nomes = [caminho.name for caminho in caminhos]
        self._galeria = self._embeddings(caminhos)

    def _rankear(self, consulta_emb: torch.Tensor, k: int) -> list[ResultadoBusca]:
        """Ordena a galeria pela similaridade de cosseno e devolve os k primeiros."""
        scores = torch.mv(self._galeria, consulta_emb[0])
        k = min(k, scores.shape[0])
        valores, indices = torch.topk(scores, k)
        return [
            ResultadoBusca(
                self._nomes[indice],
                float(valor),
                self._caminhos[indice],
            )
            for valor, indice in zip(valores.tolist(), indices.tolist(), strict=True)
        ]

    def buscar(self, consulta: Path) -> ResultadoBusca:
        """Devolve a imagem da galeria com maior cosseno em relação à consulta."""
        return self.buscar_topk(consulta, k=1)[0]

    def buscar_topk(self, consulta: Path, k: int = 3) -> list[ResultadoBusca]:
        """Devolve as k imagens da galeria mais próximas de um arquivo."""
        if self._galeria is None:
            raise RuntimeError("galeria CLIP não indexada")
        return self._rankear(self._embeddings([consulta]), k)

    def buscar_topk_imagem(self, imagem, k: int = 3) -> list[ResultadoBusca]:
        """Devolve as k imagens da galeria mais próximas de uma imagem PIL."""
        if self._galeria is None:
            raise RuntimeError("galeria CLIP não indexada")
        return self._rankear(self._vetorizar([imagem.convert("RGB")]), k)
