"""Roteador zero-shot: categoria visual e se a lupa SIFT vale a pena."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

CLIP_MODELO: Final[str] = "openai/clip-vit-base-patch32"

CATEGORIAS: Final[tuple[str, ...]] = (
    "a realistic photograph with natural textures",
    "a textured digital painting",
    "a flat vector illustration",
    "a logo or typography design",
    "a hybrid digital collage",
)

# Fotografia, pintura e colagem têm textura local que o SIFT aproveita.
# Vetor e logo são planos: o CLIP já basta no Recall@50.
REQUER_SIFT: Final[dict[str, bool]] = {
    "a realistic photograph with natural textures": True,
    "a textured digital painting": True,
    "a hybrid digital collage": True,
    "a flat vector illustration": False,
    "a logo or typography design": False,
}

ROTULO: Final[dict[str, str]] = {
    "a realistic photograph with natural textures": "fotografia",
    "a textured digital painting": "pintura digital",
    "a flat vector illustration": "ilustração vetorial",
    "a logo or typography design": "logo ou tipografia",
    "a hybrid digital collage": "colagem híbrida",
}


@dataclass(frozen=True, slots=True)
class Roteamento:
    """Saída do roteador: tag, se roda SIFT, e o embedding da consulta."""

    categoria: str
    tag: str
    requires_sift: bool
    confianca: float
    embedding: np.ndarray


class VisualRouter:
    """CLIP ViT-B/32 para zero-shot e para o vetor da busca grossa."""

    def __init__(self, modelo_id: str = CLIP_MODELO) -> None:
        self._modelo_id = modelo_id
        self._dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processador = None
        self._modelo = None

    def _carregar(self) -> None:
        if self._modelo is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processador = CLIPProcessor.from_pretrained(self._modelo_id)
        self._modelo = CLIPModel.from_pretrained(self._modelo_id)
        self._modelo.to(self._dispositivo)
        self._modelo.eval()

    def _abrir(self, consulta: Path | Image.Image) -> Image.Image:
        if isinstance(consulta, Image.Image):
            return consulta.convert("RGB")
        try:
            return Image.open(consulta).convert("RGB")
        except (OSError, UnidentifiedImageError) as erro:
            raise ValueError(f"consulta ilegível: {consulta}") from erro

    def _embedding(self, imagem: Image.Image) -> np.ndarray:
        assert self._modelo is not None and self._processador is not None
        entradas = self._processador(images=imagem, return_tensors="pt")
        entradas = {chave: valor.to(self._dispositivo) for chave, valor in entradas.items()}
        with torch.inference_mode():
            saida = self._modelo.get_image_features(**entradas)
            if hasattr(saida, "image_embeds") and saida.image_embeds is not None:
                vetor = saida.image_embeds
            elif hasattr(saida, "pooler_output") and saida.pooler_output is not None:
                vetor = saida.pooler_output
            else:
                vetor = saida
            vetor = torch.nn.functional.normalize(vetor, dim=-1)
        return vetor.detach().cpu().numpy().astype(np.float32)

    def classificar(self, consulta: Path | Image.Image) -> Roteamento:
        """Roda o zero-shot nas cinco categorias e extrai o embedding."""
        self._carregar()
        assert self._modelo is not None and self._processador is not None
        imagem = self._abrir(consulta)
        entradas = self._processador(
            text=list(CATEGORIAS),
            images=imagem,
            return_tensors="pt",
            padding=True,
        )
        entradas = {chave: valor.to(self._dispositivo) for chave, valor in entradas.items()}
        with torch.inference_mode():
            saida = self._modelo(**entradas)
            probs = saida.logits_per_image.softmax(dim=-1)[0]
        vencedor = int(torch.argmax(probs).item())
        categoria = CATEGORIAS[vencedor]
        return Roteamento(
            categoria=categoria,
            tag=ROTULO[categoria],
            requires_sift=REQUER_SIFT[categoria],
            confianca=float(probs[vencedor].item()),
            embedding=self._embedding(imagem),
        )
