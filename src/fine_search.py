"""Download sob demanda das Top-50 e verificação geométrica SIFT+RANSAC."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import requests

from src.vector_index import ResultadoGrosso

SIFT_FEATURES: Final[int] = 500
LOWE_RATIO: Final[float] = 0.75
RANSAC_REPROJ: Final[float] = 5.0
TIMEOUT_S: Final[float] = 3.0
MAX_BYTES: Final[int] = 20 * 1024 * 1024
CABECALHO: Final[dict[str, str]] = {
    "User-Agent": "art-scraping-detector/tcc (IFMG Sabara; academic use)",
}


@dataclass(frozen=True, slots=True)
class ScoreGeometrico:
    """Laudo fino de um candidato."""

    url: str
    sample_id: str
    score_clip: float
    matches_lowe: int
    inliers: int
    score: float
    caminho: Path | None
    erro: str | None = None


def _decodificar(buffer: bytes) -> np.ndarray | None:
    """Abre JPEG/PNG a partir de bytes, sem caminho no OpenCV."""
    if not buffer:
        return None
    array = np.frombuffer(buffer, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _ler_local(caminho: Path) -> np.ndarray:
    """Lê consulta local mesmo com acento no Windows."""
    try:
        bruto = np.fromfile(caminho, dtype=np.uint8)
    except OSError as erro:
        raise ValueError(f"não foi possível ler {caminho}") from erro
    imagem = cv2.imdecode(bruto, cv2.IMREAD_COLOR) if bruto.size else None
    if imagem is None:
        raise ValueError(f"consulta ilegível: {caminho}")
    return imagem


def resolver_fonte(url: str) -> Path | None:
    """Aceita caminho local, file:// ou URL http(s)."""
    if not url:
        return None
    if url.startswith("file://"):
        local = Path(url.replace("file:///", "").replace("file://", ""))
        return local if local.is_file() else None
    candidato = Path(url)
    if candidato.is_file():
        return candidato
    return None


def baixar_uma(url: str, destino: Path, timeout: float = TIMEOUT_S) -> Path | None:
    """Baixa uma URL. 404, timeout e corpo vazio devolvem None."""
    if not url.startswith(("http://", "https://")):
        return None
    try:
        resposta = requests.get(url, timeout=timeout, headers=CABECALHO, stream=True)
        if resposta.status_code != 200:
            return None
        tamanho = 0
        partes: list[bytes] = []
        for pedaco in resposta.iter_content(64 * 1024):
            tamanho += len(pedaco)
            if tamanho > MAX_BYTES:
                return None
            partes.append(pedaco)
        corpo = b"".join(partes)
        if _decodificar(corpo) is None:
            return None
        destino.write_bytes(corpo)
        return destino
    except (requests.RequestException, OSError):
        return None


def baixar_candidatos(
    candidatos: list[ResultadoGrosso],
    pasta: Path,
    trabalhadores: int = 12,
) -> dict[str, Path]:
    """Baixa as Top-K em paralelo. URLs mortas ficam de fora."""
    pasta.mkdir(parents=True, exist_ok=True)
    ok: dict[str, Path] = {}
    remotos: list[ResultadoGrosso] = []
    for item in candidatos:
        local = resolver_fonte(item.url)
        if local is not None:
            ok[item.url] = local
        elif item.url:
            remotos.append(item)
    if not remotos:
        return ok
    with ThreadPoolExecutor(max_workers=trabalhadores) as pool:
        futuros = {
            pool.submit(baixar_uma, item.url, pasta / f"{item.indice:05d}.jpg"): item
            for item in remotos
        }
        for futuro in as_completed(futuros):
            item = futuros[futuro]
            try:
                caminho = futuro.result()
            except Exception:
                continue
            if caminho is not None:
                ok[item.url] = caminho
    return ok


class SiftVerifier:
    """Cruza a consulta com cada candidato: Lowe 0,75 e homografia RANSAC."""

    def __init__(self, nfeatures: int = SIFT_FEATURES, ratio: float = LOWE_RATIO) -> None:
        self._sift = cv2.SIFT_create(nfeatures=nfeatures)
        self._ratio = ratio
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50),
        )

    def _descritores(self, imagem: np.ndarray) -> tuple[list, np.ndarray | None]:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
        return self._sift.detectAndCompute(cinza, None)

    def verificar(self, consulta: np.ndarray, candidato: np.ndarray) -> tuple[int, int]:
        """Devolve (matches do Lowe, inliers da homografia)."""
        pontos_q, desc_q = self._descritores(consulta)
        pontos_c, desc_c = self._descritores(candidato)
        if desc_q is None or desc_c is None or len(desc_q) < 2 or len(desc_c) < 2:
            return 0, 0
        try:
            pares = self._matcher.knnMatch(
                np.asarray(desc_q, dtype=np.float32),
                np.asarray(desc_c, dtype=np.float32),
                k=2,
            )
        except cv2.error:
            return 0, 0

        bons: list[cv2.DMatch] = []
        for par in pares:
            if len(par) < 2:
                continue
            melhor, segundo = par
            if melhor.distance < self._ratio * segundo.distance:
                bons.append(melhor)
        if len(bons) < 4:
            return len(bons), 0

        origem = np.float32([pontos_q[m.queryIdx].pt for m in bons])
        destino = np.float32([pontos_c[m.trainIdx].pt for m in bons])
        _homografia, mascara = cv2.findHomography(
            origem,
            destino,
            cv2.RANSAC,
            RANSAC_REPROJ,
        )
        if mascara is None:
            return len(bons), 0
        return len(bons), int(mascara.ravel().sum())

    def ranquear(
        self,
        consulta: Path,
        candidatos: list[ResultadoGrosso],
        baixados: dict[str, Path],
    ) -> list[ScoreGeometrico]:
        """Reordena os candidatos pelo número de inliers."""
        try:
            imagem_q = _ler_local(consulta)
        except ValueError as erro:
            raise RuntimeError(str(erro)) from erro

        laudos: list[ScoreGeometrico] = []
        for item in candidatos:
            caminho = baixados.get(item.url)
            if caminho is None:
                laudos.append(
                    ScoreGeometrico(
                        url=item.url,
                        sample_id=item.sample_id,
                        score_clip=item.score,
                        matches_lowe=0,
                        inliers=0,
                        score=0.0,
                        caminho=None,
                        erro="download falhou",
                    )
                )
                continue
            imagem_c = _decodificar(caminho.read_bytes())
            if imagem_c is None:
                laudos.append(
                    ScoreGeometrico(
                        url=item.url,
                        sample_id=item.sample_id,
                        score_clip=item.score,
                        matches_lowe=0,
                        inliers=0,
                        score=0.0,
                        caminho=caminho,
                        erro="arquivo ilegível",
                    )
                )
                continue
            matches, inliers = self.verificar(imagem_q, imagem_c)
            laudos.append(
                ScoreGeometrico(
                    url=item.url,
                    sample_id=item.sample_id,
                    score_clip=item.score,
                    matches_lowe=matches,
                    inliers=inliers,
                    score=float(inliers),
                    caminho=caminho,
                )
            )
        laudos.sort(key=lambda item: (item.inliers, item.matches_lowe, item.score_clip), reverse=True)
        return laudos


def lupa(
    consulta: Path,
    candidatos: list[ResultadoGrosso],
    trabalhadores: int = 12,
) -> list[ScoreGeometrico]:
    """Baixa as Top-K e devolve o ranking geométrico."""
    with tempfile.TemporaryDirectory(prefix="laion-fine-") as tmp:
        baixados = baixar_candidatos(candidatos, Path(tmp), trabalhadores=trabalhadores)
        return SiftVerifier().ranquear(consulta, candidatos, baixados)
