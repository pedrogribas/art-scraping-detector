"""Estima tempo, memória e acurácia ao sair da amostra de 500 imagens.

Não executa a busca no LAION completo. O tempo parte dos custos medidos
nesta máquina. A acurácia projetada usa um modelo gaussiano dos escores
de imagens que não são o par verdadeiro, calibrado na amostra.

Uso:
    python -m src.estimate_scale
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.adulterator import ADULTERATED_DIR, RAW_DIR, list_images
from src.similarity_evaluator import (
    SIFT_RATIO,
    ComparadorCLIP,
    ComparadorSIFT,
    ComparadorSSIM,
    _cinza_normalizado,
    _ler_bgr,
)

# laion/laion2B-en-aesthetic: subset inglês com aesthetic > 7
# (LAION, laion-aesthetic.md; conferido em 52.068.913 linhas).
N_AESTHETIC_EN = 52_068_913
# Schuhmann et al., LAION-5B (arXiv:2210.08402): 5,85 bilhões de pares.
N_LAION_5B = 5_850_000_000
N_AMOSTRA = 500

ESCALAS = (
    (500, "Amostra (500)"),
    (10_000, "10 mil"),
    (100_000, "100 mil"),
    (1_000_000, "1 milhão"),
    (N_AESTHETIC_EN, "Estético EN (52 mi)"),
    (N_LAION_5B, "LAION-5B (5,85 bi)"),
)


def _phi_acima(z: float) -> float:
    """Cauda à direita de uma normal padrão, 1 - Φ(z)."""
    if z > 8:
        # Aproximação de Mills para não zerar a cauda no float64.
        densidade = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return densidade / z
    return 0.5 * math.erfc(z / math.sqrt(2))


def _acuracia_gaussiana(escores_verdadeiros: np.ndarray, impostores: np.ndarray, n: int) -> float:
    """Fração esperada de Top-1 corretos se os impostores forem i.i.d. normais.

    Para cada consulta, P(acerto) = [Φ((s - μ) / σ)]^(n - 1). A média dessas
    probabilidades é a acurácia estimada. É otimista se o LAION tiver
    quase-duplicatas mais parecidas do que os pares desta amostra.
    """
    media = float(impostores.mean())
    desvio = float(impostores.std())
    if desvio <= 0:
        return 0.0
    probs = []
    for escore in escores_verdadeiros:
        z = (float(escore) - media) / desvio
        cauda = min(max(_phi_acima(z), 0.0), 1.0)
        if n <= 1 or cauda <= 0:
            probs.append(1.0)
            continue
        log_acerto = (n - 1) * math.log1p(-cauda) if cauda < 1 else -math.inf
        probs.append(math.exp(log_acerto) if log_acerto > -700 else 0.0)
    return 100.0 * float(np.mean(probs))


def _alinhar(brutas: list[Path], adulteradas: list[Path]) -> tuple[list[Path], list[Path]]:
    por_nome = {caminho.name: caminho for caminho in adulteradas}
    pares_raw = []
    pares_adv = []
    for caminho in brutas:
        if caminho.name in por_nome:
            pares_raw.append(caminho)
            pares_adv.append(por_nome[caminho.name])
    return pares_raw, pares_adv


def medir_ssim(brutas: list[Path], adulteradas: list[Path]) -> dict:
    comparador = ComparadorSSIM()
    inicio = time.perf_counter()
    comparador.indexar(brutas)
    index_s = time.perf_counter() - inicio

    verdadeiros = []
    impostores = []
    tempos = []
    nomes = comparador._nomes
    for ordem, consulta in enumerate(adulteradas, start=1):
        query = _cinza_normalizado(consulta)
        t0 = time.perf_counter()
        escores = np.empty(len(comparador._galeria), dtype=np.float64)
        for indice, referencia in enumerate(comparador._galeria):
            escores[indice] = structural_similarity(query, referencia, data_range=1.0)
        tempos.append(time.perf_counter() - t0)
        pos = nomes.index(consulta.name)
        verdadeiros.append(escores[pos])
        mascara = np.ones(len(escores), dtype=bool)
        mascara[pos] = False
        impostores.append(escores[mascara])
        if ordem % 50 == 0:
            print(f"  SSIM {ordem}/{len(adulteradas)}", flush=True)
    return {
        "index_s": index_s,
        "busca_s": float(np.mean(tempos)),
        "verdadeiros": np.asarray(verdadeiros),
        "impostores": np.concatenate(impostores),
    }


def medir_sift(brutas: list[Path], adulteradas: list[Path]) -> dict:
    comparador = ComparadorSIFT()
    inicio = time.perf_counter()
    comparador.indexar(brutas)
    index_s = time.perf_counter() - inicio
    descritores = int(comparador._donos.shape[0])

    extracoes = []
    matches = []
    votos_verdade = []
    votos_impostor = []
    for consulta in adulteradas:
        imagem = _ler_bgr(consulta)
        t0 = time.perf_counter()
        _pontos, descritores_q = comparador._sift.detectAndCompute(imagem, None)
        extracoes.append(time.perf_counter() - t0)
        if descritores_q is None or len(descritores_q) < 2:
            continue
        t0 = time.perf_counter()
        pares = comparador._matcher.knnMatch(np.asarray(descritores_q, dtype=np.float32), k=2)
        matches.append(time.perf_counter() - t0)
        votos = np.zeros(len(comparador._nomes), dtype=np.int32)
        for par in pares:
            if len(par) < 2:
                continue
            melhor, segundo = par
            if melhor.distance < SIFT_RATIO * segundo.distance:
                votos[comparador._donos[melhor.trainIdx]] += 1
        pos = comparador._nomes.index(consulta.name)
        votos_verdade.append(int(votos[pos]))
        votos_sem = votos.copy()
        votos_sem[pos] = 0
        votos_impostor.append(int(votos_sem.max()))
    return {
        "index_s": index_s,
        "extracao_s": float(np.mean(extracoes)),
        "match_s": float(np.mean(matches)) if matches else 0.0,
        "descritores": descritores,
        "descritores_por_imagem": descritores / max(len(brutas), 1),
        "votos_verdade": votos_verdade,
        "votos_impostor_max": votos_impostor,
    }


def medir_clip(brutas: list[Path], adulteradas: list[Path]) -> dict:
    comparador = ComparadorCLIP()
    inicio = time.perf_counter()
    comparador.indexar(brutas)
    index_s = time.perf_counter() - inicio
    galeria = comparador._galeria
    dimensao = int(galeria.shape[1])

    inicio = time.perf_counter()
    consulta = comparador._embeddings(adulteradas[:8])
    embed_s = (time.perf_counter() - inicio) / 8

    vetor = consulta[0]
    repeticoes = 30
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        torch.mv(galeria, vetor)
    scan_500_s = (time.perf_counter() - inicio) / repeticoes

    bloco = torch.nn.functional.normalize(torch.randn(100_000, dimensao), dim=-1)
    inicio = time.perf_counter()
    torch.mv(bloco, vetor)
    scan_100k_s = time.perf_counter() - inicio

    embeddings = comparador._embeddings(adulteradas)
    similaridade = embeddings @ galeria.T
    indice = np.arange(similaridade.shape[0])
    verdadeiros = similaridade[indice, indice].numpy()
    mascara = ~np.eye(similaridade.shape[0], dtype=bool)
    impostores = similaridade.numpy()[mascara]
    return {
        "index_s": index_s,
        "embed_s": embed_s,
        "scan_500_s": scan_500_s,
        "scan_por_imagem_s": scan_100k_s / 100_000,
        "dimensao": dimensao,
        "dispositivo": str(comparador._dispositivo),
        "verdadeiros": verdadeiros,
        "impostores": impostores,
    }


def _gb(n_imagens: int, bytes_por_imagem: float) -> float:
    return n_imagens * bytes_por_imagem / (1024**3)


def projetar(ssim: dict, sift: dict, clip: dict) -> dict:
    segundos_ssim = ssim["busca_s"] / N_AMOSTRA
    # O custo fixo do CLIP é o embedding da consulta; o restante cresce com a galeria.
    segundos_clip = clip["scan_por_imagem_s"]
    # FLANN com checks fixos: o casamento cresce com log2 do número de descritores.
    descritores_base = sift["descritores"]
    por_imagem = sift["descritores_por_imagem"]

    def busca_sift(n: int) -> float:
        descritores = max(por_imagem * n, 2)
        fator = math.log2(descritores) / math.log2(descritores_base)
        return sift["extracao_s"] + sift["match_s"] * fator

    linhas = []
    for n, rotulo in ESCALAS:
        linhas.append(
            {
                "n": n,
                "rotulo": rotulo,
                "ssim_busca_s": segundos_ssim * n,
                "sift_busca_s": busca_sift(n),
                "clip_busca_s": clip["embed_s"] + segundos_clip * n,
                "ssim_acuracia": _acuracia_gaussiana(ssim["verdadeiros"], ssim["impostores"], n),
                "clip_acuracia": _acuracia_gaussiana(clip["verdadeiros"], clip["impostores"], n),
                "ssim_memoria_gb": _gb(n, 128 * 128 * 4),
                "sift_memoria_gb": _gb(n, por_imagem * 128 * 4),
                "clip_memoria_gb": _gb(n, clip["dimensao"] * 4),
            }
        )
    return {
        "dispositivo_clip": clip["dispositivo"],
        "clip_dimensao": clip["dimensao"],
        "clip_embed_s": clip["embed_s"],
        "clip_scan_por_imagem_us": segundos_clip * 1e6,
        "ssim_por_imagem_us": segundos_ssim * 1e6,
        "sift_extracao_s": sift["extracao_s"],
        "sift_match_s": sift["match_s"],
        "sift_descritores_por_imagem": por_imagem,
        "sift_acuracia_na_amostra": 100.0
        * float(np.mean(np.asarray(sift["votos_verdade"]) > np.asarray(sift["votos_impostor_max"]))),
        "ssim_escore_verdadeiro_medio": float(ssim["verdadeiros"].mean()),
        "ssim_escore_impostor_medio": float(ssim["impostores"].mean()),
        "ssim_escore_impostor_dp": float(ssim["impostores"].std()),
        "clip_escore_verdadeiro_medio": float(clip["verdadeiros"].mean()),
        "clip_escore_impostor_medio": float(clip["impostores"].mean()),
        "clip_escore_impostor_dp": float(clip["impostores"].std()),
        "projecoes": linhas,
    }


def main() -> None:
    brutas, adulteradas = _alinhar(list_images(RAW_DIR), list_images(ADULTERATED_DIR))
    if len(brutas) != N_AMOSTRA:
        raise SystemExit(f"esperava {N_AMOSTRA} pares, encontrou {len(brutas)}")

    print("medindo SSIM...", flush=True)
    ssim = medir_ssim(brutas, adulteradas)
    print("medindo SIFT...", flush=True)
    sift = medir_sift(brutas, adulteradas)
    print("medindo CLIP...", flush=True)
    clip = medir_clip(brutas, adulteradas)
    relatorio = projetar(ssim, sift, clip)
    destino = Path("results") / "estimativa_laion.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    print(json.dumps(relatorio, indent=2))


if __name__ == "__main__":
    main()
