"""Exporta o índice CLIP e os exemplos para a página estática.

A busca no GitHub Pages roda no navegador: o índice das 500 já sai daqui,
com o mesmo CLIP e o mesmo corte da API.

Uso:
    python -m src.exportar_indice
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.adulterator import RAW_DIR, list_images
from src.api_server import EXEMPLOS, EXEMPLOS_DIR, LIMIAR_COSENO, LIMIAR_FOLGA, PISO_COSENO
from src.similarity_evaluator import ComparadorCLIP

DESTINO = Path(__file__).resolve().parents[1] / "web" / "public" / "indice"


def _presente(primeiro: float, segundo: float) -> tuple[bool, float]:
    folga = primeiro - segundo
    return primeiro >= LIMIAR_COSENO or (primeiro >= PISO_COSENO and folga >= LIMIAR_FOLGA), folga


def _ranking(comparador: ComparadorCLIP, consulta: Image.Image, nome: str) -> dict:
    resultados = comparador.buscar_topk_imagem(consulta, k=3)
    primeiro = resultados[0].score
    segundo = resultados[1].score if len(resultados) > 1 else 0.0
    presente, folga = _presente(primeiro, segundo)
    return {
        "nome": nome,
        "presente": presente,
        "limiar": LIMIAR_COSENO,
        "folga": round(folga, 3),
        "ranking": [
            {
                "posicao": indice,
                "arquivo": item.filename,
                "score": round(item.score, 3),
                "url": f"/imagens/raw/{item.filename}",
                "mesmo_arquivo": item.filename == nome,
            }
            for indice, item in enumerate(resultados, start=1)
        ],
    }


def main() -> None:
    caminhos = list_images(RAW_DIR)
    if len(caminhos) < 500:
        raise SystemExit(f"Esperava 500 imagens em {RAW_DIR}, achei {len(caminhos)}")

    comparador = ComparadorCLIP()
    comparador.indexar(caminhos)
    DESTINO.mkdir(parents=True, exist_ok=True)

    nomes = list(comparador._nomes)
    vetores = comparador._galeria.detach().cpu().numpy().astype("float32")
    (DESTINO / "clip_nomes.json").write_text(
        json.dumps(nomes, ensure_ascii=False),
        encoding="utf-8",
    )
    (DESTINO / "clip_vetores.bin").write_bytes(vetores.tobytes())

    exemplos = {}
    for nome, _legenda, _ausente in EXEMPLOS:
        caminho = EXEMPLOS_DIR / nome
        if not caminho.is_file():
            continue
        with Image.open(caminho) as imagem:
            exemplos[nome] = _ranking(comparador, imagem.convert("RGB"), nome)
    (DESTINO / "exemplos.json").write_text(
        json.dumps(exemplos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raw_publico = DESTINO.parents[1] / "imagens" / "raw"
    raw_publico.mkdir(parents=True, exist_ok=True)
    for caminho in caminhos:
        destino = raw_publico / caminho.name
        if not destino.is_file() or destino.stat().st_size != caminho.stat().st_size:
            destino.write_bytes(caminho.read_bytes())
    print(f"Índice: {len(nomes)} vetores em {DESTINO}")


if __name__ == "__main__":
    main()
