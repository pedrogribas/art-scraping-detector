"""API da página: busca CLIP e leitura das imagens da amostra.

Uso:
    python -m uvicorn src.api_server:app --port 8000
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.adulterator import ADULTERATED_DIR, RAW_DIR, list_images
from src.similarity_evaluator import ComparadorCLIP

EXEMPLOS_DIR = RAW_DIR.parent / "exemplos"
EXEMPLOS = (
    ("laion_001.jpg", "Xícara", False),
    ("laion_050.jpg", "Ícones", False),
    ("laion_080.jpg", "Banquete", False),
    ("laion_160.jpg", "Bordado", False),
    ("laion_330.jpg", "Pôr do sol", False),
    ("laion_400.jpg", "Cama", False),
    ("fora.jpg", "Não está", True),
)
# Nas 500 cópias degradadas, a mediana do cosseno com a original foi 0,915
# e os exemplos da página ficaram entre 0,852 e 0,989. A figura de fora
# fez 0,773 e só 0,014 acima da segunda colocada. Conta como presente se
# o primeiro lugar passa de 0,82 ou se, mesmo mais baixo, abre pelo menos
# 0,10 de vantagem sobre o segundo.
LIMIAR_COSENO = 0.82
PISO_COSENO = 0.75
LIMIAR_FOLGA = 0.10
PASTAS = {
    "raw": RAW_DIR,
    "adulterated": ADULTERATED_DIR,
    "exemplos": EXEMPLOS_DIR,
}


def carregar() -> ComparadorCLIP:
    """Indexa a galeria limpa uma vez, no sobe do servidor."""
    comparador = ComparadorCLIP()
    caminhos = list_images(RAW_DIR)
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem em {RAW_DIR}")
    comparador.indexar(caminhos)
    return comparador


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mantém o CLIP carregado enquanto a API estiver no ar."""
    try:
        app.state.comparador = carregar()
        app.state.erro = None
    except FileNotFoundError as erro:
        app.state.comparador = None
        app.state.erro = str(erro)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ranking(consulta: Image.Image, nome: str) -> dict:
    comparador: ComparadorCLIP | None = app.state.comparador
    if comparador is None:
        raise HTTPException(status_code=503, detail=app.state.erro or "Galeria indisponível")
    resultados = comparador.buscar_topk_imagem(consulta, k=3)
    primeiro = resultados[0].score
    segundo = resultados[1].score if len(resultados) > 1 else 0.0
    folga = primeiro - segundo
    presente = primeiro >= LIMIAR_COSENO or (primeiro >= PISO_COSENO and folga >= LIMIAR_FOLGA)
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


@app.get("/api/status")
def status() -> dict:
    """Diz se a galeria já foi indexada."""
    comparador = app.state.comparador
    return {
        "pronto": comparador is not None,
        "total": len(comparador._nomes) if comparador is not None else 0,
        "erro": app.state.erro,
    }


@app.get("/api/exemplos")
def exemplos() -> list[dict]:
    """Lista as consultas adulteradas prontas para testar."""
    itens = []
    for nome, legenda, ausente in EXEMPLOS:
        caminho = EXEMPLOS_DIR / nome
        if caminho.is_file():
            itens.append(
                {
                    "arquivo": nome,
                    "legenda": legenda,
                    "ausente": ausente,
                    "url": f"/imagens/exemplos/{nome}",
                }
            )
    return itens


@app.post("/api/buscar")
async def buscar(arquivo: UploadFile) -> dict:
    """Compara um upload com a galeria limpa."""
    bruto = await arquivo.read()
    if not bruto:
        raise HTTPException(status_code=400, detail="Arquivo vazio")
    try:
        imagem = Image.open(io.BytesIO(bruto)).convert("RGB")
    except UnidentifiedImageError as erro:
        raise HTTPException(status_code=400, detail="Envie um JPG ou PNG") from erro
    return _ranking(imagem, arquivo.filename or "consulta.jpg")


@app.post("/api/exemplo/{nome}")
def buscar_exemplo(nome: str) -> dict:
    """Busca uma das imagens adulteradas que já estão no projeto."""
    caminho = (EXEMPLOS_DIR / nome).resolve()
    if caminho.parent != EXEMPLOS_DIR.resolve() or not caminho.is_file():
        raise HTTPException(status_code=404, detail="Exemplo não encontrado")
    imagem = Image.open(caminho).convert("RGB")
    return _ranking(imagem, caminho.name)


@app.get("/imagens/{pasta}/{nome}")
def imagem(pasta: str, nome: str) -> FileResponse:
    """Entrega um JPEG da amostra, sem sair das pastas permitidas."""
    base = PASTAS.get(pasta)
    if base is None or Path(nome).name != nome:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    caminho = (base / nome).resolve()
    if caminho.parent != base.resolve() or not caminho.is_file():
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    return FileResponse(caminho)
