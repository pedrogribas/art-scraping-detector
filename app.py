"""Protótipo Streamlit: auditoria de arte em duas etapas.

Uso:
    streamlit run app.py

Sem um dump FAISS do LAION, a ferramenta indexa as 500 imagens locais
já medidas no TCC. URLs mortas não quebram a grade.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.adulterator import RAW_DIR
from src.category_router import VisualRouter
from src.fine_search import SiftVerifier, baixar_candidatos
from src.vector_index import FaissIndexer, indice_da_amostra

RAIZ = Path(__file__).resolve().parent
EXEMPLOS_DIR = RAIZ / "data" / "exemplos"
INDICE_NOMES = RAIZ / "web" / "public" / "indice" / "clip_nomes.json"
INDICE_VETORES = RAIZ / "web" / "public" / "indice" / "clip_vetores.bin"
INDICE_LAION = RAIZ / "indices" / "laion"
K_GROSSO = 50
K_GRADE = 12

EXEMPLOS = (
    ("laion_001.jpg", "Xícara"),
    ("laion_050.jpg", "Ícones"),
    ("laion_080.jpg", "Banquete"),
    ("laion_160.jpg", "Bordado"),
    ("laion_330.jpg", "Pôr do sol"),
    ("laion_400.jpg", "Cama"),
    ("fora.jpg", "Não está"),
)


def _css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f4ec; }
        h1, h2, h3 { font-family: Georgia, serif; }
        div[data-testid="stMetric"] { background: #fff; border-radius: 12px; padding: 0.4rem 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Carregando o índice FAISS…")
def carregar_indice() -> FaissIndexer:
    if INDICE_LAION.with_suffix(".faiss").is_file():
        indexador = FaissIndexer()
        indexador.carregar(INDICE_LAION)
        return indexador
    return indice_da_amostra(RAW_DIR, INDICE_NOMES, INDICE_VETORES)


@st.cache_resource(show_spinner="Carregando o CLIP…")
def carregar_roteador() -> VisualRouter:
    return VisualRouter()


def _abrir(caminho: Path) -> Image.Image:
    return Image.open(caminho).convert("RGB")


def _mostrar_candidato(url: str, caption: str) -> None:
    local = Path(url)
    try:
        if local.is_file():
            st.image(str(local), caption=caption, use_container_width=True)
            return
        if url.startswith(("http://", "https://")):
            st.image(url, caption=caption, use_container_width=True)
            return
        st.caption(f"indisponível · {caption}")
    except Exception:
        st.caption(f"URL fora do ar · {caption}")


def auditar(consulta: Path, indice: FaissIndexer, roteador: VisualRouter) -> None:
    inicio = time.perf_counter()
    with st.spinner("Detectando estilo visual…"):
        rota = roteador.classificar(consulta)
    with st.spinner("Buscando candidatos no banco de dados…"):
        candidatos = indice.search(rota.embedding, k=K_GROSSO)

    laudos = None
    if rota.requires_sift:
        with st.spinner("Aplicando verificação biométrica (SIFT)…"):
            pasta = RAIZ / "data" / "processed" / "_tmp_app"
            baixados = baixar_candidatos(candidatos, pasta)
            laudos = SiftVerifier().ranquear(consulta, candidatos, baixados)

    total = time.perf_counter() - inicio
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Categoria", rota.tag)
    col_b.metric("Roteador", "SIFT ativado" if rota.requires_sift else "SIFT desativado")
    col_c.metric("Tempo total", f"{total:.2f} s")
    st.caption(
        f"Zero-shot: {rota.confianca:.0%} em “{rota.categoria}”. "
        f"Recall grosso: {len(candidatos)} URLs."
    )

    visiveis: list[tuple[str, str]]
    if laudos is not None:
        visiveis = [
            (
                item.url,
                f"{item.inliers} inliers · CLIP {item.score_clip:.3f}"
                + (f" · {item.erro}" if item.erro else ""),
            )
            for item in laudos[:K_GRADE]
        ]
    else:
        visiveis = [
            (item.url, f"CLIP {item.score:.3f} · {item.sample_id}")
            for item in candidatos[:K_GRADE]
        ]

    st.subheader("Candidatos recuperados")
    for inicio_linha in range(0, len(visiveis), 4):
        colunas = st.columns(4)
        for coluna, (url, caption) in zip(colunas, visiveis[inicio_linha : inicio_linha + 4]):
            with coluna:
                _mostrar_candidato(url, caption)


def main() -> None:
    st.set_page_config(page_title="Auditor de Arte · IFMG", page_icon="🎨", layout="wide")
    _css()
    st.title("Auditor de direitos autorais")
    st.write(
        "Sobe a obra. O CLIP classifica o estilo e busca as 50 mais próximas "
        "no índice. Se a categoria tiver textura, o SIFT+RANSAC reordena."
    )

    try:
        indice = carregar_indice()
        roteador = carregar_roteador()
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as erro:
        st.error(f"Não foi possível subir o backend: {erro}")
        return

    if "consulta" not in st.session_state:
        st.session_state.consulta = None

    st.sidebar.header("A sua arte")
    enviado = st.sidebar.file_uploader("JPG ou PNG", type=("jpg", "jpeg", "png", "webp"))
    st.sidebar.caption(f"Índice com {indice.total} vetores.")

    if enviado is not None:
        pasta = RAIZ / "data" / "processed"
        pasta.mkdir(parents=True, exist_ok=True)
        destino = pasta / f"upload_{enviado.name}"
        destino.write_bytes(enviado.getvalue())
        st.session_state.consulta = destino
        try:
            st.sidebar.image(_abrir(destino), caption="Consulta", use_container_width=True)
        except (OSError, UnidentifiedImageError):
            st.sidebar.error("Arquivo ilegível.")
            st.session_state.consulta = None

    st.sidebar.markdown("Exemplos da amostra")
    for arquivo, legenda in EXEMPLOS:
        if st.sidebar.button(legenda, key=arquivo):
            st.session_state.consulta = EXEMPLOS_DIR / arquivo

    if st.session_state.consulta is None:
        st.info("Envie uma imagem ou escolha um exemplo na barra lateral. Depois clique em Auditar arte.")
        return

    if not st.sidebar.button("Auditar arte", type="primary"):
        st.write(f"Pronto para auditar: `{Path(st.session_state.consulta).name}`")
        return

    try:
        auditar(Path(st.session_state.consulta), indice, roteador)
    except Exception as erro:
        st.error(f"Falha na auditoria: {erro}")


if __name__ == "__main__":
    main()
