"""Página única do TCC: a busca fica no alto e a carta do trabalho, ao rolar.

A consulta compara uma imagem com as 500 obras limpas de ``data/raw`` usando
CLIP. Abaixo, o texto explica o problema, o experimento e a projeção de escala.

Uso:
    streamlit run app.py

Autor: Pedro Garcia Ribas - TCC Sistemas de Informação (IFMG Sabará)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.adulterator import RAW_DIR, list_images
from src.similarity_evaluator import CLIP_MODEL_ID, ComparadorCLIP

EXEMPLOS_DIR = RAW_DIR.parent / "exemplos"
EXEMPLOS = (
    ("laion_001.jpg", "Xícara"),
    ("laion_050.jpg", "Ícones"),
    ("laion_080.jpg", "Banquete"),
    ("laion_160.jpg", "Bordado"),
    ("laion_330.jpg", "Pôr do sol"),
    ("laion_400.jpg", "Helicóptero"),
)

RESULTADOS = pd.DataFrame(
    {
        "Método": ["SSIM", "SIFT", "CLIP"],
        "Acurácia (%)": [88.6, 99.2, 99.6],
        "Tempo de Busca (s)": [0.351, 0.038, 0.056],
    }
)

# Verde e vermelho do Manual de Identidade Visual do IF (HEX do manual de 2015).
VERDE = "#2f9e41"
VERMELHO = "#cd191e"
CORES_METODOS = {"SSIM": "#5c6670", "SIFT": VERMELHO, "CLIP": VERDE}
ESTIMATIVA_PATH = Path(__file__).resolve().parent / "results" / "estimativa_laion.json"

st.set_page_config(
    page_title="Auditor de Datasets · IFMG Sabará",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def aplicar_estilo() -> None:
    """Aplica a faixa institucional e o ritmo de uma carta em página única."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,460;8..60,620&family=Source+Sans+3:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: "Source Sans 3", "Segoe UI", sans-serif;
        }
        .stApp { background: #f6f3ec; }
        header[data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }
        .block-container { padding-top: 0.6rem; padding-bottom: 4rem; max-width: 820px; }
        .faixa { height: 10px; background: #2f9e41; border-bottom: 5px solid #cd191e; margin: 0 -4rem 1.4rem; }
        .kicker {
            font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
            color: #2f9e41; font-weight: 700; margin: 0;
        }
        .autor-linha { color: #3d3d3d; font-size: 0.95rem; margin: 0.15rem 0 0.8rem; }
        h1.titulo {
            font-family: "Source Serif 4", Georgia, serif;
            font-weight: 620; font-size: 2.15rem; line-height: 1.15;
            color: #141414; margin: 0 0 0.35rem;
        }
        .lead { font-size: 1.05rem; color: #333; margin: 0 0 1rem; }
        h2.secao {
            font-family: "Source Serif 4", Georgia, serif;
            font-size: 1.55rem; color: #141414;
            border-top: 1px solid #d9d3c7; padding-top: 1.6rem; margin: 2.2rem 0 0.6rem;
        }
        h2.secao span { color: #2f9e41; }
        .carta p, .prosa p { font-family: "Source Serif 4", Georgia, serif; font-size: 1.08rem; line-height: 1.55; color: #1c1c1c; }
        .data-carta { font-size: 0.92rem; color: #555; margin-bottom: 0.8rem; }
        .assinatura { margin-top: 1rem; }
        .assinatura strong { color: #2f9e41; }
        div[data-testid="stFileUploader"] {
            background: #fff; border: 1px solid #e4ddd0; border-radius: 12px; padding: 0.4rem 0.6rem 0.2rem;
        }
        </style>
        <div class="faixa"></div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Indexando a galeria com CLIP...")
def carregar_indice(pasta: str) -> ComparadorCLIP:
    """Carrega o CLIP e calcula os embeddings das imagens limpas uma só vez."""
    comparador = ComparadorCLIP()
    caminhos = list_images(Path(pasta))
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem em {pasta}")
    comparador.indexar(caminhos)
    return comparador


def _estilo_figura(figura, titulo: str):
    """Gráficos claros, com o verde e o vermelho institucionais."""
    figura.update_layout(
        template="plotly_white",
        title=titulo,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1c1c1c", "family": "Source Sans 3, sans-serif"},
        margin={"t": 64, "b": 16, "l": 8, "r": 8},
    )
    return figura


def exemplos_disponiveis() -> list[tuple[str, str]]:
    """Exemplos adulterados presentes em disco, na ordem da curadoria."""
    return [
        (nome, legenda)
        for nome, legenda in EXEMPLOS
        if (EXEMPLOS_DIR / nome).is_file()
    ]


def obter_consulta() -> tuple[Image.Image | None, str]:
    """Lê o upload ou um exemplo escolhido. Sem imagem, devolve o par vazio."""
    arquivo = st.file_uploader(
        "Insira a imagem",
        type=["jpg", "jpeg", "png"],
        help="JPG ou PNG. Uma imagem de data/adulterated ou data/exemplos recupera o original de mesmo nome.",
        label_visibility="collapsed",
    )
    if arquivo is not None:
        return Image.open(arquivo).convert("RGB"), arquivo.name

    disponiveis = exemplos_disponiveis()
    if not disponiveis:
        return None, ""

    with st.expander("Ou use um exemplo já adulterado do projeto"):
        colunas = st.columns(len(disponiveis))
        for coluna, (nome, legenda) in zip(colunas, disponiveis, strict=True):
            with coluna:
                st.image(str(EXEMPLOS_DIR / nome), width=90)
                if st.button(legenda, key=f"ex-{nome}", use_container_width=True):
                    st.session_state["exemplo"] = nome

    escolhido = st.session_state.get("exemplo")
    if escolhido and (EXEMPLOS_DIR / escolhido).is_file():
        return Image.open(EXEMPLOS_DIR / escolhido).convert("RGB"), escolhido
    return None, ""


def mostrar_busca(comparador: ComparadorCLIP, consulta: Image.Image, nome_consulta: str) -> None:
    """Mostra o Top 1 ao lado da consulta e o Top 2 e o Top 3 abaixo."""
    with st.spinner("Buscando correspondências na galeria..."):
        ranking = comparador.buscar_topk_imagem(consulta, k=3)

    if not ranking or ranking[0].caminho is None:
        st.error("A busca não retornou uma imagem da galeria.")
        return

    principal = ranking[0]
    st.subheader("Correspondência principal")
    coluna_envio, coluna_match = st.columns(2)
    with coluna_envio:
        st.image(consulta, caption=f"Consulta — {nome_consulta}", use_container_width=True)
    with coluna_match:
        st.image(
            str(principal.caminho),
            caption=f"Top 1 — {principal.filename} · cosseno {principal.score:.3f}",
            use_container_width=True,
        )

    if nome_consulta == principal.filename:
        st.success(
            f"O Top 1 é `{principal.filename}`, o mesmo nome do arquivo enviado. "
            "A obra original foi recuperada."
        )
    else:
        st.warning(
            f"O Top 1 é `{principal.filename}`, diferente de `{nome_consulta}`."
        )

    alternativas = ranking[1:]
    if not alternativas:
        return
    st.subheader("Outras correspondências")
    colunas = st.columns(len(alternativas))
    for posicao, (coluna, resultado) in enumerate(zip(colunas, alternativas, strict=True), start=2):
        with coluna:
            st.image(
                str(resultado.caminho),
                caption=f"Top {posicao} — {resultado.filename}",
                use_container_width=True,
            )
            st.metric("Similaridade de cosseno", f"{resultado.score:.3f}")


def secao(titulo: str) -> None:
    """Abre um capítulo da carta, com o verde institucional no título."""
    st.markdown(f'<h2 class="secao"><span>{titulo}</span></h2>', unsafe_allow_html=True)


def carta() -> None:
    """Abertura em forma de carta, para quem chega sem ter lido a monografia."""
    secao("Carta de entrada")
    st.markdown(
        """
        <div class="carta">
        <p class="data-carta">Sabará, 28 de setembro de 2026.</p>
        <p>À banca e a quem chega a esta página,</p>
        <p>Este trabalho nasceu de uma pergunta simples e incômoda: se uma obra
        circula pela internet e acaba dentro de um dataset de bilhões de imagens,
        ainda é possível reconhecê-la? A cópia que volta quase nunca é o arquivo
        original. Ela foi reduzida, cortada, comprimida e, muitas vezes, misturada
        a ruído. O olho humano ainda vê a mesma imagem. Um algoritmo que compara
        pixel com pixel, nem sempre.</p>
        <p>Meu nome é Pedro Garcia Ribas. Curso Sistemas de Informação no Instituto
        Federal de Minas Gerais, Campus Sabará. O que está acima deste texto é a
        prova de conceito: uma busca reversa, com o modelo CLIP, sobre 500 imagens
        reais do subset estético do LAION. O que vem abaixo é a história por trás
        do botão — o problema, o que foi medido, e o que esses 99,6% deixam de
        significar quando a galeria deixa de ter 500 arquivos.</p>
        <p class="assinatura">Pedro Garcia Ribas<br>
        <strong>Bacharelado em Sistemas de Informação · IFMG Campus Sabará</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def problema() -> None:
    """O motivo do TCC, em prosa, antes dos números."""
    secao("O problema")
    st.markdown(
        """
        <div class="prosa">
        <p>Modelos generativos são treinados em conjuntos da escala do LAION-5B,
        com 5,85 bilhões de pares imagem-texto. Uma artista que desconfia que a
        própria obra está lá não recebe o arquivo de volta do jeito que o enviou
        ao mundo. Recebe uma versão que passou por redimensionamento, recorte,
        recompressão JPEG e, às vezes, ruído. A pergunta do TCC é qual família de
        similaridade visual ainda aponta a imagem certa depois disso.</p>
        <p>Três caminhos entram na comparação. O SSIM olha a estrutura dos pixels
        alinhados. O SIFT procura detalhes locais e vota na imagem de origem. O CLIP
        transforma a imagem num vetor de 512 números e mede o cosseno. Nenhum deles
        foi treinado de novo neste trabalho. O teste é de reconhecimento, não de
        treino.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metodo() -> None:
    """Amostra, degradação e a regra de acerto."""
    secao("Como o teste foi feito")
    st.markdown(
        f"""
        <div class="prosa">
        <p>A amostra não veio de um scraper de rede social. São 500 JPEG reais do
        dataset <em>laion/laion2B-en-aesthetic</em>, as primeiras URLs do streaming
        que responderam a tempo. O subset inglês com escore estético acima de 7 tem
        52.068.913 linhas. Estas 500 não são uma amostra aleatória desse total, e
        também não são 500 pinturas de museu: o LAION mistura fotografia, ilustração
        e ícone.</p>
        <p>Cada original em <code>data/raw/</code> ganhou uma cópia de mesmo nome em
        <code>data/adulterated/</code>. A degradação é fixa, para a diferença entre
        os algoritmos não ser a diferença entre agressões: escala de 80%, corte de
        5% em cada borda, ruído gaussiano com σ = 8 e JPEG de qualidade 35. O acerto
        é o Top-1 com o mesmo nome de arquivo. O original sempre está na galeria.
        Isso é identificação em conjunto fechado, não a pergunta “esta obra está no
        LAION?”.</p>
        <p>A busca desta página usa o CLIP (<code>{CLIP_MODEL_ID}</code>). A galeria
        é indexada uma vez; as consultas seguintes só calculam o cosseno.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resultados() -> None:
    """Tabela e barras do experimento com 500 pares."""
    secao("O que as 500 imagens mostraram")
    st.markdown(
        """
        <div class="prosa">
        <p>Na galeria fechada, o CLIP acertou 498 de 500 consultas (99,6%) e o SIFT,
        496 (99,2%). O SSIM ficou em 443 (88,6%). O corte de 5% desloca o conteúdo:
        o SSIM compara posições correspondentes da matriz 128×128 e paga esse
        deslocamento. O SIFT, que vota por detalhe local, foi o mais rápido na busca.
        O tempo abaixo é só a consulta. A indexação ficou de fora da média: 1,09 s
        para o SSIM, 21,56 s para o SIFT e 43,04 s para o CLIP, este último já com
        o carregamento do modelo, em CPU, sem GPU.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        RESULTADOS.style.format({"Acurácia (%)": "{:.1f}", "Tempo de Busca (s)": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )
    coluna_acuracia, coluna_tempo = st.columns(2)
    with coluna_acuracia:
        figura = px.bar(
            RESULTADOS,
            x="Método",
            y="Acurácia (%)",
            color="Método",
            color_discrete_map=CORES_METODOS,
            text="Acurácia (%)",
        )
        figura.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        figura.update_layout(showlegend=False, yaxis_range=[0, 115])
        st.plotly_chart(
            _estilo_figura(figura, "Acurácia Top-1 na amostra de 500"),
            use_container_width=True,
        )
    with coluna_tempo:
        figura = px.bar(
            RESULTADOS,
            x="Método",
            y="Tempo de Busca (s)",
            color="Método",
            color_discrete_map=CORES_METODOS,
            text="Tempo de Busca (s)",
        )
        figura.update_traces(texttemplate="%{y:.3f} s", textposition="outside")
        figura.update_layout(showlegend=False, yaxis_range=[0, 0.48])
        st.plotly_chart(
            _estilo_figura(figura, "Tempo médio de busca por imagem"),
            use_container_width=True,
        )


def _duracao(segundos: float) -> str:
    """Formata segundos numa unidade legível."""
    if segundos < 1:
        return f"{segundos * 1000:.0f} ms"
    if segundos < 60:
        return f"{segundos:.2f} s".replace(".", ",")
    if segundos < 3600:
        return f"{segundos / 60:.1f} min".replace(".", ",")
    if segundos < 86400:
        return f"{segundos / 3600:.1f} h".replace(".", ",")
    return f"{segundos / 86400:.1f} dias".replace(".", ",")


def _gigabytes(valor: float) -> str:
    """Formata memória do índice em MB, GB ou TB."""
    if valor < 1:
        return f"{valor * 1024:.0f} MB"
    if valor < 1024:
        return f"{valor:.1f} GB".replace(".", ",")
    return f"{valor / 1024:.1f} TB".replace(".", ",")


def _pct(valor: float) -> str:
    """Formata acurácia; caudas numéricas viram um piso legível."""
    if valor < 0.01:
        return "< 0,01%"
    return f"{valor:.1f}%".replace(".", ",")


def escala() -> None:
    """Projeção para o subset estético e para o LAION-5B, com o teto do CLIP."""
    secao("Quando a galeria deixa de ter 500 imagens")
    st.markdown(
        """
        <div class="prosa">
        <p>Nenhuma busca rodou nas 52 milhões de linhas do subset nem nos 5,85
        bilhões do LAION-5B. O tempo abaixo usa o custo medido nesta CPU. A
        acurácia do CLIP é um modelo: os cossenos entre imagens que não são o par
        verdadeiro se comportam como uma normal de média 0,47 e desvio 0,09, e a
        chance de acerto é a chance de todos os impostores ficarem abaixo do par
        verdadeiro. Na amostra, o modelo devolve 99,5%, contra 99,6% medidos.
        Fora dela, é um teto. Quase-duplicatas que estas 500 imagens não contêm
        derrubariam o número. O mesmo modelo, no SSIM, previu 44,5% contra 88,6%
        medidos e foi descartado. Não há acurácia projetada para o SIFT.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ESTIMATIVA_PATH.is_file():
        st.error("Falta results/estimativa_laion.json. Rode python -m src.estimate_scale.")
        return

    relatorio = json.loads(ESTIMATIVA_PATH.read_text(encoding="utf-8"))
    projecoes = relatorio["projecoes"]
    tabela = pd.DataFrame(
        [
            {
                "Galeria": item["rotulo"],
                "SSIM": _duracao(item["ssim_busca_s"]),
                "CLIP": _duracao(item["clip_busca_s"]),
                "Acurácia CLIP": _pct(item["clip_acuracia"]),
                "Índice CLIP": _gigabytes(item["clip_memoria_gb"]),
                "Índice SIFT": _gigabytes(item["sift_memoria_gb"]),
            }
            for item in projecoes
        ]
    )
    st.dataframe(tabela, hide_index=True, use_container_width=True)

    acuracia = pd.DataFrame(
        {
            "Imagens": [item["n"] for item in projecoes],
            "Acurácia CLIP (%)": [
                item["clip_acuracia"] if item["clip_acuracia"] >= 0.01 else 0 for item in projecoes
            ],
        }
    )
    figura_acuracia = px.line(
        acuracia,
        x="Imagens",
        y="Acurácia CLIP (%)",
        markers=True,
        log_x=True,
        color_discrete_sequence=[VERDE],
    )
    figura_acuracia.update_layout(yaxis_range=[0, 105], yaxis_title="Acurácia Top-1 (%)", xaxis_title="Imagens na galeria")
    st.plotly_chart(
        _estilo_figura(figura_acuracia, "Teto da acurácia Top-1 do CLIP conforme a galeria cresce"),
        use_container_width=True,
    )

    tempos = pd.DataFrame(
        [
            {"Imagens": item["n"], "Método": metodo, "Segundos": item[chave]}
            for item in projecoes
            for metodo, chave in (
                ("SSIM", "ssim_busca_s"),
                ("SIFT", "sift_busca_s"),
                ("CLIP", "clip_busca_s"),
            )
        ]
    )
    figura_tempo = px.line(
        tempos,
        x="Imagens",
        y="Segundos",
        color="Método",
        color_discrete_map=CORES_METODOS,
        markers=True,
        log_x=True,
        log_y=True,
    )
    figura_tempo.update_layout(yaxis_title="Tempo de busca (s)", xaxis_title="Imagens na galeria")
    st.plotly_chart(
        _estilo_figura(figura_tempo, "Tempo de uma consulta. A curva do SIFT pressupõe o índice na memória"),
        use_container_width=True,
    )
    st.markdown(
        """
        <div class="prosa">
        <p>No subset estético inteiro, o teto do CLIP cai para cerca de 1,8% de
        Top-1. Cada consulta levaria cerca de 8,6 s nesta CPU, com um índice de
        99 GB. O SSIM levaria cerca de 9,2 horas por consulta, e 43 dias no
        LAION-5B. O SIFT continuaria na casa de 0,05 s só se o índice existisse:
        os descritores ocupariam 10,5 TB no subset e cerca de 1,2 PB no LAION-5B.
        A implementação testada não constrói esse índice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def limites() -> None:
    """O que a página não autoriza concluir."""
    secao("O que estes números não dizem")
    st.markdown(
        """
        <div class="prosa">
        <p>99,6% não é uma propriedade do LAION. É a taxa de acerto numa gaveta de
        500 arquivos em que o original sempre está presente, depois de uma
        degradação combinada e fixa. Não houve conjunto aberto, nem amostra
        aleatória, nem teste separado de cada agressão. Hashing perceptual, curva
        ROC e F1 constavam do plano inicial e não foram medidos. Um índice
        aproximado, como o FAISS, também não foi testado: no subset de 52 milhões
        a memória do CLIP cabe num servidor, mas isso não devolve, por si, a
        acurácia da amostra pequena.</p>
        <p>As imagens em si não estão neste repositório público. O volume e os
        direitos de terceiros no LAION impedem versioná-las. Quem clonar o código
        reconstrói a amostra com o downloader, desde que aceite os termos do
        dataset no Hugging Face.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fecho() -> None:
    """Assinatura institucional no fim da rolagem."""
    secao("Autor")
    st.markdown(
        """
        <div class="prosa">
        <p><strong>Pedro Garcia Ribas</strong><br>
        Bacharelado em Sistemas de Informação<br>
        Instituto Federal de Minas Gerais · Campus Sabará</p>
        <p>Repositório:
        <a href="https://github.com/pedrogribas/art-scraping-detector">github.com/pedrogribas/art-scraping-detector</a>.
        As cores desta página seguem o verde <code>#2f9e41</code> e o vermelho
        <code>#cd191e</code> do Manual de Identidade Visual da marca Instituto Federal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Campo de imagem no topo; a carta e os resultados aparecem ao rolar."""
    aplicar_estilo()
    st.markdown(
        """
        <p class="kicker">Instituto Federal de Minas Gerais · Campus Sabará</p>
        <p class="autor-linha">Pedro Garcia Ribas · Sistemas de Informação</p>
        <h1 class="titulo">Auditor de datasets</h1>
        <p class="lead">Insira uma imagem. A busca usa o CLIP sobre as 500 originais da amostra. A história do trabalho está logo abaixo.</p>
        """,
        unsafe_allow_html=True,
    )

    try:
        comparador = carregar_indice(str(RAW_DIR))
    except FileNotFoundError as erro:
        st.error(str(erro))
        st.stop()

    consulta, nome_consulta = obter_consulta()
    if consulta is None:
        st.caption(f"Galeria pronta: {len(comparador._nomes)} imagens. Nenhuma consulta ainda.")
    else:
        mostrar_busca(comparador, consulta, nome_consulta)

    carta()
    problema()
    metodo()
    resultados()
    escala()
    limites()
    fecho()


if __name__ == "__main__":
    main()
