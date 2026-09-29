import { useEffect, useRef, useState } from "react";
import {
  ajustarResposta,
  buscarImagem,
  carregarIndice,
  exemploPronto,
  urlPublica,
} from "./buscaClip.js";

const METODOS = [
  {
    nome: "SSIM",
    taxa: "88,6%",
    alvo: "88.6%",
    texto: "Compara luz e contraste no mesmo pixel. Um corte de 5% empurra a foto e a nota cai. Errou 57 de 500. Cada busca nas 500 levou 0,35 s.",
    fonte: "Wang et al., 2004",
  },
  {
    nome: "SIFT",
    taxa: "99,2%",
    alvo: "99.2%",
    texto: "Marca cantos e descreve a volta de cada um. Aguenta corte e redução. Acertou 496 de 500. Foi o mais rápido: 0,04 s por busca.",
    fonte: "Lowe, 2004",
  },
  {
    nome: "CLIP",
    taxa: "99,6%",
    alvo: "99.6%",
    texto: "Transforma a foto em 512 números de sentido, não de posição. É a busca desta página. Acertou 498 de 500, em 0,06 s.",
    fonte: "Radford et al., 2021",
  },
];

function Link({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

const ORIGEM = [
  {
    titulo: "O arquivo comum da web",
    texto: (
      <>
        O{" "}
        <Link href="https://commoncrawl.org/">Common Crawl</Link>{" "}
        é um arquivo público: um programa percorre páginas da internet e guarda
        uma cópia. O{" "}
        <Link href="https://laion.ai/">LAION</Link>{" "}
        não fotografou nada. Leu esse arquivo e anotou, para cada figura, o
        endereço na web e a frase que estava ao lado.
      </>
    ),
  },
  {
    titulo: "O LAION guarda o par, não a foto",
    texto: (
      <>
        Cada linha é um link e um texto associados. A imagem continua no site
        original. Quem quiser o arquivo baixa o link depois. Por isso a lista
        cabia em um computador e, ao mesmo tempo, apontava para bilhões de fotos
        que ninguém pediu para entrar. Em dezembro de 2023 o próprio LAION
        retirou os links do ar, depois de uma auditoria achar conteúdo ilegal
        no meio. As 500 imagens desta página já tinham sido baixadas antes.
      </>
    ),
  },
  {
    titulo: "Quem parecia combinar ficou",
    texto: (
      <>
        Um programa comparou a frase com a imagem e ficou só com os pares que
        batiam. Quem fez a obra não foi consultado. Empresas baixaram esses
        links e ensinaram o gerador. O recorte usado aqui está no Hugging Face:{" "}
        <Link href="https://huggingface.co/datasets/laion/laion2B-en-aesthetic">
          laion2B-en-aesthetic
        </Link>
        .
      </>
    ),
  },
];

const PASSOS = [
  ["Buscar sem varrer tudo", "Um índice como FAISS ou HNSW evita olhar o banco inteiro. Ainda falta medir o acerto que se perde."],
  ["Refazer o corte", "O “não está” desta página vale para 500 imagens. Em milhões, o vizinho errado sobe e o corte muda."],
];

function useNoScroll(limiar = 0.22) {
  const ref = useRef(null);
  const [visivel, setVisivel] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const obs = new IntersectionObserver(
      ([entrada]) => {
        if (entrada.isIntersecting) setVisivel(true);
      },
      { threshold: limiar, rootMargin: "0px 0px -8% 0px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [limiar]);
  return [ref, visivel];
}

function Surge({ className = "", ordem = 0, children }) {
  const [ref, visivel] = useNoScroll(0.16);
  return (
    <div
      ref={ref}
      className={`surge ${visivel ? "ligado visivel" : ""} ${className}`}
      style={{ "--ordem": ordem }}
    >
      {children}
    </div>
  );
}

function GraficoTempo() {
  const [ref, visivel] = useNoScroll(0.28);
  const largura = 640;
  const altura = 268;
  const esq = 86;
  const dir = 16;
  const topo = 18;
  const base = 36;
  const pontos = [
    { n: 500, rotulo: "500", ssim: 0.3507, sift: 0.0383, clip: 0.0564 },
    { n: 10000, rotulo: "10 mil", ssim: 6.35, sift: 0.0378, clip: 0.0562 },
    { n: 100000, rotulo: "100 mil", ssim: 63.55, sift: 0.0405, clip: 0.071 },
    { n: 1000000, rotulo: "1 mi", ssim: 635.5, sift: 0.0432, clip: 0.218 },
    { n: 52068913, rotulo: "52 mi", ssim: 33089, sift: 0.0478, clip: 8.58 },
    { n: 5850000000, rotulo: "5,85 bi", ssim: 3717545, sift: 0.0534, clip: 958 },
  ];
  const logN = (n) => Math.log10(n);
  const logT = (t) => Math.log10(Math.max(t, 0.03));
  const minN = logN(500);
  const maxN = logN(5850000000);
  const minT = logT(0.02);
  const maxT = logT(3717545);
  const xDe = (n) => esq + ((logN(n) - minN) / (maxN - minN)) * (largura - esq - dir);
  const yDe = (t) => topo + (1 - (logT(t) - minT) / (maxT - minT)) * (altura - topo - base);
  const pathDe = (chave) =>
    pontos.map((p, i) => `${i ? "L" : "M"}${xDe(p.n).toFixed(1)},${yDe(p[chave]).toFixed(1)}`).join(" ");
  const marcas = [
    [0.03, "0,03 s"],
    [0.1, "0,1 s"],
    [1, "1 s"],
    [60, "1 min"],
    [3600, "1 h"],
    [86400, "1 dia"],
  ];
  return (
    <figure ref={ref} className={visivel ? "grafico-linha ligado visivel" : "grafico-linha"}>
      <svg viewBox={`0 0 ${largura} ${altura}`} role="img">
        <title>Tempo de uma busca no SSIM, no SIFT e no CLIP, da amostra até o LAION-5B.</title>
        {marcas.map(([valor, rotulo]) => {
          const y = yDe(valor);
          return (
            <g key={rotulo}>
              <line className="grade" x1={esq} x2={largura - dir} y1={y} y2={y} />
              <text x={esq - 8} y={y + 4} textAnchor="end">{rotulo}</text>
            </g>
          );
        })}
        <path className="curva ssim" d={pathDe("ssim")} />
        <path className="curva sift" d={pathDe("sift")} />
        <path className="curva clip" d={pathDe("clip")} />
        {pontos.map((p, indice) => (
          <text
            key={p.rotulo}
            className="eixo"
            x={xDe(p.n)}
            y={altura - 8}
            textAnchor={indice === pontos.length - 1 ? "end" : "middle"}
          >
            {p.rotulo}
          </text>
        ))}
      </svg>
      <figcaption>
        Eixo vertical em escala de tempo. Cinza: SSIM, uma a uma. Vermelho: SIFT, rápido só se o índice couber.
        Verde: CLIP. Os dois últimos pontos do SIFT assumem memória que esta máquina não tem.
      </figcaption>
      <ul className="legenda">
        <li className="ssim">SSIM</li>
        <li className="sift">SIFT</li>
        <li className="clip">CLIP</li>
      </ul>
    </figure>
  );
}

function GraficoQueda() {
  const [ref, visivel] = useNoScroll(0.28);
  const largura = 640;
  const altura = 250;
  const esq = 46;
  const dir = 14;
  const topo = 16;
  const base = 32;
  const pontos = [
    [500, 99.6, "500"],
    [10000, 96, "10 mil"],
    [100000, 84.2, "100 mil"],
    [1000000, 52, "1 mi"],
    [52068913, 1.8, "52 mi"],
    [5850000000, 0.01, "5,85 bi"],
  ];
  const log = (n) => Math.log10(n);
  const min = log(500);
  const max = log(5850000000);
  const coords = pontos.map(([n, valor, rotulo]) => ({
    x: esq + ((log(n) - min) / (max - min)) * (largura - esq - dir),
    y: topo + (1 - valor / 100) * (altura - topo - base),
    rotulo,
    valor,
  }));
  const d = coords.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  return (
    <figure ref={ref} className={visivel ? "grafico-linha ligado visivel" : "grafico-linha"}>
      <svg viewBox={`0 0 ${largura} ${altura}`} role="img">
        <title>O acerto do CLIP cai de 99,6% em 500 imagens para 1,8% em 52 milhões.</title>
        {[0, 50, 100].map((marca) => {
          const y = topo + (1 - marca / 100) * (altura - topo - base);
          return (
            <g key={marca}>
              <line className="grade" x1={esq} x2={largura - dir} y1={y} y2={y} />
              <text x={esq - 8} y={y + 4} textAnchor="end">{marca}%</text>
            </g>
          );
        })}
        <path className="curva" d={d} />
        {coords.map((p, indice) => (
          <g key={p.rotulo}>
            <circle cx={p.x} cy={p.y} r="4.5" />
            <text className="eixo" x={p.x} y={altura - 8} textAnchor={indice === coords.length - 1 ? "end" : "middle"}>{p.rotulo}</text>
          </g>
        ))}
      </svg>
      <figcaption>
        A linha é o teto do CLIP. O ponto das 500 foi medido, 99,6%. O de 52 milhões, 1,8%, é estimativa.
      </figcaption>
    </figure>
  );
}

function Conta({ valor, casas = 0, sufixo = "" }) {
  const [ref, visivel] = useNoScroll(0.4);
  const [atual, setAtual] = useState(0);
  useEffect(() => {
    if (!visivel) return undefined;
    const inicio = performance.now();
    let quadro;
    const tick = (agora) => {
      const t = Math.min(1, (agora - inicio) / 1100);
      const suave = 1 - (1 - t) ** 3;
      setAtual(valor * suave);
      if (t < 1) quadro = requestAnimationFrame(tick);
    };
    quadro = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(quadro);
  }, [visivel, valor]);
  const texto = atual.toLocaleString("pt-BR", {
    minimumFractionDigits: casas,
    maximumFractionDigits: casas,
  });
  return (
    <span ref={ref}>
      {texto}
      {sufixo}
    </span>
  );
}

export default function App() {
  const exemplosFixos = [
    { arquivo: "laion_001.jpg", legenda: "Xícara", ausente: false, url: urlPublica("imagens/exemplos/laion_001.jpg") },
    { arquivo: "laion_050.jpg", legenda: "Ícones", ausente: false, url: urlPublica("imagens/exemplos/laion_050.jpg") },
    { arquivo: "laion_080.jpg", legenda: "Banquete", ausente: false, url: urlPublica("imagens/exemplos/laion_080.jpg") },
    { arquivo: "laion_160.jpg", legenda: "Bordado", ausente: false, url: urlPublica("imagens/exemplos/laion_160.jpg") },
    { arquivo: "laion_330.jpg", legenda: "Pôr do sol", ausente: false, url: urlPublica("imagens/exemplos/laion_330.jpg") },
    { arquivo: "laion_400.jpg", legenda: "Cama", ausente: false, url: urlPublica("imagens/exemplos/laion_400.jpg") },
    { arquivo: "fora.jpg", legenda: "Não está", ausente: true, url: urlPublica("imagens/exemplos/fora.jpg") },
  ];
  const [pronto, setPronto] = useState(false);
  const [total, setTotal] = useState(500);
  const [modo, setModo] = useState("carregando");
  const [exemplos, setExemplos] = useState(exemplosFixos);
  const [sobre, setSobre] = useState(false);
  const [buscando, setBuscando] = useState(false);
  const [erro, setErro] = useState("");
  const [preview, setPreview] = useState("");
  const [resposta, setResposta] = useState(null);
  const [progresso, setProgresso] = useState(0);
  const [fio, setFio] = useState(0);
  const arquivoRef = useRef(null);
  const trilhaRef = useRef(null);
  const apiRef = useRef(false);

  useEffect(() => {
    let ativo = true;
    async function iniciar() {
      const soPagina =
        window.location.hostname.endsWith("github.io") ||
        new URLSearchParams(window.location.search).has("estatico");
      if (!soPagina) {
        try {
          const status = await fetch("/api/status").then((r) => r.json());
          if (status.pronto && ativo) {
            apiRef.current = true;
            setModo("api");
            setPronto(true);
            setTotal(status.total);
            const lista = await fetch("/api/exemplos").then((r) => r.json());
            if (ativo) setExemplos(lista);
            return;
          }
        } catch {
          /* sem API local, cai no índice estático */
        }
      }
      try {
        const indice = await carregarIndice();
        if (!ativo) return;
        setModo("navegador");
        setPronto(true);
        setTotal(indice.nomes.length);
        setExemplos(exemplosFixos);
      } catch (falha) {
        if (ativo) {
          setModo("offline");
          setErro(falha.message);
        }
      }
    }
    iniciar();
    const aoRolar = () => {
      const altura = document.documentElement.scrollHeight - window.innerHeight;
      setProgresso(altura > 0 ? window.scrollY / altura : 0);
      const el = trilhaRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const percorrido = Math.max(0, -rect.top + window.innerHeight * 0.62);
      setFio(el.offsetHeight > 0 ? Math.min(1, percorrido / el.offsetHeight) : 0);
    };
    aoRolar();
    window.addEventListener("scroll", aoRolar, { passive: true });
    return () => {
      ativo = false;
      window.removeEventListener("scroll", aoRolar);
    };
  }, []);

  async function buscarArquivo(arquivo) {
    setErro("");
    setBuscando(true);
    setPreview(URL.createObjectURL(arquivo));
    try {
      if (apiRef.current) {
        const corpo = new FormData();
        corpo.append("arquivo", arquivo);
        const r = await fetch("/api/buscar", { method: "POST", body: corpo });
        const json = await r.json();
        if (!r.ok) throw new Error(json.detail || "Não foi possível ler a imagem");
        setResposta(json);
        return;
      }
      const json = await buscarImagem(arquivo, arquivo.name, (texto) => setErro(texto));
      setErro("");
      setResposta(json);
    } catch (falha) {
      setResposta(null);
      setErro(falha.message);
    } finally {
      setBuscando(false);
    }
  }

  async function buscarExemplo(item) {
    setErro("");
    setBuscando(true);
    setPreview(item.url);
    try {
      if (apiRef.current) {
        const r = await fetch(`/api/exemplo/${item.arquivo}`, { method: "POST" });
        const json = await r.json();
        if (!r.ok) throw new Error(json.detail || "Exemplo indisponível");
        setResposta(json);
        return;
      }
      const prontoLocal = exemploPronto(item.arquivo);
      if (prontoLocal) {
        setResposta(prontoLocal);
        return;
      }
      const json = await buscarImagem(item.url, item.arquivo, (texto) => setErro(texto));
      setErro("");
      setResposta(ajustarResposta({ ...json, nome: item.arquivo }));
    } catch (falha) {
      setResposta(null);
      setErro(falha.message);
    } finally {
      setBuscando(false);
    }
  }

  function aoSoltar(evento) {
    evento.preventDefault();
    setSobre(false);
    const arquivo = evento.dataTransfer.files?.[0];
    if (arquivo) buscarArquivo(arquivo);
  }

  const principal = resposta?.ranking?.[0];
  const outros = resposta?.ranking?.slice(1) ?? [];

  return (
    <>
      <div className="progresso" style={{ transform: `scaleX(${progresso})` }} />
      <div className="faixa" />
      <main className="trilha" ref={trilhaRef} style={{ "--fio": fio }}>
        <div className="fio" aria-hidden="true"><i /></div>
      <header className="hero">
        <Surge>
          <span className="marco" />
          <p className="kicker">IFMG Campus Sabará · Pedro Garcia Ribas</p>
          <h1>Essa imagem está no banco?</h1>
        </Surge>
        <Surge ordem={1}>
          <p className="lead">
            {pronto
              ? `A busca olha ${total} imagens reais${modo === "navegador" ? " aqui no navegador" : ""}. Role para ver de onde elas vieram.`
              : "A busca está carregando o índice das 500."}
          </p>
        </Surge>

        <form
          className={sobre ? "drop sobre" : "drop"}
          onDragOver={(evento) => {
            evento.preventDefault();
            setSobre(true);
          }}
          onDragLeave={() => setSobre(false)}
          onDrop={aoSoltar}
        >
          <div className="drop-acao">
            <button
              className="botao"
              type="button"
              disabled={!pronto || buscando}
              onClick={() => arquivoRef.current?.click()}
            >
              {pronto
                ? buscando
                  ? "Procurando…"
                  : "Escolher imagem"
                : "Carregando a busca…"}
            </button>
            <span className="dica">
              {modo === "navegador"
                ? "os exemplos já consultam as 500; um arquivo seu baixa o CLIP uma vez"
                : "ou solte um JPG aqui"}
            </span>
          </div>
          <input
            ref={arquivoRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(evento) => {
              const arquivo = evento.target.files?.[0];
              if (arquivo) buscarArquivo(arquivo);
            }}
          />
          {exemplos.length > 0 && (
            <div className="exemplos">
              {exemplos.map((item) => (
                <button
                  className={item.ausente ? "exemplo ausente" : "exemplo"}
                  type="button"
                  key={item.arquivo}
                  disabled={buscando}
                  onClick={() => buscarExemplo(item)}
                >
                  <img src={item.url} alt="" />
                  <span>{item.legenda}</span>
                </button>
              ))}
            </div>
          )}
        </form>
        {erro && <p className="erro">{erro}</p>}

        {principal && (
          <div className="resultado" key={resposta.nome + principal.score + String(resposta.presente)}>
            <div className="par">
              <figure>
                <img src={preview} alt="Imagem enviada" />
                <figcaption>A sua</figcaption>
              </figure>
              <figure className={resposta.presente ? "" : "recusada"}>
                <img src={principal.url} alt="" />
                <figcaption>
                  {resposta.presente ? "No banco" : "Mais próxima, abaixo do corte"}
                  {" · "}
                  {principal.arquivo}
                  {" · "}
                  {principal.score.toFixed(3).replace(".", ",")}
                </figcaption>
              </figure>
            </div>
            <p className={resposta.presente ? "selo" : "selo nao"}>
              {resposta.presente
                ? principal.mesmo_arquivo
                  ? "A imagem está no banco. É o mesmo arquivo."
                  : "A imagem está no banco. O primeiro lugar passou do corte."
                : "A imagem não está no banco."}
            </p>
            {!resposta.presente && (
              <p className="nota">
                O mais próximo fez {principal.score.toFixed(3).replace(".", ",")}, abaixo do corte de{" "}
                {Number(resposta.limiar).toFixed(2).replace(".", ",")}, e a vantagem sobre o segundo
                lugar foi {Number(resposta.folga).toFixed(3).replace(".", ",")}. Pouco para ser a mesma obra.
              </p>
            )}
            {outros.length > 0 && (
              <div className="outros">
                {outros.map((item) => (
                  <figure className="mini" key={item.arquivo}>
                    <img src={item.url} alt="" />
                    <figcaption>
                      {item.posicao}º lugar · {item.score.toFixed(3).replace(".", ",")}
                    </figcaption>
                  </figure>
                ))}
              </div>
            )}
          </div>
        )}
        <p className="rolagem">
          Desça
          <i />
        </p>
      </header>

      <section className="cena">
        <Surge>
          <span className="marco" />
          <p className="kicker">1 · A origem</p>
          <h2>O banco não foi doado. Foi raspado.</h2>
        </Surge>
        <Surge ordem={1}>
          <p className="gigante">
            <Conta valor={5.85} casas={2} />
            <small>
              bilhões de pares link-texto no{" "}
              <Link href="https://laion.ai/blog/laion-5b/">LAION-5B</Link>
              . Não são 5,85 bilhões de arquivos guardados.{" "}
              <Link href="https://arxiv.org/abs/2210.08402">Schuhmann et al., 2022</Link>.
            </small>
          </p>
        </Surge>
        <div className="origem">
          {ORIGEM.map((item, indice) => (
            <Surge key={item.titulo} ordem={indice}>
              <strong>{item.titulo}</strong>
              <span>{item.texto}</span>
            </Surge>
          ))}
        </div>
      </section>

      <section className="cena">
        <Surge>
          <span className="marco" />
          <p className="kicker">2 · A cópia</p>
          <h2>O que sai do modelo, às vezes, é a foto do treino.</h2>
        </Surge>
        <Surge ordem={1}>
          <p className="texto">
            <Link href="https://openaccess.thecvf.com/content/CVPR2023/html/Somepalli_Diffusion_Art_or_Digital_Forgery_Investigating_Data_Replication_in_Diffusion_CVPR_2023_paper.html">
              Somepalli e colegas
            </Link>{" "}
            geraram com o Stable Diffusion e procuraram o par no LAION.
            Em cima, a geração. Embaixo, a foto que já estava no banco. Não é só o estilo:
            a cena volta. Cerca de 1,88% passaram do limiar, e a busca só viu 12 milhões
            de imagens, uma fatia do treino.
          </p>
        </Surge>
        <Surge ordem={2} className="revela">
          <figure className="moldura">
            <img
              src="/pesquisa/somepalli-fig1.png"
              alt="Sete pares. Em cada coluna, a imagem gerada fica em cima e a foto do LAION embaixo."
            />
            <figcaption>Somepalli et al., CVPR 2023. Linha de cima gerada, linha de baixo achada no LAION.</figcaption>
          </figure>
        </Surge>
        <Surge ordem={3}>
          <p className="texto">
            <Link href="https://arxiv.org/abs/2301.13188">Carlini e colegas</Link>{" "}
            fizeram o mesmo sem abrir o gerador: 109 cópias quase iguais.
            A da direita saiu só com o nome da pessoa. A da esquerda já estava na lista.
          </p>
        </Surge>
        <Surge ordem={4} className="revela">
          <figure className="moldura estreita">
            <img
              src="/pesquisa/carlini-fig1.png"
              alt="Foto de Ann Graham Lotz no treino, ao lado da imagem quase igual gerada pelo Stable Diffusion."
            />
            <figcaption>Carlini et al., 2023. As duas imagens quase se encostam. A foto é CC BY-SA 3.0.</figcaption>
          </figure>
        </Surge>
        <Surge ordem={5}>
          <p className="texto">
            Isso só foi possível porque o LAION era público: uma lista de links, não um cofre.
            Quem pesquisou baixou as imagens e comparou com o que o gerador cuspiu.
            A lista aberta não resolve o artista. O gerador, esse sim, é fechado:
            você escreve uma frase, sai uma imagem, e não há um botão para perguntar
            se a sua obra entrou.
          </p>
        </Surge>
        <Surge ordem={6}>
          <p className="texto">
            Em 2023, artistas como Sarah Andersen processaram Stability AI, Midjourney,
            DeviantArt e Runway (
            <Link href="https://www.courtlistener.com/docket/66732129/andersens-v-stability-ai-ltd/">
              Andersen v. Stability AI
            </Link>
            ). A Getty, dona de um acervo de fotos, processou a Stability
            no Reino Unido e nos Estados Unidos (
            <Link href="https://www.judiciary.uk/judgments/getty-images-v-stability-ai/">
              Getty Images v Stability AI
            </Link>
            ). O alvo não é o LAION por ser secreto.
            O alvo são as empresas que usaram essa lista pública para treinar e vender
            um gerador. A sentença ainda não saiu. Enquanto isso, a prova que o artista
            consegue produzir é comparar figuras de fora.
          </p>
        </Surge>
      </section>

      <section className="cena">
        <Surge>
          <span className="marco" />
          <p className="kicker">3 · A prova</p>
          <h2>Por isso este trabalho existe: medir a figura sem abrir o gerador.</h2>
        </Surge>
        <Surge ordem={1}>
          <p className="texto">
            A lista oficial do LAION não está mais no ar desde dezembro de 2023.
            As 500 imagens desta página já tinham sido baixadas, de um recorte
            estético de cerca de 52 milhões. Cada original ganhou uma cópia menor,
            cortada, com ruído e JPEG ruim: o jeito como o arquivo circula.
            A pergunta de cada teste: entre as 500 limpas, a régua ainda aponta
            o arquivo de mesmo nome?
          </p>
        </Surge>
        <Surge ordem={2} className="transparente">
          <div className="duelo">
            <figure>
              <img src="/imagens/adulterated/laion_050.jpg" alt="Versão degradada do pôster de ícones." />
              <figcaption>O que a busca recebe</figcaption>
            </figure>
            <figure>
              <img src="/imagens/raw/laion_050.jpg" alt="Original do mesmo pôster no recorte do LAION." />
              <figcaption>O arquivo do banco. É o exemplo Ícones, lá em cima.</figcaption>
            </figure>
          </div>
        </Surge>
        <Surge ordem={3} className="transparente">
          <div className="chips">
            <div className="chip"><strong>80%</strong><span>do tamanho</span></div>
            <div className="chip"><strong>5%</strong><span>de cada borda</span></div>
            <div className="chip"><strong>Ruído σ = 8</strong><span>leve</span></div>
            <div className="chip"><strong>JPEG 35</strong><span>bem comprimido</span></div>
          </div>
        </Surge>
        <Surge ordem={3} className="transparente">
          <figure className="grafico-barras">
            <figcaption>Acerto e tempo nas 500, nesta máquina, com a original dentro da gaveta.</figcaption>
            {METODOS.map((item) => (
              <div className="faixa-barra" key={item.nome}>
                <span>{item.nome}</span>
                <div className="barra"><i style={{ "--alvo": item.alvo }} /></div>
                <strong>{item.taxa}</strong>
                <p>{item.texto}</p>
              </div>
            ))}
          </figure>
        </Surge>
        <Surge ordem={1}>
          <p className="texto">
            O CLIP ganhou porque compara sentido, não posição. O SIFT quase empatou
            e foi o mais rápido. O SSIM perdeu no corte. A busca desta página usa o CLIP
            e também recusa: abaixo de 0,82, e sem folga sobre o segundo lugar,
            a resposta é que a imagem não está. É o último exemplo, lá em cima.
          </p>
        </Surge>
      </section>

      <section className="cena">
        <Surge>
          <span className="marco" />
          <p className="kicker">4 · O limite</p>
          <h2>Nas 500 a régua funciona. No banco de verdade, o tempo e o acerto quebram.</h2>
        </Surge>
        <Surge ordem={1}>
          <p className="texto">
            Os 99,6% do CLIP valem só para a gaveta em que a original está presente.
            Um modelo estatístico, calibrado na amostra (99,5% previstos, 99,6% medidos),
            estima o teto quando o banco cresce. Em 52 milhões esse teto cai para 1,8%.
            Imagens parecidas passam na frente da obra. Um artista não poderia usar
            o primeiro lugar como prova no LAION real.
          </p>
        </Surge>
        <Surge ordem={2} className="transparente">
          <GraficoQueda />
        </Surge>
        <Surge ordem={3}>
          <p className="texto">
            O tempo é o outro muro. O SSIM olha uma imagem de cada vez: 0,35 s nas 500,
            9,2 horas no recorte de 52 milhões, cerca de 43 dias no LAION-5B.
            O CLIP, nesta CPU, iria a 8,6 s nas 52 milhões e a cerca de 16 minutos
            no LAION-5B, mas o acerto já teria morrido. O SIFT continuaria em fração
            de segundo só se o índice coubesse: 10,5 TB no recorte e cerca de 1,2 PB
            no LAION-5B. O índice do CLIP cabe melhor, uns 99 GB nas 52 milhões,
            e mesmo assim deixa de identificar a obra. Montar esse índice nesta máquina
            levaria cerca de 33 dias.
          </p>
        </Surge>
        <Surge ordem={4} className="transparente">
          <GraficoTempo />
        </Surge>
        <ol className="caminho">
          {PASSOS.map(([titulo, texto], indice) => (
            <Surge key={titulo} ordem={indice} className="passo-linha">
              <strong>{titulo}</strong>
              <p>{texto}</p>
            </Surge>
          ))}
        </ol>
        <Surge ordem={2}>
          <p className="nota-forte">
            Até um índice menor e um corte medido de novo, o botão do começo responde
            só para estas 500. Suba e teste Ícones, depois Não está: uma acha o arquivo,
            a outra recusa.
          </p>
        </Surge>
      </section>

      <footer className="rodape">
        <strong>Pedro Garcia Ribas</strong>
        <div>Sistemas de Informação · IFMG Campus Sabará</div>
        <p>
          <Link href="https://arxiv.org/abs/2210.08402">Schuhmann et al., 2022</Link>.{" "}
          <Link href="https://openaccess.thecvf.com/content/CVPR2023/html/Somepalli_Diffusion_Art_or_Digital_Forgery_Investigating_Data_Replication_in_Diffusion_CVPR_2023_paper.html">Somepalli et al., CVPR 2023</Link>.{" "}
          <Link href="https://arxiv.org/abs/2301.13188">Carlini et al., 2023</Link>.{" "}
          <Link href="https://www.judiciary.uk/judgments/getty-images-v-stability-ai/">Getty Images v Stability AI</Link>.{" "}
          <Link href="https://www.courtlistener.com/docket/66732129/andersens-v-stability-ai-ltd/">Andersen v. Stability AI</Link>, sem sentença.
          Wang et al., 2004. Lowe, 2004. Radford et al., 2021.
          Figuras recortadas dos artigos em acesso aberto, para esta apresentação.
        </p>
      </footer>
      </main>
    </>
  );
}
