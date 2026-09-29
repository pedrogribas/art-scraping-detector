const LIMIAR_COSENO = 0.82;
const PISO_COSENO = 0.75;
const LIMIAR_FOLGA = 0.10;
const DIM = 512;
const MODELO = "Xenova/clip-vit-base-patch32";

const BASE = import.meta.env.BASE_URL;

let indice = null;
let clip = null;
let clipCarregando = null;

export function urlGaleria(nome) {
  return urlPublica(`imagens/raw/${nome}`);
}

export function urlPublica(caminho) {
  const limpo = caminho.startsWith("/") ? caminho.slice(1) : caminho;
  return `${BASE}${limpo}`;
}

export function ajustarResposta(resposta) {
  return {
    ...resposta,
    ranking: resposta.ranking.map((item) => ({
      ...item,
      url: urlGaleria(item.arquivo),
    })),
  };
}

export async function carregarIndice() {
  if (indice) return indice;
  const [nomes, buffer, exemplos] = await Promise.all([
    fetch(urlPublica("indice/clip_nomes.json")).then((r) => {
      if (!r.ok) throw new Error("Índice CLIP ausente");
      return r.json();
    }),
    fetch(urlPublica("indice/clip_vetores.bin")).then((r) => {
      if (!r.ok) throw new Error("Vetores CLIP ausentes");
      return r.arrayBuffer();
    }),
    fetch(urlPublica("indice/exemplos.json")).then((r) => {
      if (!r.ok) throw new Error("Exemplos CLIP ausentes");
      return r.json();
    }),
  ]);
  indice = { nomes, vetores: new Float32Array(buffer), exemplos };
  if (indice.vetores.length !== nomes.length * DIM) {
    throw new Error("Índice CLIP incompleto");
  }
  return indice;
}

export function exemploPronto(arquivo) {
  const bruto = indice?.exemplos?.[arquivo];
  return bruto ? ajustarResposta(bruto) : null;
}

function normalizar(vetor) {
  let norma = 0;
  for (let i = 0; i < vetor.length; i += 1) norma += vetor[i] * vetor[i];
  norma = Math.sqrt(norma) || 1;
  const saida = new Float32Array(vetor.length);
  for (let i = 0; i < vetor.length; i += 1) saida[i] = vetor[i] / norma;
  return saida;
}

function rankear(consulta) {
  const { nomes, vetores } = indice;
  const scores = new Float32Array(nomes.length);
  for (let i = 0; i < nomes.length; i += 1) {
    let soma = 0;
    const base = i * DIM;
    for (let j = 0; j < DIM; j += 1) soma += vetores[base + j] * consulta[j];
    scores[i] = soma;
  }
  const ordem = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
  const top = ordem.slice(0, 3);
  const primeiro = scores[top[0]];
  const segundo = top.length > 1 ? scores[top[1]] : 0;
  const folga = primeiro - segundo;
  const presente = primeiro >= LIMIAR_COSENO || (primeiro >= PISO_COSENO && folga >= LIMIAR_FOLGA);
  return ajustarResposta({
    nome: "consulta.jpg",
    presente,
    limiar: LIMIAR_COSENO,
    folga: Number(folga.toFixed(3)),
    ranking: top.map((indiceItem, posicao) => ({
      posicao: posicao + 1,
      arquivo: nomes[indiceItem],
      score: Number(scores[indiceItem].toFixed(3)),
      url: `/imagens/raw/${nomes[indiceItem]}`,
      mesmo_arquivo: false,
    })),
  });
}

async function carregarModelo(aoProgresso) {
  if (clip) return clip;
  if (clipCarregando) return clipCarregando;
  clipCarregando = (async () => {
    const { AutoProcessor, CLIPVisionModelWithProjection, env } = await carregarTransformers();
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    if (aoProgresso) aoProgresso("Baixando o CLIP no navegador (uma vez)…");
    const [processor, vision] = await Promise.all([
      AutoProcessor.from_pretrained(MODELO),
      CLIPVisionModelWithProjection.from_pretrained(MODELO, {
        dtype: "fp32",
      }),
    ]);
    clip = { processor, vision };
    return clip;
  })();
  try {
    return await clipCarregando;
  } catch (erro) {
    clipCarregando = null;
    throw erro;
  }
}

async function carregarTransformers() {
  return import(
    /* @vite-ignore */
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm"
  );
}

export async function buscarImagem(fonte, nome, aoProgresso) {
  await carregarIndice();
  const { RawImage } = await carregarTransformers();
  const { processor, vision } = await carregarModelo(aoProgresso);
  if (aoProgresso) aoProgresso("Comparando com as 500…");
  const imagem = fonte instanceof Blob ? await RawImage.fromBlob(fonte) : await RawImage.read(fonte);
  const entradas = await processor(imagem);
  const { image_embeds } = await vision(entradas);
  const consulta = normalizar(image_embeds.data);
  const resposta = rankear(consulta);
  resposta.nome = nome || "consulta.jpg";
  resposta.ranking = resposta.ranking.map((item) => ({
    ...item,
    mesmo_arquivo: item.arquivo === resposta.nome,
  }));
  return resposta;
}
