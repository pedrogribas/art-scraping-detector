# Briefing para a monografia (Overleaf)

Documento factual para redigir o TCC. Use somente os números e os limites desta nota. Não invente métricas, tabelas, gráficos nem citações que não estejam aqui. O `README.md` do repositório está desatualizado: hashing perceptual, ROC/AUC, F1 e a pasta `data/processed/` **não** fazem parte do experimento realizado.

---

## Identificação

- **Título de trabalho:** Análise Comparativa de Técnicas de Similaridade Visual para Identificação de Obras Artísticas em Datasets
- **Autor:** Pedro Garcia Ribas
- **Curso:** Bacharelado em Sistemas de Informação
- **Instituição:** Instituto Federal de Minas Gerais (IFMG), Campus Sabará
- **Repositório:** https://github.com/pedrogribas/art-scraping-detector
- **Data das medições:** 28 de setembro de 2026
- **Máquina:** Windows 11, Python 3.13.4, CPU AMD64 (Family 25, Model 33), sem GPU. PyTorch 2.14.0+cpu (`cuda = False`)

Pacotes da prova de conceito: OpenCV 5.0.0, scikit-image 0.26.0, Transformers 5.17.0, Pillow 12.3.0, NumPy 2.5.3. A página pública é React no GitHub Pages (https://pedrogribas.github.io/art-scraping-detector/). Streamlit em `app.py` é rascunho antigo.

---

## O que o experimento responde

Pergunta operacional: dada uma cópia degradada de uma imagem que **está** na galeria, qual algoritmo devolve essa mesma imagem como Top-1, e quanto tempo a busca leva.

É identificação em **conjunto fechado**. O original correspondente sempre está entre as 500 imagens da galeria. Não é detecção em conjunto aberto (“a obra está ou não no LAION?”). Não houve cálculo de precisão, revocação, F1 nem ROC/AUC.

Um acerto é Top-1 com o **mesmo nome de arquivo** da consulta. O nome é preservado de propósito para servir de gabarito.

---

## Amostra de controle

- Fonte: subset inglês estético do LAION-5B, dataset Hugging Face `laion/laion2B-en-aesthetic`, split `train`, leitura em streaming.
- Critério do subset, segundo a documentação do LAION (`laion-aesthetic.md`): escore estético acima de 7, `pwatermark` abaixo de 0,8 e `punsafe` abaixo de 0,5. A parte em inglês desse recorte tem cerca de **52 milhões** de linhas. Uma contagem publicada desse parquet é **52.068.913** linhas (usar essa cifra na projeção; na frase corrida, “52 milhões” é aceitável). Conferir a nota de rodapé na fonte antes de cravar o inteiro.
- LAION-5B, no artigo de Schuhmann et al. (arXiv:2210.08402, 2022): **5,85 bilhões** de pares imagem-texto. LAION-2B-en: cerca de 2,32 bilhões de pares em inglês. O experimento **não** percorreu esses corpora; eles entram só na projeção.
- Foram gravados **500 JPEG** válidos em `data/raw/`, nomeados `laion_001.jpg` a `laion_500.jpg`, regravados com qualidade JPEG 90.
- A coleta percorre o streaming e fica com as primeiras URLs que respondem. Não é amostra aleatória das 52 milhões. Timeout de 3 s, corpo máximo de 20 MB. HTTP 404, timeout, corpo vazio e arquivo que o Pillow não reconhece são ignorados.
- O material não é só “obra de arte” de museu. O escore estético do LAION mistura fotografia, ilustração, ícones e outros conteúdos da web. Não descrever a amostra como 500 pinturas.

Referências para a outra IA citar, se já estiverem no `.bib` ou forem incluídas:

- Schuhmann, C. et al. LAION-5B: An open large-scale dataset for training next generation image-text models. arXiv:2210.08402, 2022.
- Documentação do subset estético: https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md
- Dataset: https://huggingface.co/datasets/laion/laion2B-en-aesthetic
- CLIP: Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021. Modelo usado: `openai/clip-vit-base-patch32`.
- SSIM: Wang, Z. et al. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 2004.
- SIFT: Lowe, D. G. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 2004.

Não inventar o número de página nem o BibTeX completo se ele não estiver no projeto Overleaf.

---

## Adulteração (a consulta)

Cada original em `data/raw/` gera um arquivo de mesmo nome em `data/adulterated/`. O pipeline é **fixo**, igual para as 500, para a comparação medir o algoritmo e não a intensidade da degradação. A semente do ruído é `zlib.crc32` do nome do arquivo, então a mesma original produz sempre a mesma versão degradada.

| Etapa | Parâmetro |
|---|---|
| Redimensionamento | escala 0,80, interpolação por área (`INTER_AREA`) |
| Corte | 5% de cada borda (`CROP_RATIO = 0,05`), se sobrar pelo menos 16 px |
| Ruído | gaussiano, média 0, desvio-padrão σ = 8, depois clip para 0–255 |
| JPEG | qualidade 35 |

Exemplo conferido, `laion_050.jpg`: original 450×450 px (62.273 bytes); consulta 324×324 px (39.404 bytes). A conta fecha: 450 × 0,80 = 360; corte de 5% em cada lado deixa 324. O conteúdo (a grade de ícones) é o mesmo. A consulta fica menor, com as bordas removidas, granulado leve e blocos de JPEG.

---

## Três algoritmos

A galeria (`data/raw/`) é indexada **uma vez**. O tempo de busca é só o de `buscar`, por consulta. A indexação é relatada à parte e não entra na média de busca.

### SSIM

- Família: similaridade estrutural, pixel a pixel, depois de normalizar o tamanho.
- Implementação: `skimage.metrics.structural_similarity`, `data_range=1,0`.
- Cada imagem vira cinza float32 em [0, 1], redimensionada para **128×128**.
- A busca é varredura linear: o Top-1 é o maior SSIM contra as 500 referências.
- Por que sofre com este pipeline: o corte de 5% desloca o conteúdo. O SSIM compara posições correspondentes da matriz. Não há alinhamento.

### SIFT

- Família: descritor local.
- `cv2.SIFT_create(nfeatures=500)`.
- Todos os descritores da galeria entram num único `FlannBasedMatcher`: algoritmo 1 (KD-tree), 5 árvores, `checks=50`.
- Cada descritor da consulta faz `knnMatch` com k=2. Vale o teste de razão de Lowe com limiar **0,75** (distância do 1º vizinho < 0,75 × distância do 2º).
- O vizinho aceito vota na imagem de origem. Top-1 = imagem com mais votos.
- Na recontagem para a projeção, a média foi **434,3 descritores por imagem** (teto de 500).

### CLIP

- Família: embedding de visão e linguagem.
- Modelo `openai/clip-vit-base-patch32`, inferência em **CPU**.
- Embedding L2-normalizado de dimensão **512**. Similaridade = produto interno, que equivale ao cosseno.
- A busca é o maior cosseno contra a matriz da galeria (`torch.mv` / produto de matrizes). Força bruta, sem FAISS nem índice aproximado.
- O tempo de busca publicado (0,0564 s) inclui embedar a consulta e varrer as 500. Na decomposição da projeção, embedar uma consulta leva **0,0546 s** e varrer uma imagem já indexada leva **0,164 µs**. Em N=500 o custo quase todo é o embedding.

---

## Resultados medidos (oficial, N = 500)

Fonte: `python -m src.main_eval`, 28 set 2026. Estes são os números do capítulo de resultados. Não substituí-los pelos tempos da projeção.

| Método | Acertos | Acurácia Top-1 | Indexação (s) | Busca média (s/imagem) |
|---|---:|---:|---:|---:|
| SSIM | 443/500 | 88,6% | 1,09 | 0,3507 |
| SIFT | 496/500 | 99,2% | 21,56 | 0,0383 |
| CLIP | 498/500 | 99,6% | 43,04 | 0,0564 |

Leitura que o texto pode sustentar, porque está ligada ao desenho do teste:

- O CLIP foi o mais preciso (498/500). O embedding tolera ruído e JPEG porque a comparação não é posição a posição.
- O SIFT foi quase tão preciso (496/500) e teve a busca mais curta (0,0383 s). Os descritores locais sobrevivem ao corte.
- O SSIM foi o menos preciso (443/500) e o mais lento na busca (0,3507 s). O corte de 5% desalinha as matrizes 128×128.
- A indexação do CLIP (43,04 s) inclui o carregamento do modelo. A do SIFT (21,56 s) é a extração dos descritores e o treino do FLANN. A do SSIM (1,09 s) é só o pré-processamento para 128×128.

Não afirmar que o CLIP “ignora” o corte em sentido absoluto. Duas consultas em 500 ainda erraram. Não listar quais arquivos falharam: isso não foi tabulado.

Ilustração qualitativa, não é métrica: na interface, a consulta `laion_050.jpg` recuperou o original com cosseno 0,951; `laion_001.jpg` (xícara), cosseno 0,989; `laion_330.jpg` (silhueta no pôr do sol), cosseno 0,906. São exemplos de demonstração, não a média do experimento. A média do cosseno do par verdadeiro, nas 500 consultas, é **0,910**. A média do cosseno entre pares que não são o mesmo arquivo é **0,469** (desvio-padrão **0,091**).

Escores SSIM na mesma passada de diagnóstico (não usar como acurácia): par verdadeiro médio **0,331**; impostor médio **0,103** (desvio-padrão **0,072**).

---

## Projeção de escala — o que pode entrar no texto

Nenhuma busca foi executada nas 52.068.913 imagens nem nos 5,85 bilhões. A projeção usa custos medidos nesta CPU e, só para o CLIP, um modelo estatístico. Script reprodutível: `python -m src.estimate_scale`. JSON: `results/estimativa_laion.json`.

### Tempo de busca

Premissas:

- **SSIM:** linear no tamanho da galeria. Custo medido no laço de comparação: **635,5 µs por imagem** (0,3177 s para varrer 500, só o SSIM; o 0,3507 s oficial inclui leitura de arquivo). Projeção: `T(N) = 635,5e-6 × N` segundos.
- **CLIP:** custo fixo de embedar a consulta (0,0546 s) mais varredura linear dos vetores de 512 dimensões (0,164 µs por imagem). `T(N) = 0,0546 + 0,164e-6 × N` segundos. Vale para esta CPU, força bruta, embeddings já na RAM.
- **SIFT:** extração da consulta (0,0198 s) mais o casamento FLANN. Com `checks=50`, o tempo cresce com o logaritmo do número de descritores, não com N. A fórmula usada foi `T(N) = 0,0198 + 0,0145 × log2(D_N) / log2(D_500)`, com `D_500 = 434,3 × 500`. Esse tempo **só existiria se o índice coubesse na memória**. No subset de 52 milhões e no LAION-5B, não cabe. Não escrever que o SIFT responderia em 0,05 s no LAION-5B como resultado alcançável.

### Memória do índice (ordem de grandeza, sem overhead de estrutura)

- SSIM: `N × 128 × 128 × 4` bytes (cinza float32 já redimensionado).
- SIFT: `N × 434,3 × 128 × 4` bytes (descritores float32; o KD-tree real ocupa mais).
- CLIP: `N × 512 × 4` bytes.

### Acurácia projetada — só CLIP

Para cada consulta com escore verdadeiro `s`, se os cossenos impostores fossem normais i.i.d. com média 0,4694 e desvio 0,0912,

`P(acerto) = Φ((s − μ) / σ) ^ (N − 1)`.

A acurácia estimada é a média dessa probabilidade nas 500 consultas.

Calibração, o argumento para usar o modelo: em N = 500 ele prevê **99,5%**, e o experimento mediu **99,6%**.

Limite obrigatório no texto: é um **teto otimista**. O LAION tem quase-duplicatas e vizinhos semânticos que esta amostra de 500 não representa. Isso aumenta a cauda de impostores fortes e derruba a acurácia abaixo da curva. O modelo também trata os impostores como independentes.

O mesmo modelo gaussiano no SSIM previu **44,5%** em N = 500, contra **88,6%** medidos. Foi **descartado**. Não publicar acurácia projetada de SSIM (os 17% e 11% que o script imprime não entram na monografia). Não há projeção de acurácia para o SIFT: o critério é contagem de votos, o FLANN não examina a galeria inteira, e o índice não cabe em RAM.

### Tabela para o Overleaf

Tempos de **uma consulta**. Acurácia só do modelo CLIP. Memória do índice.

| Galeria | N | SSIM | CLIP | Acurácia CLIP (modelo) | Índice CLIP | Índice SIFT | Índice SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amostra | 500 | 0,32 s | 0,055 s | 99,5% | 1 MB | 106 MB | 31 MB |
| — | 10.000 | 6,4 s | 0,056 s | 96,0% | 20 MB | 2,1 GB | 625 MB |
| — | 100.000 | 1,1 min | 0,071 s | 84,2% | 196 MB | 21 GB | 6,1 GB |
| — | 1.000.000 | 10,6 min | 0,22 s | 52,0% | 1,9 GB | 207 GB | 61 GB |
| Subset estético EN | 52.068.913 | 9,2 h | 8,6 s | 1,8% | 99 GB | 10,5 TB | 3,1 TB |
| LAION-5B | 5.850.000.000 | 43 dias | 16 min | < 0,01% | 10,9 TB | 1,2 PB | 349 TB |

Valores de origem, se a tabela precisar de mais casas:

- 52.068.913: SSIM 33.088,6 s (9,19 h); CLIP 8,58 s; acurácia 1,78%; CLIP 99,3 GB; SIFT 10.782 GB (10,5 TB); SSIM 3.178 GB (3,10 TB).
- 5,85 bilhões: SSIM 3.717.545 s (43,0 dias); CLIP 957,7 s (16,0 min); acurácia do modelo 1,8×10⁻¹⁵ %, escrever “inferior a 0,01%” ou “praticamente nula”, não a notação crua; CLIP 11.158 GB (10,9 TB); SIFT 1.211.406 GB (1,16 PB, no texto “cerca de 1,2 PB”); SSIM 357.056 GB (349 TB).

Indexar o CLIP do zero, nesta CPU, a 0,0546 s por imagem: cerca de **33 dias** para 52.068.913 imagens e cerca de **10 anos** para o LAION-5B, numa única máquina, sem contar falha de download. Isso é custo de construção do índice, separado da consulta.

### Frases que o capítulo pode usar

- Na amostra fechada de 500, o CLIP identifica a original em 99,6% das consultas (498/500), o SIFT em 99,2% (496/500) e o SSIM em 88,6% (443/500).
- O tempo médio de busca foi 0,0383 s (SIFT), 0,0564 s (CLIP) e 0,3507 s (SSIM).
- Estender a mesma busca por força bruta ao subset estético inglês (52.068.913 imagens) muda o quadro. O modelo do CLIP, calibrado na amostra, estima um teto de 1,8% de Top-1. Cada consulta levaria cerca de 8,6 s nesta CPU, com um índice de 99 GB.
- O SSIM levaria cerca de 9,2 horas por consulta nesse subset e 43 dias no LAION-5B, com índices de 3,1 TB e 349 TB.
- O SIFT permanece rápido apenas enquanto o índice existe. Os descritores ocupariam 10,5 TB no subset estético e cerca de 1,2 PB no LAION-5B. A implementação testada não constrói esse índice.
- A acurácia de 99,6% não é uma propriedade do LAION. É a taxa de acerto numa galeria de 500 arquivos em que o original sempre está presente, depois de uma degradação fixa.

### O que não escrever

- Não dizer que o experimento rodou no LAION inteiro.
- Não dizer que o CLIP manteria 99,6% em milhões de imagens.
- Não apresentar o tempo do SIFT em bilhões de imagens como medida.
- Não incluir a curva de acurácia do SSIM produzida pelo modelo normal.
- Não tratar 1,8% como piso. É teto sob independência e sob a distribuição desta amostra.
- Não afirmar que um índice aproximado (FAISS, HNSW) resolveria a acurácia. Isso não foi testado. Pode aparecer como trabalho futuro: a memória e o tempo do CLIP em 52 milhões cabem num servidor, mas a identificação Top-1 por força bruta deixa de ser confiável quando o número de impostores cresce.
- Não ressuscitar scrapers de Reddit, DeviantArt, museus ou Pexels como método executado. Foram abandonados. A amostra veio do LAION.

---

## Sugestão de figuras

1. **Resultados na amostra.** Barras de acurácia (88,6; 99,2; 99,6) e barras de tempo de busca (0,351; 0,038; 0,056 s). Título sugerido: “Acurácia Top-1 e tempo médio de busca na galeria de 500 imagens.”
2. **Acurácia do CLIP contra o tamanho da galeria**, eixo x logarítmico, pontos 99,5; 96,0; 84,2; 52,0; 1,8; ~0. Legenda: estimativa do modelo gaussiano; o ponto em 500 reproduz o valor medido (99,6%). Não plotar SSIM nem SIFT nesta figura.
3. **Tempo de busca por consulta**, eixo y logarítmico, três curvas. Anotar na curva do SIFT: “pressupõe o índice em RAM; inviável a partir de centenas de gigabytes de descritores.”
4. **Par visual** original × adulterado (por exemplo `laion_050.jpg`), com a legenda dos quatro parâmetros (escala 80%, corte 5%, σ = 8, JPEG 35).

A página pública da prova é https://pedrogribas.github.io/art-scraping-detector/ (React, GitHub Pages). Os exemplos da busca usam o CLIP medido neste briefing; o recorte “a imagem não está” é regra calibrada nas 500, não entra na tabela oficial. O `app.py` em Streamlit ficou como rascunho: não citar como interface do trabalho.

Os PNG das figuras 1 a 3 não foram exportados pelo painel antigo; se o Overleaf precisar de arquivo, gerar a partir desta tabela.

---

## Trabalho futuro honesto

- Amostra aleatória do subset, em vez das primeiras URLs que responderam.
- Conjunto aberto: consultas cuja original não está na galeria.
- Degradações separadas (só corte, só JPEG, só ruído) para atribuir o erro do SSIM.
- Índice aproximado para o CLIP e medida real de acurácia numa galeria maior que 500, para testar o teto de 1,8%.
- Perceptual hashing (aHash, pHash, dHash), citado no plano inicial, não foi implementado.
)
