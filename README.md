# Auditor de Datasets

**Análise comparativa de técnicas de similaridade visual para identificação de obras artísticas em datasets**

Trabalho de Conclusão de Curso — Bacharelado em Sistemas de Informação  
Instituto Federal de Minas Gerais (IFMG), Campus Sabará

**Autor:** Pedro Garcia Ribas  
**Repositório:** https://github.com/pedrogribas/art-scraping-detector

A página pública, com as 500 imagens e a busca no ar, é  
https://pedrogribas.github.io/art-scraping-detector/

No alto: campo para um JPG e sete exemplos (seis adulterados que **estão** no banco e um `fora.jpg` que **não está**). Ao rolar: origem do LAION, as cópias das pesquisas, a prova nas 500 e o que os 99,6% deixam de significar no banco grande.

A busca no GitHub Pages roda no navegador. Os exemplos já vêm ranqueados com o CLIP oficial desta prova. Um arquivo seu baixa o modelo uma vez (cerca de 150 MB) e consulta o índice das 500. Não há cota paga. O Hugging Face cobrou plano PRO para Space Docker; essa via não foi usada. O site é Pages, de graça.

Para apresentar no computador, com a API Python (o mesmo CLIP da tabela oficial):

```powershell
.\apresentar.ps1
```

Sobe `src/api_server.py` em http://127.0.0.1:8000 e o Vite em http://localhost:5173. Na primeira vez a API leva cerca de um minuto para indexar.

As 500 JPEG já estão em `data/raw/` (**só para fins acadêmicos** deste TCC). São obras de terceiros. Não republicar, não vender, não usar para treinar modelo. A lista oficial do LAION saiu do ar em dezembro de 2023; estes arquivos já tinham sido baixados. Veja `data/LEIA-ME.txt`.

---

## O problema

Modelos generativos são treinados em conjuntos da escala do LAION-5B (5,85 bilhões de pares imagem-texto; Schuhmann et al., 2022). Uma obra que circula na internet raramente volta idêntica ao arquivo original: foi redimensionada, cortada, recomprimida e, às vezes, recebeu ruído. A pergunta deste TCC é qual família de similaridade visual ainda devolve essa imagem como Top-1 depois de uma degradação fixa.

Três técnicas foram comparadas, sem retreinar modelo algum:

| Família | Técnica neste trabalho | O que compara |
| --- | --- | --- |
| Similaridade estrutural | SSIM em cinza 128×128 | Pixels alinhados |
| Descritor local | SIFT + FLANN, teste de Lowe 0,75 | Detalhes que votam na imagem de origem |
| Embedding | CLIP ViT-B/32, cosseno em 512 dimensões | Semântica da imagem |

Hashing perceptual (aHash, pHash, dHash), ROC/AUC e F1 constavam do plano inicial e **não foram medidos**.

## O que foi medido

Identificação em **conjunto fechado**: o original sempre está na galeria. Um acerto é o Top-1 com o mesmo nome de arquivo. Não é a pergunta “esta obra está no LAION?”.

Amostra: **500 JPEG** do subset inglês `laion/laion2B-en-aesthetic` (escore estético acima de 7; cerca de 52.068.913 linhas no subset). São as primeiras URLs do streaming que responderam, não uma amostra aleatória, e não são 500 pinturas de museu.

Cada original em `data/raw/` gerou uma cópia de mesmo nome em `data/adulterated/`, com pipeline fixo:

- escala de 80%;
- corte de 5% em cada borda;
- ruído gaussiano com σ = 8 (semente = CRC32 do nome do arquivo);
- JPEG com qualidade 35.

Medição em CPU, sem GPU, em 28 de setembro de 2026 (`python -m src.main_eval`). O tempo de busca não inclui a indexação.

| Método | Acertos | Acurácia Top-1 | Indexação | Busca média |
| --- | ---: | ---: | ---: | ---: |
| SSIM | 443/500 | 88,6% | 1,09 s | 0,3507 s |
| SIFT | 496/500 | 99,2% | 21,56 s | 0,0383 s |
| CLIP | 498/500 | 99,6% | 43,04 s | 0,0564 s |

O SSIM sofre com o corte, porque compara posições correspondentes da matriz. O SIFT foi o mais rápido na busca. O CLIP foi o mais preciso nesta gaveta de 500 arquivos.

## O que a projeção mostra — e o que ela não é

Nenhuma busca rodou nas 52 milhões de linhas do subset nem nos 5,85 bilhões do LAION-5B. `python -m src.estimate_scale` projeta tempo, memória e, só para o CLIP, um teto de acurácia. O modelo gaussiano dos cossenos impostores (média 0,47, desvio 0,09) devolve 99,5% em N = 500, contra 99,6% medidos. Fora da amostra é um **teto otimista**: quase-duplicatas que estas 500 imagens não contêm derrubariam o número. O mesmo modelo no SSIM previu 44,5% contra 88,6% medidos e foi descartado. Não há acurácia projetada para o SIFT.

| Galeria | Busca SSIM | Busca CLIP | Teto CLIP | Índice CLIP | Índice SIFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0,32 s | 0,05 s | 99,5% | 1 MB | 106 MB |
| 52.068.913 | 9,2 h | 8,6 s | 1,8% | 99 GB | 10,5 TB |
| 5,85 bilhões | 43 dias | 16 min | < 0,01% | 10,9 TB | cerca de 1,2 PB |

O tempo do SIFT ficaria na casa de 0,05 s só se o índice coubesse na memória. Não cabe. A implementação testada não constrói um índice de terabytes.

O briefing usado na redação da monografia está em [`briefing-tcc-overleaf.md`](briefing-tcc-overleaf.md), com os limites que o texto não deve ultrapassar.

## Página

A página viva é o React em `web/`, nas cores do Manual de Identidade Visual da marca Instituto Federal: verde `#2f9e41` e vermelho `#cd191e`. A API está em `src/api_server.py`.

1. No topo, o campo para inserir um JPG ou PNG e os sete exemplos (incluindo `fora.jpg`).
2. Conta como presente se o primeiro cosseno passa de **0,82**, ou se fica acima de **0,75** e abre pelo menos **0,10** sobre o segundo. Nas 500, a mediana do par verdadeiro foi 0,915; `fora.jpg` fez 0,773 com folga 0,014 e foi recusada. Essa regra vale para esta gaveta, não para o LAION inteiro.
3. Ao rolar: origem do LAION, as figuras de Somepalli e Carlini, a prova nas 500 e o limite no banco grande, com gráficos de barra e de linha.

O build estático mora em `docs/` (GitHub Pages). O índice CLIP da página (`web/public/indice/`) sai de `python -m src.exportar_indice`. O Streamlit em `app.py` é rascunho antigo.

## Como reproduzir

Python 3.10 ou superior. O ambiente usado no experimento foi o Python 3.13.4.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

O clone já traz `data/raw/` (500), `data/adulterated/`, `data/exemplos/` (os sete da página) e o índice estático. Não precisa baixar de novo para ver a prova.

```bash
python -m src.main_eval                   # acuracia e tempo dos tres metodos
python -m src.estimate_scale              # projecao; grava results/estimativa_laion.json
cd web
npm install
cd ..
.\apresentar.ps1
```

A lista oficial do LAION não está mais no ar. `python -m src.scraper.laion_downloader` só faria sentido se os links ainda respondessem (muitos já falham). Quem for repetir a coleta: aceite os termos em https://huggingface.co/datasets/laion/laion2B-en-aesthetic, rode `hf auth login` e, se quiser, exporte `HF_TOKEN` (veja `.env.example`). Não commite o token. O downloader para em 500 JPEG válidos, timeout de 3 s, corpo máximo de 20 MB.

Para regenerar as cópias degradadas ou o índice da página:

```bash
python -m src.adulterator
python -m src.exportar_indice
```

## Estrutura

```
art-scraping-detector/
├── apresentar.ps1                  # API :8000 + pagina :5173
├── app.py                          # rascunho Streamlit antigo
├── briefing-tcc-overleaf.md        # limites para a monografia
├── docs/                           # build do GitHub Pages
├── requirements.txt
├── web/                            # React; indice em public/indice/
├── src/api_server.py               # busca CLIP local e corte "nao esta"
├── src/exportar_indice.py          # gera o indice da pagina estatica
├── results/estimativa_laion.json
└── data/                           # raw, adulterated, exemplos; so academico
```

As 500 JPEG entram no Git **só para fins acadêmicos**. São obras de terceiros. Não republicar, não vender, não usar para treinar modelo.

## Limites que o leitor precisa levar

- 99,6% vale para 500 arquivos em que o original sempre está presente.
- A amostra é de conveniência (as primeiras URLs que baixaram), não um sorteio das 52 milhões.
- A degradação é uma combinação só. Não se sabe, por este teste, quanto do erro do SSIM vem do corte e quanto vem do JPEG.
- O tempo e a memória da projeção são desta CPU, com força bruta. Não houve FAISS nem outra busca aproximada.
- O índice do SIFT na escala do LAION é uma conta de armazenamento, não uma busca executada.

## Referências de método

- Schuhmann, C. et al. *LAION-5B: An open large-scale dataset for training next generation image-text models*. arXiv:2210.08402, 2022.
- Subset estético: https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md
- Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021. Modelo: `openai/clip-vit-base-patch32`.
- Wang, Z. et al. *Image quality assessment: from error visibility to structural similarity*. IEEE TIP, 2004.
- Lowe, D. G. *Distinctive image features from scale-invariant keypoints*. IJCV, 2004.
- Cores da página: Manual de Identidade Visual da marca Instituto Federal (verde `#2f9e41`, vermelho `#cd191e`).

---

Desenvolvido por **Pedro Garcia Ribas** — IFMG Campus Sabará.
