# Art Scraping Detector

**Análise Comparativa de Técnicas de Similaridade Visual para Identificação de Obras Artísticas em Datasets**

Trabalho de Conclusão de Curso — Bacharelado em Sistemas de Informação
Instituto Federal de Minas Gerais (IFMG), *Campus* Sabará

**Autor:** Pedro Garcia Ribas

---

## 1. Escopo acadêmico

Com a popularização de modelos generativos treinados em larga escala, cresceu a
necessidade de verificar se uma obra específica está presente em um determinado
conjunto de dados. O problema é não trivial porque a imagem raramente aparece
idêntica ao original: ela costuma ter sofrido redimensionamento, recorte,
recompressão com perdas, ajuste de cor ou inserção de marca d'água.

Este trabalho investiga **quais técnicas de similaridade visual são mais
eficazes para identificar uma obra artística dentro de um dataset**, mesmo após
essas transformações. A avaliação comparativa contempla três famílias de
abordagens:

| Família | Técnicas previstas | Característica |
| --- | --- | --- |
| Hashing perceptual | aHash, pHash, dHash, wHash | Baixo custo, robusto a compressão |
| Descritores locais | ORB, SIFT/SURF | Robusto a recorte e rotação |
| Aprendizado profundo | *Embeddings* CNN, CLIP | Robusto a alterações semânticas |

As métricas de comparação são precisão, revocação, F1-score, curva ROC/AUC e
tempo médio de consulta por imagem.

## 2. Etapa atual: construção da Amostra de Controle

Este repositório está na **primeira etapa** do projeto: a coleta do dataset.

A **Amostra de Controle** é um conjunto de aproximadamente **1.500 imagens de
arte digital** obtidas de fontes públicas. Ela funciona como o *ground truth* do
experimento: cada imagem original ficará em `data/raw/` e, na etapa seguinte,
gerará versões deliberadamente adulteradas em `data/processed/` (recorte,
rotação, ruído, recompressão, marca d'água). Como a correspondência entre
original e versão adulterada é conhecida por construção, é possível medir
objetivamente o acerto de cada técnica.

### Fontes de dados e justificativa metodológica

Plataformas como **ArtStation** e **Pinterest** foram avaliadas e descartadas:
ambas empregam proteção agressiva da Cloudflare e renderizam as galerias via
JavaScript, o que tornaria a coleta instável e dependente de automação de
navegador. Optou-se por fontes com acesso programático legítimo:

| Fonte | Método de acesso | Módulo |
| --- | --- | --- |
| Reddit (r/DigitalArt, r/Art, r/conceptart e outros) | API oficial autenticada (`praw`) | `reddit_scraper.py` |
| DeviantArt | Feed RSS público (`backend.deviantart.com/rss.xml`) | `deviantart_scraper.py` |
| Unsplash / Openverse | API oficial ou parsing de HTML (`requests` + `BeautifulSoup`) | `web_scraper.py` |

A diversidade de fontes é intencional: evita que o dataset fique enviesado pelo
estilo predominante de uma única comunidade.

> **Nota sobre o Unsplash.** O site responde HTTP 403 ao acesso sem credenciais
> (proteção Cloudflare). Por isso o `web_scraper` trabalha em cascata: usa a API
> oficial do Unsplash quando há `UNSPLASH_ACCESS_KEY` configurada e, caso
> contrário, recorre ao **Openverse** — agregador de imagens Creative Commons da
> WordPress Foundation, com API aberta e sem cadastro, que indexa Flickr, museus
> e acervos digitais. A troca é automática e registrada no log.

### Considerações éticas e legais

* A coleta usa apenas conteúdo público e vias oficiais de acesso, respeitando os
  termos de uso de cada plataforma.
* O pipeline aplica *delays* aleatórios entre requisições e trata explicitamente
  respostas HTTP 429, evitando sobrecarregar os servidores de origem.
* As imagens são utilizadas **exclusivamente para fins acadêmicos**, não são
  redistribuídas e **não são versionadas** neste repositório (ver `.gitignore`).
* Autoria e URL de origem de cada imagem ficam registradas em `data/manifest.json`,
  garantindo rastreabilidade e crédito.

## 3. Tecnologias

| Tecnologia | Uso no projeto |
| --- | --- |
| Python 3.10+ | Linguagem base |
| `praw` | Cliente oficial da API do Reddit |
| `requests` | Requisições HTTP com sessão persistente |
| `beautifulsoup4` + `lxml` | Parsing de HTML e de feeds RSS |
| `python-dotenv` | Isolamento de credenciais em `.env` |
| `tqdm` | Barra de progresso da coleta |
| `hashlib` (SHA-256) | Deduplicação por conteúdo binário |
| `logging` | Monitoramento no terminal e em arquivo |

## 4. Estrutura do projeto

```
art-scraping-detector/
├── data/
│   ├── raw/                     # 1.500 imagens originais coletadas
│   ├── processed/               # versões adulteradas (etapa seguinte)
│   └── manifest.json            # metadados e estado da coleta (gerado)
├── logs/
│   └── scraping.log             # histórico de execução (gerado)
├── src/
│   └── scraper/
│       ├── config.py            # User-Agents, delays, .env, paths, logging
│       ├── models.py            # ImageCandidate e DownloadResult
│       ├── downloader.py        # download, validação, dedup e manifesto
│       ├── reddit_scraper.py    # coleta no Reddit (praw)
│       ├── deviantart_scraper.py# coleta no DeviantArt (RSS)
│       ├── web_scraper.py       # scraping genérico: Unsplash e Openverse
│       └── main_scraper.py      # orquestrador do pipeline
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Instalação

**Pré-requisito:** Python 3.10 ou superior.

```bash
git clone https://github.com/<seu-usuario>/art-scraping-detector.git
cd art-scraping-detector

# Ambiente virtual (Windows / PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Ambiente virtual (Linux / macOS)
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuração das credenciais

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Para obter as credenciais do Reddit (obrigatórias para a fonte principal):

1. Acesse <https://www.reddit.com/prefs/apps> e clique em **create another app**.
2. Escolha o tipo **script** e informe `http://localhost:8080` como *redirect uri*.
3. Copie o `client_id` (texto sob o nome do app) e o `client_secret` para o `.env`.

A chave do Unsplash (`UNSPLASH_ACCESS_KEY`) é **opcional**: sem ela, o
`web_scraper` opera em modo HTML público automaticamente.

## 6. Execução

```bash
# Coleta padrão: até 1.500 imagens, todas as fontes
python -m src.scraper.main_scraper

# Teste rápido sem baixar arquivos (valida as fontes)
python -m src.scraper.main_scraper --dry-run --target 20

# Meta e fontes personalizadas
python -m src.scraper.main_scraper --target 500 --sources reddit deviantart

# Logs detalhados para depuração
python -m src.scraper.main_scraper --log-level DEBUG
```

| Argumento | Descrição | Padrão |
| --- | --- | --- |
| `--target` | Total de imagens desejado em `data/raw/` | `1500` |
| `--sources` | Fontes a utilizar (`reddit`, `deviantart`, `web`), na ordem informada | todas |
| `--dry-run` | Lista os candidatos sem baixar nada | desativado |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` ou `ERROR` | `INFO` |

O `--dry-run` é a forma recomendada de validar credenciais e conectividade
antes de disparar a coleta completa, que leva algumas horas por causa dos
*delays* entre requisições.

## 7. Como o pipeline funciona

1. **Descoberta** — cada módulo de fonte gera `ImageCandidate` (URL, título,
   autor, página de origem) de forma preguiçosa, sem carregar tudo em memória.
2. **Download** — o `ImageDownloader` baixa o binário com até 3 tentativas e
   *backoff* exponencial, tratando timeouts, links quebrados e HTTP 429.
3. **Validação** — o formato real é confirmado pelos *magic bytes* (JPEG, PNG,
   WebP), e arquivos fora da faixa de 15 KB a 25 MB são descartados, eliminando
   ícones, avatares e downloads corrompidos.
4. **Deduplicação** — em dois níveis: URL já visitada e hash SHA-256 do
   conteúdo, o que remove reposts entre subreddits e entre plataformas.
5. **Padronização** — os arquivos são salvos como
   `arte_<fonte>_<sequencial>.<ext>`, por exemplo `arte_reddit_0001.jpg` e
   `arte_deviantart_0002.png`.
6. **Encerramento automático** — a execução para assim que `data/raw/` atinge a
   meta configurada.

**Retomada de execução:** o `data/manifest.json` registra tudo o que já foi
coletado. Se a coleta for interrompida (Ctrl+C, queda de rede ou bloqueio
temporário), basta executar o comando novamente — o pipeline continua de onde
parou, sem duplicar imagens nem reiniciar a numeração.

Ao final, o terminal exibe um relatório com o total coletado, a distribuição por
fonte, os motivos de descarte e o tempo de execução.

## 8. Próximas etapas do TCC

- [x] Arquitetura do projeto e pipeline de coleta
- [ ] Geração das versões adulteradas em `data/processed/`
- [ ] Implementação dos algoritmos de hashing perceptual
- [ ] Implementação dos descritores locais (ORB/SIFT)
- [ ] Implementação da busca por *embeddings* de redes neurais
- [ ] Avaliação comparativa e análise estatística dos resultados
- [ ] Redação da monografia

## 9. Licença e uso

O **código-fonte** deste repositório é disponibilizado para fins acadêmicos e
educacionais. As **imagens coletadas** permanecem sob os direitos autorais de
seus respectivos criadores, não são redistribuídas e são usadas apenas como
insumo experimental desta pesquisa.

---

Desenvolvido por **Pedro Garcia Ribas** — IFMG *Campus* Sabará.
