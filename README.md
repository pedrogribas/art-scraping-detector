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

A **Amostra de Controle** é um conjunto de até **5.000 imagens** obtidas de
fontes públicas. Ela funciona como o *ground truth* do experimento: cada imagem
original fica em `data/raw/` e, na etapa seguinte, gerará versões
deliberadamente adulteradas em `data/processed/` (recorte, rotação, ruído,
recompressão, marca d'água). Como a correspondência entre original e versão
adulterada é conhecida por construção, é possível medir objetivamente o acerto
de cada técnica.

### Composição pretendida

A amostra mistura quatro naturezas visuais distintas, o que evita que os
resultados reflitam as características de um único estilo:

| Natureza | Origem | Peso na meta |
| --- | --- | --- |
| Arte clássica (pinturas, gravuras, aquarelas) | Met e Art Institute of Chicago | 44% |
| Ilustração 2D digital | Safebooru | 20% |
| Render 3D e arte digital contemporânea | Pexels | 16% |
| Arte digital diversa (comunidades) | DeviantArt e Flickr | 20% |

### Fontes de dados e justificativa metodológica

Plataformas como **ArtStation**, **Pinterest** e **Unsplash** foram avaliadas e
descartadas: todas empregam proteção agressiva da Cloudflare e renderizam as
galerias via JavaScript, o que tornaria a coleta instável e dependente de
automação de navegador. A coleta pelo Reddit, usada na versão inicial do
pipeline, foi removida porque o portal de desenvolvedores da plataforma esteve
indisponível para emissão de credenciais.

Optou-se por fontes com acesso programático legítimo e consumo em lote:

| Fonte | Método de acesso | Credencial | Módulo |
| --- | --- | --- | --- |
| Metropolitan Museum of Art | API REST/JSON pública | não | `museum_api_scraper.py` |
| Art Institute of Chicago | API REST/JSON + servidor IIIF | não | `museum_api_scraper.py` |
| Safebooru | API pública no padrão Gelbooru | não | `booru_scraper.py` |
| Pexels | API oficial | **sim** (gratuita) | `pexels_scraper.py` |
| DeviantArt e Flickr | Feeds RSS públicos | não | `rss_scraper.py` |
| Openverse / Unsplash | API aberta (reserva) | opcional | `web_scraper.py` |

### Particularidades tratadas no código

Cada fonte tem restrições próprias, mapeadas em testes e documentadas junto ao
código que lida com elas:

* o servidor IIIF do **Art Institute** responde HTTP 403 sem o header
  `AIC-User-Agent` com contato válido;
* o endpoint `/artworks/search` do mesmo museu recusa páginas acima da décima,
  limitando cada consulta a mil resultados — esgotadas as consultas, o módulo
  migra para a listagem `/artworks`, que não tem esse teto;
* o **Met** não pagina: a busca devolve todos os identificadores de uma vez e
  exige uma requisição de metadados por obra, feita sob demanda;
* o acervo do Met inclui esculturas e cerâmica, descartadas pela classificação
  declarada na obra — o parâmetro `medium` da API não serve para isso, porque
  combinar vários valores com `|` funciona como E lógico e reduz a busca a
  poucas dezenas de resultados;
* no **Safebooru**, cada conjunto de tags tem profundidade própria (`highres
  digital_media` esgota por volta da página 30), o que motivou o uso de vários
  conjuntos;
* o **Pexels** limita o plano gratuito a 200 requisições por hora, situação
  detectada pelo HTTP 429 e tratada como fim da fonte.

### Considerações éticas e legais

* A coleta usa apenas conteúdo público e vias oficiais de acesso, respeitando os
  termos de uso de cada plataforma.
* As obras dos museus são restritas às marcadas como **domínio público** pelas
  próprias instituições.
* O pipeline aplica *delays* aleatórios entre requisições, com intervalo maior
  (de 1 a 3 segundos) nos feeds RSS, e trata explicitamente respostas HTTP 429.
* As imagens são utilizadas **exclusivamente para fins acadêmicos**, não são
  redistribuídas e **não são versionadas** neste repositório (ver `.gitignore`).
* Autoria, licença e URL de origem de cada imagem ficam registradas em
  `data/manifest.json`, garantindo rastreabilidade e crédito.

## 3. Tecnologias

| Tecnologia | Uso no projeto |
| --- | --- |
| Python 3.10+ | Linguagem base |
| `requests` | Requisições HTTP com sessão persistente |
| `beautifulsoup4` + `lxml` | Parsing de feeds RSS |
| `python-dotenv` | Isolamento de credenciais em `.env` |
| `tqdm` | Barra de progresso por módulo de coleta |
| `hashlib` (SHA-256) | Deduplicação por conteúdo binário |
| `logging` | Monitoramento no terminal e em arquivo |

## 4. Estrutura do projeto

```
art-scraping-detector/
├── data/
│   ├── raw/                      # imagens originais coletadas
│   ├── processed/                # versões adulteradas (etapa seguinte)
│   └── manifest.json             # metadados e estado da coleta (gerado)
├── logs/
│   └── scraping.log              # histórico de execução (gerado)
├── src/
│   └── scraper/
│       ├── config.py             # fontes, cotas, delays, .env e logging
│       ├── models.py             # ImageCandidate e DownloadResult
│       ├── downloader.py         # download, validação, dedup e manifesto
│       ├── museum_api_scraper.py # Met + Art Institute of Chicago
│       ├── booru_scraper.py      # Safebooru
│       ├── pexels_scraper.py     # Pexels (requer chave)
│       ├── rss_scraper.py        # DeviantArt + Flickr (RSS)
│       ├── web_scraper.py        # Openverse/Unsplash (reserva)
│       └── main_scraper.py       # orquestrador balanceado
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

### Configuração

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Quatro das cinco fontes funcionam **sem nenhuma credencial**. O `.env` é usado
para dois ajustes:

1. `PEXELS_API_KEY` — chave gratuita obtida em <https://www.pexels.com/api/>.
   Sem ela, o módulo do Pexels é ignorado e sua cota é redistribuída entre as
   demais fontes.
2. `AIC_CONTACT` — e-mail de contato enviado ao Art Institute of Chicago.
   Recomendado por cortesia, já que a instituição pede identificação de quem
   consome o acervo.

## 6. Execução

```bash
# Coleta padrão: até 5.000 imagens, todos os módulos, cotas balanceadas
python -m src.scraper.main_scraper

# Teste rápido sem baixar arquivos (valida credenciais e conectividade)
python -m src.scraper.main_scraper --dry-run --target 24

# Meta e módulos personalizados
python -m src.scraper.main_scraper --target 1000 --sources museum booru

# Logs detalhados para depuração
python -m src.scraper.main_scraper --log-level DEBUG
```

| Argumento | Descrição | Padrão |
| --- | --- | --- |
| `--target` | Total de imagens desejado em `data/raw/` | `5000` |
| `--sources` | Módulos: `museum`, `booru`, `pexels`, `rss`, `web` | todos |
| `--dry-run` | Lista os candidatos sem baixar nada | desativado |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` ou `ERROR` | `INFO` |

O `--dry-run` é a forma recomendada de validar as fontes antes de disparar a
coleta completa, que leva horas por causa dos *delays* entre requisições.

## 7. Como o pipeline funciona

1. **Distribuição das cotas** — a meta é repartida entre os módulos conforme os
   pesos de `config.SOURCE_WEIGHTS`, preservando a composição pretendida.
2. **Rodada balanceada** — cada módulo coleta até a sua cota, com barra de
   progresso própria. Um módulo indisponível não interrompe os demais.
3. **Rodada de compensação** — se a meta não foi atingida (por exemplo, quando o
   Pexels está sem chave), o que faltou é redistribuído entre os módulos que
   ainda têm material, incluindo a fonte de reserva. Como os geradores são
   preguiçosos e preservados entre as rodadas, a coleta continua da página onde
   havia parado, sem repetir requisições.
4. **Download** — até 3 tentativas com *backoff* exponencial, tratando timeouts,
   links quebrados e HTTP 429.
5. **Validação** — o formato real é confirmado pelos *magic bytes* (JPEG, PNG,
   WebP), e arquivos fora da faixa de 15 KB a 25 MB são descartados, eliminando
   ícones, avatares e downloads corrompidos.
6. **Deduplicação** — em dois níveis: URL já visitada e hash SHA-256 do
   conteúdo, o que remove reposts dentro de uma fonte e entre plataformas.
7. **Padronização** — cada fonte tem numeração própria, no formato
   `<fonte>_<sequencial>.<ext>`: `metmuseum_0001.jpg`, `artic_0001.jpg`,
   `safebooru_0001.jpg`, `pexels_0001.jpg`, `deviantart_0001.jpg`,
   `flickr_0001.jpg`. São quatro dígitos porque uma única fonte pode passar de
   mil imagens na meta de 5.000.
8. **Encerramento automático** — a execução para assim que `data/raw/` atinge a
   meta configurada.

**Retomada de execução:** o `data/manifest.json` registra tudo o que já foi
coletado. Se a coleta for interrompida (Ctrl+C, queda de rede ou bloqueio
temporário), basta executar o comando novamente — o pipeline continua de onde
parou, sem duplicar imagens nem reiniciar a numeração.

Ao final, o terminal exibe um relatório com o total coletado, a distribuição por
fonte, o rendimento de cada módulo, os motivos de descarte e o tempo de
execução.

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
seus respectivos criadores ou sob as licenças declaradas pelas instituições de
origem; não são redistribuídas e são usadas apenas como insumo experimental
desta pesquisa.

---

Desenvolvido por **Pedro Garcia Ribas** — IFMG *Campus* Sabará.
