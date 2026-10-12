# PAVD: guia de uso e decisões metodológicas

O PAVD (Protocolo de Auditoria Visual de Datasets) é uma cadeia técnica para
**localizar, verificar, rejeitar e documentar correspondências visuais**. Ele
não decide autoria, licença, infração, scraping ilícito ou se um modelo foi
treinado com uma imagem.

## Princípio central

O protocolo separa quatro perguntas que não devem ser confundidas:

1. **A imagem consulta é parecida com algum item da galeria?**
2. **O candidato recuperado preserva correspondência local/geométrica?**
3. **A consulta pode estar ausente da galeria?**
4. **Quais dados e parâmetros permitem reproduzir a decisão?**

O fluxo global usa CLIP/FAISS para descoberta e SIFT/Lowe/RANSAC apenas para
verificação dos candidatos. SSIM permanece como baseline pareado, pois é
interpretável, mas não é escalável para uma galeria grande.

## Modos de entrada

### Diretório local

Use quando os arquivos estão disponíveis localmente:

```powershell
python -m pavd.cli.build_gallery --source data/raw
python -m pavd.cli.generate_queries --source data/raw --limit 500 --resume
python -m pavd.cli.classify_categories --source data/raw
python -m pavd.cli.build_clip_index --source data/raw --limit 500
```

### Manifesto de uma base

Um manifesto Parquet pode conter `gallery_id`, `source_id`, `source_url`,
`caption`, `aesthetic_score` e, se permitido, campos de proveniência. O PAVD
preserva o vínculo `posição do índice -> gallery_id -> origem`; nunca usa a
posição do vetor como identidade sem esse mapeamento.

### Dataset baseado em URLs

Para o LAION, o fluxo padrão não baixa imagens:

```powershell
python -m pavd.cli.build_remote_manifest --limit 100
```

Esse comando produz um catálogo URL-only. Metadados não são embeddings:
portanto, não permitem declarar Recall visual, MRR ou ANN Recall sem que um
índice de embeddings pré-computado e auditável seja fornecido. O download é
uma rota opcional, bloqueada por padrão:

```powershell
python -m pavd.cli.download_gallery --target 100000 --allow-download
```

## Por que estes métodos?

| Componente | Escolha | Justificativa | Limitação |
|---|---|---|---|
| SSIM | Cinza, 128x128 | Baseline simples para estrutura alinhada | Sensível a crop, escala e deslocamento |
| SIFT | 500 features, Lowe 0,75, RANSAC | Evidência local e geométrica interpretável | Falha em imagens planas ou sem textura |
| CLIP | ViT-B/32, vetor normalizado, cosseno | Recuperação global robusta a alterações semânticas/visuais | Similaridade semântica não prova identidade |
| FAISS | Flat no controle; HNSW/IVF/PQ em escala | Separa busca exata de busca aproximada | ANN pode perder o vizinho verdadeiro |
| Top-50 | Orçamento de reranking | Limita o custo local e preserva candidatos | Um item fora do Top-50 não pode ser recuperado pelo SIFT |
| Open set | Score absoluto + margem | Permite rejeitar consultas ausentes | Limiares dependem da galeria e devem ser calibrados |

## Por que estas categorias?

As classes primárias foram escolhidas para cobrir os modos visuais presentes na
galeria e o comportamento esperado dos métodos: fotografia, pintura digital,
arte vetorial, logo, colagem e ambígua/mista. A última classe evita forçar uma
imagem híbrida em uma taxonomia inadequada.

As classes primárias são estimadas por CLIP zero-shot com prompts textuais.
Não vieram dos metadados LAION e não são rótulos ontológicos. Cada registro
contém confiança e `manual_review_status=pending`.

Os atributos secundários não são mutuamente exclusivos:

- modo de cor: colorida, pouco saturada/escala de cinza, preto e branco;
- composição: paisagem, retrato, quadrada;
- textura: baixa, média, alta;
- densidade de bordas e saturação média;
- texto visível, rosto visível e fundo uniforme/complexo.

Cor, proporção, bordas e textura são medidas de imagem. Texto, rosto e fundo
são estimativas zero-shot e devem ser tratados como variáveis auxiliares.

## Protocolo experimental

### Bloco A — 500 itens

As 500 imagens oficiais têm suas consultas degradadas e a fonte correta está
presente na galeria. O resultado é fechado: Top-1 mede recuperação do mesmo
arquivo-fonte, não presença no LAION inteiro.

```powershell
python -m pavd.cli.run_tcc_evaluation
python -m pavd.cli.build_tcc_report
```

Saídas:

- `results/pavd/block_a_per_query.csv`;
- `results/pavd/block_a_summary.csv`;
- `results/pavd/block_a_top1.png`;
- `results/pavd/block_a_by_category.png`;
- distribuições de categoria, cor e textura;
- JSON com parâmetros e status da execução.

### Bloco B — piloto URL-only com 100 itens

As 500 consultas continuam sendo as mesmas. O experimento atual acrescenta
100 itens URL-only às 500 fontes oficiais. O relatório válido é de
**cobertura do catálogo, proveniência e custo estimado**, não de desempenho
visual medido. Para medir CLIP/FAISS, é obrigatório fornecer embeddings
pré-computados alinhados ao manifesto ou tornar os bytes acessíveis.

A expansão para aproximadamente 100 mil itens permanece como etapa futura e
não é representada pelos resultados atuais.

O relatório separa:

- medido: número de linhas do manifesto e campos disponíveis;
- calculado: memória de embeddings, custo de produto interno e tamanho de
  índice;
- não disponível: Top-1, Recall@K, MRR, ranks e ANN Recall sem embeddings.

## Reprodutibilidade e evidência

Cada execução deve registrar:

- versão do código e configuração YAML;
- modelo, dimensão e precisão do embedding;
- tipo e parâmetros FAISS;
- seed e pipeline de degradação;
- `gallery_id`, `source_id`, URL e hashes quando houver arquivo local;
- score, rank, Top-K e diagnóstico SIFT;
- status de ausência, revisão ou exclusão.

Os resultados de 500 publicados no README são específicos da prova local.
Nenhum número de 100 itens URL-only ou de 100 mil itens deve ser preenchido
como métrica visual medida sem embeddings correspondentes.

## Limitações éticas e jurídicas

Não redistribua imagens de terceiros. Presença em um manifesto ou
correspondência visual não prova autoria, titularidade, licença, uso em
treinamento ou infração. Consulte `ETHICS.md`.
