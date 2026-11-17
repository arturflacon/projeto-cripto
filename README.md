# Análise do Mercado de Criptomoedas — Solana

Projeto integrador da disciplina de Aprendizagem de Máquina. Pipeline completo de
coleta, limpeza, análise exploratória, agrupamento, classificação e regressão sobre
a série histórica diária da **Solana**.

> **Trabalho acadêmico.** Nada neste repositório é recomendação de investimento. Os
> modelos produzem estimativas condicionadas ao período analisado, e a avaliação
> formal mostra desempenho **próximo do acaso** para previsão de direção — ver
> [Resultados](#resultados).

## Trabalho final (entrega)

O notebook de entrega é **[`notebooks/trabalho_final.ipynb`](notebooks/trabalho_final.ipynb)**,
organizado pelas 7 etapas do plano de atividades (base → EDA → pré-processamento → divisão →
treinamento → avaliação → uso do modelo), com código comentado, interpretação após cada saída e
um apêndice de regressão. Ele já está salvo **com as saídas**.

- **Base:** [`data/snapshot/solana_365d.csv`](data/snapshot/solana_365d.csv) — Solana, 365 dias
  (2025-09-01 a 2026-08-31), congelada a partir da API pública CoinGecko,
  endpoint [`/coins/{id}/market_chart`](https://docs.coingecko.com/reference/coins-id-market-chart)
  (`https://api.coingecko.com/api/v3/coins/solana/market_chart?vs_currency=usd&days=365`).
- **Problema:** classificação binária — "o preço da Solana fecha amanhã acima de hoje?"
- **Resultado oficial (split temporal 70/30):** Perceptron, acurácia balanceada 0,575 no teste
  (48/83 acertos, p ≈ 0,09 contra o acaso). Contraste com split embaralhado incluído.
- **Item novo:** atributos de 2026-08-31 → classificado como alta para 2026-09-01.

Para reexecutar do zero (não usa a API, exceto numa célula opcional da Etapa 7):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/trabalho_final.ipynb
```

## Restrição central: um único ativo

O projeto usa apenas a Solana (id CoinGecko `solana`). Como há um só ativo, o
agrupamento e o "mapa do mercado" não são feitos **entre ativos** — eles são
reenquadrados sobre o **tempo**:

- cada observação diária vira um vetor de 13 atributos;
- o agrupamento identifica **regimes temporais** (calmo, tendência, estresse);
- o "mapa" é uma projeção **PCA das observações diárias**, colorida por regime;
- classificação e regressão operam sobre a própria série da Solana.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reprodutibilidade

A API da CoinGecko devolve uma janela **móvel** dos últimos 365 dias: rodar a coleta
em outra data produz outra série, e os números dos relatórios deixam de bater.

Por isso o repositório versiona um **snapshot congelado** em
`data/snapshot/solana_365d.csv` (2025-09-01 a 2026-08-31). Para reproduzir
exatamente os resultados entregues:

```bash
USAR_SNAPSHOT=1 python src/collect.py     # Windows PowerShell: $env:USAR_SNAPSHOT=1
```

Sem a variável, `collect.py` busca dados novos na API — é o diferencial de
"atualização automática" do projeto, não o caminho da reprodução.

## Pipeline

Rodar em ordem, a partir da raiz do projeto (com `USAR_SNAPSHOT=1` para reproduzir os
números entregues):

| # | Comando | Produz |
|---|---|---|
| 1 | `python src/collect.py` | `data/processed/precos.parquet` |
| 2 | `python src/preprocess.py` | `precos_limpo.parquet` (contagens de qualidade no terminal) |
| 3 | `python src/features.py` | `atributos.parquet` (274 linhas × 13 atributos) |
| 4 | `python src/clustering.py` | `regimes.parquet` (resumo dos regimes no terminal) |
| 5 | `python src/classification.py` | `reports/classificacao.md` |
| 6 | `python src/regression.py` | resultados no terminal |
| 7 | `python src/prever.py` | `previsoes.parquet` + `reports/previsoes.md` / `.png` — classificação do dia seguinte |

Análise exploratória (Encontros 5-6), com 7 visualizações e interpretação escrita:

```bash
jupyter notebook notebooks/eda_solana.ipynb
```

Painel interativo (Encontro 13) — indicadores, regimes, eventos atípicos e a
classificação do dia seguinte:

```bash
streamlit run app/dashboard.py
```

## Verificação de vazamento temporal

O argumento central do trabalho é metodológico: **os atributos usam apenas dados até
o instante `t`; só o rótulo olha para o futuro, e só no treino.** Isso não fica
apenas escrito — é verificado:

```bash
python tests/test_vazamento.py
```

São 9 verificações. A principal recomputa os atributos sobre uma série **truncada** e
exige que os valores anteriores ao corte sejam idênticos — se algum atributo olhasse
adiante, truncar mudaria o passado. O teste foi validado contra quatro tipos de
vazamento injetado de propósito (`shift` negativo, janela centrada, normalização com
estatística global e ranking sobre a série inteira), e detecta os quatro.

## Estrutura

```
projeto-cripto/
├── data/
│   ├── raw/          # respostas brutas da API (não versionado)
│   ├── snapshot/     # série congelada, VERSIONADA — base da reprodução
│   └── processed/    # tabelas geradas pelo pipeline (não versionado)
├── notebooks/
│   ├── eda_solana.ipynb      # EDA (Encontros 5-6)
│   └── trabalho_final.ipynb  # notebook de entrega (7 etapas)
├── src/
│   ├── collect.py            # coleta CoinGecko / snapshot   (Enc. 3)
│   ├── preprocess.py         # limpeza e relatório de qualidade (Enc. 4)
│   ├── features.py           # 13 atributos + rótulo, sem vazamento (Enc. 7)
│   ├── clustering.py         # regimes: K-Means, Ward, DBSCAN  (Enc. 8-9)
│   ├── classification.py     # direção do dia seguinte        (Enc. 10-11)
│   ├── regression.py         # volatilidade futura            (Enc. 12)
│   └── prever.py             # classificação do dia seguinte, walk-forward
├── tests/
│   └── test_vazamento.py     # 9 verificações de vazamento temporal
├── app/
│   └── dashboard.py          # painel Streamlit               (Enc. 13)
├── reports/                  # relatórios .md
├── requirements.txt
└── README.md
```

## Dados

`collect.py` baixa a série diária da Solana (endpoint `/coins/{id}/market_chart`,
`days=365` — acima de 90 dias a API já devolve granularidade diária). A tabela tem
uma observação por (ativo, data):

| coluna | descrição |
|---|---|
| `id` | identificador CoinGecko (`solana`) |
| `simbolo` | símbolo (`sol`) |
| `nome` | nome do ativo (`Solana`) |
| `data` | data da observação (UTC, sem hora) |
| `preco` | preço em `VS_CURRENCY` (padrão `usd`) |
| `capitalizacao` | capitalização de mercado |
| `volume` | volume negociado no dia |
| `fonte` | origem dos dados (`coingecko`) |
| `coletado_em` | momento da coleta (UTC, ISO 8601) |

Parâmetros no topo de `src/collect.py`: `MOEDA_ID` (ou a variável de ambiente
`COINGECKO_COIN_ID`), `DIAS`, `VS_CURRENCY`. Chave Demo opcional da CoinGecko
(100 req/min) pela variável `COINGECKO_API_KEY`.

## Resultados

Números da base congelada (`USAR_SNAPSHOT=1`). Relatórios em `reports/`
(`classificacao.md`, `previsoes.md`) e no notebook de entrega. Em resumo:

**Qualidade dos dados** — a série veio limpa: 365 dias, sem duplicatas, sem lacunas
e sem valores inválidos.

**EDA** — três análises independentes convergem para o mesmo diagnóstico: a série
carrega informação sobre a **magnitude** dos movimentos e quase nenhuma sobre a
**direção**. A autocorrelação dos retornos é indistinguível de ruído branco, mas a
de |retornos| é significativa nas defasagens 1-3 e 5; o volume tem correlação
+0,003 com a direção e +0,600 com a magnitude.

**Regimes** (k=3 por silhouette) — R0 "em alta" (11 dias), R1 "lateral" (202 dias) e
R2 "turbulento, em queda" (61 dias). O regime R2 isola sozinho os três episódios de
estresse do período, embora o agrupamento não receba nenhuma informação de data.

**Classificação da direção** — o melhor modelo (Perceptron) alcançou acurácia
balanceada de 0,575 contra 0,500 do melhor baseline. Com 83 dias de teste isso fica
a **1,4 desvios** do acaso: dentro do que uma moeda honesta produziria. O resultado
é consistente com o que a EDA já indicava.

**Regressão de volatilidade** — nenhuma das 16 configurações testadas (4 horizontes ×
4 modelos) obteve R² médio positivo no walk-forward do treino. O relatório documenta
uma armadilha instrutiva: escolher o horizonte pelo R² **do teste** levaria a
reportar R² = +0,398, mas o treino coloca essa mesma configuração em 14º lugar entre
16 — era sorte da janela de teste, não capacidade preditiva.

### A leitura honesta

Prever a direção diária de um criptoativo, sem vazamento e com split temporal, fica
perto de cara-ou-coroa. **Isso não é falha de execução** — é a estrutura do dado, e a
EDA já antecipava. O que sustenta o trabalho é o rigor do protocolo: atributos só com
passado (verificado por teste automatizado), rótulo isolado no futuro, divisão
cronológica, padronização dentro do `Pipeline`, baselines medidos antes dos modelos e
seleção de hiperparâmetros que nunca enxerga o conjunto de teste.
