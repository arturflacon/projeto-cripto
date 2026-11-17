# Projeto — Sistema de Análise do Mercado de Criptomoedas (foco: SOLANA)

Projeto integrador da disciplina de Aprendizagem de Máquina, baseado numa apostila.
Pipeline educacional de coleta → preparação → EDA → agrupamento → classificação →
(opcional) regressão/anomalias → visualização. **Trabalho acadêmico: nada aqui é
recomendação de investimento.**

---

## ⚠️ Restrição central

O projeto usa **apenas a Solana** (id CoinGecko `solana`). Nenhum outro ativo é
coletado nem modelado.

**Implicação metodológica:** como há um único ativo, agrupamento e "mapa do mercado"
NÃO são feitos entre ativos. Eles são reenquadrados sobre o **tempo**:
- cada observação diária da Solana vira um vetor de atributos;
- o agrupamento identifica **regimes/estados temporais** (ex.: calmo, tendência, estresse);
- o "mapa" vira uma projeção **PCA das observações diárias** colorida por regime;
- classificação e detecção de anomalias operam sobre a **série temporal** da própria
  Solana (direção/retorno futuro, eventos atípicos).

---

## Estado atual do repositório (atualizado em 2026-11-16 — pipeline COMPLETO)

Todas as etapas estão implementadas e funcionando. Não há stubs.

```
projeto-cripto/
├── data/
│   ├── raw/          # respostas brutas da API (não versionadas)
│   ├── snapshot/     # solana_365d.csv — base OFICIAL congelada (versionada)
│   └── processed/    # Parquets gerados pelo pipeline (não versionados)
├── notebooks/
│   ├── eda_solana.ipynb      # EDA (Enc. 5-6), 7 visualizações
│   └── trabalho_final.ipynb  # NOTEBOOK DE ENTREGA — 7 etapas do plano de atividades
├── src/
│   ├── collect.py        # coleta CoinGecko (só `solana`) ou snapshot (USAR_SNAPSHOT=1)
│   ├── preprocess.py     # limpeza/qualidade -> precos_limpo.parquet (contagens no terminal)
│   ├── features.py       # 13 atributos só com passado + rótulo y -> atributos.parquet
│   ├── clustering.py     # regimes (K-Means, Ward, DBSCAN) + PCA -> regimes.parquet
│   ├── classification.py # direção do dia seguinte -> reports/classificacao.md
│   ├── regression.py     # volatilidade futura (resultados no terminal)
│   └── prever.py         # walk-forward do dia seguinte -> reports/previsoes.md/.png
├── tests/test_vazamento.py  # 9 verificações de vazamento temporal (todas passam)
├── app/dashboard.py         # painel Streamlit
├── reports/                 # classificacao.md, previsoes.md, previsoes.png
├── requirements.txt
└── README.md
```

**Base oficial:** `data/snapshot/solana_365d.csv` (2025-09-01 a 2026-08-31). Para reproduzir os
números entregues, rodar o pipeline com `USAR_SNAPSHOT=1` (sem a variável, `collect.py` busca a
janela móvel atual da API e os números mudam). 365 dias -> 274 linhas com atributos e rótulo
(90 de aquecimento da janela de 90 dias + 1 último dia sem rótulo).

**Resultados de referência (snapshot):**
- Classificação (split temporal 70/30: 191 treino / 83 teste; semente 42, KNN k=15, árvore
  max_depth=3, TimeSeriesSplit 5): **Perceptron** selecionado no walk-forward do treino
  (0,519 ± 0,083); no teste acc. balanceada **0,575**, F1 0,624, 48/83 acertos, binomial p = 0,094
  — não distinguível do acaso.
- Regimes: k=3 (R0 em alta 11 dias, R1 lateral 202, R2 turbulento 61).
- Regressão: nenhuma das 16 configurações com R² médio > 0 no treino; selecionado KNN h=7.
- `prever.py` seleciona o classificador numa janela diferente e escolhe **SVM (RBF)** —
  divergência esperada e documentada no notebook.

**Notebook de entrega (`notebooks/trabalho_final.ipynb`):** reaproveita `collect`/`preprocess`/
`features` por import; o código das etapas 4-7 (split, pipelines, treino, métricas, uso do
modelo) fica visível no notebook, com os mesmos hiperparâmetros de `classification.py` — os
números batem com `reports/classificacao.md`. Inclui contraste com split embaralhado (seção 6.5)
e classificação do item novo 2026-08-31 -> 2026-09-01. Executar com
`jupyter nbconvert --to notebook --execute --inplace notebooks/trabalho_final.ipynb`.

**Pendências conhecidas (fora do escopo da entrega):** `app/dashboard.py` cita
`reports/comparacao_classificadores.md`, mas o relatório real é `reports/classificacao.md`.

---

## Convenções de código (seguir SEMPRE)

- Comentários e docstrings **em português**.
- `pathlib` para caminhos (`RAIZ = Path(__file__).resolve().parents[1]`).
- Type hints; `from __future__ import annotations`; `main()` sob `if __name__ == "__main__"`.
- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, pyarrow, requests. (Streamlit/Plotly
  só na etapa de dashboard.)
- Estilo **incremental**: baseline funcional primeiro, refinar depois. Mudanças pequenas e
  verificáveis, uma etapa por vez.
- **Sem vazamento temporal**: atributos usam só dados até o instante t; apenas o rótulo
  olha o futuro, e só no treino.
- Reprodutibilidade: sementes fixas, parâmetros registrados.
- Linguagem educacional: nunca "vai subir" / "melhor investimento"; usar "classificado pelo
  modelo" / "estimativa condicionada ao período".
- Reutilizar as mesmas funções de conversão e o mesmo estilo do `collect.py`.

---

## Roadmap original (todas as etapas abaixo já foram implementadas)

- **Enc. 5–6 — EDA** (`notebooks/`): distribuição de preços/retornos (escala log), volatilidade
  ao longo do tempo, drawdown, autocorrelação dos retornos, relação volume × retorno × volatilidade.
  Opcional: correlação da Solana com o BTC apenas como **série de referência** (baixar BTC só para
  isso, sem modelá-lo). ≥5 visualizações, cada uma com interpretação escrita.
- **Enc. 7 — `features.py`**: vetor de atributos por dia usando só o passado (retorno 1/7/30d,
  retorno acumulado, médias móveis e distância à média, volatilidade histórica/semidesvio, drawdown,
  razão volume/capitalização, assimetria/curtose em janela). Sem vazamento.
- **Enc. 8–9 — `clustering.py`**: agrupar as observações diárias em **regimes** (K-Means, hierárquico,
  DBSCAN); escolher k por silhouette + cotovelo + dendrograma; interpretar os regimes; mapa PCA
  colorido por regime.
- **Enc. 10–11 — `classification.py`**: alvo = direção/retorno futuro (ex.: retorno do próximo dia > 0,
  ou > mediana histórica). Baseline (repetir último retorno / sempre alta). ≥3 classificadores
  (KNN, SVM, árvore, Naive Bayes, perceptron) com **TimeSeriesSplit / walk-forward**. Métricas:
  acurácia balanceada, F1, matriz de confusão. Análise de erros.
- **Enc. 12 — extensão**: regressão de volatilidade futura OU detecção de anomalias (z-score/IQR,
  Isolation Forest/LOF, DBSCAN) na série da Solana.
- **Enc. 13 — `app/`**: dashboard (Streamlit) com indicadores, regimes e eventos atípicos da Solana.

---

## Histórico da conversa (resumo)

- Definido que a fonte é a **CoinGecko** (plano Demo: 100 req/min, 10k chamadas/mês; com
  `days > 90` o `market_chart` já devolve granularidade diária — não passar `interval`).
- Decidido que **dado estático/CSV/Parquet é preferível** a bater na API a cada execução
  (reprodutibilidade para a banca). A API só entra pelo diferencial de "atualização automática".
- A previsão do dia seguinte é montada transformando a série numa tabela supervisionada:
  `y = (close.shift(-1) > close)` como rótulo; atributos só com dados passados; split **temporal**.
  Evitar enquadramento "short/long" (é recomendação — penalizado pela apostila).
- Esqueleto do repositório e `collect.py` já criados nesta conversa.