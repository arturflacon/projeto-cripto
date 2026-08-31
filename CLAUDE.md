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

## Estado atual do repositório

```
projeto-cripto/
├── data/
│   ├── raw/          # respostas brutas da API (não versionadas)
│   └── processed/    # tabelas em Parquet
├── notebooks/        # análise exploratória
├── src/
│   ├── collect.py        # PRONTO — ver nota abaixo
│   ├── preprocess.py     # stub (TAREFA ATUAL)
│   ├── features.py       # stub
│   ├── clustering.py     # stub
│   └── classification.py # stub
├── app/
├── reports/
├── requirements.txt
└── README.md
```

**`collect.py` já funciona**, mas está configurado para baixar as N maiores moedas por
capitalização. **Passo 0 desta sessão: ajustar `collect.py` para coletar apenas `solana`** —
trocar a seleção por market cap por um único id fixo/configurável, mantendo o resto
(endpoint `/coins/{id}/market_chart`, `days=365`, salvar JSON bruto em `data/raw/` e a
tabela longa em Parquet).

**Tabela gerada `data/processed/precos.parquet`** (uma observação por ativo-data):
`id, simbolo, nome, data, preco, capitalizacao, volume, fonte, coletado_em`.

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

## TAREFA ATUAL — Encontro 4: `preprocess.py` (limpeza e qualidade)

Ler a coleta bruta e produzir uma versão limpa + um relatório de qualidade.

**Entrada:** `data/processed/precos.parquet`
**Saídas:**
- `data/processed/precos_limpo.parquet` — série diária contínua e limpa da Solana
- `reports/relatorio_qualidade_dados.md` — o "relatório de problemas" do Encontro 4

**Etapas (seção 7 da apostila):**
1. Carregar o parquet, ordenar por data, garantir tipos (`data` como datetime; valores numéricos).
2. Padronizar datas (UTC, sem hora) e **remover duplicidades** por (id, data), registrando a contagem.
3. Construir um **índice diário contínuo** de `min(data)` a `max(data)` e reindexar para expor **lacunas** (dias ausentes).
4. **Relatar lacunas:** total de dias ausentes, número de blocos de lacuna, maior lacuna contígua, e as datas.
5. **Política de preenchimento** (parametrizável, ex.: `LIMIAR_LACUNA_DIAS = 3`):
   - lacunas curtas (≤ limiar): interpolar preço/capitalização e `ffill` no volume, **marcando** as linhas numa coluna booleana `preenchido`;
   - lacunas longas (> limiar): **não** preencher artificialmente (a apostila proíbe) — deixar NaN e reportar; oferecer função separada para recortar ao maior trecho contíguo.
6. **Sanidade de valores:** remover/reportar preço ≤ 0, capitalização ≤ 0, volume negativo.
7. **"Ativos removidos"** (mecanismo genérico): se um ativo tiver menos que `N_MIN_OBS`
   observações ou fração de lacunas acima de um limite, removê-lo e registrar `id + motivo`
   numa tabela do relatório. (Com Solana isso provavelmente não dispara, mas manter o mecanismo.)
8. Escrever o Parquet limpo e o relatório `.md` com as contagens: nº de observações, período,
   duplicatas removidas, dias ausentes, maior lacuna, linhas preenchidas, ativos removidos e motivo.

Ao terminar: rodar o script e mostrar o relatório gerado.

---

## Roadmap (próximos encontros — já reenquadrado para Solana única)

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