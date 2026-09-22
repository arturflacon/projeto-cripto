# Regressão da volatilidade futura — Encontro 12

Extensão do projeto. **Trabalho acadêmico**: estimativas condicionadas ao período analisado, não recomendação de investimento.

- Gerado em: `2026-09-22T01:07:52+00:00`
- Entrada: `data\processed\precos_limpo.parquet`
- Alvo: desvio-padrão dos retornos dos **7 dias seguintes**
- Horizonte e modelo selecionados **só com o treino**
- Semente fixa: `42`

## Por que volatilidade, e não direção

A escolha vem da EDA (`notebooks/eda_solana.ipynb`), que encontrou dois indícios independentes na mesma direção:

| evidência | direção | magnitude |
| --- | --- | --- |
| autocorrelação (lags 1-3, 5) | dentro da banda (ruído) | 0,16 / 0,17 / 0,13 / 0,12 |
| correlação com volume | +0,003 | +0,600 |

A série informa *quanto* o preço se move, quase nada sobre *para onde*. Esta etapa testa se essa estrutura é suficiente para **estimar** a volatilidade futura melhor que referências triviais.

## Desenho experimental

| conjunto | dias | período | volatilidade média do alvo |
| --- | --- | --- | --- |
| treino | 187 | 2025-11-30 a 2026-06-04 | 3.08% |
| teste | 81 | 2026-06-05 a 2026-08-24 | 2.55% |
| total | 268 | 2025-11-30 a 2026-08-24 | 2.92% |

> **Sobreposição do alvo.** Como o alvo é uma janela de 7 dias, linhas vizinhas compartilham até 6 dias. O número de janelas realmente independentes é da ordem de 38, não 268 — qualquer diferença pequena entre métodos está dentro do ruído. É também mais uma razão para o split ser temporal: embaralhar colocaria janelas quase idênticas nos dois lados da divisão.

> **Deslocamento de nível.** A volatilidade média do teste é -17.1% em relação ao treino, com variância bem menor. Isso penaliza o R² de **todos** os estimadores, inclusive dos baselines — voltamos a isso na discussão.

## 1. Baselines (medidos antes dos modelos)

| baseline | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Baseline: persistência (vol_30d) | 0.0115 | 0.0090 | -0.278 |
| Baseline: persistência (vol_7d) | 0.0119 | 0.0089 | -0.385 |
| Baseline: média do treino | 0.0114 | 0.0097 | -0.271 |

> A correlação entre `vol_30d` (hoje) e a volatilidade realizada dos 7 dias seguintes é **+0.301** — é o agrupamento de volatilidade da EDA aparecendo como capacidade preditiva direta. Fraca, mas real; para comparar, a correlação equivalente para *direção* é praticamente zero.

## 2. Seleção de horizonte e modelo (walk-forward, só no treino)

O horizonte do alvo é uma decisão de modelagem: janelas curtas medem a volatilidade com muito erro amostral (o desvio-padrão de 7 retornos tem ~29% de erro relativo), janelas longas alisam o sinal. Por isso ele é selecionado junto com o modelo, entre 4 horizontes x 4 modelos — **sempre dentro do treino**:

| horizonte | modelo | R² médio (walk-forward) | desvio |
| --- | --- | --- | --- |
| 7 | Árvore de decisão | -0.549 | 1.047 |
| 7 | KNN | -0.679 | 0.604 |
| 7 | Ridge (linear) | -0.924 | 1.580 |
| 14 | Árvore de decisão | -1.979 | 3.550 |
| 14 | KNN | -3.100 | 4.605 |
| 14 | Ridge (linear) | -3.226 | 4.771 |
| 7 | SVR (RBF) | -3.403 | 3.161 |
| 14 | SVR (RBF) | -6.275 | 8.921 |
| 21 | KNN | -6.558 | 8.289 |
| 21 | Árvore de decisão | -7.554 | 9.781 |
| 30 | KNN | -7.772 | 4.656 |
| 30 | SVR (RBF) | -9.564 | 6.661 |
| 21 | SVR (RBF) | -10.650 | 14.684 |
| 30 | Árvore de decisão | -12.397 | 13.424 |
| 21 | Ridge (linear) | -15.427 | 19.012 |
| 30 | Ridge (linear) | -16.636 | 17.024 |

> Selecionado: **Árvore de decisão** com horizonte de **7 dias** — a melhor das configurações testadas. Note que **todas** têm R² médio negativo no walk-forward: nenhuma combinação supera, de forma estável, prever a média da própria partição.

## 3. Avaliação final no teste retido

| modelo | RMSE | MAE | R² |
| --- | --- | --- | --- |
| KNN | 0.0112 | 0.0094 | -0.2256 |
| Ridge (linear) | 0.0185 | 0.0112 | -2.3098 |
| Árvore de decisão | 0.0192 | 0.0170 | -2.5864 |
| SVR (RBF) | 0.0236 | 0.0214 | -4.4194 |

**Ganho do modelo selecionado sobre os baselines** (redução do erro quadrático):

| comparação | ganho |
| --- | --- |
| Árvore de decisão vs Baseline: persistência (vol_30d) | -180.6% |
| Árvore de decisão vs Baseline: persistência (vol_7d) | -159.0% |
| Árvore de decisão vs Baseline: média do treino | -182.2% |

## 4. A armadilha que o protocolo evitou

Durante o desenvolvimento, calcular o R² **no teste** para cada horizonte produziu esta tabela:

| horizonte | melhor modelo no teste | R² no teste |
| --- | --- | --- |
| 7 | KNN | -0.226 |
| 14 | Ridge (linear) | +0.280 |
| 21 | KNN | +0.255 |
| 30 | Árvore de decisão | -0.001 |

Escolher o horizonte por essa tabela levaria a reportar **R² = +0.280** com h=14 — um resultado aparentemente bom. Mas o walk-forward no treino coloca a melhor configuração de h=14 em **4º lugar** entre 16, com R² médio de -1.979.

> Ou seja: aquele R² positivo era **sorte da janela de teste específica**, não capacidade preditiva. Um trabalho que escolhesse o horizonte olhando o resultado final reportaria um sucesso inexistente. É exatamente esse tipo de erro que a seleção restrita ao treino previne — e ele é sutil, porque não envolve vazamento nos atributos nem no rótulo, apenas na **escolha de hiperparâmetro**.

## 5. Discussão

O modelo selecionado (**Árvore de decisão**, h=7) **não** superou o melhor baseline (*Baseline: média do treino*): erro quadrático 182.2% maior.

**Todos os R² são negativos — inclusive os dos baselines.** Isso merece explicação, porque parece pior do que é. O R² compara o erro do estimador com o de prever a **média do conjunto de teste**. Como a volatilidade do teste ficou -17.1% abaixo da do treino, essa média é um alvo que nenhum método honesto poderia conhecer de antemão. R² negativo aqui indica **deslocamento de nível entre os períodos**, não que os modelos sejam absurdos — o baseline "média do treino", que é o estimador mais conservador possível, também fica negativo.

**O sinal existe, mas é fraco e não sobrevive à validação.** A correlação de +0.301 entre volatilidade passada e futura confirma o agrupamento de volatilidade que a EDA encontrou — é real, e é muito maior que o sinal disponível para direção. Mas transformá-lo em estimativa que supere referências triviais, com 274 observações diárias sobrepostas de um único ativo, não funcionou: **nenhuma das 16 configurações testadas teve R² médio positivo no walk-forward do treino**.

**O que este resultado significa para o projeto.** A hipótese da EDA — "magnitude é mais previsível que direção" — se confirma no diagnóstico (correlação de 0,30 contra ~0) mas não se converte em desempenho preditivo aqui. A conclusão honesta não é "volatilidade é imprevisível", e sim: *com esta amostra, estes atributos e este protocolo, não conseguimos demonstrar previsibilidade além da persistência*. Literatura de finanças obtém ganhos reais nessa tarefa com modelos próprios da área (GARCH, volatilidade realizada intradiária) e séries muito mais longas — fora do escopo desta disciplina.

**Limitações.** (i) Janelas sobrepostas reduzem o número efetivo de amostras a ~11 no teste; (ii) o período é de queda acentuada, e o teste é sistematicamente mais calmo que o treino; (iii) volatilidade estimada não é risco garantido — são estimativas condicionadas ao período analisado, sem valor de recomendação.
