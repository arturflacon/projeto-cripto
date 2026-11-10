# Previsões × preço real — Solana

**Trabalho acadêmico.** As previsões abaixo são estimativas de modelos estatísticos a partir de padrões passados, condicionadas ao período analisado. Não são previsão confiável do mercado nem recomendação de investimento.

- Gerado em: `2026-11-10T00:47:22+00:00`
- Período comparado: **27/09/2026 a 10/11/2026** (45 dias com preço real conhecido)
- Modelos: direção `Naive Bayes` · preço `KNN`
- Preços: instantâneos diários de 00:00 UTC (CoinGecko), em USD

## Resumo

|  | resultado |
| --- | --- |
| Direção (sobe / não sobe) | **26 de 45** dias classificados corretamente (58%) |
| Por acaso (moeda honesta) | entre 16 e 29 acertos em 95% das vezes |
| Erro médio do preço estimado | 1.65 USD (1.6%) |
| Erro de "amanhã = hoje" (persistência) | 1.65 USD (1.6%) |

> A taxa de acerto da direção está **dentro** da faixa do acaso: não se distingue de cara-ou-coroa. O preço estimado **não** errou menos que simplesmente repetir o preço de hoje — resultado esperado no horizonte de um dia.

**Próximo dia (11/11/2026, ainda não aconteceu):** direção classificada como **baixa/estável**; preço estimado **85.55 USD** (-1.21% sobre 86.60 USD).

## Gráfico

![Preço real × estimado](previsoes.png)

## Dia a dia

Cada linha foi prevista **antes** de o dia acontecer: o modelo foi treinado só com os dias anteriores. *Véspera* é o preço do dia anterior (o ponto de partida); *erro* = estimado − real.

| dia | véspera (USD) | estimado (USD) | real (USD) | erro | variação real | direção classificada | direção real | acertou? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27/09/2026 | 122.08 | 122.67 | 121.38 | +1.29 (+1.1%) | -0.6% | baixa/estável | baixa/estável | ✓ |
| 28/09/2026 | 121.38 | 121.17 | 121.96 | -0.78 (-0.6%) | +0.5% | baixa/estável | alta | ✗ |
| 29/09/2026 | 121.96 | 121.98 | 118.82 | +3.16 (+2.7%) | -2.6% | alta | baixa/estável | ✗ |
| 30/09/2026 | 118.82 | 118.50 | 119.09 | -0.60 (-0.5%) | +0.2% | alta | alta | ✓ |
| 01/10/2026 | 119.09 | 118.45 | 118.04 | +0.41 (+0.3%) | -0.9% | alta | baixa/estável | ✗ |
| 02/10/2026 | 118.04 | 117.94 | 118.38 | -0.44 (-0.4%) | +0.3% | alta | alta | ✓ |
| 03/10/2026 | 118.38 | 118.35 | 118.59 | -0.24 (-0.2%) | +0.2% | alta | alta | ✓ |
| 04/10/2026 | 118.59 | 117.80 | 119.60 | -1.80 (-1.5%) | +0.9% | alta | alta | ✓ |
| 05/10/2026 | 119.60 | 119.81 | 121.52 | -1.71 (-1.4%) | +1.6% | alta | alta | ✓ |
| 06/10/2026 | 121.52 | 121.08 | 120.76 | +0.32 (+0.3%) | -0.6% | alta | baixa/estável | ✗ |
| 07/10/2026 | 120.76 | 120.49 | 120.68 | -0.19 (-0.2%) | -0.1% | alta | baixa/estável | ✗ |
| 08/10/2026 | 120.68 | 120.59 | 116.23 | +4.37 (+3.8%) | -3.7% | alta | baixa/estável | ✗ |
| 09/10/2026 | 116.23 | 116.31 | 109.51 | +6.80 (+6.2%) | -5.8% | alta | baixa/estável | ✗ |
| 10/10/2026 | 109.51 | 109.27 | 109.07 | +0.20 (+0.2%) | -0.4% | alta | baixa/estável | ✗ |
| 11/10/2026 | 109.07 | 109.39 | 110.00 | -0.61 (-0.6%) | +0.8% | alta | alta | ✓ |
| 12/10/2026 | 110.00 | 109.61 | 111.25 | -1.64 (-1.5%) | +1.1% | alta | alta | ✓ |
| 13/10/2026 | 111.25 | 111.16 | 105.64 | +5.52 (+5.2%) | -5.0% | alta | baixa/estável | ✗ |
| 14/10/2026 | 105.64 | 105.54 | 106.75 | -1.21 (-1.1%) | +1.1% | alta | alta | ✓ |
| 15/10/2026 | 106.75 | 106.73 | 104.19 | +2.53 (+2.4%) | -2.4% | alta | baixa/estável | ✗ |
| 16/10/2026 | 104.19 | 104.11 | 105.34 | -1.23 (-1.2%) | +1.1% | alta | alta | ✓ |
| 17/10/2026 | 105.34 | 104.65 | 104.32 | +0.34 (+0.3%) | -1.0% | alta | baixa/estável | ✗ |
| 18/10/2026 | 104.32 | 103.47 | 103.36 | +0.11 (+0.1%) | -0.9% | alta | baixa/estável | ✗ |
| 19/10/2026 | 103.36 | 102.45 | 104.00 | -1.55 (-1.5%) | +0.6% | alta | alta | ✓ |
| 20/10/2026 | 104.00 | 103.24 | 104.98 | -1.74 (-1.7%) | +0.9% | alta | alta | ✓ |
| 21/10/2026 | 104.98 | 104.77 | 106.39 | -1.62 (-1.5%) | +1.3% | alta | alta | ✓ |
| 22/10/2026 | 106.39 | 106.19 | 106.94 | -0.75 (-0.7%) | +0.5% | alta | alta | ✓ |
| 23/10/2026 | 106.94 | 106.73 | 103.54 | +3.19 (+3.1%) | -3.2% | baixa/estável | baixa/estável | ✓ |
| 24/10/2026 | 103.54 | 102.66 | 102.74 | -0.08 (-0.1%) | -0.8% | baixa/estável | baixa/estável | ✓ |
| 25/10/2026 | 102.74 | 102.38 | 102.27 | +0.11 (+0.1%) | -0.5% | baixa/estável | baixa/estável | ✓ |
| 26/10/2026 | 102.27 | 101.99 | 101.24 | +0.75 (+0.7%) | -1.0% | baixa/estável | baixa/estável | ✓ |
| 27/10/2026 | 101.24 | 101.05 | 100.75 | +0.29 (+0.3%) | -0.5% | baixa/estável | baixa/estável | ✓ |
| 28/10/2026 | 100.75 | 100.50 | 101.20 | -0.70 (-0.7%) | +0.4% | baixa/estável | alta | ✗ |
| 29/10/2026 | 101.20 | 100.92 | 102.32 | -1.40 (-1.4%) | +1.1% | baixa/estável | alta | ✗ |
| 30/10/2026 | 102.32 | 101.92 | 95.47 | +6.44 (+6.7%) | -6.7% | baixa/estável | baixa/estável | ✓ |
| 31/10/2026 | 95.47 | 93.81 | 96.25 | -2.44 (-2.5%) | +0.8% | baixa/estável | alta | ✗ |
| 01/11/2026 | 96.25 | 95.34 | 95.42 | -0.08 (-0.1%) | -0.9% | baixa/estável | baixa/estável | ✓ |
| 02/11/2026 | 95.42 | 94.77 | 94.48 | +0.29 (+0.3%) | -1.0% | baixa/estável | baixa/estável | ✓ |
| 03/11/2026 | 94.48 | 93.75 | 91.84 | +1.90 (+2.1%) | -2.8% | baixa/estável | baixa/estável | ✓ |
| 04/11/2026 | 91.84 | 90.78 | 92.78 | -2.00 (-2.2%) | +1.0% | baixa/estável | alta | ✗ |
| 05/11/2026 | 92.78 | 92.06 | 95.51 | -3.45 (-3.6%) | +2.9% | baixa/estável | alta | ✗ |
| 06/11/2026 | 95.51 | 94.66 | 89.23 | +5.43 (+6.1%) | -6.6% | baixa/estável | baixa/estável | ✓ |
| 07/11/2026 | 89.23 | 88.80 | 90.06 | -1.26 (-1.4%) | +0.9% | baixa/estável | alta | ✗ |
| 08/11/2026 | 90.06 | 89.04 | 89.70 | -0.66 (-0.7%) | -0.4% | baixa/estável | baixa/estável | ✓ |
| 09/11/2026 | 89.70 | 88.58 | 86.41 | +2.17 (+2.5%) | -3.7% | baixa/estável | baixa/estável | ✓ |
| 10/11/2026 | 86.41 | 86.17 | 86.60 | -0.43 (-0.5%) | +0.2% | baixa/estável | alta | ✗ |
| **11/11/2026** | 86.60 | **85.55** | aguardando | — | — | **baixa/estável** | aguardando |  |

## Como os modelos foram escolhidos

`TimeSeriesSplit` com 5 partições sobre os 229 dias **anteriores** ao período comparado — nenhum dia da tabela acima participou da escolha.

**Direção** (critério: maior acurácia balanceada média; 0,5 = acaso):

| modelo | acurácia balanceada média | desvio |
| --- | --- | --- |
| Naive Bayes | 0.521 | 0.019 |
| Perceptron | 0.513 | 0.059 |
| KNN | 0.512 | 0.041 |
| Árvore de decisão | 0.505 | 0.035 |
| SVM (RBF) | 0.501 | 0.042 |

**Preço** (critério: menor RMSE médio do retorno diário):

| modelo | RMSE médio do retorno | desvio |
| --- | --- | --- |
| Baseline: retorno zero | 0.0287 | 0.0082 |
| KNN | 0.0295 | 0.0082 |
| SVR (RBF) | 0.0321 | 0.0059 |
| Árvore de decisão | 0.0392 | 0.0121 |
| Ridge (linear) | 0.0511 | 0.0150 |

## Ressalvas

- "Amanhã" é o próximo instantâneo de 00:00 UTC, não o fechamento de um pregão.
- O preço estimado parte do preço da véspera, por isso acompanha de perto a curva real: o que importa é o erro comparado ao de repetir o preço de hoje.
- Cada linha usa um modelo reajustado só com o passado, então a tabela não é uma medida otimista de dentro da amostra — mas o período é curto e não sustenta conclusões sobre o futuro.
