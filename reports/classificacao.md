# Classificação do dia seguinte — Solana

**Trabalho acadêmico.** Os resultados abaixo são estimativas condicionadas ao período analisado. Não são previsão confiável do mercado nem recomendação de investimento.

- Gerado em: `2026-11-17T00:22:58+00:00`
- Entrada: `data\processed\atributos.parquet`
- Alvo: `direcao` · janelas 7/30/90 dias · 13 atributos
- Split temporal 70%/30%: treino **191 dias** (2025-11-30 a 2026-06-08) · teste **83 dias** (2026-06-09 a 2026-08-30)
- Classe 1 no treino: 46.1% · no teste: 51.8%
- Semente: 42 · TimeSeriesSplit: 5 partições · KNN k=15 · árvore max_depth=3

## 1. Comparação walk-forward (somente no treino)

Modelos e baselines avaliados nas **mesmas partições** do `TimeSeriesSplit`; valores = média ± desvio-padrão entre partições. É esta tabela que escolhe o modelo — o teste não participa da escolha.

| modelo | tipo | acc. balanceada | F1 |
| --- | --- | --- | --- |
| **Perceptron** | modelo | 0.519 ± 0.083 | 0.367 ± 0.141 |
| Baseline: persistência | baseline | 0.517 ± 0.067 | 0.482 ± 0.083 |
| Baseline: classe majoritária (treino) | baseline | 0.500 ± 0.000 | 0.000 ± 0.000 |
| Baseline: sempre alta | baseline | 0.500 ± 0.000 | 0.623 ± 0.089 |
| Árvore de decisão | modelo | 0.495 ± 0.032 | 0.497 ± 0.044 |
| SVM (RBF) | modelo | 0.480 ± 0.023 | 0.293 ± 0.255 |
| KNN | modelo | 0.478 ± 0.046 | 0.252 ± 0.242 |
| Naive Bayes | modelo | 0.462 ± 0.050 | 0.151 ± 0.208 |

Modelo selecionado: **Perceptron**.

## 2. Teste retido — modelo × métrica

Cada modelo é reajustado no treino inteiro e avaliado **uma vez** no trecho final. Δ = diferença de acurácia balanceada para o melhor baseline.

| modelo | tipo | acc. balanceada | Δ vs baseline | F1 | precisão | recall | acurácia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Perceptron** | modelo | 0.575 | +0.075 | 0.624 | 0.580 | 0.674 | 0.578 |
| Baseline: sempre alta | baseline | 0.500 | +0.000 | 0.683 | 0.518 | 1.000 | 0.518 |
| Baseline: classe majoritária (treino) | baseline | 0.500 | +0.000 | 0.000 | 0.000 | 0.000 | 0.482 |
| SVM (RBF) | modelo | 0.499 | -0.001 | 0.432 | 0.516 | 0.372 | 0.494 |
| Árvore de decisão | modelo | 0.483 | -0.017 | 0.482 | 0.500 | 0.465 | 0.482 |
| Baseline: persistência | baseline | 0.481 | -0.019 | 0.506 | 0.500 | 0.512 | 0.482 |
| KNN | modelo | 0.458 | -0.042 | 0.471 | 0.476 | 0.465 | 0.458 |
| Naive Bayes | modelo | 0.419 | -0.081 | 0.222 | 0.350 | 0.163 | 0.410 |

## 3. Matriz de confusão — Perceptron

|  | previsto 0 | previsto 1 |
| --- | --- | --- |
| **observado 0** | 19 | 21 |
| **observado 1** | 14 | 29 |

Acertos: 48 de 83 dias (57.8%). Teste binomial contra um palpite 50/50: **p = 0.094**. Com 83 dias, a margem do IC 95% de uma acurácia perto de 50% é de ±10.8%.

## 4. Análise de erros — Perceptron

**Por classe real**

| classe real | dias | taxa de acerto |
| --- | --- | --- |
| 0 | 40 | 47.5% |
| 1 | 43 | 67.4% |

**Por tercil de volatilidade (`vol_30d` em t)**

| faixa | dias | taxa de acerto |
| --- | --- | --- |
| baixo | 28 | 64.3% |
| médio | 27 | 63.0% |
| alto | 28 | 46.4% |

**Por tercil do tamanho do movimento seguinte (|ret t→t+1|)**

| faixa | dias | taxa de acerto |
| --- | --- | --- |
| baixo | 28 | 50.0% |
| médio | 27 | 44.4% |
| alto | 28 | 78.6% |

O tamanho do movimento seguinte só é conhecido depois do fato: serve para entender *onde* o modelo erra, nunca como atributo.

## 5. Leitura dos resultados

- No teste, o modelo selecionado (Perceptron) ficou +0.075 em acurácia balanceada em relação ao melhor baseline. O número de acertos **não** é estatisticamente distinguível de um palpite 50/50 (p ≥ 0,05): a diferença cabe no acaso de um teste com poucos dias.
- A dispersão entre partições do walk-forward mostra a instabilidade: o desempenho de um mesmo modelo varia vários pontos de um trecho para outro.
- Prever a direção **diária** de uma criptomoeda, sem vazamento e com split temporal, fica próximo de cara-ou-coroa. Empatar com o baseline não é falha do trabalho: é o comportamento esperado do domínio, e o valor desta etapa está no rigor do protocolo (split temporal, escala ajustada só no treino, baselines nas mesmas partições, análise de erros).
- Extensões com sinal mais estável: alvo `acima_mediana` (desempenho relativo) e a regressão de volatilidade futura (`src/regression.py`).
