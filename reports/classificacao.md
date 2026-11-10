# Classificação do dia seguinte — Solana

**Trabalho acadêmico.** Os resultados abaixo são estimativas condicionadas ao período analisado. Não são previsão confiável do mercado nem recomendação de investimento.

- Gerado em: `2026-11-10T00:47:15+00:00`
- Entrada: `data\processed\atributos.parquet`
- Alvo: `direcao` · janelas 7/30/90 dias · 13 atributos
- Split temporal 70%/30%: treino **191 dias** (2026-02-09 a 2026-08-18) · teste **83 dias** (2026-08-19 a 2026-11-09)
- Classe 1 no treino: 48.2% · no teste: 53.0%
- Semente: 42 · TimeSeriesSplit: 5 partições · KNN k=15 · árvore max_depth=3

## 1. Comparação walk-forward (somente no treino)

Modelos e baselines avaliados nas **mesmas partições** do `TimeSeriesSplit`; valores = média ± desvio-padrão entre partições. É esta tabela que escolhe o modelo — o teste não participa da escolha.

| modelo | tipo | acc. balanceada | F1 |
| --- | --- | --- | --- |
| **Perceptron** | modelo | 0.553 ± 0.069 | 0.539 ± 0.109 |
| Árvore de decisão | modelo | 0.539 ± 0.057 | 0.549 ± 0.142 |
| KNN | modelo | 0.520 ± 0.043 | 0.528 ± 0.122 |
| Baseline: sempre alta | baseline | 0.500 ± 0.000 | 0.632 ± 0.058 |
| Baseline: classe majoritária (treino) | baseline | 0.500 ± 0.000 | 0.371 ± 0.308 |
| Naive Bayes | modelo | 0.498 ± 0.040 | 0.542 ± 0.098 |
| SVM (RBF) | modelo | 0.481 ± 0.028 | 0.433 ± 0.128 |
| Baseline: persistência | baseline | 0.473 ± 0.037 | 0.436 ± 0.073 |

Modelo selecionado: **Perceptron**.

## 2. Teste retido — modelo × métrica

Cada modelo é reajustado no treino inteiro e avaliado **uma vez** no trecho final. Δ = diferença de acurácia balanceada para o melhor baseline.

| modelo | tipo | acc. balanceada | Δ vs baseline | F1 | precisão | recall | acurácia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Árvore de decisão | modelo | 0.568 | +0.068 | 0.707 | 0.569 | 0.932 | 0.590 |
| **Perceptron** | modelo | 0.567 | +0.067 | 0.269 | 0.875 | 0.159 | 0.542 |
| KNN | modelo | 0.532 | +0.032 | 0.394 | 0.591 | 0.295 | 0.518 |
| Naive Bayes | modelo | 0.511 | +0.011 | 0.044 | 1.000 | 0.023 | 0.482 |
| Baseline: sempre alta | baseline | 0.500 | +0.000 | 0.693 | 0.530 | 1.000 | 0.530 |
| Baseline: classe majoritária (treino) | baseline | 0.500 | +0.000 | 0.000 | 0.000 | 0.000 | 0.470 |
| SVM (RBF) | modelo | 0.500 | +0.000 | 0.000 | 0.000 | 0.000 | 0.470 |
| Baseline: persistência | baseline | 0.468 | -0.032 | 0.500 | 0.500 | 0.500 | 0.470 |

## 3. Matriz de confusão — Perceptron

|  | previsto 0 | previsto 1 |
| --- | --- | --- |
| **observado 0** | 38 | 1 |
| **observado 1** | 37 | 7 |

Acertos: 45 de 83 dias (54.2%). Teste binomial contra um palpite 50/50: **p = 0.255**. Com 83 dias, a margem do IC 95% de uma acurácia perto de 50% é de ±10.8%.

## 4. Análise de erros — Perceptron

**Por classe real**

| classe real | dias | taxa de acerto |
| --- | --- | --- |
| 0 | 39 | 97.4% |
| 1 | 44 | 15.9% |

**Por tercil de volatilidade (`vol_30d` em t)**

| faixa | dias | taxa de acerto |
| --- | --- | --- |
| baixo | 28 | 57.1% |
| médio | 27 | 59.3% |
| alto | 28 | 46.4% |

**Por tercil do tamanho do movimento seguinte (|ret t→t+1|)**

| faixa | dias | taxa de acerto |
| --- | --- | --- |
| baixo | 28 | 53.6% |
| médio | 27 | 44.4% |
| alto | 28 | 64.3% |

O tamanho do movimento seguinte só é conhecido depois do fato: serve para entender *onde* o modelo erra, nunca como atributo.

## 5. Leitura dos resultados

- No teste, o modelo selecionado (Perceptron) ficou +0.067 em acurácia balanceada em relação ao melhor baseline. O número de acertos **não** é estatisticamente distinguível de um palpite 50/50 (p ≥ 0,05): a diferença cabe no acaso de um teste com poucos dias.
- A dispersão entre partições do walk-forward mostra a instabilidade: o desempenho de um mesmo modelo varia vários pontos de um trecho para outro.
- Prever a direção **diária** de uma criptomoeda, sem vazamento e com split temporal, fica próximo de cara-ou-coroa. Empatar com o baseline não é falha do trabalho: é o comportamento esperado do domínio, e o valor desta etapa está no rigor do protocolo (split temporal, escala ajustada só no treino, baselines nas mesmas partições, análise de erros).
- Extensões com sinal mais estável: alvo `acima_mediana` (desempenho relativo) e a regressão de volatilidade futura (`src/regression.py`).
