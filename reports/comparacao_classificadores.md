# Classificação da direção do dia seguinte — Solana

Encontros 10-11. **Trabalho acadêmico**: os resultados são estimativas condicionadas ao período analisado e não constituem recomendação de investimento.

- Gerado em: `2026-09-22T01:07:49+00:00`
- Entrada: `data\processed\atributos.parquet`
- Alvo: `direcao`
- Semente fixa: `42`

## Desenho experimental

| conjunto | dias | período | proporção de alta |
| --- | --- | --- | --- |
| treino | 191 | 2025-11-30 a 2026-06-08 | 46.1% |
| teste | 83 | 2026-06-09 a 2026-08-30 | 51.8% |
| total | 274 | 2025-11-30 a 2026-08-30 | 47.8% |

> A divisão é **cronológica**: o teste é o trecho final da série. Nenhum embaralhamento é usado — embaralhar aqui seria vazamento. A padronização (`StandardScaler`) está dentro do `Pipeline`, então média e desvio saem só do treino de cada partição.

## 1. Baselines (medidos antes dos modelos)

| baseline | acurácia balanceada | F1 | precisão | recall | acurácia |
| --- | --- | --- | --- | --- | --- |
| Baseline: sempre alta | 0.500 | 0.683 | 0.518 | 1.000 | 0.518 |
| Baseline: classe majoritária (treino) | 0.500 | 0.000 | 0.000 | 0.000 | 0.482 |
| Baseline: persistência | 0.481 | 0.506 | 0.500 | 0.512 | 0.482 |

> Referência obrigatória: um classificador só agrega valor se superar estes números. Note que *sempre alta* tem acurácia balanceada 0,5 por construção (acerta 100% de uma classe e 0% da outra) — é justamente por isso que a acurácia balanceada é a métrica principal aqui.

## 2. Comparação walk-forward (`TimeSeriesSplit`, 5 partições)

Rodado **apenas dentro do treino**, para que a escolha do melhor modelo não enxergue o teste. Média ± desvio entre partições:

| modelo | acurácia balanceada (média ± dp) | F1 (média ± dp) |
| --- | --- | --- |
| Perceptron | 0.519 ± 0.083 | 0.367 ± 0.141 |
| Árvore de decisão | 0.495 ± 0.032 | 0.497 ± 0.044 |
| SVM (RBF) | 0.480 ± 0.023 | 0.293 ± 0.255 |
| KNN | 0.478 ± 0.046 | 0.252 ± 0.242 |
| Naive Bayes | 0.462 ± 0.050 | 0.151 ± 0.208 |

> Modelo selecionado: **Perceptron** (maior acurácia balanceada média no walk-forward). O desvio entre partições é grande em relação à diferença entre modelos — sinal de que o ranking é instável.

## 3. Avaliação final no teste retido

| modelo | acurácia balanceada | F1 | precisão | recall | acurácia |
| --- | --- | --- | --- | --- | --- |
| Perceptron | 0.575 | 0.624 | 0.580 | 0.674 | 0.578 |
| SVM (RBF) | 0.499 | 0.432 | 0.516 | 0.372 | 0.494 |
| Árvore de decisão | 0.483 | 0.482 | 0.500 | 0.465 | 0.482 |
| KNN | 0.458 | 0.471 | 0.476 | 0.465 | 0.458 |
| Naive Bayes | 0.419 | 0.222 | 0.350 | 0.163 | 0.410 |

> Cada modelo foi reajustado no treino inteiro e avaliado **uma única vez** no teste.

## 4. Matriz de confusão — Perceptron

|  | classificado 0 (baixa/estável) | classificado 1 (alta) |
| --- | --- | --- |
| **observado 0** | 19 | 21 |
| **observado 1** | 14 | 29 |

## 5. Análise de erros

**Distribuição das previsões**

| métrica | valor |
| --- | --- |
| dias de teste | 83 |
| previsões de alta | 50 |
| altas reais | 43 |

**Taxa de acerto por classe real**

| classe real | dias | taxa de acerto |
| --- | --- | --- |
| 0 | 40 | 0.475 |
| 1 | 43 | 0.674 |

**Taxa de acerto por regime de volatilidade (tercis de `vol_30d`)**

| faixa de volatilidade | dias | taxa de acerto |
| --- | --- | --- |
| baixo | 28 | 0.643 |
| médio | 27 | 0.630 |
| alto | 28 | 0.464 |

**Taxa de acerto por tamanho do movimento realizado (|retorno t+1|)**

| faixa do movimento | dias | taxa de acerto |
| --- | --- | --- |
| baixo | 28 | 0.500 |
| médio | 27 | 0.444 |
| alto | 28 | 0.786 |

## 6. Discussão

O modelo selecionado (**Perceptron**) obteve acurácia balanceada **0.575** no teste retido, contra **0.500** do melhor baseline — ou seja, superou o melhor baseline em +0.075.

**Viés de classe.** O modelo classificou como alta 60.2% dos 83 dias de teste, enquanto apenas 51.8% foram de fato de alta: ele é otimista por construção do ajuste. Isso aparece na assimetria de acerto — 67.4% nos dias que subiram contra 47.5% nos que não subiram. Num período de tendência de queda como o analisado, esse viés é frágil: ele depende de a proporção de altas se manter.

**Regime de volatilidade.** O acerto cai de 64.3% (volatilidade baixa) e 63.0% (média) para 46.4% nos dias de volatilidade alta — coerente com a intuição de que regimes turbulentos são menos previsíveis.

**Tamanho do movimento.** O acerto é 50.0% nos movimentos pequenos, 44.4% nos médios e 78.6% nos grandes. Acertar movimentos pequenos tem pouco significado prático (a diferença entre subir e cair é quase arbitrária ali), então concentrar acerto nos movimentos grandes é o padrão mais interessante — embora com poucas dezenas de dias possa ser coincidência.

**Significância.** Com 83 dias de teste, o erro padrão da acurácia sob a hipótese de cara-ou-coroa é 0.055. A acurácia observada (0.578) está a 1.4 desvios de 0,5 — ou seja, **dentro do que o acaso produz**; não é evidência de capacidade preditiva real.

Vale o mesmo para a comparação entre modelos: diferenças de poucos pontos estão dentro do ruído, e é por isso que o walk-forward reporta o desvio entre partições, e não só a média.

**Enquadramento honesto.** Prever a direção **diária** de um criptoativo, com split temporal e sem vazamento, fica próximo de cara-ou-coroa. Isso é esperado: se um padrão simples e estável existisse nesses atributos, ele seria arbitrado rapidamente. O resultado modesto **não indica falha de execução** — indica que o sinal disponível nesse horizonte, com esses atributos e esse tamanho de amostra, é fraco. O que sustenta o trabalho é o rigor do protocolo: atributos só com passado (verificado por recomputação em série truncada), rótulo isolado no futuro, divisão cronológica, padronização dentro do `Pipeline`, baseline medido antes e métricas balanceadas.

**Extensões possíveis** (mais promissoras que direção pura): alvo `acima_mediana` em `features.py` (desempenho relativo, mais estável) ou regressão da **volatilidade futura**, que é bem mais previsível que o sinal do retorno.
