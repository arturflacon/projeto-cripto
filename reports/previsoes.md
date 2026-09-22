# Previsões do dia seguinte — Solana

**Trabalho acadêmico.** As previsões abaixo são estimativas de modelos estatísticos a partir de padrões passados, condicionadas ao período analisado. Não são previsão confiável do mercado nem recomendação de investimento.

- Gerado em: `2026-09-22T01:14:58+00:00`
- Entrada: `data\processed\precos_limpo.parquet`
- Janela exibida: **2026-09-08 a 2026-09-22** (15 dias com resultado conhecido) + o dia seguinte
- Classificador: `Perceptron` · Regressor: `KNN`

## Protocolo

Walk-forward expansivo: para cada dia `d` da janela, o modelo é ajustado **apenas com os dias anteriores a `d`** e então prevê `d`. Os atributos de `d` usam só dados até `d`; o dia seguinte (`d+1`) entra apenas como gabarito, depois da previsão — nunca como atributo. A linha do dia seguinte real é só mais um passo do mesmo laço, então o histórico é comparável a ela.

## 1. Direção do dia seguinte — Perceptron

| dia | véspera (USD) | dia (USD) | variação | classificado | observado |  |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-08 | 106.49 | 103.77 | -2.6% | alta | baixa/estável | ✗ |
| 2026-09-09 | 103.77 | 103.36 | -0.4% | alta | baixa/estável | ✗ |
| 2026-09-10 | 103.36 | 101.57 | -1.7% | baixa/estável | baixa/estável | ✓ |
| 2026-09-11 | 101.57 | 98.63 | -2.9% | baixa/estável | baixa/estável | ✓ |
| 2026-09-12 | 98.63 | 102.39 | +3.8% | baixa/estável | alta | ✗ |
| 2026-09-13 | 102.39 | 101.75 | -0.6% | alta | baixa/estável | ✗ |
| 2026-09-14 | 101.75 | 99.33 | -2.4% | baixa/estável | baixa/estável | ✓ |
| 2026-09-15 | 99.33 | 102.51 | +3.2% | baixa/estável | alta | ✗ |
| 2026-09-16 | 102.51 | 96.87 | -5.5% | baixa/estável | baixa/estável | ✓ |
| 2026-09-17 | 96.87 | 98.54 | +1.7% | baixa/estável | alta | ✗ |
| 2026-09-18 | 98.54 | 101.57 | +3.1% | baixa/estável | alta | ✗ |
| 2026-09-19 | 101.57 | 112.70 | +11.0% | baixa/estável | alta | ✗ |
| 2026-09-20 | 112.70 | 111.02 | -1.5% | baixa/estável | baixa/estável | ✓ |
| 2026-09-21 | 111.02 | 111.15 | +0.1% | baixa/estável | alta | ✗ |
| 2026-09-22 | 111.15 | 118.87 | +6.9% | alta | alta | ✓ |
| **2026-09-23** | 118.87 | — | — | **baixa/estável** | (pendente) |  |

> Acertos na janela: **6/15** (40%). Com 15 dias, uma moeda honesta fica entre 5 e 10 acertos na maior parte das vezes — a janela é curta demais para distinguir habilidade de acaso.

## 2. Preço do dia seguinte — KNN

O regressor estima o **retorno** de `t -> t+1`; o preço estimado é `preço de hoje × (1 + retorno estimado)`. Erro = estimado − observado.

| dia | véspera (USD) | estimado (USD) | observado (USD) | erro (USD) | erro % | direção |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-08 | 106.49 | 107.11 | 103.77 | +3.34 | +3.2% | ✗ |
| 2026-09-09 | 103.77 | 103.43 | 103.36 | +0.07 | +0.1% | ✓ |
| 2026-09-10 | 103.36 | 102.91 | 101.57 | +1.34 | +1.3% | ✓ |
| 2026-09-11 | 101.57 | 100.72 | 98.63 | +2.09 | +2.1% | ✓ |
| 2026-09-12 | 98.63 | 98.33 | 102.39 | -4.06 | -4.0% | ✗ |
| 2026-09-13 | 102.39 | 101.81 | 101.75 | +0.06 | +0.1% | ✓ |
| 2026-09-14 | 101.75 | 101.32 | 99.33 | +1.99 | +2.0% | ✓ |
| 2026-09-15 | 99.33 | 99.30 | 102.51 | -3.21 | -3.1% | ✗ |
| 2026-09-16 | 102.51 | 102.35 | 96.87 | +5.48 | +5.7% | ✓ |
| 2026-09-17 | 96.87 | 96.24 | 98.54 | -2.30 | -2.3% | ✗ |
| 2026-09-18 | 98.54 | 98.25 | 101.57 | -3.32 | -3.3% | ✗ |
| 2026-09-19 | 101.57 | 101.37 | 112.70 | -11.33 | -10.1% | ✗ |
| 2026-09-20 | 112.70 | 112.54 | 111.02 | +1.52 | +1.4% | ✓ |
| 2026-09-21 | 111.02 | 111.14 | 111.15 | -0.02 | -0.0% | ✓ |
| 2026-09-22 | 111.15 | 111.19 | 118.87 | -7.68 | -6.5% | ✓ |
| **2026-09-23** | 118.87 | **118.77** (-0.09%) | (pendente) | — | — |  |

**Erro na janela, contra o baseline de persistência** ("o preço de amanhã é o de hoje"):

|  | MAE (USD) | RMSE (USD) | erro % médio |
| --- | --- | --- | --- |
| KNN | 3.19 | 4.35 | 3.00% |
| Baseline: persistência | 3.26 | 4.32 | 3.08% |

> Direção acertada pelo sinal do retorno estimado: **9/15**. O regressor errou **menos** que a persistência nesta janela — diferença pequena, sobre poucos dias, que pode não se repetir.

### Como o regressor foi escolhido

`TimeSeriesSplit` com 5 partições sobre os 259 dias **anteriores** à janela exibida — nenhum dia previsto acima participou da escolha. Critério: menor RMSE médio do retorno diário.

| modelo | RMSE médio do retorno | desvio |
| --- | --- | --- |
| Baseline: retorno zero | 0.0319 | 0.0071 |
| KNN | 0.0330 | 0.0068 |
| SVR (RBF) | 0.0337 | 0.0098 |
| Árvore de decisão | 0.0455 | 0.0142 |
| Ridge (linear) | 0.0467 | 0.0217 |

> Se o baseline *retorno zero* aparece no topo, nenhum regressor aprendeu algo além de "amanhã ≈ hoje" — o escolhido é apenas o menos ruim entre os modelos.

## Ressalvas

- Os preços são instantâneos de 00:00 UTC; "amanhã" é o próximo instantâneo, não o fechamento de um pregão.
- A avaliação formal, em teste retido de 30% da série, mostra desempenho **próximo do acaso** para direção; para o preço, a persistência é uma referência que modelos simples raramente superam no horizonte de um dia.
- Cada linha usa um modelo reajustado só com o passado, então o histórico não é uma medida otimista de dentro da amostra — mas a janela é curta e não sustenta conclusões sobre o futuro.
