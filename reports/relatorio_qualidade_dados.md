# Relatório de qualidade dos dados — Encontro 4

Trabalho acadêmico (Aprendizagem de Máquina). Nada aqui é recomendação de investimento.

- Gerado em: `2026-09-22T01:14:48+00:00`
- Entrada: `data\processed\precos.parquet`
- Saída limpa: `data\processed\precos_limpo.parquet`
- Período coberto: **2025-09-23 a 2026-09-22**

## Parâmetros usados

- `LIMIAR_LACUNA_DIAS` = 3
- `N_MIN_OBS` = 60
- `FRACAO_MAX_LACUNAS` = 0.2

## Resumo geral

| Métrica | Valor |
| --- | --- |
| Observações no arquivo bruto | 365 |
| Duplicatas (id, data) removidas | 0 |
| Observações após limpeza (série contínua) | 365 |
| Ativos mantidos | 1 |
| Ativos removidos | 0 |

## Lacunas por ativo

| id | dias no período | dias ausentes | blocos de lacuna | maior lacuna (dias) | fração de lacunas |
| --- | --- | --- | --- | --- | --- |
| solana | 365 | 0 | 0 | 0 | 0.0% |

### `solana` — detalhe das lacunas

Sem lacunas: série diária já contígua no período.

## Preenchimento aplicado

| id | linhas preenchidas (lacunas curtas) | lacunas longas | dias em lacuna longa (NaN) |
| --- | --- | --- | --- |
| solana | 0 | 0 | 0 |

> Lacunas curtas: preço/capitalização por interpolação linear no tempo e volume por `ffill`, com a coluna booleana `preenchido = True`.
> Lacunas longas: **não preenchidas** (a apostila proíbe preenchimento artificial de trechos longos); use `recortar_maior_trecho_contiguo()` para obter uma subsérie sem NaN quando necessário.

## Sanidade de valores

| id | preço <= 0 | capitalização <= 0 | volume < 0 |
| --- | --- | --- | --- |
| solana | 0 | 0 | 0 |

> Valores fora de faixa são convertidos para NaN e contam como lacuna.

## Ativos removidos

Nenhum ativo removido. (O mecanismo genérico — `N_MIN_OBS` e `FRACAO_MAX_LACUNAS` — permanece ativo para coletas futuras.)
