"""prever.py — Classificação do dia seguinte da Solana.

Responde à pergunta prática: *"o que o modelo diz sobre amanhã?"*

Diferença em relação a ``classification.py``: lá o objetivo é **comparar** modelos
num teste retido; aqui o objetivo é **usar** o modelo escolhido, dia a dia, como
seria usado de verdade.

Protocolo (walk-forward expansivo — a mesma regra para todos os dias):
  para cada dia `d`, o modelo é reajustado **apenas com os dias anteriores a `d`**
  e então classifica `d`. Nenhuma linha é classificada por um modelo que já a viu.
  O dia seguinte (que ainda não aconteceu) é só mais um passo desse mesmo laço —
  por isso o histórico exibido é comparável à previsão de amanhã, e não uma
  medida otimista de dentro da amostra.

Saídas:
  data/processed/previsoes.parquet  -> histórico classificado + o dia seguinte
  (e um resumo no terminal)

**Trabalho acadêmico.** A saída é a classe atribuída por um modelo estatístico a
partir de padrões passados, condicionada ao período analisado. Não é previsão
confiável do mercado nem recomendação de investimento — a avaliação em
``reports/comparacao_classificadores.md`` mostra desempenho próximo do acaso.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import classification
import features

# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA = RAIZ / "data" / "processed" / "previsoes.parquet"

# Modelo usado nas previsões. O padrão é o vencedor do walk-forward feito em
# `classification.py` (seleção que não olhou o teste retido). Qualquer chave de
# `classification.construir_modelos()` serve.
MODELO = "Perceptron"

# Quantos dias recentes reclassificar para exibir o comportamento do modelo.
DIAS_HISTORICO = 15

ROTULOS = {0: "baixa/estável", 1: "alta"}


# ----------------------------------------------------------------------
# 1. Dados: histórico rotulado + o dia a classificar
# ----------------------------------------------------------------------
def carregar_para_previsao() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o histórico com rótulo do último dia (ainda sem "amanhã").

    Retorna ``(rotulados, a_prever)``. ``a_prever`` normalmente tem uma linha:
    o dia mais recente da coleta, cujos atributos já estão completos mas cujo
    resultado só será conhecido no dia seguinte.
    """
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {ENTRADA}\nRode antes `python src/preprocess.py`."
        )
    precos = pd.read_parquet(ENTRADA)
    tabela = features.montar_tabela(precos, manter_sem_rotulo=True)

    rotulados = tabela[tabela["y"].notna()].reset_index(drop=True)
    a_prever = tabela[tabela["y"].isna()].reset_index(drop=True)
    return rotulados, a_prever


# ----------------------------------------------------------------------
# 2. Walk-forward expansivo
# ----------------------------------------------------------------------
def _ajustar_e_classificar(
    treino: pd.DataFrame, alvo: pd.DataFrame, nome_modelo: str
) -> int:
    """Ajusta um Pipeline novo no treino e classifica as linhas de ``alvo``."""
    colunas = features.colunas_atributos()
    modelo = classification.construir_modelos()[nome_modelo]
    modelo.fit(treino[colunas], treino["y"].astype(int))
    return int(modelo.predict(alvo[colunas])[0])


def classificar_historico(
    rotulados: pd.DataFrame, n_dias: int = DIAS_HISTORICO, nome_modelo: str = MODELO
) -> pd.DataFrame:
    """Reclassifica os ``n_dias`` finais, cada um com um modelo ajustado só no passado."""
    inicio = max(len(rotulados) - n_dias, 1)
    registros = []

    for pos in range(inicio, len(rotulados)):
        dia = rotulados.iloc[[pos]]
        previsto = _ajustar_e_classificar(rotulados.iloc[:pos], dia, nome_modelo)
        observado = int(dia["y"].iloc[0])
        registros.append({
            "data_base": dia["data"].iloc[0],
            "data_alvo": dia["data"].iloc[0] + pd.Timedelta(days=1),
            "preco_base": float(dia["preco"].iloc[0]),
            "classificado": previsto,
            "observado": observado,
            "acertou": bool(previsto == observado),
            "dias_treino": pos,
        })

    return pd.DataFrame(registros)


def classificar_dia_seguinte(
    rotulados: pd.DataFrame, a_prever: pd.DataFrame, nome_modelo: str = MODELO
) -> dict:
    """Classifica o dia seguinte usando **todo** o histórico rotulado disponível."""
    if a_prever.empty:
        raise SystemExit(
            "Não há dia pendente para classificar — a coleta não inclui um dia "
            "posterior ao último rótulo. Rode `python src/collect.py` para atualizar."
        )
    dia = a_prever.iloc[[-1]]
    previsto = _ajustar_e_classificar(rotulados, dia, nome_modelo)
    return {
        "data_base": dia["data"].iloc[0],
        "data_alvo": dia["data"].iloc[0] + pd.Timedelta(days=1),
        "preco_base": float(dia["preco"].iloc[0]),
        "classificado": previsto,
        "observado": pd.NA,
        "acertou": pd.NA,
        "dias_treino": len(rotulados),
    }


# ----------------------------------------------------------------------
# 3. Apresentação
# ----------------------------------------------------------------------
def imprimir_historico(historico: pd.DataFrame) -> None:
    """Mostra os dias recentes já com resultado conhecido."""
    print(
        f"\nÚltimos {len(historico)} dias — cada um classificado por um modelo "
        "ajustado só com os dias anteriores:\n"
    )
    print(f"  {'dia alvo':<12} {'preço base':>11}  {'classificado':<14} "
          f"{'observado':<14} {'':>3}")
    print("  " + "-" * 60)
    for _, r in historico.iterrows():
        marca = "ok" if r["acertou"] else "X"
        print(
            f"  {str(r['data_alvo'].date()):<12} {r['preco_base']:>11.2f}  "
            f"{ROTULOS[r['classificado']]:<14} {ROTULOS[r['observado']]:<14} {marca:>3}"
        )
    acertos = int(historico["acertou"].sum())
    print(
        f"\n  acertos nesse trecho: {acertos}/{len(historico)} "
        f"({acertos / len(historico):.0%}) — amostra pequena demais para "
        "concluir qualquer coisa."
    )


def imprimir_dia_seguinte(previsao: dict, nome_modelo: str) -> None:
    """Mostra a classificação do dia que ainda não aconteceu."""
    print("\n" + "=" * 62)
    print("  CLASSIFICAÇÃO DO DIA SEGUINTE")
    print("=" * 62)
    print(f"  dados até:      {previsao['data_base'].date()} "
          f"(preço {previsao['preco_base']:.2f} USD)")
    print(f"  dia classificado: {previsao['data_alvo'].date()}")
    print(f"  modelo:         {nome_modelo}, ajustado em "
          f"{previsao['dias_treino']} dias")
    print(f"\n  -> classificado como: {ROTULOS[previsao['classificado']].upper()}")
    print("=" * 62)
    print(
        "  Leitura correta: o modelo atribuiu essa classe a partir de padrões\n"
        "  do período analisado. NÃO é uma afirmação sobre o que vai acontecer,\n"
        "  nem recomendação de investimento. O desempenho medido em teste é\n"
        "  próximo do acaso (ver reports/comparacao_classificadores.md)."
    )


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    rotulados, a_prever = carregar_para_previsao()
    print(
        f"Histórico rotulado: {len(rotulados)} dias "
        f"({rotulados['data'].min().date()} a {rotulados['data'].max().date()})"
    )

    historico = classificar_historico(rotulados)
    imprimir_historico(historico)

    previsao = classificar_dia_seguinte(rotulados, a_prever)
    imprimir_dia_seguinte(previsao, MODELO)

    todas = pd.concat([historico, pd.DataFrame([previsao])], ignore_index=True)
    # a linha do dia seguinte não tem resultado: tipos nuláveis evitam que
    # `observado` vire float e `acertou` vire object na concatenação
    todas["observado"] = todas["observado"].astype("Int64")
    todas["acertou"] = todas["acertou"].astype("boolean")
    todas.to_parquet(SAIDA, index=False)
    print(f"\nTabela salva em: {SAIDA}")


if __name__ == "__main__":
    main()
