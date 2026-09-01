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
    rotulados: pd.DataFrame,
    precos_por_data: pd.Series,
    n_dias: int = DIAS_HISTORICO,
    nome_modelo: str = MODELO,
) -> pd.DataFrame:
    """Reclassifica os ``n_dias`` finais, cada um com um modelo ajustado só no passado.

    ``precos_por_data`` cobre a série inteira, inclusive o dia pendente. É o que
    permite mostrar o preço do dia-alvo até na última linha do histórico: aquele
    preço não está em ``rotulados`` (o último dia da série não tem rótulo), mas
    existe — é justamente o que tornou o dia anterior rotulável.
    """
    inicio = max(len(rotulados) - n_dias, 1)
    registros = []

    for pos in range(inicio, len(rotulados)):
        dia = rotulados.iloc[[pos]]
        previsto = _ajustar_e_classificar(rotulados.iloc[:pos], dia, nome_modelo)
        observado = int(dia["y"].iloc[0])
        data_base = dia["data"].iloc[0]
        data_alvo = data_base + pd.Timedelta(days=1)
        registros.append({
            "data_base": data_base,
            "data_alvo": data_alvo,
            "preco_base": float(dia["preco"].iloc[0]),
            "preco_alvo": float(precos_por_data[data_alvo]),
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
        "preco_alvo": pd.NA,          # o dia-alvo ainda não aconteceu
        "classificado": previsto,
        "observado": pd.NA,
        "acertou": pd.NA,
        "dias_treino": len(rotulados),
    }


# ----------------------------------------------------------------------
# 3. Apresentação
# ----------------------------------------------------------------------
LARGURA = 78


def imprimir_historico(historico: pd.DataFrame) -> None:
    """Mostra os dias recentes já com resultado conhecido.

    A tabela exibe os **dois** preços (véspera e dia-alvo) de propósito: mostrar
    só o preço base ao lado da data-alvo faz parecer que aquele é o preço do dia
    rotulado, quando ele é o do dia anterior.
    """
    print("\n  DIAS RECENTES (resultado já conhecido)\n")
    print(f"  {'dia':<12}{'véspera':>9}{'':4}{'dia':>8}{'var.':>8}   "
          f"{'classificado':<15}{'observado':<15}")
    print("  " + "-" * LARGURA)
    for _, r in historico.iterrows():
        marca = "ok" if r["acertou"] else "X"
        variacao = r["preco_alvo"] / r["preco_base"] - 1
        print(
            # seta em ASCII: o console do Windows usa cp1252 e não codifica '→'
            f"  {str(r['data_alvo'].date()):<12}{r['preco_base']:>9.2f} -> "
            f"{r['preco_alvo']:>8.2f}{variacao:>+8.1%}   "
            f"{ROTULOS[r['classificado']]:<15}{ROTULOS[r['observado']]:<15}{marca}"
        )
    acertos = int(historico["acertou"].sum())
    print("  " + "-" * LARGURA)
    print(f"  acertos: {acertos}/{len(historico)} "
          f"({acertos / len(historico):.0%})")


def imprimir_dia_seguinte(previsao: dict, nome_modelo: str) -> None:
    """Mostra a classificação do dia que ainda não aconteceu."""
    print("\n  " + "=" * LARGURA)
    print(f"  PREVISÃO — {previsao['data_alvo'].date()} (ainda não aconteceu)")
    print(f"  partindo de {previsao['preco_base']:.2f} USD "
          f"em {previsao['data_base'].date()}  ->  "
          f"{ROTULOS[previsao['classificado']].upper()}")
    print("  " + "=" * LARGURA)
    print(
        f"  {nome_modelo} · {previsao['dias_treino']} dias de treino · "
        "preços são snapshots de 00:00 UTC"
    )
    # a ressalva fica em uma linha, mas não sai: é classe atribuída por modelo,
    # não afirmação sobre o mercado (ver reports/comparacao_classificadores.md)
    print("  Classe atribuída por modelo estatístico — desempenho próximo do acaso.")


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    rotulados, a_prever = carregar_para_previsao()
    # série completa (com o dia pendente) para resolver o preço de cada dia-alvo
    precos_por_data = (
        pd.concat([rotulados, a_prever], ignore_index=True)
        .set_index("data")["preco"]
    )
    historico = classificar_historico(rotulados, precos_por_data)
    imprimir_historico(historico)

    previsao = classificar_dia_seguinte(rotulados, a_prever)
    imprimir_dia_seguinte(previsao, MODELO)

    todas = pd.concat([historico, pd.DataFrame([previsao])], ignore_index=True)
    # a linha do dia seguinte não tem resultado: tipos nuláveis evitam que
    # `observado` vire float e `acertou` vire object na concatenação
    todas["observado"] = todas["observado"].astype("Int64")
    todas["acertou"] = todas["acertou"].astype("boolean")
    todas.to_parquet(SAIDA, index=False)   # consumido pelo dashboard
    print()


if __name__ == "__main__":
    main()
