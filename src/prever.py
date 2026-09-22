"""prever.py — Previsões do dia seguinte da Solana (direção e preço).

Responde à pergunta prática: *"o que os modelos dizem sobre amanhã?"* — e mostra
como eles teriam se saído nos últimos dias, sob a mesma regra.

Duas previsões por dia:
  * **classificação** da direção (sobe / não sobe), com o classificador vencedor
    de ``classification.py``;
  * **regressão** do preço: um regressor estima o retorno de ``t -> t+1`` e o
    preço de amanhã sai de ``preco_hoje * (1 + retorno_estimado)``.

Protocolo (walk-forward expansivo — a mesma regra para todos os dias):
  para cada dia `d`, o modelo é reajustado **apenas com os dias anteriores a `d`**
  e então prevê `d`. Nenhuma linha é prevista por um modelo que já a viu, e o dia
  seguinte a `d` nunca entra nos atributos — ele só aparece como gabarito depois.
  O dia seguinte (que ainda não aconteceu) é só mais um passo desse mesmo laço —
  por isso o histórico exibido é comparável à previsão de amanhã.

Classificador e regressor são escolhidos por ``TimeSeriesSplit`` **antes** da
janela exibida, com dados estritamente anteriores a ela — a escolha não enxerga
os dias previstos.

Saídas:
  data/processed/previsoes.parquet            -> direção (consumido pelo dashboard)
  data/processed/previsoes_regressao.parquet  -> preço estimado
  reports/previsoes.md                        -> o relatório de previsões

**Trabalho acadêmico.** As saídas são estimativas de modelos estatísticos a partir
de padrões passados, condicionadas ao período analisado. Não são previsão
confiável do mercado nem recomendação de investimento.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

import classification
import features
import regression
from preprocess import _tabela_md as tabela_md

# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA = RAIZ / "data" / "processed" / "previsoes.parquet"
SAIDA_REGRESSAO = RAIZ / "data" / "processed" / "previsoes_regressao.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "previsoes.md"

# Modelos fixados à mão (chaves de `classification.construir_modelos()` e
# `regression.construir_modelos()`). ``None`` = escolher por walk-forward nos
# dias anteriores à janela exibida, que é o padrão.
CLASSIFICADOR = None
REGRESSOR = None

# Quantos dias recentes re-prever para exibir o comportamento dos modelos.
DIAS_HISTORICO = 15

# Partições do walk-forward usado para escolher os modelos (só com dados
# anteriores à janela exibida).
N_SPLITS_SELECAO = 5

ROTULOS = {0: "baixa/estável", 1: "alta"}


# ----------------------------------------------------------------------
# 1. Dados: histórico rotulado + o dia a prever
# ----------------------------------------------------------------------
def carregar_para_previsao() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Separa o histórico com rótulo do último dia (ainda sem "amanhã").

    Retorna ``(rotulados, a_prever, precos_por_data)``. ``a_prever`` normalmente
    tem uma linha: o dia mais recente da coleta, cujos atributos já estão
    completos mas cujo resultado só será conhecido no dia seguinte.
    ``precos_por_data`` cobre a série inteira, inclusive o dia pendente.
    """
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {ENTRADA}\nRode antes `python src/preprocess.py`."
        )
    precos = pd.read_parquet(ENTRADA)
    tabela = features.montar_tabela(precos, manter_sem_rotulo=True)

    precos_por_data = tabela.set_index("data")["preco"]
    tabela = _adicionar_retorno_futuro(tabela, precos_por_data)

    rotulados = tabela[tabela["y"].notna()].reset_index(drop=True)
    a_prever = tabela[tabela["y"].isna()].reset_index(drop=True)
    return rotulados, a_prever, precos_por_data


def _adicionar_retorno_futuro(
    tabela: pd.DataFrame, precos_por_data: pd.Series
) -> pd.DataFrame:
    """Alvo contínuo da regressão: retorno de ``t -> t+1``.

    É o **único** olhar para o futuro, e só serve de gabarito no ajuste — nunca
    entra como atributo. No último dia da série fica ``NaN`` (amanhã não existe).
    """
    amanha = tabela["data"].map(
        lambda d: precos_por_data.get(d + pd.Timedelta(days=1), np.nan)
    )
    saida = tabela.copy()
    saida["ret_futuro"] = amanha.to_numpy() / saida["preco"].to_numpy() - 1.0
    return saida


# ----------------------------------------------------------------------
# 2. Classificação da direção — walk-forward expansivo
# ----------------------------------------------------------------------
def _ajustar_e_classificar(
    treino: pd.DataFrame, alvo: pd.DataFrame, nome_modelo: str
) -> int:
    """Ajusta um Pipeline novo no treino e classifica as linhas de ``alvo``."""
    colunas = features.colunas_atributos()
    modelo = classification.construir_modelos()[nome_modelo]
    modelo.fit(treino[colunas], treino["y"].astype(int))
    return int(modelo.predict(alvo[colunas])[0])


def selecionar_classificador(
    treino: pd.DataFrame, n_splits: int = N_SPLITS_SELECAO
) -> tuple[str, pd.DataFrame]:
    """Escolhe o classificador por ``TimeSeriesSplit`` — só com o trecho recebido.

    Chamada com os dias **anteriores** à janela exibida. Critério: maior acurácia
    balanceada média entre partições (o mesmo de ``classification.py``).
    """
    colunas = features.colunas_atributos()
    X = treino[colunas].to_numpy()
    y = treino["y"].astype(int).to_numpy()
    particoes = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    linhas = []
    for nome in classification.construir_modelos():
        acuracias = []
        for idx_treino, idx_val in particoes:
            modelo = classification.construir_modelos()[nome]   # Pipeline novo por partição
            modelo.fit(X[idx_treino], y[idx_treino])
            acuracias.append(balanced_accuracy_score(y[idx_val], modelo.predict(X[idx_val])))
        linhas.append({"modelo": nome, "acuracia_balanceada_media": float(np.mean(acuracias)),
                       "acuracia_balanceada_desvio": float(np.std(acuracias))})

    selecao = (
        pd.DataFrame(linhas)
        .sort_values("acuracia_balanceada_media", ascending=False)
        .reset_index(drop=True)
    )
    return str(selecao.iloc[0]["modelo"]), selecao


def classificar_historico(
    rotulados: pd.DataFrame,
    precos_por_data: pd.Series,
    nome_modelo: str,
    n_dias: int = DIAS_HISTORICO,
) -> pd.DataFrame:
    """Reclassifica os ``n_dias`` finais, cada um com um modelo ajustado só no passado.

    A tabela guarda os **dois** preços (véspera e dia-alvo): o preço do dia-alvo
    da última linha não está em ``rotulados`` (o último dia da série não tem
    rótulo), mas existe em ``precos_por_data`` — é justamente o que tornou o dia
    anterior rotulável.
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
    rotulados: pd.DataFrame, a_prever: pd.DataFrame, nome_modelo: str
) -> dict:
    """Classifica o dia seguinte usando **todo** o histórico rotulado disponível."""
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
# 3. Regressão do preço — walk-forward expansivo
# ----------------------------------------------------------------------
def _ajustar_e_estimar(
    treino: pd.DataFrame, alvo: pd.DataFrame, nome_modelo: str
) -> float:
    """Ajusta um regressor novo no treino e estima o retorno de ``t -> t+1``."""
    colunas = features.colunas_atributos()
    modelo = regression.construir_modelos()[nome_modelo]
    modelo.fit(treino[colunas], treino["ret_futuro"])
    return float(modelo.predict(alvo[colunas])[0])


def selecionar_regressor(
    treino: pd.DataFrame, n_splits: int = N_SPLITS_SELECAO
) -> tuple[str, pd.DataFrame]:
    """Escolhe o regressor por ``TimeSeriesSplit`` — só com o trecho recebido.

    Chamada com os dias **anteriores** à janela exibida, para que a escolha não
    enxergue nenhum dia que depois será previsto. O critério é o menor RMSE
    médio do retorno entre partições. A tabela devolvida inclui, para
    referência, o baseline *retorno zero* (preço de amanhã = preço de hoje).
    """
    colunas = features.colunas_atributos()
    X = treino[colunas].to_numpy()
    y = treino["ret_futuro"].to_numpy()
    particoes = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    linhas = []
    for nome in regression.construir_modelos():
        rmses = []
        for idx_treino, idx_val in particoes:
            modelo = regression.construir_modelos()[nome]   # Pipeline novo por partição
            modelo.fit(X[idx_treino], y[idx_treino])
            previsto = modelo.predict(X[idx_val])
            rmses.append(float(np.sqrt(np.mean((y[idx_val] - previsto) ** 2))))
        linhas.append({"modelo": nome, "rmse_media": float(np.mean(rmses)),
                       "rmse_desvio": float(np.std(rmses))})

    # baseline: prever retorno zero em cada partição de validação
    rmses_zero = [float(np.sqrt(np.mean(y[idx_val] ** 2))) for _, idx_val in particoes]
    linhas.append({"modelo": "Baseline: retorno zero",
                   "rmse_media": float(np.mean(rmses_zero)),
                   "rmse_desvio": float(np.std(rmses_zero))})

    selecao = pd.DataFrame(linhas).sort_values("rmse_media").reset_index(drop=True)
    candidatos = selecao[~selecao["modelo"].str.startswith("Baseline")]
    return str(candidatos.iloc[0]["modelo"]), selecao


def estimar_historico(
    rotulados: pd.DataFrame,
    precos_por_data: pd.Series,
    nome_modelo: str,
    n_dias: int = DIAS_HISTORICO,
) -> pd.DataFrame:
    """Re-estima o preço dos ``n_dias`` finais, cada um com um regressor ajustado só no passado."""
    inicio = max(len(rotulados) - n_dias, 1)
    registros = []

    for pos in range(inicio, len(rotulados)):
        dia = rotulados.iloc[[pos]]
        ret_estimado = _ajustar_e_estimar(rotulados.iloc[:pos], dia, nome_modelo)
        preco_base = float(dia["preco"].iloc[0])
        data_base = dia["data"].iloc[0]
        data_alvo = data_base + pd.Timedelta(days=1)
        preco_alvo = float(precos_por_data[data_alvo])
        preco_estimado = preco_base * (1.0 + ret_estimado)
        registros.append({
            "data_base": data_base,
            "data_alvo": data_alvo,
            "preco_base": preco_base,
            "preco_estimado": preco_estimado,
            "preco_alvo": preco_alvo,
            "ret_estimado": ret_estimado,
            "ret_observado": float(dia["ret_futuro"].iloc[0]),
            "erro": preco_estimado - preco_alvo,
            "erro_pct": preco_estimado / preco_alvo - 1.0,
            "direcao_ok": bool((ret_estimado > 0) == (preco_alvo > preco_base)),
            "dias_treino": pos,
        })

    return pd.DataFrame(registros)


def estimar_dia_seguinte(
    rotulados: pd.DataFrame, a_prever: pd.DataFrame, nome_modelo: str
) -> dict:
    """Estima o preço do dia seguinte usando **todo** o histórico disponível."""
    dia = a_prever.iloc[[-1]]
    ret_estimado = _ajustar_e_estimar(rotulados, dia, nome_modelo)
    preco_base = float(dia["preco"].iloc[0])
    return {
        "data_base": dia["data"].iloc[0],
        "data_alvo": dia["data"].iloc[0] + pd.Timedelta(days=1),
        "preco_base": preco_base,
        "preco_estimado": preco_base * (1.0 + ret_estimado),
        "preco_alvo": np.nan,         # o dia-alvo ainda não aconteceu
        "ret_estimado": ret_estimado,
        "ret_observado": np.nan,
        "erro": np.nan,
        "erro_pct": np.nan,
        "direcao_ok": pd.NA,
        "dias_treino": len(rotulados),
    }


def resumir_regressao(historico: pd.DataFrame) -> dict[str, float]:
    """Erro do regressor na janela exibida, lado a lado com o baseline de persistência.

    *Persistência* = "o preço de amanhã é o de hoje". É a referência honesta:
    em séries de preço ela é notoriamente difícil de superar no dia seguinte.
    """
    erro = historico["erro"].to_numpy()
    erro_persist = (historico["preco_base"] - historico["preco_alvo"]).to_numpy()
    alvo = historico["preco_alvo"].to_numpy()
    return {
        "n": len(historico),
        "mae": float(np.mean(np.abs(erro))),
        "rmse": float(np.sqrt(np.mean(erro ** 2))),
        "mape": float(np.mean(np.abs(erro) / alvo)),
        "mae_persistencia": float(np.mean(np.abs(erro_persist))),
        "rmse_persistencia": float(np.sqrt(np.mean(erro_persist ** 2))),
        "mape_persistencia": float(np.mean(np.abs(erro_persist) / alvo)),
        "acertos_direcao": int(historico["direcao_ok"].sum()),
    }


# ----------------------------------------------------------------------
# 4. Apresentação no terminal
# ----------------------------------------------------------------------
LARGURA = 78


def imprimir_historico(historico: pd.DataFrame) -> None:
    """Direção: dias recentes já com resultado conhecido."""
    print("\n  DIREÇÃO — dias recentes (resultado já conhecido)\n")
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


def imprimir_historico_regressao(historico: pd.DataFrame, resumo: dict) -> None:
    """Preço: dias recentes já com resultado conhecido."""
    print("\n  PREÇO — dias recentes (resultado já conhecido)\n")
    print(f"  {'dia':<12}{'véspera':>9}{'estimado':>10}{'observado':>11}"
          f"{'erro':>9}{'erro %':>8}   direção")
    print("  " + "-" * LARGURA)
    for _, r in historico.iterrows():
        marca = "ok" if r["direcao_ok"] else "X"
        print(
            f"  {str(r['data_alvo'].date()):<12}{r['preco_base']:>9.2f}"
            f"{r['preco_estimado']:>10.2f}{r['preco_alvo']:>11.2f}"
            f"{r['erro']:>+9.2f}{r['erro_pct']:>+8.1%}   {marca}"
        )
    print("  " + "-" * LARGURA)
    print(f"  erro médio absoluto: {resumo['mae']:.2f} USD ({resumo['mape']:.1%})  |  "
          f"persistência: {resumo['mae_persistencia']:.2f} USD "
          f"({resumo['mape_persistencia']:.1%})")
    print(f"  direção acertada pelo sinal do retorno estimado: "
          f"{resumo['acertos_direcao']}/{resumo['n']}")


def imprimir_dia_seguinte(
    previsao: dict, previsao_reg: dict, nome_classificador: str, nome_regressor: str
) -> None:
    """Mostra as duas previsões do dia que ainda não aconteceu."""
    print("\n  " + "=" * LARGURA)
    print(f"  PREVISÃO — {previsao['data_alvo'].date()} (ainda não aconteceu)")
    print(f"  partindo de {previsao['preco_base']:.2f} USD em {previsao['data_base'].date()}")
    print(f"    direção ({nome_classificador}): "
          f"{ROTULOS[previsao['classificado']].upper()}")
    print(f"    preço   ({nome_regressor}): "
          f"{previsao_reg['preco_estimado']:.2f} USD "
          f"({previsao_reg['ret_estimado']:+.2%})")
    print("  " + "=" * LARGURA)
    print(f"  {previsao['dias_treino']} dias de treino · preços são snapshots de 00:00 UTC")
    # a ressalva fica em uma linha, mas não sai: é estimativa de modelo, não
    # afirmação sobre o mercado
    print("  Estimativas de modelos estatísticos — desempenho próximo do acaso.")


# ----------------------------------------------------------------------
# 5. Relatório
# ----------------------------------------------------------------------
def escrever_relatorio(
    caminho: Path,
    *,
    historico: pd.DataFrame,
    previsao: dict,
    historico_reg: pd.DataFrame,
    previsao_reg: dict,
    resumo_reg: dict,
    nome_classificador: str,
    nome_regressor: str,
    selecao_clf: pd.DataFrame,
    selecao_reg: pd.DataFrame,
    n_treino_selecao: int,
) -> None:
    """Escreve o relatório de previsões em Markdown."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p: list[str] = []

    p.append("# Previsões do dia seguinte — Solana\n")
    p.append(
        "**Trabalho acadêmico.** As previsões abaixo são estimativas de modelos "
        "estatísticos a partir de padrões passados, condicionadas ao período "
        "analisado. Não são previsão confiável do mercado nem recomendação de "
        "investimento.\n"
    )
    p.append(f"- Gerado em: `{agora}`")
    p.append(f"- Entrada: `{ENTRADA.relative_to(RAIZ)}`")
    p.append(
        f"- Janela exibida: **{historico['data_alvo'].min().date()} a "
        f"{historico['data_alvo'].max().date()}** ({len(historico)} dias com "
        "resultado conhecido) + o dia seguinte"
    )
    p.append(f"- Classificador: `{nome_classificador}` · Regressor: `{nome_regressor}` "
             "(escolhidos na seção 3)")
    p.append("")

    p.append("## Protocolo\n")
    p.append(
        "Walk-forward expansivo: para cada dia `d` da janela, o modelo é ajustado "
        "**apenas com os dias anteriores a `d`** e então prevê `d`. Os atributos "
        "de `d` usam só dados até `d`; o dia seguinte (`d+1`) entra apenas como "
        "gabarito, depois da previsão — nunca como atributo. A linha do dia "
        "seguinte real é só mais um passo do mesmo laço, então o histórico é "
        "comparável a ela.\n"
    )

    # --- 1. direção
    p.append(f"## 1. Direção do dia seguinte — {nome_classificador}\n")
    p.append(
        tabela_md(
            ["dia", "véspera (USD)", "dia (USD)", "variação", "classificado",
             "observado", ""],
            [
                [
                    r["data_alvo"].date(), f"{r['preco_base']:.2f}",
                    f"{r['preco_alvo']:.2f}",
                    f"{r['preco_alvo'] / r['preco_base'] - 1:+.1%}",
                    ROTULOS[r["classificado"]], ROTULOS[r["observado"]],
                    "✓" if r["acertou"] else "✗",
                ]
                for _, r in historico.iterrows()
            ] + [[
                f"**{previsao['data_alvo'].date()}**", f"{previsao['preco_base']:.2f}",
                "—", "—", f"**{ROTULOS[previsao['classificado']]}**", "(pendente)", "",
            ]],
        )
    )
    acertos = int(historico["acertou"].sum())
    p.append(
        f"\n> Acertos na janela: **{acertos}/{len(historico)}** "
        f"({acertos / len(historico):.0%}). Com {len(historico)} dias, uma moeda "
        f"honesta fica entre {max(0, len(historico) // 2 - 2)} e "
        f"{len(historico) // 2 + 3} acertos na maior parte das vezes — a "
        "janela é curta demais para distinguir habilidade de acaso.\n"
    )

    # --- 2. preço
    p.append(f"## 2. Preço do dia seguinte — {nome_regressor}\n")
    p.append(
        "O regressor estima o **retorno** de `t -> t+1`; o preço estimado é "
        "`preço de hoje × (1 + retorno estimado)`. Erro = estimado − observado.\n"
    )
    p.append(
        tabela_md(
            ["dia", "véspera (USD)", "estimado (USD)", "observado (USD)", "erro (USD)",
             "erro %", "direção"],
            [
                [
                    r["data_alvo"].date(), f"{r['preco_base']:.2f}",
                    f"{r['preco_estimado']:.2f}", f"{r['preco_alvo']:.2f}",
                    f"{r['erro']:+.2f}", f"{r['erro_pct']:+.1%}",
                    "✓" if r["direcao_ok"] else "✗",
                ]
                for _, r in historico_reg.iterrows()
            ] + [[
                f"**{previsao_reg['data_alvo'].date()}**",
                f"{previsao_reg['preco_base']:.2f}",
                f"**{previsao_reg['preco_estimado']:.2f}** "
                f"({previsao_reg['ret_estimado']:+.2%})",
                "(pendente)", "—", "—", "",
            ]],
        )
    )
    p.append("\n**Erro na janela, contra o baseline de persistência** "
             "(\"o preço de amanhã é o de hoje\"):\n")
    p.append(
        tabela_md(
            ["", "MAE (USD)", "RMSE (USD)", "erro % médio"],
            [
                [nome_regressor, f"{resumo_reg['mae']:.2f}", f"{resumo_reg['rmse']:.2f}",
                 f"{resumo_reg['mape']:.2%}"],
                ["Baseline: persistência", f"{resumo_reg['mae_persistencia']:.2f}",
                 f"{resumo_reg['rmse_persistencia']:.2f}",
                 f"{resumo_reg['mape_persistencia']:.2%}"],
            ],
        )
    )
    melhor_que_persist = resumo_reg["mae"] < resumo_reg["mae_persistencia"]
    p.append(
        f"\n> Direção acertada pelo sinal do retorno estimado: "
        f"**{resumo_reg['acertos_direcao']}/{resumo_reg['n']}**. "
        + (
            "O regressor errou **menos** que a persistência nesta janela — "
            "diferença pequena, sobre poucos dias, que pode não se repetir.\n"
            if melhor_que_persist
            else "O regressor **não** superou a persistência nesta janela: repetir "
            "o preço de hoje teria errado menos. É o resultado esperado para "
            "preço no dia seguinte.\n"
        )
    )

    # --- 3. seleção
    p.append("## 3. Como os modelos foram escolhidos\n")
    p.append(
        f"`TimeSeriesSplit` com {N_SPLITS_SELECAO} partições sobre os "
        f"{n_treino_selecao} dias **anteriores** à janela exibida — nenhum dia "
        "previsto acima participou da escolha.\n"
    )
    p.append("**Classificador** (critério: maior acurácia balanceada média):\n")
    p.append(
        tabela_md(
            ["modelo", "acurácia balanceada média", "desvio"],
            [[r["modelo"], f"{r['acuracia_balanceada_media']:.3f}",
              f"{r['acuracia_balanceada_desvio']:.3f}"]
             for _, r in selecao_clf.iterrows()],
        )
    )
    p.append("\n**Regressor** (critério: menor RMSE médio do retorno diário):\n")
    p.append(
        tabela_md(
            ["modelo", "RMSE médio do retorno", "desvio"],
            [[r["modelo"], f"{r['rmse_media']:.4f}", f"{r['rmse_desvio']:.4f}"]
             for _, r in selecao_reg.iterrows()],
        )
    )
    p.append(
        "\n> Acurácia balanceada de 0,5 é o acaso. Se o baseline *retorno zero* "
        "aparece no topo da tabela de regressores, nenhum deles aprendeu algo além "
        "de \"amanhã ≈ hoje\" — o escolhido é apenas o menos ruim entre os modelos.\n"
    )

    p.append("## Ressalvas\n")
    p.append(
        "- Os preços são instantâneos de 00:00 UTC; \"amanhã\" é o próximo "
        "instantâneo, não o fechamento de um pregão.\n"
        "- A avaliação formal, em teste retido de 30% da série, mostra desempenho "
        "**próximo do acaso** para direção; para o preço, a persistência é uma "
        "referência que modelos simples raramente superam no horizonte de um dia.\n"
        "- Cada linha usa um modelo reajustado só com o passado, então o histórico "
        "não é uma medida otimista de dentro da amostra — mas a janela é curta e "
        "não sustenta conclusões sobre o futuro.\n"
    )

    caminho.write_text("\n".join(p), encoding="utf-8")


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)

    rotulados, a_prever, precos_por_data = carregar_para_previsao()
    if a_prever.empty:
        raise SystemExit(
            "Não há dia pendente para prever — a coleta não inclui um dia "
            "posterior ao último rótulo. Rode `python src/collect.py` para atualizar."
        )

    # --- escolha dos modelos: só com os dias anteriores à janela exibida
    inicio_janela = max(len(rotulados) - DIAS_HISTORICO, 1)
    anteriores = rotulados.iloc[:inicio_janela]
    nome_clf, selecao_clf = selecionar_classificador(anteriores)
    nome_reg, selecao_reg = selecionar_regressor(anteriores)
    if CLASSIFICADOR:
        nome_clf = CLASSIFICADOR
    if REGRESSOR:
        nome_reg = REGRESSOR
    print(f"  modelos escolhidos por walk-forward em {inicio_janela} dias anteriores "
          f"à janela: {nome_clf} (direção), {nome_reg} (preço)")

    # --- direção
    historico = classificar_historico(rotulados, precos_por_data, nome_clf)
    imprimir_historico(historico)
    previsao = classificar_dia_seguinte(rotulados, a_prever, nome_clf)

    # --- preço
    historico_reg = estimar_historico(rotulados, precos_por_data, nome_reg)
    resumo_reg = resumir_regressao(historico_reg)
    imprimir_historico_regressao(historico_reg, resumo_reg)
    previsao_reg = estimar_dia_seguinte(rotulados, a_prever, nome_reg)

    imprimir_dia_seguinte(previsao, previsao_reg, nome_clf, nome_reg)

    # --- saídas
    todas = pd.concat([historico, pd.DataFrame([previsao])], ignore_index=True)
    # a linha do dia seguinte não tem resultado: tipos nuláveis evitam que
    # `observado` vire float e `acertou` vire object na concatenação
    todas["observado"] = todas["observado"].astype("Int64")
    todas["acertou"] = todas["acertou"].astype("boolean")
    todas.to_parquet(SAIDA, index=False)   # consumido pelo dashboard

    todas_reg = pd.concat([historico_reg, pd.DataFrame([previsao_reg])], ignore_index=True)
    todas_reg["direcao_ok"] = todas_reg["direcao_ok"].astype("boolean")
    todas_reg.to_parquet(SAIDA_REGRESSAO, index=False)

    escrever_relatorio(
        SAIDA_RELATORIO,
        historico=historico,
        previsao=previsao,
        historico_reg=historico_reg,
        previsao_reg=previsao_reg,
        resumo_reg=resumo_reg,
        nome_classificador=nome_clf,
        nome_regressor=nome_reg,
        selecao_clf=selecao_clf,
        selecao_reg=selecao_reg,
        n_treino_selecao=inicio_janela,
    )
    print(f"\n  Relatório: {SAIDA_RELATORIO}\n")


if __name__ == "__main__":
    main()
