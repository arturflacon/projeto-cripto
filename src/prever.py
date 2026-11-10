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

Uso:
  python src/prever.py              -> janela padrão (DIAS_HISTORICO = 45 dias)
  python src/prever.py --dias 90    -> outra janela, quando precisar

Saídas:
  data/processed/previsoes.parquet            -> direção (consumido pelo dashboard)
  data/processed/previsoes_regressao.parquet  -> preço estimado
  reports/previsoes.md                        -> o relatório: previsões × preço real, dia a dia
  reports/previsoes.png                       -> gráfico do relatório

**Trabalho acadêmico.** As saídas são estimativas de modelos estatísticos a partir
de padrões passados, condicionadas ao período analisado. Não são previsão
confiável do mercado nem recomendação de investimento.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binom

matplotlib.use("Agg")                    # gera o PNG sem abrir janela
import matplotlib.dates as mdates        # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
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
SAIDA_GRAFICO = RAIZ / "reports" / "previsoes.png"

# Modelos fixados à mão (chaves de `classification.construir_modelos()` e
# `regression.construir_modelos()`). ``None`` = escolher por walk-forward nos
# dias anteriores à janela exibida, que é o padrão.
CLASSIFICADOR = None
REGRESSOR = None

# Quantos dias recentes re-prever para exibir o comportamento dos modelos
# (padrão; pode ser trocado na linha de comando com `--dias N`).
DIAS_HISTORICO = 45

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
# Cores do gráfico (paleta de referência validada para daltonismo: azul x laranja).
# Acerto/erro usa cores de status, mas a informação vai na FORMA do marcador
# (círculo = acertou, X = errou) — verde e vermelho sozinhos não bastam para
# quem tem deuteranopia.
COR_REAL = "#2a78d6"
COR_ESTIMADO = "#eb6834"
COR_ACERTO = "#0ca30c"
COR_ERRO = "#d03b3b"
COR_SUPERFICIE = "#fcfcfb"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SEC = "#52514e"
COR_GRADE = "#e6e5e0"


def consolidar(
    historico: pd.DataFrame,
    previsao: dict,
    historico_reg: pd.DataFrame,
    previsao_reg: dict,
) -> pd.DataFrame:
    """Junta direção e preço numa tabela única: uma linha por dia-alvo.

    A última linha é o dia seguinte, ainda sem preço real (``NaN``).
    """
    direcao = pd.concat([historico, pd.DataFrame([previsao])], ignore_index=True)
    preco = pd.concat([historico_reg, pd.DataFrame([previsao_reg])], ignore_index=True)

    tabela = direcao[["data_alvo", "preco_base", "preco_alvo", "classificado",
                      "observado", "acertou"]].merge(
        preco[["data_alvo", "preco_estimado", "ret_estimado", "erro", "erro_pct"]],
        on="data_alvo", how="inner",
    )
    for coluna in ("preco_base", "preco_alvo", "preco_estimado", "ret_estimado",
                   "erro", "erro_pct"):
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela["variacao_real"] = tabela["preco_alvo"] / tabela["preco_base"] - 1.0
    return tabela.sort_values("data_alvo").reset_index(drop=True)


def desenhar_grafico(tabela: pd.DataFrame, caminho: Path, nome_regressor: str,
                     nome_classificador: str) -> None:
    """Gráfico do relatório: preço real × estimado (em cima) e acertos de direção (embaixo).

    Dois painéis com o mesmo eixo de datas — nunca dois eixos y no mesmo painel.
    """
    conhecidos = tabela[tabela["preco_alvo"].notna()]
    pendente = tabela[tabela["preco_alvo"].isna()]

    fig, (ax, ax_dir) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor(COR_SUPERFICIE)
    for eixo in (ax, ax_dir):
        eixo.set_facecolor(COR_SUPERFICIE)
        for lado in ("top", "right", "left"):
            eixo.spines[lado].set_visible(False)
        eixo.spines["bottom"].set_color(COR_GRADE)
        eixo.tick_params(colors=COR_TEXTO_SEC, labelsize=9, length=0)

    # --- painel de preço
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(conhecidos["data_alvo"], conhecidos["preco_alvo"], color=COR_REAL,
            linewidth=2, marker="o", markersize=4, label="preço real", zorder=3)
    ax.plot(conhecidos["data_alvo"], conhecidos["preco_estimado"], color=COR_ESTIMADO,
            linewidth=2, linestyle="--", marker="o", markersize=3.5,
            label=f"preço estimado ({nome_regressor})", zorder=2)

    # rótulos diretos só nas pontas (o último real e a estimativa pendente)
    ultimo = conhecidos.iloc[-1]
    ax.annotate(f"real {ultimo['preco_alvo']:.2f}",
                (ultimo["data_alvo"], ultimo["preco_alvo"]),
                xytext=(6, 8), textcoords="offset points",
                fontsize=9, color=COR_TEXTO)
    if not pendente.empty:
        amanha = pendente.iloc[-1]
        ax.plot([ultimo["data_alvo"], amanha["data_alvo"]],
                [ultimo["preco_alvo"], amanha["preco_estimado"]],
                color=COR_ESTIMADO, linewidth=1.5, linestyle=":", zorder=2)
        ax.plot(amanha["data_alvo"], amanha["preco_estimado"], marker="o",
                markersize=9, markerfacecolor=COR_SUPERFICIE,
                markeredgecolor=COR_ESTIMADO, markeredgewidth=2, zorder=4)
        ax.annotate(f"{amanha['data_alvo']:%d/%m} estimado\n{amanha['preco_estimado']:.2f}",
                    (amanha["data_alvo"], amanha["preco_estimado"]),
                    xytext=(8, -26), textcoords="offset points",
                    fontsize=9, color=COR_TEXTO)

    ax.set_ylabel("USD", color=COR_TEXTO_SEC, fontsize=9)
    legenda = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for texto in legenda.get_texts():
        texto.set_color(COR_TEXTO)

    # --- painel de direção: forma do marcador carrega o acerto
    acertos = conhecidos[conhecidos["acertou"].astype(bool)]
    erros = conhecidos[~conhecidos["acertou"].astype(bool)]
    ax_dir.scatter(acertos["data_alvo"], np.zeros(len(acertos)), marker="o", s=40,
                   color=COR_ACERTO, label="direção: acertou", zorder=3)
    ax_dir.scatter(erros["data_alvo"], np.zeros(len(erros)), marker="x", s=46,
                   linewidths=2, color=COR_ERRO, label="direção: errou", zorder=3)
    ax_dir.set_yticks([])
    ax_dir.set_ylim(-1, 1)
    ax_dir.set_ylabel(nome_classificador, color=COR_TEXTO_SEC, fontsize=8,
                      rotation=0, ha="right", va="center")
    legenda_dir = ax_dir.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55),
                                ncol=2, frameon=False, fontsize=9)
    for texto in legenda_dir.get_texts():
        texto.set_color(COR_TEXTO)

    # folga à direita para o rótulo da estimativa pendente não ser cortado
    ax_dir.set_xlim(tabela["data_alvo"].min() - pd.Timedelta(days=1),
                    tabela["data_alvo"].max() + pd.Timedelta(days=4))
    ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax_dir.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))

    fig.suptitle(
        f"Solana — preço real × estimado pelo modelo, "
        f"{conhecidos['data_alvo'].min():%d/%m/%Y} a {conhecidos['data_alvo'].max():%d/%m/%Y}",
        x=0.06, ha="left", fontsize=12, color=COR_TEXTO,
    )
    fig.text(0.06, 0.915, "Trabalho acadêmico — estimativas de modelo, não recomendação "
             "de investimento.", fontsize=8.5, color=COR_TEXTO_SEC)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.14)

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=130, facecolor=COR_SUPERFICIE)
    plt.close(fig)


def escrever_relatorio(
    caminho: Path,
    *,
    tabela: pd.DataFrame,
    resumo_reg: dict,
    nome_classificador: str,
    nome_regressor: str,
    selecao_clf: pd.DataFrame,
    selecao_reg: pd.DataFrame,
    n_treino_selecao: int,
    grafico: Path,
) -> None:
    """Escreve o relatório de previsões em Markdown (tabela única + gráfico)."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conhecidos = tabela[tabela["preco_alvo"].notna()]
    pendente = tabela[tabela["preco_alvo"].isna()]
    n = len(conhecidos)
    acertos = int(conhecidos["acertou"].astype(bool).sum())
    # faixa central de 95% dos acertos de uma moeda honesta (binomial n, 0,5)
    acaso_min, acaso_max = (int(v) for v in binom.interval(0.95, n, 0.5))
    dentro_do_acaso = acaso_min <= acertos <= acaso_max
    melhor_que_persist = resumo_reg["mae"] < resumo_reg["mae_persistencia"]
    p: list[str] = []

    p.append("# Previsões × preço real — Solana\n")
    p.append(
        "**Trabalho acadêmico.** As previsões abaixo são estimativas de modelos "
        "estatísticos a partir de padrões passados, condicionadas ao período "
        "analisado. Não são previsão confiável do mercado nem recomendação de "
        "investimento.\n"
    )
    p.append(f"- Gerado em: `{agora}`")
    p.append(
        f"- Período comparado: **{conhecidos['data_alvo'].min():%d/%m/%Y} a "
        f"{conhecidos['data_alvo'].max():%d/%m/%Y}** ({n} dias com preço real conhecido)"
    )
    p.append(f"- Modelos: direção `{nome_classificador}` · preço `{nome_regressor}`")
    p.append("- Preços: instantâneos diários de 00:00 UTC (CoinGecko), em USD\n")

    # --- resumo
    p.append("## Resumo\n")
    p.append(tabela_md(
        ["", "resultado"],
        [
            ["Direção (sobe / não sobe)",
             f"**{acertos} de {n}** dias classificados corretamente ({acertos / n:.0%})"],
            ["Por acaso (moeda honesta)",
             f"entre {acaso_min} e {acaso_max} acertos em 95% das vezes"],
            ["Erro médio do preço estimado",
             f"{resumo_reg['mae']:.2f} USD ({resumo_reg['mape']:.1%})"],
            ["Erro de \"amanhã = hoje\" (persistência)",
             f"{resumo_reg['mae_persistencia']:.2f} USD "
             f"({resumo_reg['mape_persistencia']:.1%})"],
        ],
    ))
    p.append(
        "\n> "
        + ("A taxa de acerto da direção está **dentro** da faixa do acaso: não se "
           "distingue de cara-ou-coroa. " if dentro_do_acaso else
           "A taxa de acerto da direção ficou **fora** da faixa do acaso nesta janela, "
           "o que não garante que se repita em outro período. ")
        + ("O preço estimado errou **menos** que repetir o preço de hoje, por pouca "
           "diferença.\n" if melhor_que_persist else
           "O preço estimado **não** errou menos que simplesmente repetir o preço de "
           "hoje — resultado esperado no horizonte de um dia.\n")
    )

    if not pendente.empty:
        amanha = pendente.iloc[-1]
        p.append(
            f"**Próximo dia ({amanha['data_alvo']:%d/%m/%Y}, ainda não aconteceu):** "
            f"direção classificada como **{ROTULOS[int(amanha['classificado'])]}**; "
            f"preço estimado **{amanha['preco_estimado']:.2f} USD** "
            f"({amanha['ret_estimado']:+.2%} sobre {amanha['preco_base']:.2f} USD).\n"
        )

    # --- gráfico
    p.append("## Gráfico\n")
    p.append(f"![Preço real × estimado]({grafico.name})\n")

    # --- tabela dia a dia
    p.append("## Dia a dia\n")
    p.append(
        "Cada linha foi prevista **antes** de o dia acontecer: o modelo foi treinado só "
        "com os dias anteriores. *Véspera* é o preço do dia anterior (o ponto de "
        "partida); *erro* = estimado − real.\n"
    )
    linhas = []
    for _, r in tabela.iterrows():
        if pd.isna(r["preco_alvo"]):
            linhas.append([
                f"**{r['data_alvo']:%d/%m/%Y}**", f"{r['preco_base']:.2f}",
                f"**{r['preco_estimado']:.2f}**", "aguardando", "—", "—",
                f"**{ROTULOS[int(r['classificado'])]}**", "aguardando", "",
            ])
            continue
        linhas.append([
            f"{r['data_alvo']:%d/%m/%Y}", f"{r['preco_base']:.2f}",
            f"{r['preco_estimado']:.2f}", f"{r['preco_alvo']:.2f}",
            f"{r['erro']:+.2f} ({r['erro_pct']:+.1%})", f"{r['variacao_real']:+.1%}",
            ROTULOS[int(r["classificado"])], ROTULOS[int(r["observado"])],
            "✓" if bool(r["acertou"]) else "✗",
        ])
    p.append(tabela_md(
        ["dia", "véspera (USD)", "estimado (USD)", "real (USD)", "erro",
         "variação real", "direção classificada", "direção real", "acertou?"],
        linhas,
    ))
    p.append("")

    # --- como os modelos foram escolhidos
    p.append("## Como os modelos foram escolhidos\n")
    p.append(
        f"`TimeSeriesSplit` com {N_SPLITS_SELECAO} partições sobre os "
        f"{n_treino_selecao} dias **anteriores** ao período comparado — nenhum dia "
        "da tabela acima participou da escolha.\n"
    )
    p.append("**Direção** (critério: maior acurácia balanceada média; 0,5 = acaso):\n")
    p.append(tabela_md(
        ["modelo", "acurácia balanceada média", "desvio"],
        [[r["modelo"], f"{r['acuracia_balanceada_media']:.3f}",
          f"{r['acuracia_balanceada_desvio']:.3f}"]
         for _, r in selecao_clf.iterrows()],
    ))
    p.append("\n**Preço** (critério: menor RMSE médio do retorno diário):\n")
    p.append(tabela_md(
        ["modelo", "RMSE médio do retorno", "desvio"],
        [[r["modelo"], f"{r['rmse_media']:.4f}", f"{r['rmse_desvio']:.4f}"]
         for _, r in selecao_reg.iterrows()],
    ))
    p.append("")

    p.append("## Ressalvas\n")
    p.append(
        "- \"Amanhã\" é o próximo instantâneo de 00:00 UTC, não o fechamento de um pregão.\n"
        "- O preço estimado parte do preço da véspera, por isso acompanha de perto a "
        "curva real: o que importa é o erro comparado ao de repetir o preço de hoje.\n"
        "- Cada linha usa um modelo reajustado só com o passado, então a tabela não é "
        "uma medida otimista de dentro da amostra — mas o período é curto e não "
        "sustenta conclusões sobre o futuro.\n"
    )

    caminho.write_text("\n".join(p), encoding="utf-8")


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def ler_argumentos() -> argparse.Namespace:
    """Lê a linha de comando (tamanho da janela exibida)."""
    parser = argparse.ArgumentParser(
        description="Previsões do dia seguinte da Solana com histórico dia a dia."
    )
    parser.add_argument(
        "--dias", type=int, default=DIAS_HISTORICO,
        help=f"quantos dias recentes re-prever no relatório (padrão: {DIAS_HISTORICO})",
    )
    args = parser.parse_args()
    if args.dias < 1:
        parser.error("--dias deve ser pelo menos 1")
    return args


def main() -> None:
    n_dias = ler_argumentos().dias
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)

    rotulados, a_prever, precos_por_data = carregar_para_previsao()
    if a_prever.empty:
        raise SystemExit(
            "Não há dia pendente para prever — a coleta não inclui um dia "
            "posterior ao último rótulo. Rode `python src/collect.py` para atualizar."
        )

    # --- escolha dos modelos: só com os dias anteriores à janela exibida
    # a seleção precisa de dias anteriores suficientes para N_SPLITS_SELECAO partições
    n_dias = min(n_dias, len(rotulados) - 2 * (N_SPLITS_SELECAO + 1))
    if n_dias < 1:
        raise SystemExit("Histórico curto demais para separar seleção e janela exibida.")
    inicio_janela = len(rotulados) - n_dias
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
    historico = classificar_historico(rotulados, precos_por_data, nome_clf, n_dias)
    imprimir_historico(historico)
    previsao = classificar_dia_seguinte(rotulados, a_prever, nome_clf)

    # --- preço
    historico_reg = estimar_historico(rotulados, precos_por_data, nome_reg, n_dias)
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

    tabela = consolidar(historico, previsao, historico_reg, previsao_reg)
    desenhar_grafico(tabela, SAIDA_GRAFICO, nome_reg, nome_clf)
    escrever_relatorio(
        SAIDA_RELATORIO,
        tabela=tabela,
        resumo_reg=resumo_reg,
        nome_classificador=nome_clf,
        nome_regressor=nome_reg,
        selecao_clf=selecao_clf,
        selecao_reg=selecao_reg,
        n_treino_selecao=inicio_janela,
        grafico=SAIDA_GRAFICO,
    )
    print(f"\n  Relatório: {SAIDA_RELATORIO}")
    print(f"  Gráfico:   {SAIDA_GRAFICO}")
    print("  (no VS Code, abra o .md e tecle Ctrl+Shift+V para ver formatado)\n")


if __name__ == "__main__":
    main()
