"""regression.py — Regressão da volatilidade futura (Encontro 12).

Extensão do projeto. A escolha do alvo **não é arbitrária**: a EDA
(`notebooks/eda_solana.ipynb`) mostrou que a série carrega informação sobre a
*magnitude* dos movimentos e quase nenhuma sobre a *direção* —

  * autocorrelação dos retornos ≈ ruído branco, mas de |retornos| significativa
    nas defasagens 1, 2, 3 e 5 (agrupamento de volatilidade);
  * correlação do volume com a direção = +0,003, com a magnitude = +0,600.

Alvo: **volatilidade realizada dos próximos H dias** (desvio-padrão dos retornos
diários dessa janela futura). O horizonte H **também é selecionado**, junto com o
modelo, usando apenas o treino — ele é uma decisão de modelagem, não um dado do
problema.

Entrada:  ``data/processed/precos_limpo.parquet``
Saída:
  reports/relatorio_volatilidade.md

Mesmo rigor dos Encontros 10-11: atributos só com passado, split cronológico,
padronização dentro do Pipeline, baselines medidos antes e seleção que nunca
enxerga o teste.

Trabalho acadêmico: nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import features
from classification import dividir_temporal
from preprocess import _tabela_md as tabela_md

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "relatorio_volatilidade.md"

SEMENTE = 42
# O horizonte do alvo é uma escolha de modelagem, não um dado do problema: uma
# janela curta estima a volatilidade com muito erro amostral (o desvio-padrão de
# 7 retornos tem ~29% de erro relativo), uma janela longa alisa o sinal. Por isso
# o horizonte é **selecionado junto com o modelo**, e só com o treino.
HORIZONTES = (7, 14, 21, 30)
N_SPLITS = 5             # partições do walk-forward dentro do treino
K_VIZINHOS = 15
PROFUNDIDADE_ARVORE = 3

np.random.seed(SEMENTE)


def construir_modelos() -> dict[str, Pipeline]:
    """Os regressores comparados, todos com padronização no Pipeline."""
    return {
        "Ridge (linear)": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", Ridge(alpha=1.0, random_state=SEMENTE)),
        ]),
        "KNN": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", KNeighborsRegressor(n_neighbors=K_VIZINHOS)),
        ]),
        "Árvore de decisão": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", DecisionTreeRegressor(
                max_depth=PROFUNDIDADE_ARVORE, random_state=SEMENTE)),
        ]),
        "SVR (RBF)": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", SVR(kernel="rbf")),
        ]),
    }


# ----------------------------------------------------------------------
# 1. Alvo: volatilidade realizada futura
# ----------------------------------------------------------------------
def volatilidade_futura(ret_1d: pd.Series, horizonte: int) -> pd.Series:
    """Desvio-padrão dos retornos dos próximos ``horizonte`` dias.

    ``rolling(h).std()`` na posição ``t+h`` é o desvio de ``ret[t+1 … t+h]``;
    ``shift(-h)`` traz esse valor para a linha ``t``. Assim a linha ``t`` carrega
    a volatilidade que **ainda vai acontecer** — é o rótulo, e só ele olha adiante.

    As últimas ``horizonte`` linhas ficam ``NaN`` (janela futura incompleta) e são
    descartadas.
    """
    return ret_1d.rolling(horizonte).std().shift(-horizonte)


def montar_tabela(horizonte: int) -> pd.DataFrame:
    """Monta X (os mesmos 13 atributos do projeto) + y (volatilidade futura)."""
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {ENTRADA}\nRode antes `python src/preprocess.py`."
        )
    precos = pd.read_parquet(ENTRADA)

    partes = []
    for _, grupo in precos.groupby("id", sort=True):
        com_atributos = features.construir_atributos(grupo)
        com_atributos["y"] = volatilidade_futura(com_atributos["ret_1d"], horizonte)
        partes.append(com_atributos)

    tabela = pd.concat(partes, ignore_index=True)
    colunas = ["id", "data", "preco", *features.colunas_atributos(), "y"]
    tabela = tabela[colunas].dropna(subset=[*features.colunas_atributos(), "y"])
    return tabela.sort_values(["id", "data"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Métricas
# ----------------------------------------------------------------------
def avaliar(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE e R² (R² compara implicitamente com prever a média do conjunto)."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def ganho_sobre(referencia: dict[str, float], modelo: dict[str, float]) -> float:
    """Redução proporcional do erro quadrático em relação a um baseline.

    Positivo = o modelo erra menos que a referência; negativo = erra mais.
    """
    return 1.0 - (modelo["rmse"] ** 2) / (referencia["rmse"] ** 2)


# ----------------------------------------------------------------------
# 3. Baselines (medidos ANTES dos modelos)
# ----------------------------------------------------------------------
def calcular_baselines(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Referências ingênuas para a volatilidade futura.

    * **persistência**: repetir a volatilidade recente observada até hoje. É a
      referência séria — em volatilidade, persistir é notoriamente difícil de
      superar, justamente por causa do agrupamento que a EDA mostrou.
    * **média do treino**: prever sempre a volatilidade média do período de treino.
    """
    y_teste = teste["y"].to_numpy()
    return {
        "Baseline: persistência (vol_30d)": avaliar(y_teste, teste["vol_30d"].to_numpy()),
        "Baseline: persistência (vol_7d)": avaliar(y_teste, teste["vol_7d"].to_numpy()),
        "Baseline: média do treino": avaliar(
            y_teste, np.full_like(y_teste, treino["y"].mean())
        ),
    }


# ----------------------------------------------------------------------
# 4. Seleção conjunta de horizonte e modelo — SÓ COM O TREINO
# ----------------------------------------------------------------------
def selecionar(horizontes=HORIZONTES, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Avalia cada par (horizonte, modelo) por walk-forward dentro do treino.

    O critério é o **R² médio** entre partições, e não o RMSE: RMSE de horizontes
    diferentes não é comparável (as escalas do alvo diferem), enquanto o R² é
    adimensional.

    Nada aqui toca o conjunto de teste.
    """
    colunas = features.colunas_atributos()
    linhas = []

    for horizonte in horizontes:
        treino, _ = dividir_temporal(montar_tabela(horizonte))
        X = treino[colunas].to_numpy()
        y = treino["y"].to_numpy()
        particoes = list(TimeSeriesSplit(n_splits=n_splits).split(X))

        for nome in construir_modelos():
            r2s, rmses = [], []
            for idx_treino, idx_val in particoes:
                modelo = construir_modelos()[nome]   # Pipeline novo a cada partição
                modelo.fit(X[idx_treino], y[idx_treino])
                previsto = modelo.predict(X[idx_val])
                r2s.append(r2_score(y[idx_val], previsto))
                rmses.append(np.sqrt(mean_squared_error(y[idx_val], previsto)))
            linhas.append({
                "horizonte": horizonte,
                "modelo": nome,
                "r2_media": float(np.mean(r2s)),
                "r2_desvio": float(np.std(r2s)),
                "rmse_media": float(np.mean(rmses)),
            })

    return pd.DataFrame(linhas).sort_values("r2_media", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 5. Avaliação final no teste retido
# ----------------------------------------------------------------------
def avaliar_no_teste(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Ajusta cada modelo no treino inteiro e avalia uma única vez no teste."""
    colunas = features.colunas_atributos()
    X_treino, y_treino = treino[colunas].to_numpy(), treino["y"].to_numpy()
    X_teste, y_teste = teste[colunas].to_numpy(), teste["y"].to_numpy()

    linhas, previsoes = [], {}
    for nome, modelo in construir_modelos().items():
        modelo.fit(X_treino, y_treino)
        previsto = modelo.predict(X_teste)
        previsoes[nome] = previsto
        linhas.append({"modelo": nome, **avaliar(y_teste, previsto)})

    return pd.DataFrame(linhas), previsoes


def r2_no_teste_por_horizonte(horizontes=HORIZONTES) -> pd.DataFrame:
    """R² no teste de cada horizonte — **apenas para a demonstração metodológica**.

    Esta tabela mostra o que teria acontecido se o horizonte fosse escolhido
    olhando o teste. Ela **não** participa de nenhuma decisão do pipeline; existe
    para documentar a armadilha evitada (seção 5 do relatório).
    """
    colunas = features.colunas_atributos()
    linhas = []
    for horizonte in horizontes:
        treino, teste = dividir_temporal(montar_tabela(horizonte))
        melhor_r2, melhor_nome = -np.inf, ""
        for nome, modelo in construir_modelos().items():
            modelo.fit(treino[colunas], treino["y"])
            r2 = r2_score(teste["y"], modelo.predict(teste[colunas]))
            if r2 > melhor_r2:
                melhor_r2, melhor_nome = r2, nome
        linhas.append({
            "horizonte": horizonte,
            "melhor_no_teste": melhor_nome,
            "r2_teste": float(melhor_r2),
        })
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)

    print("1. Selecionando horizonte e modelo por walk-forward no treino")
    print(f"   ({len(HORIZONTES)} horizontes x {len(construir_modelos())} modelos, "
          f"{N_SPLITS} partições)...")
    selecao = selecionar()
    for _, r in selecao.head(5).iterrows():
        print(f"   h={int(r['horizonte']):2d}  {r['modelo']:20s} "
              f"R²={r['r2_media']:+.3f} ± {r['r2_desvio']:.3f}")
    print(f"   ... ({len(selecao)} configurações no total)")

    horizonte = int(selecao.iloc[0]["horizonte"])
    melhor = selecao.iloc[0]["modelo"]
    print(f"\n   -> selecionado (sem olhar o teste): {melhor}, horizonte {horizonte}d")

    tabela = montar_tabela(horizonte)
    treino, teste = dividir_temporal(tabela)
    correlacao = float(tabela["vol_30d"].corr(tabela["y"]))
    print(f"\n2. Tabela: {len(tabela)} linhas ({len(treino)} treino / {len(teste)} teste)")
    print(f"   corr(vol_30d hoje, volatilidade futura) = {correlacao:+.3f}")
    print(f"   volatilidade média: treino {treino['y'].mean():.2%}  "
          f"teste {teste['y'].mean():.2%} "
          f"({teste['y'].mean() / treino['y'].mean() - 1:+.1%})")

    print("\n3. Baselines (antes dos modelos):")
    baselines = calcular_baselines(treino, teste)
    for nome, m in baselines.items():
        print(f"   {nome:36s} RMSE={m['rmse']:.4f}  R²={m['r2']:+.3f}")

    print("\n4. Avaliação final no teste retido:")
    resultados, _ = avaliar_no_teste(treino, teste)
    for _, r in resultados.sort_values("rmse").iterrows():
        print(f"   {r['modelo']:20s} RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  "
              f"R²={r['r2']:+.3f}")

    metricas = dict(resultados.set_index("modelo").loc[melhor])
    ganhos = {
        f"{melhor} vs {nome}": ganho_sobre(m, metricas)
        for nome, m in baselines.items()
    }
    print("\n   ganho sobre baselines (redução do erro quadrático):")
    for nome, valor in ganhos.items():
        print(f"      {nome:52s} {valor:+.1%}")

    print("\n5. Armadilha (só para o relatório — não decide nada):")
    armadilha = r2_no_teste_por_horizonte()
    for _, r in armadilha.iterrows():
        print(f"   h={int(r['horizonte']):2d}: melhor no teste seria "
              f"{r['melhor_no_teste']:20s} R²={r['r2_teste']:+.3f}")

    escrever_relatorio(
        SAIDA_RELATORIO,
        horizonte=horizonte,
        melhor=melhor,
        tabela=tabela,
        treino=treino,
        teste=teste,
        baselines=baselines,
        selecao=selecao,
        resultados=resultados,
        ganhos=ganhos,
        correlacao=correlacao,
        armadilha=armadilha,
    )

    print(f"\nRelatório: {SAIDA_RELATORIO}")


if __name__ == "__main__":
    main()
