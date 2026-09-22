"""classification.py — Classificação da direção do dia seguinte (Encontros 10-11).

Lê ``data/processed/atributos.parquet`` e compara classificadores na tarefa
"a Solana fecha amanhã acima de hoje?", sempre com **split temporal**.

Protocolo (nesta ordem, para não contaminar a avaliação):
  1. divisão cronológica 70% treino / 30% teste — o teste é o trecho **final**;
  2. baselines medidos ANTES dos modelos (sempre alta, classe majoritária,
     persistência) — um modelo só tem valor se superá-los;
  3. comparação entre modelos por ``TimeSeriesSplit`` (walk-forward) rodado
     **somente dentro do treino** — a seleção do melhor modelo nunca olha o teste;
  4. o modelo escolhido é reajustado no treino inteiro e avaliado **uma vez** no teste;
  5. matriz de confusão e análise de erros do melhor modelo.

Padronização (``StandardScaler``) vive dentro de um ``Pipeline``, então é ajustada
só com o treino de cada partição — nunca com o teste.

Sem arquivo de saída: baselines, comparação, matriz de confusão e análise de
erros vão para o terminal. O classificador vencedor é o usado por ``prever.py``.

Trabalho acadêmico: nada aqui é recomendação de investimento. As previsões são
estimativas condicionadas ao período analisado, não afirmações sobre o futuro.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import features

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "atributos.parquet"
ENTRADA_PRECOS = RAIZ / "data" / "processed" / "precos_limpo.parquet"

SEMENTE = 42
FRACAO_TREINO = 0.70    # os 30% finais (cronológicos) ficam para teste
N_SPLITS = 5            # partições do walk-forward dentro do treino
K_VIZINHOS = 15         # KNN: k alto suaviza o ruído de uma série curta
PROFUNDIDADE_ARVORE = 3  # árvore rasa: ~190 dias de treino overfitam fácil

np.random.seed(SEMENTE)


def construir_modelos() -> dict[str, Pipeline]:
    """Os classificadores comparados, todos com padronização no Pipeline."""
    return {
        "KNN": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", KNeighborsClassifier(n_neighbors=K_VIZINHOS)),
        ]),
        "SVM (RBF)": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", SVC(kernel="rbf", random_state=SEMENTE)),
        ]),
        "Árvore de decisão": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", DecisionTreeClassifier(
                max_depth=PROFUNDIDADE_ARVORE, random_state=SEMENTE)),
        ]),
        "Naive Bayes": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", GaussianNB()),
        ]),
        "Perceptron": Pipeline([
            ("escala", StandardScaler()),
            ("modelo", Perceptron(random_state=SEMENTE)),
        ]),
    }


# ----------------------------------------------------------------------
# 1. Dados e split temporal
# ----------------------------------------------------------------------
def carregar_tabela() -> pd.DataFrame:
    """Lê a tabela supervisionada já ordenada no tempo."""
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {ENTRADA}\nRode antes `python src/features.py`."
        )
    return pd.read_parquet(ENTRADA).sort_values("data").reset_index(drop=True)


def dividir_temporal(
    tabela: pd.DataFrame, fracao: float = FRACAO_TREINO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em treino (início) e teste (final) — **nunca** de forma aleatória."""
    corte = int(len(tabela) * fracao)
    return tabela.iloc[:corte].copy(), tabela.iloc[corte:].copy()


# ----------------------------------------------------------------------
# 2. Métricas
# ----------------------------------------------------------------------
def avaliar(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas de classificação binária (acurácia balanceada em primeiro lugar)."""
    return {
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "acuracia": float(np.mean(y_true == y_pred)),
    }


# ----------------------------------------------------------------------
# 3. Baselines (medidos ANTES dos modelos)
# ----------------------------------------------------------------------
def calcular_baselines(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Baselines ingênuos avaliados no mesmo conjunto de teste dos modelos.

    * **sempre alta**: prevê 1 todos os dias;
    * **classe majoritária (treino)**: prevê sempre a classe mais frequente no treino;
    * **persistência**: repete a direção observada de ontem para hoje
      (``ret_1d > 0``), informação disponível em `t`.
    """
    y_teste = teste["y"].to_numpy()

    sempre_alta = np.ones_like(y_teste)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(treino[features.colunas_atributos()], treino["y"])
    majoritaria = dummy.predict(teste[features.colunas_atributos()])

    persistencia = (teste["ret_1d"] > 0).astype(int).to_numpy()

    return {
        "Baseline: sempre alta": avaliar(y_teste, sempre_alta),
        "Baseline: classe majoritária (treino)": avaliar(y_teste, majoritaria),
        "Baseline: persistência": avaliar(y_teste, persistencia),
    }


# ----------------------------------------------------------------------
# 4. Comparação walk-forward (só dentro do treino)
# ----------------------------------------------------------------------
def comparar_walk_forward(
    treino: pd.DataFrame, n_splits: int = N_SPLITS
) -> pd.DataFrame:
    """Roda ``TimeSeriesSplit`` no treino e devolve média e desvio por modelo."""
    colunas = features.colunas_atributos()
    X = treino[colunas].to_numpy()
    y = treino["y"].to_numpy()
    particoes = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    linhas = []
    for nome in construir_modelos():
        acuracias, f1s = [], []
        for idx_treino, idx_val in particoes:
            # Pipeline novo a cada partição: nenhum estado (nem a escala) atravessa folds
            modelo_fold = construir_modelos()[nome]
            modelo_fold.fit(X[idx_treino], y[idx_treino])
            previsto = modelo_fold.predict(X[idx_val])
            acuracias.append(balanced_accuracy_score(y[idx_val], previsto))
            f1s.append(f1_score(y[idx_val], previsto, zero_division=0))
        linhas.append({
            "modelo": nome,
            "acuracia_balanceada_media": float(np.mean(acuracias)),
            "acuracia_balanceada_desvio": float(np.std(acuracias)),
            "f1_media": float(np.mean(f1s)),
            "f1_desvio": float(np.std(f1s)),
        })

    return (
        pd.DataFrame(linhas)
        .sort_values("acuracia_balanceada_media", ascending=False)
        .reset_index(drop=True)
    )


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


# ----------------------------------------------------------------------
# 6. Análise de erros
# ----------------------------------------------------------------------
def retorno_futuro(teste: pd.DataFrame) -> pd.Series:
    """Retorno realizado de t -> t+1 (só para analisar o erro, nunca como atributo)."""
    precos = (
        pd.read_parquet(ENTRADA_PRECOS)
        .set_index("data")["preco"]
        .sort_index()
    )
    amanha = teste["data"].map(lambda d: precos.get(d + pd.Timedelta(days=1), np.nan))
    return (amanha.to_numpy() / teste["preco"].to_numpy()) - 1.0


def analisar_erros(
    teste: pd.DataFrame, previsto: np.ndarray, nome: str
) -> dict[str, pd.DataFrame]:
    """Onde o modelo erra: por classe, por volatilidade e por tamanho do movimento."""
    analise = teste.copy()
    analise["previsto"] = previsto
    analise["acertou"] = (analise["previsto"] == analise["y"]).astype(int)
    analise["ret_futuro"] = retorno_futuro(teste)
    analise["mov_abs"] = analise["ret_futuro"].abs()

    por_classe = (
        analise.groupby("y")
        .agg(dias=("y", "size"), taxa_acerto=("acertou", "mean"))
        .reset_index()
        .rename(columns={"y": "classe_real"})
    )

    def por_tercil(coluna: str) -> pd.DataFrame:
        rotulos = ["baixo", "médio", "alto"]
        faixa = pd.qcut(analise[coluna], 3, labels=rotulos)
        return (
            analise.groupby(faixa, observed=True)
            .agg(dias=("acertou", "size"), taxa_acerto=("acertou", "mean"))
            .reset_index()
            .rename(columns={coluna: "faixa"})
        )

    return {
        "por_classe": por_classe,
        "por_volatilidade": por_tercil("vol_30d"),
        "por_movimento": por_tercil("mov_abs"),
        "distribuicao": pd.DataFrame({
            "métrica": ["dias de teste", "previsões de alta", "altas reais"],
            "valor": [
                len(analise),
                int(analise["previsto"].sum()),
                int(analise["y"].sum()),
            ],
        }),
        "tabela": analise,
        "nome": nome,
    }


def matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriz de confusão 2x2 do melhor modelo (linhas = observado, colunas = previsto)."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    tabela = carregar_tabela()
    treino, teste = dividir_temporal(tabela)
    print(
        f"Split temporal: treino {len(treino)} dias "
        f"({treino['data'].min().date()} a {treino['data'].max().date()}) | "
        f"teste {len(teste)} dias "
        f"({teste['data'].min().date()} a {teste['data'].max().date()})"
    )

    print("\n1. Baselines (antes dos modelos):")
    baselines = calcular_baselines(treino, teste)
    for nome, m in baselines.items():
        print(f"   {nome:42s} acc_bal={m['acuracia_balanceada']:.3f} f1={m['f1']:.3f}")

    print(f"\n2. Walk-forward no treino (TimeSeriesSplit, {N_SPLITS} partições):")
    walk_forward = comparar_walk_forward(treino)
    for _, r in walk_forward.iterrows():
        print(
            f"   {r['modelo']:20s} acc_bal={r['acuracia_balanceada_media']:.3f} "
            f"± {r['acuracia_balanceada_desvio']:.3f}"
        )

    melhor = walk_forward.iloc[0]["modelo"]
    print(f"\n   -> modelo selecionado (sem olhar o teste): {melhor}")

    print("\n3. Avaliação final no teste retido:")
    resultados_teste, previsoes = avaliar_no_teste(treino, teste)
    for _, r in resultados_teste.sort_values(
        "acuracia_balanceada", ascending=False
    ).iterrows():
        print(
            f"   {r['modelo']:20s} acc_bal={r['acuracia_balanceada']:.3f} "
            f"f1={r['f1']:.3f}"
        )

    print(f"\n4. Matriz de confusão — {melhor} (linhas = observado, colunas = previsto):")
    matriz = matriz_confusao(teste["y"].to_numpy(), previsoes[melhor])
    print(f"   obs. 0 -> [{matriz[0, 0]:3d} {matriz[0, 1]:3d}]")
    print(f"   obs. 1 -> [{matriz[1, 0]:3d} {matriz[1, 1]:3d}]")

    print("\n5. Análise de erros (taxa de acerto):")
    erros = analisar_erros(teste, previsoes[melhor], melhor)
    for _, r in erros["por_classe"].iterrows():
        print(f"   classe real {int(r['classe_real'])}: {r['taxa_acerto']:.1%} "
              f"({int(r['dias'])} dias)")
    for chave, rotulo in (("por_volatilidade", "vol_30d"), ("por_movimento", "|ret t+1|")):
        faixas = ", ".join(f"{r['faixa']} {r['taxa_acerto']:.1%}"
                           for _, r in erros[chave].iterrows())
        print(f"   por tercil de {rotulo}: {faixas}")


if __name__ == "__main__":
    main()
