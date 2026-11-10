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

Saída: ``reports/classificacao.md`` — baselines, comparação walk-forward (modelos
e baselines lado a lado, média ± desvio), tabela modelo × métrica no teste,
matriz de confusão e análise de erros do modelo selecionado. O mesmo resumo é
impresso no terminal.

Trabalho acadêmico: nada aqui é recomendação de investimento. As previsões são
estimativas condicionadas ao período analisado, não afirmações sobre o futuro.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
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
from preprocess import _tabela_md

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "atributos.parquet"
ENTRADA_PRECOS = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "classificacao.md"

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
def prever_baselines(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Previsões dos baselines ingênuos para as linhas de ``teste``.

    * **sempre alta**: prevê 1 todos os dias;
    * **classe majoritária (treino)**: prevê sempre a classe mais frequente no treino;
    * **persistência**: repete o rótulo de ontem (``y_anterior``), já conhecido
      em `t`. No alvo ``direcao`` isso equivale a ``ret_1d > 0``; no alvo
      ``acima_mediana`` é "ontem ficou acima da mediana?" — o mesmo critério do
      alvo, e não a direção pura. Se faltar o rótulo de ontem, usa a majoritária.

    Recebe ``treino`` e ``teste`` porque serve tanto ao teste retido quanto a
    cada partição do walk-forward (a majoritária é sempre a do treino da vez).
    """
    n = len(teste)
    sempre_alta = np.ones(n, dtype=int)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(treino[features.colunas_atributos()], treino["y"])
    majoritaria = dummy.predict(teste[features.colunas_atributos()]).astype(int)

    persistencia = (
        teste["y_anterior"].astype("Float64")
        .fillna(float(majoritaria[0]) if n else 0.0)
        .astype(int).to_numpy()
    )

    return {
        "Baseline: sempre alta": sempre_alta,
        "Baseline: classe majoritária (treino)": majoritaria,
        "Baseline: persistência": persistencia,
    }


def calcular_baselines(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Métricas dos baselines no mesmo conjunto de teste dos modelos."""
    y_teste = teste["y"].to_numpy()
    return {
        nome: avaliar(y_teste, previsto)
        for nome, previsto in prever_baselines(treino, teste).items()
    }


# ----------------------------------------------------------------------
# 4. Comparação walk-forward (só dentro do treino)
# ----------------------------------------------------------------------
def comparar_walk_forward(
    treino: pd.DataFrame, n_splits: int = N_SPLITS
) -> pd.DataFrame:
    """Roda ``TimeSeriesSplit`` no treino e devolve média e desvio por modelo.

    Os baselines são avaliados **nas mesmas partições**, para que a comparação
    modelo × baseline use a mesma régua (média ± desvio sobre os mesmos folds).
    A coluna ``tipo`` distingue "modelo" de "baseline".
    """
    colunas = features.colunas_atributos()
    X = treino[colunas].to_numpy()
    y = treino["y"].to_numpy()
    particoes = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    # nome -> (tipo, lista de acurácias balanceadas, lista de F1)
    placar: dict[str, tuple[str, list[float], list[float]]] = {}

    def registrar(nome: str, tipo: str, y_val: np.ndarray, previsto: np.ndarray) -> None:
        _, accs, f1s = placar.setdefault(nome, (tipo, [], []))
        accs.append(balanced_accuracy_score(y_val, previsto))
        f1s.append(f1_score(y_val, previsto, zero_division=0))

    for idx_treino, idx_val in particoes:
        fold_treino, fold_val = treino.iloc[idx_treino], treino.iloc[idx_val]
        for nome, previsto in prever_baselines(fold_treino, fold_val).items():
            registrar(nome, "baseline", y[idx_val], previsto)
        for nome in construir_modelos():
            # Pipeline novo a cada partição: nenhum estado (nem a escala) atravessa folds
            modelo_fold = construir_modelos()[nome]
            modelo_fold.fit(X[idx_treino], y[idx_treino])
            registrar(nome, "modelo", y[idx_val], modelo_fold.predict(X[idx_val]))

    linhas = [
        {
            "modelo": nome,
            "tipo": tipo,
            "acuracia_balanceada_media": float(np.mean(accs)),
            "acuracia_balanceada_desvio": float(np.std(accs)),
            "f1_media": float(np.mean(f1s)),
            "f1_desvio": float(np.std(f1s)),
        }
        for nome, (tipo, accs, f1s) in placar.items()
    ]

    return (
        pd.DataFrame(linhas)
        .sort_values("acuracia_balanceada_media", ascending=False)
        .reset_index(drop=True)
    )


def selecionar_melhor(walk_forward: pd.DataFrame) -> str:
    """Melhor **modelo** pelo walk-forward (baselines não concorrem à seleção)."""
    return str(walk_forward[walk_forward["tipo"] == "modelo"].iloc[0]["modelo"])


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
def tabela_teste_comparativa(
    baselines: dict[str, dict[str, float]], resultados_teste: pd.DataFrame
) -> pd.DataFrame:
    """Modelos e baselines numa só tabela (modelo × métrica) do teste retido.

    ``delta_vs_melhor_baseline`` = acurácia balanceada do modelo menos a do
    melhor baseline — a comparação que decide se o modelo tem algum valor.
    """
    linhas_base = pd.DataFrame(
        [{"modelo": nome, "tipo": "baseline", **m} for nome, m in baselines.items()]
    )
    linhas_modelo = resultados_teste.assign(tipo="modelo")
    tabela = pd.concat([linhas_base, linhas_modelo], ignore_index=True)

    referencia = linhas_base["acuracia_balanceada"].max()
    tabela["delta_vs_melhor_baseline"] = tabela["acuracia_balanceada"] - referencia
    return tabela.sort_values("acuracia_balanceada", ascending=False).reset_index(drop=True)


def teste_contra_acaso(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Os acertos no teste são distinguíveis de um palpite 50/50?

    Teste binomial unilateral sobre a acurácia simples: probabilidade de um
    classificador que acerta 50% dos dias obter esse número de acertos ou mais.
    Com poucas dezenas de dias de teste, diferenças de alguns pontos percentuais
    costumam ficar dentro do acaso — por isso o número é reportado.
    """
    n = len(y_true)
    acertos = int((y_true == y_pred).sum())
    return {
        "n": n,
        "acertos": acertos,
        "p_valor": float(binomtest(acertos, n, 0.5, alternative="greater").pvalue),
        # meia-largura aproximada do IC 95% de uma proporção perto de 50%
        "margem_ic95": float(1.96 * np.sqrt(0.25 / n)),
    }


def _f(valor: float, casas: int = 3) -> str:
    """Formata número para as tabelas Markdown."""
    return f"{valor:.{casas}f}"


def escrever_relatorio(
    caminho: Path,
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    walk_forward: pd.DataFrame,
    comparativo_teste: pd.DataFrame,
    melhor: str,
    matriz: np.ndarray,
    erros: dict,
    acaso: dict[str, float],
) -> None:
    """Grava ``reports/classificacao.md`` com todas as tabelas da etapa."""
    melhor_teste = comparativo_teste.set_index("modelo").loc[melhor]
    p: list[str] = []

    p.append("# Classificação do dia seguinte — Solana\n")
    p.append(
        "**Trabalho acadêmico.** Os resultados abaixo são estimativas condicionadas "
        "ao período analisado. Não são previsão confiável do mercado nem "
        "recomendação de investimento.\n"
    )
    p.append(f"- Gerado em: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    p.append(f"- Entrada: `{ENTRADA.relative_to(RAIZ)}`")
    p.append(
        f"- Alvo: `{features.ALVO}` · janelas {features.JANELA_CURTA}/"
        f"{features.JANELA_LONGA}/{features.JANELA_ACUM} dias · "
        f"{len(features.colunas_atributos())} atributos"
    )
    p.append(
        f"- Split temporal {FRACAO_TREINO:.0%}/{1 - FRACAO_TREINO:.0%}: treino "
        f"**{len(treino)} dias** ({treino['data'].min().date()} a "
        f"{treino['data'].max().date()}) · teste **{len(teste)} dias** "
        f"({teste['data'].min().date()} a {teste['data'].max().date()})"
    )
    p.append(
        f"- Classe 1 no treino: {treino['y'].mean():.1%} · no teste: "
        f"{teste['y'].mean():.1%}"
    )
    p.append(
        f"- Semente: {SEMENTE} · TimeSeriesSplit: {N_SPLITS} partições · "
        f"KNN k={K_VIZINHOS} · árvore max_depth={PROFUNDIDADE_ARVORE}\n"
    )

    p.append("## 1. Comparação walk-forward (somente no treino)\n")
    p.append(
        "Modelos e baselines avaliados nas **mesmas partições** do `TimeSeriesSplit`; "
        "valores = média ± desvio-padrão entre partições. É esta tabela que escolhe "
        "o modelo — o teste não participa da escolha.\n"
    )
    p.append(_tabela_md(
        ["modelo", "tipo", "acc. balanceada", "F1"],
        [
            [
                f"**{r['modelo']}**" if r["modelo"] == melhor else r["modelo"],
                r["tipo"],
                f"{_f(r['acuracia_balanceada_media'])} ± {_f(r['acuracia_balanceada_desvio'])}",
                f"{_f(r['f1_media'])} ± {_f(r['f1_desvio'])}",
            ]
            for _, r in walk_forward.iterrows()
        ],
    ))
    p.append(f"\nModelo selecionado: **{melhor}**.\n")

    p.append("## 2. Teste retido — modelo × métrica\n")
    p.append(
        "Cada modelo é reajustado no treino inteiro e avaliado **uma vez** no trecho "
        "final. Δ = diferença de acurácia balanceada para o melhor baseline.\n"
    )
    p.append(_tabela_md(
        ["modelo", "tipo", "acc. balanceada", "Δ vs baseline", "F1",
         "precisão", "recall", "acurácia"],
        [
            [
                f"**{r['modelo']}**" if r["modelo"] == melhor else r["modelo"],
                r["tipo"],
                _f(r["acuracia_balanceada"]),
                f"{r['delta_vs_melhor_baseline']:+.3f}",
                _f(r["f1"]),
                _f(r["precisao"]),
                _f(r["recall"]),
                _f(r["acuracia"]),
            ]
            for _, r in comparativo_teste.iterrows()
        ],
    ))
    p.append("")

    p.append(f"## 3. Matriz de confusão — {melhor}\n")
    p.append(_tabela_md(
        ["", "previsto 0", "previsto 1"],
        [
            ["**observado 0**", int(matriz[0, 0]), int(matriz[0, 1])],
            ["**observado 1**", int(matriz[1, 0]), int(matriz[1, 1])],
        ],
    ))
    p.append(
        f"\nAcertos: {acaso['acertos']} de {acaso['n']} dias "
        f"({acaso['acertos'] / acaso['n']:.1%}). Teste binomial contra um palpite "
        f"50/50: **p = {acaso['p_valor']:.3f}**. Com {acaso['n']} dias, a margem do "
        f"IC 95% de uma acurácia perto de 50% é de ±{acaso['margem_ic95']:.1%}.\n"
    )

    p.append(f"## 4. Análise de erros — {melhor}\n")
    p.append("**Por classe real**\n")
    p.append(_tabela_md(
        ["classe real", "dias", "taxa de acerto"],
        [[int(r["classe_real"]), int(r["dias"]), f"{r['taxa_acerto']:.1%}"]
         for _, r in erros["por_classe"].iterrows()],
    ))
    for chave, titulo in (
        ("por_volatilidade", "Por tercil de volatilidade (`vol_30d` em t)"),
        ("por_movimento", "Por tercil do tamanho do movimento seguinte (|ret t→t+1|)"),
    ):
        p.append(f"\n**{titulo}**\n")
        p.append(_tabela_md(
            ["faixa", "dias", "taxa de acerto"],
            [[r["faixa"], int(r["dias"]), f"{r['taxa_acerto']:.1%}"]
             for _, r in erros[chave].iterrows()],
        ))
    p.append(
        "\nO tamanho do movimento seguinte só é conhecido depois do fato: serve para "
        "entender *onde* o modelo erra, nunca como atributo.\n"
    )

    p.append("## 5. Leitura dos resultados\n")
    delta = melhor_teste["delta_vs_melhor_baseline"]
    significativo = acaso["p_valor"] < 0.05
    p.append(
        f"- No teste, o modelo selecionado ({melhor}) ficou {delta:+.3f} em acurácia "
        "balanceada em relação ao melhor baseline. "
        + (
            "O número de acertos é estatisticamente distinguível de um palpite 50/50 "
            "(p < 0,05), mas ainda assim é uma estimativa condicionada a um único "
            "trecho de teste."
            if significativo else
            "O número de acertos **não** é estatisticamente distinguível de um palpite "
            "50/50 (p ≥ 0,05): a diferença cabe no acaso de um teste com poucos dias."
        )
    )
    p.append(
        "- A dispersão entre partições do walk-forward mostra a instabilidade: o "
        "desempenho de um mesmo modelo varia vários pontos de um trecho para outro."
    )
    p.append(
        "- Prever a direção **diária** de uma criptomoeda, sem vazamento e com split "
        "temporal, fica próximo de cara-ou-coroa. Empatar com o baseline não é falha "
        "do trabalho: é o comportamento esperado do domínio, e o valor desta etapa "
        "está no rigor do protocolo (split temporal, escala ajustada só no treino, "
        "baselines nas mesmas partições, análise de erros)."
    )
    p.append(
        "- Extensões com sinal mais estável: alvo `acima_mediana` (desempenho "
        "relativo) e a regressão de volatilidade futura (`src/regression.py`).\n"
    )

    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(p), encoding="utf-8")


def main() -> None:
    tabela = carregar_tabela()
    treino, teste = dividir_temporal(tabela)
    print(
        f"Split temporal: treino {len(treino)} dias "
        f"({treino['data'].min().date()} a {treino['data'].max().date()}) | "
        f"teste {len(teste)} dias "
        f"({teste['data'].min().date()} a {teste['data'].max().date()})"
    )

    print("\n1. Baselines no teste (antes dos modelos):")
    baselines = calcular_baselines(treino, teste)
    for nome, m in baselines.items():
        print(f"   {nome:42s} acc_bal={m['acuracia_balanceada']:.3f} f1={m['f1']:.3f}")

    print(f"\n2. Walk-forward no treino (TimeSeriesSplit, {N_SPLITS} partições):")
    walk_forward = comparar_walk_forward(treino)
    for _, r in walk_forward.iterrows():
        print(
            f"   {r['modelo']:42s} acc_bal={r['acuracia_balanceada_media']:.3f} "
            f"± {r['acuracia_balanceada_desvio']:.3f}"
        )

    melhor = selecionar_melhor(walk_forward)
    print(f"\n   -> modelo selecionado (sem olhar o teste): {melhor}")

    print("\n3. Avaliação final no teste retido (Δ = vs melhor baseline):")
    resultados_teste, previsoes = avaliar_no_teste(treino, teste)
    comparativo = tabela_teste_comparativa(baselines, resultados_teste)
    for _, r in comparativo.iterrows():
        print(
            f"   {r['modelo']:42s} acc_bal={r['acuracia_balanceada']:.3f} "
            f"Δ={r['delta_vs_melhor_baseline']:+.3f} f1={r['f1']:.3f}"
        )

    y_teste = teste["y"].to_numpy()
    print(f"\n4. Matriz de confusão — {melhor} (linhas = observado, colunas = previsto):")
    matriz = matriz_confusao(y_teste, previsoes[melhor])
    print(f"   obs. 0 -> [{matriz[0, 0]:3d} {matriz[0, 1]:3d}]")
    print(f"   obs. 1 -> [{matriz[1, 0]:3d} {matriz[1, 1]:3d}]")
    acaso = teste_contra_acaso(y_teste, previsoes[melhor])
    print(
        f"   acertos {acaso['acertos']}/{acaso['n']} | binomial vs 50%: "
        f"p = {acaso['p_valor']:.3f} | IC 95% ±{acaso['margem_ic95']:.1%}"
    )

    print("\n5. Análise de erros (taxa de acerto):")
    erros = analisar_erros(teste, previsoes[melhor], melhor)
    for _, r in erros["por_classe"].iterrows():
        print(f"   classe real {int(r['classe_real'])}: {r['taxa_acerto']:.1%} "
              f"({int(r['dias'])} dias)")
    for chave, rotulo in (("por_volatilidade", "vol_30d"), ("por_movimento", "|ret t+1|")):
        faixas = ", ".join(f"{r['faixa']} {r['taxa_acerto']:.1%}"
                           for _, r in erros[chave].iterrows())
        print(f"   por tercil de {rotulo}: {faixas}")

    escrever_relatorio(
        SAIDA_RELATORIO, treino, teste, walk_forward, comparativo,
        melhor, matriz, erros, acaso,
    )
    print(f"\nRelatório: {SAIDA_RELATORIO}")


if __name__ == "__main__":
    main()
