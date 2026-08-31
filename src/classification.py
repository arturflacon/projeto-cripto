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

Saídas:
  reports/comparacao_classificadores.md
  reports/matriz_confusao.png

Trabalho acadêmico: nada aqui é recomendação de investimento. As previsões são
estimativas condicionadas ao período analisado, não afirmações sobre o futuro.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
from preprocess import _tabela_md as tabela_md

matplotlib.use("Agg")  # backend sem janela: salva a figura direto em arquivo

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "atributos.parquet"
ENTRADA_PRECOS = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "comparacao_classificadores.md"
SAIDA_FIGURA = RAIZ / "reports" / "matriz_confusao.png"

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


def salvar_matriz_confusao(
    y_true: np.ndarray, y_pred: np.ndarray, nome: str, destino: Path
) -> np.ndarray:
    """Salva a matriz de confusão do melhor modelo como figura."""
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, eixo = plt.subplots(figsize=(4.2, 3.8))
    eixo.imshow(matriz, cmap="Blues")
    rotulos = ["baixa/estável (0)", "alta (1)"]
    eixo.set_xticks([0, 1], rotulos, fontsize=8)
    eixo.set_yticks([0, 1], rotulos, fontsize=8, rotation=90, va="center")
    eixo.set_xlabel("classificado pelo modelo")
    eixo.set_ylabel("observado")
    eixo.set_title(f"Matriz de confusão — {nome}", fontsize=10)
    # limiar no meio da faixa de cores (o colormap normaliza entre min e max),
    # para o texto não sair branco sobre célula clara
    limiar = (int(matriz.min()) + int(matriz.max())) / 2
    for i in range(2):
        for j in range(2):
            eixo.text(
                j, i, str(matriz[i, j]), ha="center", va="center",
                color="white" if matriz[i, j] > limiar else "black",
                fontsize=14,
            )
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return matriz


# ----------------------------------------------------------------------
# 7. Relatório
# ----------------------------------------------------------------------
def _fmt(df: pd.DataFrame, casas: int = 3) -> list[list]:
    """Formata um DataFrame para as linhas da tabela Markdown.

    Formata coluna a coluna (e não célula a célula) porque ``to_numpy`` em um
    DataFrame misto promove tudo a float — o que imprimiria contagens como
    ``40.000``.
    """
    formatado = df.copy()
    for coluna in formatado.columns:
        if pd.api.types.is_float_dtype(formatado[coluna]):
            formatado[coluna] = formatado[coluna].map(lambda v: f"{v:.{casas}f}")
        else:
            formatado[coluna] = formatado[coluna].astype(str)
    return formatado.to_numpy().tolist()


def escrever_relatorio(
    caminho: Path,
    *,
    tabela: pd.DataFrame,
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
    walk_forward: pd.DataFrame,
    resultados_teste: pd.DataFrame,
    melhor: str,
    matriz: np.ndarray,
    erros: dict,
) -> None:
    """Escreve a comparação completa em Markdown."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p: list[str] = []

    p.append("# Classificação da direção do dia seguinte — Solana\n")
    p.append(
        "Encontros 10-11. **Trabalho acadêmico**: os resultados são estimativas "
        "condicionadas ao período analisado e não constituem recomendação de "
        "investimento.\n"
    )
    p.append(f"- Gerado em: `{agora}`")
    p.append(f"- Entrada: `{ENTRADA.relative_to(RAIZ)}`")
    p.append(f"- Alvo: `{features.ALVO}`")
    p.append(f"- Semente fixa: `{SEMENTE}`")
    p.append("")

    p.append("## Desenho experimental\n")
    p.append(
        _linha_tabela_split(tabela, treino, teste)
    )
    p.append("")
    p.append(
        "> A divisão é **cronológica**: o teste é o trecho final da série. "
        "Nenhum embaralhamento é usado — embaralhar aqui seria vazamento. "
        "A padronização (`StandardScaler`) está dentro do `Pipeline`, então "
        "média e desvio saem só do treino de cada partição.\n"
    )

    p.append("## 1. Baselines (medidos antes dos modelos)\n")
    p.append(
        tabela_md(
            ["baseline", "acurácia balanceada", "F1", "precisão", "recall", "acurácia"],
            [
                [nome, f"{m['acuracia_balanceada']:.3f}", f"{m['f1']:.3f}",
                 f"{m['precisao']:.3f}", f"{m['recall']:.3f}", f"{m['acuracia']:.3f}"]
                for nome, m in baselines.items()
            ],
        )
    )
    p.append(
        "\n> Referência obrigatória: um classificador só agrega valor se superar "
        "estes números. Note que *sempre alta* tem acurácia balanceada 0,5 por "
        "construção (acerta 100% de uma classe e 0% da outra) — é justamente por "
        "isso que a acurácia balanceada é a métrica principal aqui.\n"
    )

    p.append(f"## 2. Comparação walk-forward (`TimeSeriesSplit`, {N_SPLITS} partições)\n")
    p.append(
        "Rodado **apenas dentro do treino**, para que a escolha do melhor modelo "
        "não enxergue o teste. Média ± desvio entre partições:\n"
    )
    p.append(
        tabela_md(
            ["modelo", "acurácia balanceada (média ± dp)", "F1 (média ± dp)"],
            [
                [
                    r["modelo"],
                    f"{r['acuracia_balanceada_media']:.3f} ± "
                    f"{r['acuracia_balanceada_desvio']:.3f}",
                    f"{r['f1_media']:.3f} ± {r['f1_desvio']:.3f}",
                ]
                for _, r in walk_forward.iterrows()
            ],
        )
    )
    p.append(
        f"\n> Modelo selecionado: **{melhor}** (maior acurácia balanceada média "
        "no walk-forward). O desvio entre partições é grande em relação à "
        "diferença entre modelos — sinal de que o ranking é instável.\n"
    )

    p.append("## 3. Avaliação final no teste retido\n")
    p.append(
        tabela_md(
            ["modelo", "acurácia balanceada", "F1", "precisão", "recall", "acurácia"],
            _fmt(resultados_teste.sort_values("acuracia_balanceada", ascending=False)),
        )
    )
    p.append(
        "\n> Cada modelo foi reajustado no treino inteiro e avaliado **uma única vez** "
        "no teste.\n"
    )

    p.append(f"## 4. Matriz de confusão — {melhor}\n")
    p.append(
        tabela_md(
            ["", "classificado 0 (baixa/estável)", "classificado 1 (alta)"],
            [
                ["**observado 0**", int(matriz[0, 0]), int(matriz[0, 1])],
                ["**observado 1**", int(matriz[1, 0]), int(matriz[1, 1])],
            ],
        )
    )
    p.append(f"\n![Matriz de confusão]({SAIDA_FIGURA.name})\n")

    p.append("## 5. Análise de erros\n")
    p.append("**Distribuição das previsões**\n")
    p.append(tabela_md(["métrica", "valor"], erros["distribuicao"].to_numpy().tolist()))
    p.append("\n**Taxa de acerto por classe real**\n")
    p.append(
        tabela_md(["classe real", "dias", "taxa de acerto"],
                  _fmt(erros["por_classe"]))
    )
    p.append("\n**Taxa de acerto por regime de volatilidade (tercis de `vol_30d`)**\n")
    p.append(
        tabela_md(["faixa de volatilidade", "dias", "taxa de acerto"],
                  _fmt(erros["por_volatilidade"]))
    )
    p.append("\n**Taxa de acerto por tamanho do movimento realizado (|retorno t+1|)**\n")
    p.append(
        tabela_md(["faixa do movimento", "dias", "taxa de acerto"],
                  _fmt(erros["por_movimento"]))
    )
    p.append("")

    p.append("## 6. Discussão\n")
    p.append(_discussao(baselines, resultados_teste, melhor, erros))

    caminho.write_text("\n".join(p), encoding="utf-8")


def _linha_tabela_split(
    tabela: pd.DataFrame, treino: pd.DataFrame, teste: pd.DataFrame
) -> str:
    """Tabela com o desenho do split temporal."""
    return tabela_md(
        ["conjunto", "dias", "período", "proporção de alta"],
        [
            ["treino", len(treino),
             f"{treino['data'].min().date()} a {treino['data'].max().date()}",
             f"{treino['y'].mean():.1%}"],
            ["teste", len(teste),
             f"{teste['data'].min().date()} a {teste['data'].max().date()}",
             f"{teste['y'].mean():.1%}"],
            ["total", len(tabela),
             f"{tabela['data'].min().date()} a {tabela['data'].max().date()}",
             f"{tabela['y'].mean():.1%}"],
        ],
    )


def _discussao(
    baselines: dict[str, dict[str, float]],
    resultados_teste: pd.DataFrame,
    melhor: str,
    erros: dict,
) -> str:
    """Leitura honesta dos resultados, com os números do experimento."""
    melhor_baseline = max(b["acuracia_balanceada"] for b in baselines.values())
    linha = resultados_teste.set_index("modelo").loc[melhor]
    diferenca = linha["acuracia_balanceada"] - melhor_baseline
    superou = diferenca > 0

    veredito = (
        f"superou o melhor baseline em {diferenca:+.3f}"
        if superou
        else f"**não** superou o melhor baseline ({diferenca:+.3f})"
    )
    analise = erros["tabela"]
    n_teste = len(analise)
    frac_alta_prevista = analise["previsto"].mean()
    frac_alta_real = analise["y"].mean()

    # Sob a hipótese nula "o modelo é uma moeda honesta", a acurácia tem erro
    # padrão sqrt(0.25/n). Serve para dimensionar se a vantagem é ruído.
    erro_padrao = float(np.sqrt(0.25 / n_teste))
    desvios = (linha["acuracia"] - 0.5) / erro_padrao

    acerto_classe = erros["por_classe"].set_index("classe_real")["taxa_acerto"]
    vol = erros["por_volatilidade"].set_index("faixa")["taxa_acerto"]
    mov = erros["por_movimento"].set_index("faixa")["taxa_acerto"]

    return (
        f"O modelo selecionado (**{melhor}**) obteve acurácia balanceada "
        f"**{linha['acuracia_balanceada']:.3f}** no teste retido, contra "
        f"**{melhor_baseline:.3f}** do melhor baseline — ou seja, {veredito}.\n\n"
        f"**Viés de classe.** O modelo classificou como alta {frac_alta_prevista:.1%} "
        f"dos {n_teste} dias de teste, enquanto apenas {frac_alta_real:.1%} foram de "
        f"fato de alta: ele é otimista por construção do ajuste. Isso aparece na "
        f"assimetria de acerto — {acerto_classe.get(1, float('nan')):.1%} nos dias "
        f"que subiram contra {acerto_classe.get(0, float('nan')):.1%} nos que não "
        "subiram. Num período de tendência de queda como o analisado, esse viés é "
        "frágil: ele depende de a proporção de altas se manter.\n\n"
        f"**Regime de volatilidade.** O acerto cai de {vol.get('baixo', float('nan')):.1%} "
        f"(volatilidade baixa) e {vol.get('médio', float('nan')):.1%} (média) para "
        f"{vol.get('alto', float('nan')):.1%} nos dias de volatilidade alta — "
        "coerente com a intuição de que regimes turbulentos são menos previsíveis.\n\n"
        f"**Tamanho do movimento.** O acerto é {mov.get('baixo', float('nan')):.1%} nos "
        f"movimentos pequenos, {mov.get('médio', float('nan')):.1%} nos médios e "
        f"{mov.get('alto', float('nan')):.1%} nos grandes. Acertar movimentos pequenos "
        "tem pouco significado prático (a diferença entre subir e cair é quase "
        "arbitrária ali), então concentrar acerto nos movimentos grandes é o padrão "
        "mais interessante — embora com poucas dezenas de dias possa ser coincidência.\n\n"
        f"**Significância.** Com {n_teste} dias de teste, o erro padrão da acurácia sob "
        f"a hipótese de cara-ou-coroa é {erro_padrao:.3f}. A acurácia observada "
        f"({linha['acuracia']:.3f}) está a {desvios:.1f} desvios de 0,5 — "
        + (
            "ou seja, **dentro do que o acaso produz**; não é evidência de "
            "capacidade preditiva real.\n\n"
            if abs(desvios) < 2
            else "acima do limiar usual de 2 desvios, mas ainda sobre uma amostra "
            "pequena e um único período.\n\n"
        )
        + "Vale o mesmo para a comparação entre modelos: diferenças de poucos pontos "
        "estão dentro do ruído, e é por isso que o walk-forward reporta o desvio "
        "entre partições, e não só a média.\n\n"
        "**Enquadramento honesto.** Prever a direção **diária** de um criptoativo, "
        "com split temporal e sem vazamento, fica próximo de cara-ou-coroa. Isso é "
        "esperado: se um padrão simples e estável existisse nesses atributos, ele "
        "seria arbitrado rapidamente. O resultado modesto **não indica falha de "
        "execução** — indica que o sinal disponível nesse horizonte, com esses "
        "atributos e esse tamanho de amostra, é fraco. O que sustenta o trabalho é "
        "o rigor do protocolo: atributos só com passado (verificado por "
        "recomputação em série truncada), rótulo isolado no futuro, divisão "
        "cronológica, padronização dentro do `Pipeline`, baseline medido antes e "
        "métricas balanceadas.\n\n"
        "**Extensões possíveis** (mais promissoras que direção pura): alvo "
        "`acima_mediana` em `features.py` (desempenho relativo, mais estável) ou "
        "regressão da **volatilidade futura**, que é bem mais previsível que o "
        "sinal do retorno.\n"
    )


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)

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

    matriz = salvar_matriz_confusao(
        teste["y"].to_numpy(), previsoes[melhor], melhor, SAIDA_FIGURA
    )
    erros = analisar_erros(teste, previsoes[melhor], melhor)

    escrever_relatorio(
        SAIDA_RELATORIO,
        tabela=tabela,
        treino=treino,
        teste=teste,
        baselines=baselines,
        walk_forward=walk_forward,
        resultados_teste=resultados_teste,
        melhor=melhor,
        matriz=matriz,
        erros=erros,
    )

    print(f"\nRelatório: {SAIDA_RELATORIO}")
    print(f"Figura:    {SAIDA_FIGURA}")


if __name__ == "__main__":
    main()
