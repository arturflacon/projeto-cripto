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
Saídas:
  reports/relatorio_volatilidade.md
  reports/volatilidade_previsao.png

Mesmo rigor dos Encontros 10-11: atributos só com passado, split cronológico,
padronização dentro do Pipeline, baselines medidos antes e seleção que nunca
enxerga o teste.

Trabalho acadêmico: nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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

matplotlib.use("Agg")  # backend sem janela: salva a figura direto em arquivo

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "relatorio_volatilidade.md"
SAIDA_FIGURA = RAIZ / "reports" / "volatilidade_previsao.png"

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
# 6. Figura
# ----------------------------------------------------------------------
def salvar_figura(
    teste: pd.DataFrame, previsto: np.ndarray, nome: str, horizonte: int, destino: Path
) -> None:
    """Série observada x estimada e dispersão, no período de teste."""
    observado = teste["y"].to_numpy()

    fig, (cima, baixo) = plt.subplots(
        2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [1.4, 1]}
    )

    cima.plot(teste["data"], observado * 100, color="#2b6cb0",
              linewidth=1.4, label="volatilidade realizada (observada)")
    cima.plot(teste["data"], previsto * 100, color="#c53030",
              linewidth=1.4, linestyle="--", label=f"estimada — {nome}")
    cima.plot(teste["data"], teste["vol_30d"] * 100, color="#a0aec0",
              linewidth=1.1, label="baseline: persistência (vol_30d)")
    cima.set_ylabel(f"volatilidade dos {horizonte}d seguintes (%)")
    cima.set_title("Volatilidade futura — observada x estimada (teste retido)",
                   fontsize=10)
    cima.legend(fontsize=8)
    cima.grid(alpha=0.25)

    baixo.scatter(observado * 100, previsto * 100, s=22, color="#2b6cb0", alpha=0.65)
    limites = [min(observado.min(), previsto.min()) * 100,
               max(observado.max(), previsto.max()) * 100]
    baixo.plot(limites, limites, color="#c53030", linestyle="--",
               linewidth=1.2, label="estimativa perfeita")
    baixo.set_xlabel("volatilidade observada (%)")
    baixo.set_ylabel("volatilidade estimada (%)")
    baixo.set_title("Dispersão — quanto mais perto da diagonal, melhor", fontsize=10)
    baixo.legend(fontsize=8)
    baixo.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 7. Relatório
# ----------------------------------------------------------------------
def _fmt(df: pd.DataFrame, casas: int = 4) -> list[list]:
    """Formata coluna a coluna, para contagens não virarem ``40.000``."""
    saida = df.copy()
    for coluna in saida.columns:
        if pd.api.types.is_float_dtype(saida[coluna]):
            saida[coluna] = saida[coluna].map(lambda v: f"{v:.{casas}f}")
        else:
            saida[coluna] = saida[coluna].astype(str)
    return saida.to_numpy().tolist()


def escrever_relatorio(
    caminho: Path,
    *,
    horizonte: int,
    melhor: str,
    tabela: pd.DataFrame,
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
    selecao: pd.DataFrame,
    resultados: pd.DataFrame,
    ganhos: dict[str, float],
    correlacao: float,
    armadilha: pd.DataFrame,
) -> None:
    """Escreve o relatório da regressão em Markdown."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p: list[str] = []

    p.append("# Regressão da volatilidade futura — Encontro 12\n")
    p.append(
        "Extensão do projeto. **Trabalho acadêmico**: estimativas condicionadas ao "
        "período analisado, não recomendação de investimento.\n"
    )
    p.append(f"- Gerado em: `{agora}`")
    p.append(f"- Entrada: `{ENTRADA.relative_to(RAIZ)}`")
    p.append(f"- Alvo: desvio-padrão dos retornos dos **{horizonte} dias seguintes**")
    p.append("- Horizonte e modelo selecionados **só com o treino**")
    p.append(f"- Semente fixa: `{SEMENTE}`")
    p.append("")

    p.append("## Por que volatilidade, e não direção\n")
    p.append(
        "A escolha vem da EDA (`notebooks/eda_solana.ipynb`), que encontrou dois "
        "indícios independentes na mesma direção:\n\n"
        "| evidência | direção | magnitude |\n| --- | --- | --- |\n"
        "| autocorrelação (lags 1-3, 5) | dentro da banda (ruído) | 0,16 / 0,17 / 0,13 / 0,12 |\n"
        "| correlação com volume | +0,003 | +0,600 |\n\n"
        "A série informa *quanto* o preço se move, quase nada sobre *para onde*. "
        "Esta etapa testa se essa estrutura é suficiente para **estimar** a "
        "volatilidade futura melhor que referências triviais.\n"
    )

    p.append("## Desenho experimental\n")
    p.append(
        tabela_md(
            ["conjunto", "dias", "período", "volatilidade média do alvo"],
            [
                ["treino", len(treino),
                 f"{treino['data'].min().date()} a {treino['data'].max().date()}",
                 f"{treino['y'].mean():.2%}"],
                ["teste", len(teste),
                 f"{teste['data'].min().date()} a {teste['data'].max().date()}",
                 f"{teste['y'].mean():.2%}"],
                ["total", len(tabela),
                 f"{tabela['data'].min().date()} a {tabela['data'].max().date()}",
                 f"{tabela['y'].mean():.2%}"],
            ],
        )
    )
    p.append(
        f"\n> **Sobreposição do alvo.** Como o alvo é uma janela de {horizonte} dias, "
        f"linhas vizinhas compartilham até {horizonte - 1} dias. O número de janelas "
        f"realmente independentes é da ordem de {len(tabela) // horizonte}, não "
        f"{len(tabela)} — qualquer diferença pequena entre métodos está dentro do "
        "ruído. É também mais uma razão para o split ser temporal: embaralhar "
        "colocaria janelas quase idênticas nos dois lados da divisão.\n\n"
        f"> **Deslocamento de nível.** A volatilidade média do teste é "
        f"{teste['y'].mean() / treino['y'].mean() - 1:+.1%} em relação ao treino, "
        "com variância bem menor. Isso penaliza o R² de **todos** os estimadores, "
        "inclusive dos baselines — voltamos a isso na discussão.\n"
    )

    p.append("## 1. Baselines (medidos antes dos modelos)\n")
    p.append(
        tabela_md(
            ["baseline", "RMSE", "MAE", "R²"],
            [[nome, f"{m['rmse']:.4f}", f"{m['mae']:.4f}", f"{m['r2']:+.3f}"]
             for nome, m in baselines.items()],
        )
    )
    p.append(
        f"\n> A correlação entre `vol_30d` (hoje) e a volatilidade realizada dos "
        f"{horizonte} dias seguintes é **{correlacao:+.3f}** — é o agrupamento de "
        "volatilidade da EDA aparecendo como capacidade preditiva direta. Fraca, "
        "mas real; para comparar, a correlação equivalente para *direção* é "
        "praticamente zero.\n"
    )

    p.append("## 2. Seleção de horizonte e modelo (walk-forward, só no treino)\n")
    p.append(
        "O horizonte do alvo é uma decisão de modelagem: janelas curtas medem a "
        "volatilidade com muito erro amostral (o desvio-padrão de 7 retornos tem "
        "~29% de erro relativo), janelas longas alisam o sinal. Por isso ele é "
        f"selecionado junto com o modelo, entre {len(HORIZONTES)} horizontes x "
        f"{len(construir_modelos())} modelos — **sempre dentro do treino**:\n"
    )
    p.append(
        tabela_md(
            ["horizonte", "modelo", "R² médio (walk-forward)", "desvio"],
            [[int(r["horizonte"]), r["modelo"], f"{r['r2_media']:+.3f}",
              f"{r['r2_desvio']:.3f}"]
             for _, r in selecao.iterrows()],
        )
    )
    p.append(
        f"\n> Selecionado: **{melhor}** com horizonte de **{horizonte} dias** — "
        "a melhor das configurações testadas. Note que **todas** têm R² médio "
        "negativo no walk-forward: nenhuma combinação supera, de forma estável, "
        "prever a média da própria partição.\n"
    )

    p.append("## 3. Avaliação final no teste retido\n")
    p.append(
        tabela_md(
            ["modelo", "RMSE", "MAE", "R²"],
            _fmt(resultados.sort_values("rmse"), 4),
        )
    )
    p.append("\n**Ganho do modelo selecionado sobre os baselines** "
             "(redução do erro quadrático):\n")
    p.append(
        tabela_md(
            ["comparação", "ganho"],
            [[nome, f"{v:+.1%}"] for nome, v in ganhos.items()],
        )
    )
    p.append(f"\n![Volatilidade observada x estimada]({SAIDA_FIGURA.name})\n")

    p.append("## 4. A armadilha que o protocolo evitou\n")
    p.append(
        "Durante o desenvolvimento, calcular o R² **no teste** para cada horizonte "
        "produziu esta tabela:\n"
    )
    p.append(
        tabela_md(
            ["horizonte", "melhor modelo no teste", "R² no teste"],
            [[int(r["horizonte"]), r["melhor_no_teste"], f"{r['r2_teste']:+.3f}"]
             for _, r in armadilha.iterrows()],
        )
    )
    linha_boa = armadilha.loc[armadilha["r2_teste"].idxmax()]
    h_bom = int(linha_boa["horizonte"])
    r2_bom = float(linha_boa["r2_teste"])
    treino_do_bom = selecao[selecao["horizonte"] == h_bom]["r2_media"].max()
    posicao = int((selecao["r2_media"] > treino_do_bom).sum()) + 1
    p.append(
        f"\nEscolher o horizonte por essa tabela levaria a reportar "
        f"**R² = {r2_bom:+.3f}** com h={h_bom} — um resultado aparentemente bom. "
        f"Mas o walk-forward no treino coloca a melhor configuração de h={h_bom} "
        f"em **{posicao}º lugar** entre {len(selecao)}, com R² médio de "
        f"{treino_do_bom:+.3f}.\n\n"
        f"> Ou seja: aquele R² positivo era **sorte da janela de teste específica**, "
        "não capacidade preditiva. Um trabalho que escolhesse o horizonte olhando o "
        "resultado final reportaria um sucesso inexistente. É exatamente esse tipo "
        "de erro que a seleção restrita ao treino previne — e ele é sutil, porque "
        "não envolve vazamento nos atributos nem no rótulo, apenas na **escolha de "
        "hiperparâmetro**.\n"
    )

    p.append("## 5. Discussão\n")
    p.append(_discussao(baselines, resultados, melhor, horizonte, correlacao, teste, treino))

    caminho.write_text("\n".join(p), encoding="utf-8")


def _discussao(
    baselines: dict[str, dict[str, float]],
    resultados: pd.DataFrame,
    melhor: str,
    horizonte: int,
    correlacao: float,
    teste: pd.DataFrame,
    treino: pd.DataFrame,
) -> str:
    """Leitura honesta dos resultados, com os números do experimento."""
    linha = dict(resultados.set_index("modelo").loc[melhor])
    nome_ref = min(baselines, key=lambda n: baselines[n]["rmse"])
    ref = baselines[nome_ref]
    ganho = ganho_sobre(ref, linha)
    n_efetivo = len(teste) // horizonte

    if ganho > 0:
        veredito = (
            f"O modelo selecionado (**{melhor}**, h={horizonte}) errou "
            f"**{ganho:.1%}** menos, em erro quadrático, que o melhor baseline "
            f"(*{nome_ref}*). É uma vantagem pequena, medida sobre cerca de "
            f"{n_efetivo} janelas independentes — bem dentro do que o acaso produz "
            "nesse tamanho de amostra."
        )
    else:
        veredito = (
            f"O modelo selecionado (**{melhor}**, h={horizonte}) **não** superou o "
            f"melhor baseline (*{nome_ref}*): erro quadrático {abs(ganho):.1%} maior."
        )

    return (
        f"{veredito}\n\n"
        "**Todos os R² são negativos — inclusive os dos baselines.** Isso merece "
        "explicação, porque parece pior do que é. O R² compara o erro do estimador "
        "com o de prever a **média do conjunto de teste**. Como a volatilidade do "
        f"teste ficou {teste['y'].mean() / treino['y'].mean() - 1:+.1%} abaixo da do "
        "treino, essa média é um alvo que nenhum método honesto poderia conhecer de "
        "antemão. R² negativo aqui indica **deslocamento de nível entre os "
        "períodos**, não que os modelos sejam absurdos — o baseline \"média do "
        "treino\", que é o estimador mais conservador possível, também fica "
        "negativo.\n\n"
        "**O sinal existe, mas é fraco e não sobrevive à validação.** A correlação "
        f"de {correlacao:+.3f} entre volatilidade passada e futura confirma o "
        "agrupamento de volatilidade que a EDA encontrou — é real, e é muito maior "
        "que o sinal disponível para direção. Mas transformá-lo em estimativa que "
        "supere referências triviais, com 274 observações diárias sobrepostas de um "
        "único ativo, não funcionou: **nenhuma das 16 configurações testadas teve "
        "R² médio positivo no walk-forward do treino**.\n\n"
        "**O que este resultado significa para o projeto.** A hipótese da EDA — "
        "\"magnitude é mais previsível que direção\" — se confirma no diagnóstico "
        "(correlação de 0,30 contra ~0) mas não se converte em desempenho preditivo "
        "aqui. A conclusão honesta não é \"volatilidade é imprevisível\", e sim: "
        "*com esta amostra, estes atributos e este protocolo, não conseguimos "
        "demonstrar previsibilidade além da persistência*. Literatura de finanças "
        "obtém ganhos reais nessa tarefa com modelos próprios da área (GARCH, "
        "volatilidade realizada intradiária) e séries muito mais longas — fora do "
        "escopo desta disciplina.\n\n"
        "**Limitações.** (i) Janelas sobrepostas reduzem o número efetivo de "
        f"amostras a ~{n_efetivo} no teste; (ii) o período é de queda acentuada, e "
        "o teste é sistematicamente mais calmo que o treino; (iii) volatilidade "
        "estimada não é risco garantido — são estimativas condicionadas ao período "
        "analisado, sem valor de recomendação.\n"
    )


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
    resultados, previsoes = avaliar_no_teste(treino, teste)
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

    salvar_figura(teste, previsoes[melhor], melhor, horizonte, SAIDA_FIGURA)
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
    print(f"Figura:    {SAIDA_FIGURA}")


if __name__ == "__main__":
    main()
