"""dashboard.py — Painel da análise da Solana (Encontro 13).

Reúne o que o pipeline produziu: indicadores do período, regimes temporais,
eventos atípicos e a classificação do dia seguinte.

Rodar (da raiz do projeto):
    streamlit run app/dashboard.py

O painel apenas **lê** os arquivos já gerados — não treina nada nem chama a API.
Se algum arquivo faltar, a seção correspondente explica qual script rodar.

**Trabalho acadêmico.** Nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
DIR_PROC = RAIZ / "data" / "processed"

CORES_REGIME = ["#3182ce", "#dd6b20", "#38a169", "#805ad5", "#d53f8c"]

st.set_page_config(page_title="Solana — Análise de Mercado", layout="wide")


# ----------------------------------------------------------------------
# Carregamento (em cache: o painel só lê arquivos prontos)
# ----------------------------------------------------------------------
@st.cache_data
def carregar(nome: str) -> pd.DataFrame | None:
    """Lê um Parquet de ``data/processed``; devolve ``None`` se não existir."""
    caminho = DIR_PROC / nome
    if not caminho.exists():
        return None
    return pd.read_parquet(caminho)


def aviso_ausente(arquivo: str, script: str) -> None:
    """Mensagem padrão quando uma etapa do pipeline ainda não rodou."""
    st.warning(
        f"`data/processed/{arquivo}` não encontrado. "
        f"Rode `python src/{script}` para gerar esta seção."
    )


# ----------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------
st.title("Análise do mercado de criptomoedas — Solana")
st.caption(
    "Projeto integrador de Aprendizagem de Máquina · pipeline: coleta → limpeza → "
    "atributos → regimes → classificação → volatilidade"
)
st.error(
    "**Trabalho acadêmico.** Todos os números são estimativas estatísticas "
    "condicionadas ao período analisado. Nada aqui é recomendação de investimento, "
    "e a avaliação dos modelos (ver `reports/`) mostra desempenho próximo do acaso "
    "para previsão de direção.",
    icon="⚠️",
)

precos = carregar("precos_limpo.parquet")
if precos is None:
    aviso_ausente("precos_limpo.parquet", "preprocess.py")
    st.stop()

precos = precos.sort_values("data").reset_index(drop=True)
atributos = carregar("atributos.parquet")
regimes = carregar("regimes.parquet")
previsoes = carregar("previsoes.parquet")

# ----------------------------------------------------------------------
# Filtro de período
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Período")
    inicio_min = precos["data"].min().date()
    fim_max = precos["data"].max().date()
    faixa = st.date_input(
        "Intervalo analisado",
        value=(inicio_min, fim_max),
        min_value=inicio_min,
        max_value=fim_max,
    )
    if isinstance(faixa, tuple) and len(faixa) == 2:
        inicio, fim = faixa
    else:                                   # seleção ainda incompleta
        inicio, fim = inicio_min, fim_max

    st.divider()
    st.caption(
        f"Fonte: CoinGecko · {len(precos)} dias coletados\n\n"
        f"Cobertura: {inicio_min} a {fim_max}"
    )

mascara = (precos["data"].dt.date >= inicio) & (precos["data"].dt.date <= fim)
janela = precos[mascara]

if janela.empty:
    st.warning("Nenhum dia no intervalo selecionado.")
    st.stop()


# ----------------------------------------------------------------------
# 1. Indicadores
# ----------------------------------------------------------------------
st.subheader("Indicadores do período")

preco_ini, preco_fim = janela["preco"].iloc[0], janela["preco"].iloc[-1]
retornos = janela["preco"] / janela["preco"].shift(1) - 1
drawdown = (janela["preco"] / janela["preco"].cummax() - 1).iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Preço final (USD)", f"{preco_fim:,.2f}",
            f"{preco_fim / preco_ini - 1:+.1%} no período")
col2.metric("Máxima (USD)", f"{janela['preco'].max():,.2f}",
            help=f"em {janela.loc[janela['preco'].idxmax(), 'data'].date()}")
col3.metric("Mínima (USD)", f"{janela['preco'].min():,.2f}",
            help=f"em {janela.loc[janela['preco'].idxmin(), 'data'].date()}")
col4.metric("Volatilidade diária", f"{retornos.std():.2%}",
            help="desvio-padrão dos retornos diários no intervalo")
col5.metric("Drawdown atual", f"{drawdown:.1%}",
            help="queda em relação ao maior preço do intervalo")

st.caption(
    "Indicadores descritivos do intervalo selecionado — descrevem o passado, "
    "não projetam o futuro."
)

# ----------------------------------------------------------------------
# 2. Preço e regimes
# ----------------------------------------------------------------------
st.divider()
st.subheader("Preço e regimes de mercado")

if regimes is None:
    aviso_ausente("regimes.parquet", "clustering.py")
else:
    reg = regimes[
        (regimes["data"].dt.date >= inicio) & (regimes["data"].dt.date <= fim)
    ].copy()
    reg["regime"] = "R" + reg["regime_kmeans"].astype(str) + " — " + reg["interpretacao"]

    figura = px.scatter(
        reg, x="data", y="preco", color="regime",
        color_discrete_sequence=CORES_REGIME,
        labels={"data": "", "preco": "preço (USD)", "regime": "regime"},
    )
    figura.add_trace(
        go.Scatter(
            x=reg["data"], y=reg["preco"], mode="lines",
            line={"color": "rgba(120,120,120,0.45)", "width": 1},
            name="preço", hoverinfo="skip", showlegend=False,
        )
    )
    figura.update_traces(marker={"size": 7}, selector={"mode": "markers"})
    figura.update_layout(
        height=420, legend={"orientation": "h", "y": -0.18},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(figura, width="stretch")

    st.markdown(
        "Cada dia é colorido pelo **regime** que o agrupamento (K-Means sobre 13 "
        "atributos) atribuiu. Os regimes foram descobertos sem informação de data — "
        "que eles formem blocos coerentes no tempo é resultado, não imposição."
    )

    resumo = (
        reg.groupby(["regime_kmeans", "interpretacao"])
        .agg(dias=("preco", "size"), preco_medio=("preco", "mean"))
        .reset_index()
        .rename(columns={
            "regime_kmeans": "regime", "interpretacao": "interpretação",
            "preco_medio": "preço médio (USD)",
        })
    )
    resumo["preço médio (USD)"] = resumo["preço médio (USD)"].round(2)
    st.dataframe(resumo, width="stretch", hide_index=True)

# ----------------------------------------------------------------------
# 3. Eventos atípicos
# ----------------------------------------------------------------------
st.divider()
st.subheader("Eventos atípicos")

if regimes is None:
    aviso_ausente("regimes.parquet", "clustering.py")
else:
    atipicos = reg[reg["regime_dbscan"] == -1].copy()
    st.markdown(
        f"O DBSCAN não obriga todo dia a pertencer a um grupo: dias em regiões "
        f"esparsas do espaço de atributos ficam sem grupo (`-1`). "
        f"**{len(atipicos)} de {len(reg)} dias** ({len(atipicos) / len(reg):.0%}) "
        "do intervalo foram marcados assim — são candidatos a evento atípico."
    )

    if atipicos.empty:
        st.info("Nenhum dia atípico no intervalo selecionado.")
    else:
        linha = go.Figure()
        linha.add_trace(go.Scatter(
            x=reg["data"], y=reg["preco"], mode="lines",
            line={"color": "#a0aec0", "width": 1.2}, name="preço",
        ))
        linha.add_trace(go.Scatter(
            x=atipicos["data"], y=atipicos["preco"], mode="markers",
            marker={"color": "#c53030", "size": 8, "symbol": "x"},
            name="dia atípico (DBSCAN)",
        ))
        linha.update_layout(
            height=320, yaxis_title="preço (USD)",
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        st.plotly_chart(linha, width="stretch")

        if atributos is not None:
            detalhe = atipicos.merge(
                atributos[["data", "ret_1d", "vol_30d", "razao_vol_cap"]],
                on="data", how="left",
            )
            detalhe = detalhe[["data", "preco", "ret_1d", "vol_30d", "razao_vol_cap"]]
            detalhe.columns = ["data", "preço (USD)", "retorno do dia",
                               "volatilidade 30d", "volume/cap"]
            detalhe["retorno do dia"] = detalhe["retorno do dia"].map(
                lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
            detalhe["volatilidade 30d"] = detalhe["volatilidade 30d"].map(
                lambda v: f"{v:.2%}" if pd.notna(v) else "—")
            detalhe["volume/cap"] = detalhe["volume/cap"].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            detalhe["preço (USD)"] = detalhe["preço (USD)"].round(2)
            st.dataframe(detalhe, width="stretch", hide_index=True)

        st.caption(
            "\"Atípico\" significa *estatisticamente distante do comportamento "
            "usual do período* — não um evento noticioso identificado, nem sinal "
            "de nada."
        )

# ----------------------------------------------------------------------
# 4. Classificação do dia seguinte
# ----------------------------------------------------------------------
st.divider()
st.subheader("Classificação do dia seguinte")

if previsoes is None:
    aviso_ausente("previsoes.parquet", "prever.py")
else:
    pendente = previsoes[previsoes["observado"].isna()]
    historico = previsoes[previsoes["observado"].notna()]

    if pendente.empty:
        st.info("Nenhum dia pendente. Rode `python src/prever.py` para atualizar.")
    else:
        linha_prev = pendente.iloc[-1]
        rotulo = "ALTA" if linha_prev["classificado"] == 1 else "BAIXA/ESTÁVEL"

        col_a, col_b, col_c = st.columns([1.2, 1, 1])
        col_a.metric(f"Dia {linha_prev['data_alvo'].date()}", rotulo)
        col_b.metric("Dados até", str(linha_prev["data_base"].date()),
                     help=f"preço base {linha_prev['preco_base']:.2f} USD")
        col_c.metric("Dias de treino", int(linha_prev["dias_treino"]))

    if not historico.empty:
        acertos = int(historico["acertou"].sum())
        st.markdown(
            f"Nos **{len(historico)} dias** anteriores exibidos, cada um "
            f"classificado por um modelo ajustado apenas com os dias anteriores a "
            f"ele, houve **{acertos} acertos** "
            f"({acertos / len(historico):.0%})."
        )
        tabela = historico.copy()
        tabela["classificado"] = tabela["classificado"].map(
            {1: "alta", 0: "baixa/estável"})
        tabela["observado"] = tabela["observado"].map(
            {1: "alta", 0: "baixa/estável"})
        tabela = tabela[["data_alvo", "preco_base", "classificado",
                         "observado", "acertou"]]
        tabela.columns = ["dia", "preço base (USD)", "classificado",
                          "observado", "acertou"]
        tabela["preço base (USD)"] = tabela["preço base (USD)"].round(2)
        st.dataframe(tabela, width="stretch", hide_index=True)

    st.warning(
        "**Como ler.** \"ALTA\" significa apenas que o modelo colocou este dia do "
        "lado positivo da sua fronteira de decisão, a partir de padrões passados. "
        "A avaliação formal (`reports/comparacao_classificadores.md`) mostra "
        "acurácia a 1,4 desvios do acaso — ou seja, dentro do que uma moeda honesta "
        "produziria. Não use isto para decidir nada.",
        icon="⚠️",
    )

# ----------------------------------------------------------------------
# 5. Volatilidade
# ----------------------------------------------------------------------
st.divider()
st.subheader("Volatilidade ao longo do tempo")

if atributos is None:
    aviso_ausente("atributos.parquet", "features.py")
else:
    vol = atributos[
        (atributos["data"].dt.date >= inicio) & (atributos["data"].dt.date <= fim)
    ]
    figura_vol = go.Figure()
    figura_vol.add_trace(go.Scatter(
        x=vol["data"], y=vol["vol_7d"] * 100, mode="lines",
        line={"color": "#a0aec0", "width": 1.1}, name="volatilidade 7d",
    ))
    figura_vol.add_trace(go.Scatter(
        x=vol["data"], y=vol["vol_30d"] * 100, mode="lines",
        line={"color": "#c53030", "width": 1.8}, name="volatilidade 30d",
    ))
    figura_vol.update_layout(
        height=320, yaxis_title="volatilidade (%)",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(figura_vol, width="stretch")

    st.markdown(
        "A volatilidade **não é constante** e se agrupa em blocos: dias agitados "
        "vêm colados em outros dias agitados. A EDA mediu esse efeito "
        "(autocorrelação de |retornos| significativa nas defasagens 1-3 e 5), e é "
        "por isso que a série informa mais sobre a *magnitude* do movimento do que "
        "sobre a sua *direção*."
    )

# ----------------------------------------------------------------------
# Rodapé
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "Relatórios completos em `reports/`: qualidade dos dados, regimes, comparação "
    "de classificadores e regressão de volatilidade. "
    "Análise exploratória em `notebooks/eda_solana.ipynb`."
)
