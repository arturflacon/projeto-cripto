"""clustering.py — Agrupamento em regimes temporais (Encontros 8-9).

Como o projeto usa **um único ativo**, não há como agrupar ativos entre si. O
agrupamento é reenquadrado sobre o **tempo**: cada observação diária da Solana é
um vetor de 13 atributos, e o agrupamento procura **regimes** — estados
recorrentes do mercado (calmo, tendência, estresse).

O "mapa do mercado" da apostila vira uma projeção **PCA das observações diárias**
colorida por regime.

Entrada:  ``data/processed/atributos.parquet``
Saídas:
  data/processed/regimes.parquet     -> rótulo de regime por dia (3 métodos)
  reports/relatorio_regimes.md       -> escolha de k, perfis e interpretação
  reports/escolha_k.png              -> cotovelo + silhouette
  reports/dendrograma.png            -> agrupamento hierárquico (Ward)
  reports/mapa_pca_regimes.png       -> o "mapa", colorido por regime
  reports/regimes_no_tempo.png       -> a série de preço colorida por regime

Métodos: K-Means, hierárquico (Ward) e DBSCAN. O agrupamento acontece no espaço
completo dos 13 atributos padronizados; o PCA é só para **visualizar**.

Trabalho acadêmico: nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import features
from preprocess import _blocos_contiguos, _tabela_md as tabela_md

matplotlib.use("Agg")  # backend sem janela: salva as figuras direto em arquivo

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "atributos.parquet"
SAIDA_REGIMES = RAIZ / "data" / "processed" / "regimes.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "relatorio_regimes.md"
FIG_ESCOLHA_K = RAIZ / "reports" / "escolha_k.png"
FIG_DENDROGRAMA = RAIZ / "reports" / "dendrograma.png"
FIG_MAPA_PCA = RAIZ / "reports" / "mapa_pca_regimes.png"
FIG_TEMPO = RAIZ / "reports" / "regimes_no_tempo.png"

SEMENTE = 42
K_MIN, K_MAX = 2, 8          # faixa testada para o cotovelo/silhouette
N_INIT = 10                  # reinícios do K-Means (evita mínimo local)

# DBSCAN: eps varrido numa grade; min_samples fixo pela regra usual (~2 x dims,
# mas limitado para não exigir vizinhanças grandes demais numa amostra curta).
DBSCAN_MIN_SAMPLES = 8
DBSCAN_EPS_GRADE = np.arange(1.5, 5.01, 0.25)

CORES = plt.get_cmap("tab10")


# ----------------------------------------------------------------------
# 1. Dados
# ----------------------------------------------------------------------
def carregar() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Lê a tabela de atributos e devolve a matriz padronizada.

    O agrupamento é **não supervisionado**: a coluna ``y`` existe no arquivo mas
    é ignorada aqui — ela só entra na etapa de classificação.
    """
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {ENTRADA}\nRode antes `python src/features.py`."
        )
    tabela = pd.read_parquet(ENTRADA).sort_values("data").reset_index(drop=True)
    colunas = features.colunas_atributos()

    # padronizar é obrigatório: todos os métodos aqui são baseados em distância,
    # e sem isso os atributos de maior escala dominariam a geometria
    matriz = StandardScaler().fit_transform(tabela[colunas])
    return tabela, matriz, colunas


# ----------------------------------------------------------------------
# 2. Escolha de k: cotovelo + silhouette
# ----------------------------------------------------------------------
def avaliar_k(matriz: np.ndarray) -> pd.DataFrame:
    """Roda K-Means para cada k na faixa e mede inércia e silhouette."""
    linhas = []
    for k in range(K_MIN, K_MAX + 1):
        modelo = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEMENTE)
        rotulos = modelo.fit_predict(matriz)
        linhas.append({
            "k": k,
            "inercia": float(modelo.inertia_),
            "silhouette": float(silhouette_score(matriz, rotulos)),
        })
    return pd.DataFrame(linhas)


def escolher_k(avaliacao: pd.DataFrame) -> int:
    """Escolhe k pelo maior silhouette (critério objetivo e reprodutível).

    O cotovelo é lido na figura — ele é visual e ambíguo por natureza, então
    serve de confirmação, não de critério automático.
    """
    return int(avaliacao.loc[avaliacao["silhouette"].idxmax(), "k"])


def figura_escolha_k(avaliacao: pd.DataFrame, k_escolhido: int, destino: Path) -> None:
    """Cotovelo (inércia) e silhouette lado a lado."""
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(10, 3.8))

    esq.plot(avaliacao["k"], avaliacao["inercia"], marker="o", color="#2b6cb0")
    esq.set_xlabel("k (nº de regimes)")
    esq.set_ylabel("inércia (soma dos quadrados intra-grupo)")
    esq.set_title("Método do cotovelo", fontsize=10)
    esq.grid(alpha=0.3)

    dir_.plot(avaliacao["k"], avaliacao["silhouette"], marker="o", color="#2f855a")
    dir_.axvline(k_escolhido, color="#c53030", linestyle="--", linewidth=1,
                 label=f"k escolhido = {k_escolhido}")
    dir_.set_xlabel("k (nº de regimes)")
    dir_.set_ylabel("silhouette médio")
    dir_.set_title("Coeficiente de silhouette", fontsize=10)
    dir_.legend(fontsize=8)
    dir_.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def figura_dendrograma(matriz: np.ndarray, destino: Path) -> None:
    """Dendrograma do agrupamento hierárquico (ligação de Ward)."""
    ligacao = linkage(matriz, method="ward")

    fig, eixo = plt.subplots(figsize=(10, 4))
    dendrogram(ligacao, ax=eixo, no_labels=True, color_threshold=None)
    eixo.set_title(
        "Dendrograma — agrupamento hierárquico (Ward)\n"
        "cada folha é um dia; a altura indica a distância na fusão",
        fontsize=10,
    )
    eixo.set_ylabel("distância de fusão")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Os três métodos de agrupamento
# ----------------------------------------------------------------------
def agrupar_kmeans(matriz: np.ndarray, k: int) -> np.ndarray:
    """K-Means: centróides esféricos, todo dia recebe um regime."""
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=SEMENTE).fit_predict(matriz)


def agrupar_hierarquico(matriz: np.ndarray, k: int) -> np.ndarray:
    """Hierárquico aglomerativo com ligação de Ward (mesmo k do K-Means)."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matriz)


def agrupar_dbscan(matriz: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """DBSCAN com ``eps`` varrido numa grade.

    Diferente dos outros dois, o DBSCAN **não** obriga todo dia a pertencer a um
    grupo: pontos em regiões esparsas viram ruído (rótulo ``-1``). Esses dias
    atípicos são a matéria-prima da detecção de anomalias do Encontro 12.

    Escolhe o ``eps`` de maior silhouette entre as configurações que produzem
    ao menos 2 grupos e no máximo 40% de ruído.
    """
    linhas, melhor, melhor_sil = [], None, -np.inf

    for eps in DBSCAN_EPS_GRADE:
        rotulos = DBSCAN(eps=float(eps), min_samples=DBSCAN_MIN_SAMPLES).fit_predict(matriz)
        grupos = sorted(set(rotulos) - {-1})
        frac_ruido = float(np.mean(rotulos == -1))

        sil = np.nan
        valido = len(grupos) >= 2 and frac_ruido <= 0.40
        if valido:
            # silhouette calculado só sobre os pontos agrupados
            nucleo = rotulos != -1
            sil = float(silhouette_score(matriz[nucleo], rotulos[nucleo]))
            if sil > melhor_sil:
                melhor_sil, melhor = sil, rotulos

        linhas.append({
            "eps": round(float(eps), 2),
            "grupos": len(grupos),
            "fracao_ruido": frac_ruido,
            "silhouette": sil,
        })

    grade = pd.DataFrame(linhas)
    if melhor is None:
        # nenhuma configuração válida: devolve a de maior nº de grupos para relatar
        melhor = DBSCAN(
            eps=float(DBSCAN_EPS_GRADE[len(DBSCAN_EPS_GRADE) // 2]),
            min_samples=DBSCAN_MIN_SAMPLES,
        ).fit_predict(matriz)
    return melhor, grade


# ----------------------------------------------------------------------
# 4. Perfil e interpretação dos regimes
# ----------------------------------------------------------------------
def perfil_regimes(
    tabela: pd.DataFrame, rotulos: np.ndarray, colunas: list[str]
) -> pd.DataFrame:
    """Média de cada atributo por regime, em **unidades originais** (interpretável)."""
    com_rotulo = tabela[colunas].copy()
    com_rotulo["regime"] = rotulos
    perfil = com_rotulo.groupby("regime").mean()
    perfil.insert(0, "dias", com_rotulo.groupby("regime").size())
    return perfil.reset_index()


def nomear_regimes(tabela: pd.DataFrame, perfil: pd.DataFrame) -> dict[int, str]:
    """Nome descritivo por regime: turbulência x direção.

    Heurística explícita: a média de ``vol_30d`` e ``ret_30d`` de cada regime é
    posicionada nos **tercis da distribuição diária** — não na mediana entre as
    médias dos regimes. Esse detalhe importa: com poucos regimes, a mediana
    entre eles cai exatamente sobre um deles, que passaria a se comparar consigo
    mesmo e jamais receberia o rótulo "turbulento" ou "em alta".

    Os tercis também permitem uma faixa do meio ("lateral", "volatilidade
    típica"), em vez de forçar todo regime a um dos dois extremos.

    É um rótulo legível — a interpretação de verdade está na tabela de perfis.
    """
    tercis_vol = tabela["vol_30d"].quantile([1 / 3, 2 / 3]).to_numpy()
    tercis_ret = tabela["ret_30d"].quantile([1 / 3, 2 / 3]).to_numpy()

    def faixa(valor: float, limites: np.ndarray, rotulos: tuple[str, str, str]) -> str:
        if valor < limites[0]:
            return rotulos[0]
        if valor > limites[1]:
            return rotulos[2]
        return rotulos[1]

    nomes = {}
    for _, linha in perfil.iterrows():
        regime = int(linha["regime"])
        if regime == -1:
            nomes[regime] = "ruído (dias atípicos)"
            continue
        turbulencia = faixa(
            linha["vol_30d"], tercis_vol, ("calmo", "volatilidade típica", "turbulento")
        )
        direcao = faixa(
            linha["ret_30d"], tercis_ret, ("em queda", "lateral", "em alta")
        )
        nomes[regime] = f"{turbulencia}, {direcao}"
    return nomes


def blocos_por_regime(tabela: pd.DataFrame, rotulos: np.ndarray) -> pd.DataFrame:
    """Quantos trechos contíguos de dias cada regime ocupa.

    Reutiliza ``_blocos_contiguos`` do ``preprocess.py``. Um regime coerente no
    tempo aparece em poucos blocos longos; se aparecer espalhado em dezenas de
    blocos de 1 dia, ele não descreve um "estado" e sim ruído diário.
    """
    linhas = []
    for regime in sorted(set(rotulos)):
        datas = pd.DatetimeIndex(tabela.loc[rotulos == regime, "data"])
        blocos = _blocos_contiguos(datas)
        linhas.append({
            "regime": int(regime),
            "dias": len(datas),
            "blocos": len(blocos),
            "maior_bloco": max((len(b) for b in blocos), default=0),
            "primeiro_dia": datas.min().date().isoformat() if len(datas) else "-",
            "ultimo_dia": datas.max().date().isoformat() if len(datas) else "-",
        })
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# 5. O "mapa": projeção PCA
# ----------------------------------------------------------------------
def figura_mapa_pca(
    matriz: np.ndarray, rotulos: np.ndarray, nomes: dict[int, str], destino: Path
) -> tuple[PCA, np.ndarray]:
    """Projeta as observações diárias em 2 componentes e colore por regime."""
    pca = PCA(n_components=2, random_state=SEMENTE)
    coords = pca.fit_transform(matriz)

    fig, eixo = plt.subplots(figsize=(7, 5.5))
    for i, regime in enumerate(sorted(set(rotulos))):
        marca = rotulos == regime
        cinza = regime == -1
        eixo.scatter(
            coords[marca, 0], coords[marca, 1],
            s=26,
            color="#a0aec0" if cinza else CORES(i % 10),
            marker="x" if cinza else "o",
            alpha=0.55 if cinza else 0.8,
            label=f"R{regime} — {nomes.get(regime, '')}" if not cinza
            else f"ruído ({int(marca.sum())} dias)",
        )
    var = pca.explained_variance_ratio_
    eixo.set_xlabel(f"PC1 ({var[0]:.0%} da variância)")
    eixo.set_ylabel(f"PC2 ({var[1]:.0%} da variância)")
    eixo.set_title(
        "Mapa das observações diárias da Solana\n"
        "projeção PCA colorida por regime",
        fontsize=10,
    )
    eixo.legend(fontsize=7.5, loc="best")
    eixo.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return pca, coords


def figura_regimes_no_tempo(
    tabela: pd.DataFrame, rotulos: np.ndarray, nomes: dict[int, str], destino: Path
) -> None:
    """A série de preço com cada dia colorido pelo seu regime.

    É a figura que mostra se os regimes têm sentido temporal — trechos coerentes
    em vez de cores alternando a cada dia.
    """
    fig, eixo = plt.subplots(figsize=(11, 4))
    eixo.plot(tabela["data"], tabela["preco"], color="#4a5568", linewidth=0.8, zorder=1)

    for i, regime in enumerate(sorted(set(rotulos))):
        marca = rotulos == regime
        cinza = regime == -1
        eixo.scatter(
            tabela.loc[marca, "data"], tabela.loc[marca, "preco"],
            s=22, zorder=2,
            color="#a0aec0" if cinza else CORES(i % 10),
            marker="x" if cinza else "o",
            label=f"R{regime} — {nomes.get(regime, '')}" if not cinza else "ruído",
        )

    eixo.set_ylabel("preço (USD)")
    eixo.set_title(
        "Regimes ao longo do tempo — Solana\n"
        "cada dia colorido pelo regime atribuído pelo K-Means",
        fontsize=10,
    )
    eixo.legend(fontsize=7.5, ncols=2)
    eixo.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 6. Relatório
# ----------------------------------------------------------------------
def _fmt(df: pd.DataFrame, casas: int = 3) -> list[list]:
    """Formata coluna a coluna, para contagens não virarem ``40.000``."""
    saida = df.copy()
    for coluna in saida.columns:
        if pd.api.types.is_float_dtype(saida[coluna]):
            saida[coluna] = saida[coluna].map(
                lambda v: "—" if pd.isna(v) else f"{v:.{casas}f}"
            )
        else:
            saida[coluna] = saida[coluna].astype(str)
    return saida.to_numpy().tolist()


def escrever_relatorio(
    caminho: Path,
    *,
    tabela: pd.DataFrame,
    avaliacao: pd.DataFrame,
    k: int,
    perfil: pd.DataFrame,
    nomes: dict[int, str],
    blocos: pd.DataFrame,
    grade_dbscan: pd.DataFrame,
    concordancia: dict[str, float],
    silhouettes: dict[str, float],
    variancia_pca: np.ndarray,
    cargas_pca: pd.DataFrame,
) -> None:
    """Escreve o relatório de agrupamento em Markdown."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p: list[str] = []

    p.append("# Regimes temporais da Solana — Encontros 8-9\n")
    p.append(
        "**Trabalho acadêmico.** Os regimes são grupos estatísticos de dias com "
        "atributos parecidos, não categorias oficiais de mercado nem sinais de "
        "investimento.\n"
    )
    p.append(f"- Gerado em: `{agora}`")
    p.append(f"- Entrada: `{ENTRADA.relative_to(RAIZ)}`")
    p.append(
        f"- Período: **{tabela['data'].min().date()} a {tabela['data'].max().date()}** "
        f"({len(tabela)} dias, {len(features.colunas_atributos())} atributos)"
    )
    p.append(f"- Semente fixa: `{SEMENTE}`")
    p.append("")

    p.append("## Reenquadramento: agrupar tempo, não ativos\n")
    p.append(
        "Com um único ativo não existe \"mapa entre criptomoedas\". Cada **dia** "
        "vira um ponto de 13 dimensões e o agrupamento procura **regimes**: "
        "estados recorrentes do mercado. O mapa da apostila vira a projeção PCA "
        "das observações diárias, colorida por regime.\n"
    )

    p.append("## 1. Escolha do número de regimes (k)\n")
    p.append(tabela_md(["k", "inércia", "silhouette"], _fmt(avaliacao)))
    p.append(f"\n![Cotovelo e silhouette]({FIG_ESCOLHA_K.name})\n")
    p.append(
        f"> **k = {k}** foi escolhido pelo maior silhouette médio — critério "
        "objetivo e reprodutível. O cotovelo confirma visualmente: a inércia cai "
        "rápido no início e depois achata, sem um joelho nítido (típico de dados "
        "contínuos, sem grupos naturalmente separados).\n"
    )
    p.append(f"![Dendrograma]({FIG_DENDROGRAMA.name})\n")
    p.append(
        "> O dendrograma mostra a estrutura hierárquica: cada folha é um dia e a "
        "altura da fusão indica quão distantes os grupos estavam. Cortar em "
        "alturas diferentes produz mais ou menos regimes.\n"
    )

    p.append("## 2. Perfil dos regimes (K-Means)\n")
    p.append(
        "Média de cada atributo por regime, em unidades originais. É aqui que os "
        "grupos ganham significado:\n"
    )
    interesse = ["regime", "dias", "ret_1d", "ret_30d", "vol_30d",
                 "dist_mm_30", "drawdown", "razao_vol_cap"]
    resumo = perfil[interesse].copy()
    resumo.insert(1, "interpretação", resumo["regime"].map(nomes))
    p.append(
        tabela_md(
            ["regime", "interpretação", "dias", "ret_1d", "ret_30d", "vol_30d",
             "dist_mm_30", "drawdown", "razao_vol_cap"],
            _fmt(resumo),
        )
    )
    p.append(
        "\n> A coluna *interpretação* vem de uma heurística explícita "
        "(`nomear_regimes`): a média de `vol_30d` e `ret_30d` de cada regime é "
        "posicionada nos **tercis da distribuição diária**. Usar os tercis dos "
        "dias — e não a mediana entre as médias dos regimes — evita que o regime "
        "central se compare consigo mesmo, e abre uma faixa intermediária "
        "(*lateral*, *volatilidade típica*) em vez de forçar todo regime a um "
        "extremo. É um rótulo legível, não uma classificação teórica.\n"
    )

    p.append("## 3. Coerência temporal\n")
    p.append(
        "Um regime que descreve um *estado* deve ocupar trechos contíguos. Se "
        "aparecesse espalhado em dezenas de blocos de um dia, seria ruído diário:\n"
    )
    p.append(
        tabela_md(
            ["regime", "dias", "blocos contíguos", "maior bloco", "primeiro dia",
             "último dia"],
            _fmt(blocos),
        )
    )
    p.append(f"\n![Regimes no tempo]({FIG_TEMPO.name})\n")

    p.append("## 4. O mapa (PCA)\n")
    p.append(f"![Mapa PCA]({FIG_MAPA_PCA.name})\n")
    p.append(
        f"> As duas primeiras componentes explicam "
        f"**{variancia_pca[:2].sum():.0%}** da variância "
        f"(PC1 {variancia_pca[0]:.0%}, PC2 {variancia_pca[1]:.0%}). O mapa é uma "
        "sombra 2D de um espaço de 13 dimensões — grupos que parecem sobrepostos "
        "podem estar separados nas dimensões não mostradas.\n"
    )
    p.append("**Atributos que mais pesam em cada componente:**\n")
    p.append(tabela_md(["atributo", "PC1", "PC2"], _fmt(cargas_pca)))
    p.append("")

    p.append("## 5. Comparação entre os três métodos\n")
    p.append(
        tabela_md(
            ["método", "silhouette"],
            [[nome, f"{v:.3f}" if not np.isnan(v) else "—"]
             for nome, v in silhouettes.items()],
        )
    )
    p.append("\n**Concordância entre partições (índice de Rand ajustado):**\n")
    p.append(
        tabela_md(
            ["par de métodos", "ARI"],
            [[par, f"{v:.3f}"] for par, v in concordancia.items()],
        )
    )
    p.append(
        "\n> ARI = 1 significa partições idênticas; ARI ≈ 0, concordância ao "
        "acaso. Métodos que concordam sugerem que a estrutura encontrada é do "
        "dado, não do algoritmo.\n"
    )
    p.append("**Grade de busca do DBSCAN** (`min_samples` = "
             f"{DBSCAN_MIN_SAMPLES}):\n")
    p.append(
        tabela_md(
            ["eps", "grupos", "fração de ruído", "silhouette"],
            _fmt(grade_dbscan),
        )
    )
    p.append(
        "\n> O DBSCAN não obriga todo dia a pertencer a um grupo: dias em regiões "
        "esparsas viram **ruído** (`-1`). Esses dias atípicos são a matéria-prima "
        "da detecção de anomalias do Encontro 12.\n"
    )

    p.append("## 6. Ressalvas metodológicas\n")
    p.append(
        "**Observações não são independentes.** Dias consecutivos compartilham "
        "janelas móveis (uma média de 30 dias muda pouco de um dia para o outro), "
        "então pontos vizinhos no tempo são parecidos por construção. Isso infla "
        "a coerência temporal dos regimes e torna o silhouette otimista — ele "
        "mede separação geométrica, não descoberta de estados independentes.\n\n"
        "**Não há regimes \"verdadeiros\" para comparar.** O agrupamento é não "
        "supervisionado: não existe gabarito. Silhouette e ARI medem consistência "
        "interna, não acerto.\n\n"
        "**Os regimes descrevem o passado analisado.** Eles resumem o período "
        "coletado e não implicam que os mesmos estados voltarão, nem em que "
        "ordem.\n\n"
        "**k é uma escolha, não uma descoberta.** A ausência de cotovelo nítido "
        "indica que os dias formam um contínuo, não grupos naturalmente "
        "separados. O k adotado é o corte mais defensável pelo silhouette dentro "
        f"da faixa testada ({K_MIN}–{K_MAX}), não um número intrínseco ao mercado.\n"
    )

    caminho.write_text("\n".join(p), encoding="utf-8")


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_REGIMES.parent.mkdir(parents=True, exist_ok=True)

    tabela, matriz, colunas = carregar()
    print(f"{len(tabela)} dias x {len(colunas)} atributos padronizados")

    print(f"\n1. Testando k de {K_MIN} a {K_MAX}...")
    avaliacao = avaliar_k(matriz)
    for _, r in avaliacao.iterrows():
        print(f"   k={int(r['k'])}  inercia={r['inercia']:8.1f}  "
              f"silhouette={r['silhouette']:.3f}")
    k = escolher_k(avaliacao)
    print(f"   -> k escolhido (maior silhouette): {k}")

    figura_escolha_k(avaliacao, k, FIG_ESCOLHA_K)
    figura_dendrograma(matriz, FIG_DENDROGRAMA)

    print("\n2. Agrupando pelos três métodos...")
    rotulos_km = agrupar_kmeans(matriz, k)
    rotulos_hi = agrupar_hierarquico(matriz, k)
    rotulos_db, grade_dbscan = agrupar_dbscan(matriz)

    def _sil(rotulos: np.ndarray) -> float:
        nucleo = rotulos != -1
        if len(set(rotulos[nucleo])) < 2:
            return float("nan")
        return float(silhouette_score(matriz[nucleo], rotulos[nucleo]))

    silhouettes = {
        "K-Means": _sil(rotulos_km),
        "Hierárquico (Ward)": _sil(rotulos_hi),
        "DBSCAN (só pontos agrupados)": _sil(rotulos_db),
    }
    for nome, valor in silhouettes.items():
        print(f"   {nome:32s} silhouette={valor:.3f}")
    print(f"   DBSCAN: {len(set(rotulos_db) - {-1})} grupos, "
          f"{int((rotulos_db == -1).sum())} dias marcados como ruído")

    concordancia = {
        "K-Means x Hierárquico": adjusted_rand_score(rotulos_km, rotulos_hi),
        "K-Means x DBSCAN": adjusted_rand_score(rotulos_km, rotulos_db),
        "Hierárquico x DBSCAN": adjusted_rand_score(rotulos_hi, rotulos_db),
    }

    print("\n3. Perfis e figuras...")
    perfil = perfil_regimes(tabela, rotulos_km, colunas)
    nomes = nomear_regimes(tabela, perfil)
    for regime, nome in nomes.items():
        dias = int(perfil.loc[perfil["regime"] == regime, "dias"].iloc[0])
        print(f"   R{regime}: {nome:24s} ({dias} dias)")

    blocos = blocos_por_regime(tabela, rotulos_km)
    pca, _ = figura_mapa_pca(matriz, rotulos_km, nomes, FIG_MAPA_PCA)
    figura_regimes_no_tempo(tabela, rotulos_km, nomes, FIG_TEMPO)

    cargas = pd.DataFrame({
        "atributo": colunas,
        "PC1": pca.components_[0],
        "PC2": pca.components_[1],
    }).sort_values("PC1", key=abs, ascending=False)

    saida = tabela[["id", "data", "preco"]].copy()
    saida["regime_kmeans"] = rotulos_km
    saida["regime_hierarquico"] = rotulos_hi
    saida["regime_dbscan"] = rotulos_db
    saida["interpretacao"] = saida["regime_kmeans"].map(nomes)
    saida.to_parquet(SAIDA_REGIMES, index=False)

    escrever_relatorio(
        SAIDA_RELATORIO,
        tabela=tabela,
        avaliacao=avaliacao,
        k=k,
        perfil=perfil,
        nomes=nomes,
        blocos=blocos,
        grade_dbscan=grade_dbscan,
        concordancia=concordancia,
        silhouettes=silhouettes,
        variancia_pca=pca.explained_variance_ratio_,
        cargas_pca=cargas,
    )

    print(f"\nRegimes:   {SAIDA_REGIMES}")
    print(f"Relatório: {SAIDA_RELATORIO}")
    print(f"Figuras:   {FIG_ESCOLHA_K.name}, {FIG_DENDROGRAMA.name}, "
          f"{FIG_MAPA_PCA.name}, {FIG_TEMPO.name}")


if __name__ == "__main__":
    main()
