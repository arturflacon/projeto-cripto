"""clustering.py — Agrupamento em regimes temporais (Encontros 8-9).

Como o projeto usa **um único ativo**, não há como agrupar ativos entre si. O
agrupamento é reenquadrado sobre o **tempo**: cada observação diária da Solana é
um vetor de 13 atributos, e o agrupamento procura **regimes** — estados
recorrentes do mercado (calmo, tendência, estresse).

O "mapa do mercado" da apostila vira uma projeção **PCA das observações diárias**,
resumida no terminal pela variância explicada e pelas cargas de cada atributo.

Entrada:  ``data/processed/atributos.parquet``
Saída:    ``data/processed/regimes.parquet`` -> rótulo de regime por dia (3 métodos)
          (escolha de k, perfis, concordância entre métodos e PCA vão para o terminal)

Métodos: K-Means, hierárquico (Ward) e DBSCAN. O agrupamento acontece no espaço
completo dos 13 atributos padronizados; o PCA é só para **descrever** o resultado.

Trabalho acadêmico: nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import features
from preprocess import _blocos_contiguos

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "atributos.parquet"
SAIDA_REGIMES = RAIZ / "data" / "processed" / "regimes.parquet"

SEMENTE = 42
K_MIN, K_MAX = 2, 8          # faixa testada para o cotovelo/silhouette
N_INIT = 10                  # reinícios do K-Means (evita mínimo local)

# DBSCAN: eps varrido numa grade; min_samples fixo pela regra usual (~2 x dims,
# mas limitado para não exigir vizinhanças grandes demais numa amostra curta).
DBSCAN_MIN_SAMPLES = 8
DBSCAN_EPS_GRADE = np.arange(1.5, 5.01, 0.25)


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

    O cotovelo é lido na tabela de inércia — ele é ambíguo por natureza, então
    serve de confirmação, não de critério automático.
    """
    return int(avaliacao.loc[avaliacao["silhouette"].idxmax(), "k"])


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
def projetar_pca(matriz: np.ndarray) -> PCA:
    """Ajusta a PCA de 2 componentes sobre as observações diárias.

    A projeção é resumida pela variância explicada e pelas cargas de cada
    atributo — isto é, quais atributos mais separam os dias no "mapa".
    """
    return PCA(n_components=2, random_state=SEMENTE).fit(matriz)


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
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

    print("\n2. Agrupando pelos três métodos...")
    rotulos_km = agrupar_kmeans(matriz, k)
    rotulos_hi = agrupar_hierarquico(matriz, k)
    rotulos_db, _ = agrupar_dbscan(matriz)

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
    print("   concordância entre partições (ARI):")
    for par, valor in concordancia.items():
        print(f"      {par:24s} {valor:.3f}")

    print("\n3. Perfis dos regimes (K-Means) e coerência temporal...")
    perfil = perfil_regimes(tabela, rotulos_km, colunas)
    nomes = nomear_regimes(tabela, perfil)
    blocos = blocos_por_regime(tabela, rotulos_km).set_index("regime")
    for regime, nome in nomes.items():
        b = blocos.loc[regime]
        print(f"   R{regime}: {nome:28s} {b['dias']:3d} dias em {b['blocos']:2d} "
              f"bloco(s), maior bloco {b['maior_bloco']} dias")

    print("\n4. Projeção PCA (o \"mapa\")...")
    pca = projetar_pca(matriz)
    var = pca.explained_variance_ratio_
    print(f"   PC1 {var[0]:.0%} + PC2 {var[1]:.0%} = {var.sum():.0%} da variância")
    cargas = pd.DataFrame({
        "atributo": colunas,
        "PC1": pca.components_[0],
        "PC2": pca.components_[1],
    })
    for pc in ("PC1", "PC2"):
        top = cargas.reindex(cargas[pc].abs().sort_values(ascending=False).index).head(3)
        print(f"   {pc} pesa mais em: " +
              ", ".join(f"{r['atributo']} ({r[pc]:+.2f})" for _, r in top.iterrows()))

    saida = tabela[["id", "data", "preco"]].copy()
    saida["regime_kmeans"] = rotulos_km
    saida["regime_hierarquico"] = rotulos_hi
    saida["regime_dbscan"] = rotulos_db
    saida["interpretacao"] = saida["regime_kmeans"].map(nomes)
    saida.to_parquet(SAIDA_REGIMES, index=False)

    print(f"\nRegimes: {SAIDA_REGIMES}")


if __name__ == "__main__":
    main()
