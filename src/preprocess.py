"""preprocess.py — Limpeza e qualidade dos dados (Encontro 4).

Lê ``data/processed/precos.parquet`` (saída de ``collect.py``) e produz:

* ``data/processed/precos_limpo.parquet`` — série diária contínua e limpa da Solana;
* ``reports/relatorio_qualidade_dados.md`` — o "relatório de problemas" do Encontro 4.

Etapas (seção 7 da apostila):
  1. carregar, ordenar por data, garantir tipos;
  2. padronizar datas (UTC, sem hora) e remover duplicidades por (id, data);
  3. construir índice diário contínuo e reindexar para expor lacunas;
  4. relatar lacunas (total, blocos, maior lacuna contígua, datas);
  5. política de preenchimento parametrizável (lacunas curtas x longas);
  6. sanidade de valores (preço/capitalização <= 0, volume negativo);
  7. mecanismo genérico de "ativos removidos" (poucas observações ou muitas lacunas);
  8. escrever o Parquet limpo e o relatório ``.md`` com as contagens.

Trabalho acadêmico: nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "precos.parquet"
SAIDA_PARQUET = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA_RELATORIO = RAIZ / "reports" / "relatorio_qualidade_dados.md"

# Lacuna "curta" (<= limiar) é interpolada; lacuna "longa" (> limiar) fica NaN.
LIMIAR_LACUNA_DIAS = 3

# Mecanismo genérico de exclusão de ativos.
N_MIN_OBS = 60                  # mínimo de observações reais (não preenchidas)
FRACAO_MAX_LACUNAS = 0.20       # fração máxima de dias ausentes no período

COLUNAS_NUMERICAS = ["preco", "capitalizacao", "volume"]
COLUNAS_SAIDA = [
    "id", "simbolo", "nome", "data",
    "preco", "capitalizacao", "volume",
    "fonte", "coletado_em", "preenchido",
]


# ----------------------------------------------------------------------
# 1. Carregamento e tipagem
# ----------------------------------------------------------------------
def carregar(caminho: Path) -> pd.DataFrame:
    """Lê o Parquet bruto, ordena por (id, data) e garante os tipos básicos."""
    df = pd.read_parquet(caminho)

    # 'data' como datetime sem fuso; a CoinGecko entrega buckets diários em UTC,
    # então tratamos cada valor como a meia-noite UTC daquele dia (hora removida).
    df["data"] = pd.to_datetime(df["data"], utc=False).dt.normalize()

    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    for coluna in ("id", "simbolo", "nome", "fonte"):
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("string")

    return df.sort_values(["id", "data"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Duplicidades por (id, data)
# ----------------------------------------------------------------------
def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove linhas repetidas por (id, data), mantendo a primeira. Conta as removidas."""
    antes = len(df)
    limpo = df.drop_duplicates(subset=["id", "data"], keep="first").reset_index(drop=True)
    return limpo, antes - len(limpo)


# ----------------------------------------------------------------------
# 3-4. Índice diário contínuo e diagnóstico de lacunas
# ----------------------------------------------------------------------
def _blocos_contiguos(datas: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Agrupa um conjunto de datas em blocos contíguos (passo de 1 dia)."""
    if len(datas) == 0:
        return []
    ordenadas = datas.sort_values()
    cortes = np.where(np.diff(ordenadas.values) != np.timedelta64(1, "D"))[0] + 1
    return [bloco for bloco in np.split(ordenadas, cortes) if len(bloco) > 0]


def reindexar_diario(df_ativo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reindexa a série de um ativo para um índice diário contínuo.

    Retorna o DataFrame reindexado (linhas ausentes ficam NaN nos numéricos) e
    um dicionário com o diagnóstico de lacunas do ativo.
    """
    identificador = df_ativo["id"].iloc[0]
    simbolo = df_ativo["simbolo"].iloc[0]
    nome = df_ativo["nome"].iloc[0]

    inicio, fim = df_ativo["data"].min(), df_ativo["data"].max()
    indice = pd.date_range(inicio, fim, freq="D", name="data")

    reindexado = (
        df_ativo.set_index("data")
        .reindex(indice)
        .reset_index()
    )
    # metadados constantes do ativo são reaplicados nas linhas criadas
    reindexado["id"] = identificador
    reindexado["simbolo"] = simbolo
    reindexado["nome"] = nome

    datas_ausentes = indice[reindexado["preco"].isna().to_numpy()]
    blocos = _blocos_contiguos(pd.DatetimeIndex(datas_ausentes))

    diagnostico = {
        "id": identificador,
        "dias_periodo": len(indice),
        "dias_ausentes": int(len(datas_ausentes)),
        "n_blocos": len(blocos),
        "maior_lacuna": max((len(b) for b in blocos), default=0),
        "blocos": [
            {
                "inicio": b[0].date().isoformat(),
                "fim": b[-1].date().isoformat(),
                "dias": len(b),
            }
            for b in blocos
        ],
        "fracao_lacunas": (len(datas_ausentes) / len(indice)) if len(indice) else 0.0,
    }
    return reindexado, diagnostico


# ----------------------------------------------------------------------
# 5. Política de preenchimento
# ----------------------------------------------------------------------
def preencher_lacunas(
    df_ativo: pd.DataFrame, limiar: int = LIMIAR_LACUNA_DIAS
) -> tuple[pd.DataFrame, dict]:
    """Preenche apenas lacunas curtas (<= ``limiar``), marcando as linhas.

    * lacunas curtas: interpola preço/capitalização (linear no tempo) e faz
      ``ffill`` no volume; a coluna booleana ``preenchido`` marca essas linhas;
    * lacunas longas: **não** são preenchidas (a apostila proíbe) — ficam NaN.
    """
    df_ativo = df_ativo.sort_values("data").reset_index(drop=True)
    ausente = df_ativo["preco"].isna()
    df_ativo["preenchido"] = False

    linhas_preenchidas = 0
    lacunas_longas: list[dict] = []

    for bloco in _blocos_contiguos(pd.DatetimeIndex(df_ativo.loc[ausente, "data"])):
        mascara = df_ativo["data"].isin(bloco)
        if len(bloco) <= limiar:
            df_ativo.loc[mascara, "preenchido"] = True
            linhas_preenchidas += int(mascara.sum())
        else:
            lacunas_longas.append(
                {
                    "inicio": bloco[0].date().isoformat(),
                    "fim": bloco[-1].date().isoformat(),
                    "dias": len(bloco),
                }
            )

    # interpola só onde marcamos como preenchível; o resto permanece NaN
    curtas = df_ativo["preenchido"].to_numpy()
    for coluna in ("preco", "capitalizacao"):
        interp = df_ativo[coluna].interpolate(method="linear", limit_area="inside")
        df_ativo.loc[curtas, coluna] = interp[curtas]
    volume_ffill = df_ativo["volume"].ffill()
    df_ativo.loc[curtas, "volume"] = volume_ffill[curtas]

    resumo = {
        "linhas_preenchidas": linhas_preenchidas,
        "lacunas_longas": lacunas_longas,
        "dias_lacuna_longa": sum(l["dias"] for l in lacunas_longas),
    }
    return df_ativo, resumo


def recortar_maior_trecho_contiguo(df_ativo: pd.DataFrame) -> pd.DataFrame:
    """Recorta a série ao maior trecho diário contíguo **sem NaN** de preço.

    Função auxiliar oferecida para a etapa de modelagem quando uma série
    totalmente contínua for necessária. Não é aplicada por ``main()``.
    """
    df_ativo = df_ativo.sort_values("data").reset_index(drop=True)
    validos = df_ativo.index[df_ativo["preco"].notna()]
    blocos = _blocos_contiguos(pd.DatetimeIndex(df_ativo.loc[validos, "data"]))
    if not blocos:
        return df_ativo.iloc[0:0]
    maior = max(blocos, key=len)
    return df_ativo[df_ativo["data"].isin(maior)].reset_index(drop=True)


# ----------------------------------------------------------------------
# 6. Sanidade de valores
# ----------------------------------------------------------------------
def sanidade_valores(df_ativo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Converte para NaN (e conta) preço/capitalização <= 0 e volume negativo."""
    contagens = {
        "preco_nao_positivo": int((df_ativo["preco"] <= 0).sum()),
        "capitalizacao_nao_positiva": int((df_ativo["capitalizacao"] <= 0).sum()),
        "volume_negativo": int((df_ativo["volume"] < 0).sum()),
    }
    df_ativo.loc[df_ativo["preco"] <= 0, "preco"] = np.nan
    df_ativo.loc[df_ativo["capitalizacao"] <= 0, "capitalizacao"] = np.nan
    df_ativo.loc[df_ativo["volume"] < 0, "volume"] = np.nan
    return df_ativo, contagens


# ----------------------------------------------------------------------
# 7. Mecanismo genérico de "ativos removidos"
# ----------------------------------------------------------------------
def avaliar_ativo(diagnostico: dict, obs_reais: int) -> str | None:
    """Retorna o motivo da remoção do ativo, ou ``None`` para mantê-lo."""
    motivos = []
    if obs_reais < N_MIN_OBS:
        motivos.append(f"poucas observações reais ({obs_reais} < {N_MIN_OBS})")
    if diagnostico["fracao_lacunas"] > FRACAO_MAX_LACUNAS:
        motivos.append(
            f"fração de lacunas alta ({diagnostico['fracao_lacunas']:.1%} "
            f"> {FRACAO_MAX_LACUNAS:.0%})"
        )
    return "; ".join(motivos) if motivos else None


# ----------------------------------------------------------------------
# 8. Relatório
# ----------------------------------------------------------------------
def _tabela_md(cabecalho: list[str], linhas: list[list]) -> str:
    """Monta uma tabela Markdown simples."""
    sep = ["---"] * len(cabecalho)
    corpo = [cabecalho, sep] + [[str(c) for c in linha] for linha in linhas]
    return "\n".join("| " + " | ".join(linha) + " |" for linha in corpo)


def escrever_relatorio(
    caminho: Path,
    *,
    periodo: tuple[pd.Timestamp, pd.Timestamp],
    n_obs_original: int,
    n_obs_final: int,
    duplicatas: int,
    diagnosticos: list[dict],
    resumos_preenchimento: dict[str, dict],
    sanidade: dict[str, dict],
    removidos: list[dict],
    mantidos: list[str],
) -> None:
    """Escreve o relatório de qualidade em Markdown."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    partes: list[str] = []
    partes.append("# Relatório de qualidade dos dados — Encontro 4\n")
    partes.append(
        "Trabalho acadêmico (Aprendizagem de Máquina). Nada aqui é recomendação "
        "de investimento.\n"
    )
    partes.append(f"- Gerado em: `{agora}`")
    partes.append(f"- Entrada: `{ENTRADA.relative_to(RAIZ)}`")
    partes.append(f"- Saída limpa: `{SAIDA_PARQUET.relative_to(RAIZ)}`")
    partes.append(
        f"- Período coberto: **{periodo[0].date()} a {periodo[1].date()}**"
    )
    partes.append("")
    partes.append("## Parâmetros usados\n")
    partes.append(f"- `LIMIAR_LACUNA_DIAS` = {LIMIAR_LACUNA_DIAS}")
    partes.append(f"- `N_MIN_OBS` = {N_MIN_OBS}")
    partes.append(f"- `FRACAO_MAX_LACUNAS` = {FRACAO_MAX_LACUNAS}")
    partes.append("")

    partes.append("## Resumo geral\n")
    partes.append(
        _tabela_md(
            ["Métrica", "Valor"],
            [
                ["Observações no arquivo bruto", n_obs_original],
                ["Duplicatas (id, data) removidas", duplicatas],
                ["Observações após limpeza (série contínua)", n_obs_final],
                ["Ativos mantidos", len(mantidos)],
                ["Ativos removidos", len(removidos)],
            ],
        )
    )
    partes.append("")

    partes.append("## Lacunas por ativo\n")
    linhas = []
    for d in diagnosticos:
        linhas.append(
            [
                d["id"],
                d["dias_periodo"],
                d["dias_ausentes"],
                d["n_blocos"],
                d["maior_lacuna"],
                f"{d['fracao_lacunas']:.1%}",
            ]
        )
    partes.append(
        _tabela_md(
            ["id", "dias no período", "dias ausentes", "blocos de lacuna",
             "maior lacuna (dias)", "fração de lacunas"],
            linhas,
        )
    )
    partes.append("")

    for d in diagnosticos:
        partes.append(f"### `{d['id']}` — detalhe das lacunas\n")
        if not d["blocos"]:
            partes.append("Sem lacunas: série diária já contígua no período.\n")
            continue
        partes.append(
            _tabela_md(
                ["início", "fim", "dias", "classificação"],
                [
                    [
                        b["inicio"],
                        b["fim"],
                        b["dias"],
                        "curta (interpolada)"
                        if b["dias"] <= LIMIAR_LACUNA_DIAS
                        else "longa (mantida como NaN)",
                    ]
                    for b in d["blocos"]
                ],
            )
        )
        partes.append("")

    partes.append("## Preenchimento aplicado\n")
    linhas = []
    for identificador, r in resumos_preenchimento.items():
        linhas.append(
            [
                identificador,
                r["linhas_preenchidas"],
                len(r["lacunas_longas"]),
                r["dias_lacuna_longa"],
            ]
        )
    partes.append(
        _tabela_md(
            ["id", "linhas preenchidas (lacunas curtas)",
             "lacunas longas", "dias em lacuna longa (NaN)"],
            linhas,
        )
    )
    partes.append(
        "\n> Lacunas curtas: preço/capitalização por interpolação linear no tempo "
        "e volume por `ffill`, com a coluna booleana `preenchido = True`.\n"
        "> Lacunas longas: **não preenchidas** (a apostila proíbe preenchimento "
        "artificial de trechos longos); use `recortar_maior_trecho_contiguo()` "
        "para obter uma subsérie sem NaN quando necessário.\n"
    )

    partes.append("## Sanidade de valores\n")
    partes.append(
        _tabela_md(
            ["id", "preço <= 0", "capitalização <= 0", "volume < 0"],
            [
                [
                    identificador,
                    s["preco_nao_positivo"],
                    s["capitalizacao_nao_positiva"],
                    s["volume_negativo"],
                ]
                for identificador, s in sanidade.items()
            ],
        )
    )
    partes.append("\n> Valores fora de faixa são convertidos para NaN e contam como lacuna.\n")

    partes.append("## Ativos removidos\n")
    if removidos:
        partes.append(
            _tabela_md(
                ["id", "motivo"],
                [[r["id"], r["motivo"]] for r in removidos],
            )
        )
    else:
        partes.append(
            "Nenhum ativo removido. (O mecanismo genérico — `N_MIN_OBS` e "
            "`FRACAO_MAX_LACUNAS` — permanece ativo para coletas futuras.)"
        )
    partes.append("")

    caminho.write_text("\n".join(partes), encoding="utf-8")


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo de entrada não encontrado: {ENTRADA}\n"
            "Rode primeiro `python src/collect.py`."
        )
    SAIDA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_RELATORIO.parent.mkdir(parents=True, exist_ok=True)

    bruto = carregar(ENTRADA)
    n_obs_original = len(bruto)
    periodo = (bruto["data"].min(), bruto["data"].max())

    sem_dup, n_duplicatas = remover_duplicatas(bruto)

    diagnosticos: list[dict] = []
    resumos_preenchimento: dict[str, dict] = {}
    sanidade: dict[str, dict] = {}
    removidos: list[dict] = []
    mantidos: list[str] = []
    partes_limpas: list[pd.DataFrame] = []

    for identificador, grupo in sem_dup.groupby("id", sort=True):
        reindexado, diagnostico = reindexar_diario(grupo)
        diagnosticos.append(diagnostico)

        preenchido, resumo = preencher_lacunas(reindexado)
        resumos_preenchimento[identificador] = resumo

        limpo_ativo, contagens = sanidade_valores(preenchido)
        sanidade[identificador] = contagens

        obs_reais = int((limpo_ativo["preco"].notna() & ~limpo_ativo["preenchido"]).sum())
        motivo = avaliar_ativo(diagnostico, obs_reais)
        if motivo:
            removidos.append({"id": identificador, "motivo": motivo})
            print(f"  ativo removido: {identificador} ({motivo})")
            continue

        mantidos.append(identificador)
        limpo_ativo["fonte"] = grupo["fonte"].iloc[0] if "fonte" in grupo else "coingecko"
        limpo_ativo["coletado_em"] = (
            grupo["coletado_em"].iloc[0] if "coletado_em" in grupo else pd.NaT
        )
        partes_limpas.append(limpo_ativo[COLUNAS_SAIDA])

    if not partes_limpas:
        raise SystemExit("Todos os ativos foram removidos pelos critérios de qualidade.")

    final = (
        pd.concat(partes_limpas, ignore_index=True)
        .sort_values(["id", "data"])
        .reset_index(drop=True)
    )
    final.to_parquet(SAIDA_PARQUET, index=False)

    escrever_relatorio(
        SAIDA_RELATORIO,
        periodo=periodo,
        n_obs_original=n_obs_original,
        n_obs_final=len(final),
        duplicatas=n_duplicatas,
        diagnosticos=diagnosticos,
        resumos_preenchimento=resumos_preenchimento,
        sanidade=sanidade,
        removidos=removidos,
        mantidos=mantidos,
    )

    print(f"\nParquet limpo: {SAIDA_PARQUET}  ({len(final)} linhas)")
    print(f"Relatório:     {SAIDA_RELATORIO}")


if __name__ == "__main__":
    main()
