import sqlite3

import pandas as pd

from src.config import CAMINHO_DADOS_RAW


TEMPORADA_API = 2026


def _formatar_partidas(partidas, usar_data_utc=False):
    if partidas.empty:
        return pd.DataFrame()

    if usar_data_utc:
        datas = pd.to_datetime(partidas["data"], utc=True).dt.tz_convert(
            "America/Sao_Paulo"
        )
    else:
        datas = pd.to_datetime(partidas["data"], errors="coerce")

    jogos = pd.DataFrame({
        "ID": None,
        "rodata": partidas["rodada"],
        "data": datas.dt.strftime("%d/%m/%Y"),
        "hora": datas.dt.strftime("%H:%M"),
        "mandante": partidas["mandante"],
        "visitante": partidas["visitante"],
        "formacao_mandante": "",
        "formacao_visitante": "",
        "tecnico_mandante": "",
        "tecnico_visitante": "",
        "vencedor": [
            mandante if gols_m > gols_v
            else visitante if gols_v > gols_m
            else "-"
            for mandante, visitante, gols_m, gols_v in zip(
                partidas["mandante"],
                partidas["visitante"],
                partidas["gols_mandante"],
                partidas["gols_visitante"],
            )
        ],
        "arena": "",
        "mandante_Placar": partidas["gols_mandante"],
        "visitante_Placar": partidas["gols_visitante"],
        "mandante_Estado": "",
        "visitante_Estado": "",
    })

    return jogos


def carregar_dados():
    if not CAMINHO_DADOS_RAW.exists():
        raise FileNotFoundError(f"CSV não encontrado: {CAMINHO_DADOS_RAW}")

    historico = pd.read_csv(CAMINHO_DADOS_RAW)

    caminho_banco = CAMINHO_DADOS_RAW.parents[1] / "brasileirao.db"
    if not caminho_banco.exists():
        return historico

    with sqlite3.connect(caminho_banco) as conexao:
        tabelas = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            conexao,
        )["name"].tolist()

        partidas_novas = []

        if "partidas_api" in tabelas:
            api = pd.read_sql_query(
                """
                SELECT data, mandante, visitante,
                       mandante_placar AS gols_mandante,
                       visitante_placar AS gols_visitante,
                       0 AS rodada
                FROM partidas_api
                WHERE temporada = ?
                  AND status IN ('FT', 'AET', 'PEN')
                  AND mandante_placar IS NOT NULL
                  AND visitante_placar IS NOT NULL
                """,
                conexao,
                params=(TEMPORADA_API,),
            )
            partidas_novas.append(_formatar_partidas(api, usar_data_utc=True))

        if "jogos_2026" in tabelas:
            jogos_2026 = pd.read_sql_query(
                """
                SELECT data, rodada, mandante, visitante,
                       gols_mandante, gols_visitante
                FROM jogos_2026
                WHERE status = 'ENCERRADO'
                  AND gols_mandante IS NOT NULL
                  AND gols_visitante IS NOT NULL
                """,
                conexao,
            )
            partidas_novas.append(_formatar_partidas(jogos_2026))

    partidas_novas = [partidas for partidas in partidas_novas if not partidas.empty]
    if not partidas_novas:
        return historico

    dados = pd.concat([historico, *partidas_novas], ignore_index=True)
    dados["_chave_data"] = pd.to_datetime(
        dados["data"], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    dados = dados.drop_duplicates(
        subset=["_chave_data", "mandante", "visitante"],
        keep="last",
    ).drop(columns="_chave_data")

    ids = pd.to_numeric(dados["ID"], errors="coerce")
    proximo_id = int(ids.max()) + 1 if ids.notna().any() else 1
    sem_id = ids.isna()
    dados.loc[sem_id, "ID"] = range(proximo_id, proximo_id + sem_id.sum())
    dados["ID"] = pd.to_numeric(dados["ID"])

    return dados
