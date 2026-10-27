import sqlite3
from pathlib import Path

import requests


RAIZ_PROJETO = Path(__file__).resolve().parents[2]
DB_PATH = RAIZ_PROJETO / "dados" / "brasileirao.db"

URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "football.json/master/2026/br.1.json"
)


def importar_jogos():
    print("Buscando jogos do Brasileirão 2026...")

    resposta = requests.get(URL, timeout=30)
    resposta.raise_for_status()

    dados = resposta.json()
    jogos = dados.get("matches", [])

    if not jogos:
        print("Nenhum jogo encontrado. Banco não alterado.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jogos_2026 (
                data TEXT NOT NULL,
                rodada TEXT,
                mandante TEXT NOT NULL,
                visitante TEXT NOT NULL,
                gols_mandante INTEGER,
                gols_visitante INTEGER,
                status TEXT,
                PRIMARY KEY (data, mandante, visitante)
            )
        """)

        for jogo in jogos:
            data = jogo.get("date")
            rodada = jogo.get("round")
            mandante = jogo.get("team1")
            visitante = jogo.get("team2")

            placar = jogo.get("score")
            gols_mandante = None
            gols_visitante = None

            if isinstance(placar, dict):
                gols = placar.get("ft")

                if isinstance(gols, list) and len(gols) == 2:
                    gols_mandante = gols[0]
                    gols_visitante = gols[1]

            elif isinstance(placar, list) and len(placar) == 2:
                gols_mandante = placar[0]
                gols_visitante = placar[1]

            if jogo.get("status") == "postponed":
                status = "ADIADO"
            elif (
                gols_mandante is not None
                and gols_visitante is not None
            ):
                status = "ENCERRADO"
            else:
                status = "SEM_PLACAR"

            conn.execute("""
                INSERT INTO jogos_2026 (
                    data,
                    rodada,
                    mandante,
                    visitante,
                    gols_mandante,
                    gols_visitante,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(data, mandante, visitante)
                DO UPDATE SET
                    rodada = excluded.rodada,
                    gols_mandante = excluded.gols_mandante,
                    gols_visitante = excluded.gols_visitante,
                    status = excluded.status
            """, (
                data,
                rodada,
                mandante,
                visitante,
                gols_mandante,
                gols_visitante,
                status
            ))

    print(f"Jogos encontrados na fonte: {len(jogos)}")
    print("Importação concluída.")
    print("A tabela histórica partidas não foi modificada.")


if __name__ == "__main__":
    try:
        importar_jogos()
    except requests.RequestException as erro:
        print(f"Erro ao consultar a fonte: {erro}")
    except sqlite3.Error as erro:
        print(f"Erro no banco de dados: {erro}")
    except (ValueError, TypeError) as erro:
        print(f"Erro ao interpretar os dados: {erro}")