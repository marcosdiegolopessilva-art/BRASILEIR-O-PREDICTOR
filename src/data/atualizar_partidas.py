import os
import sqlite3
from pathlib import Path

import requests
from dotenv import load_dotenv

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
DB_PATH = RAIZ_PROJETO / "dados" / "brasileirao.db"

load_dotenv(RAIZ_PROJETO / ".env")
API_KEY = os.getenv("API_FOOTBALL_KEY")

def atualizar_partidas_recentes():
    if not API_KEY:
        print("ERRO: configure API_FOOTBALL_KEY no arquivo .env")
        return

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {"league": 71, "season": 2024}

    try:
        resposta = requests.get(
            url, headers=headers, params=params, timeout=30
        )
        resposta.raise_for_status()
        dados = resposta.json()

        if dados.get("errors"):
            print("Erro informado pela API:", dados["errors"])
            return

        partidas = dados.get("response", [])
        if not partidas:
            print("Nenhuma partida retornada pela API.")
            return

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS partidas_api (
                    fixture_id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL,
                    mandante TEXT NOT NULL,
                    visitante TEXT NOT NULL,
                    mandante_placar INTEGER,
                    visitante_placar INTEGER,
                    status TEXT,
                    temporada INTEGER
                )
            """)

            for item in partidas:
                fixture = item["fixture"]
                equipes = item["teams"]
                gols = item["goals"]

                conn.execute("""
                    INSERT INTO partidas_api (
                        fixture_id, data, mandante, visitante,
                        mandante_placar, visitante_placar,
                        status, temporada
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fixture_id) DO UPDATE SET
                        data = excluded.data,
                        mandante_placar = excluded.mandante_placar,
                        visitante_placar = excluded.visitante_placar,
                        status = excluded.status
                """, (
                    fixture["id"],
                    fixture["date"],
                    equipes["home"]["name"],
                    equipes["away"]["name"],
                    gols["home"],
                    gols["away"],
                    fixture["status"]["short"],
                    2026
                ))

        print(f"Atualização concluída: {len(partidas)} partidas processadas.")
        print("A tabela histórica partidas não foi substituída.")

    except requests.RequestException as erro:
        print("Erro na comunicação com a API:", erro)
    except (KeyError, ValueError) as erro:
        print("Erro ao interpretar os dados recebidos:", erro)

if __name__ == "__main__":
    atualizar_partidas_recentes()