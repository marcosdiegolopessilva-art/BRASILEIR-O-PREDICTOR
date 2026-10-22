from dotenv import load_dotenv
import os
import requests

# Carrega as variáveis do arquivo .env na raiz do projeto
load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

URL = "https://v3.football.api-sports.io/fixtures"

PARAMETROS = {
    "league": 71,  # Campeonato Brasileiro Série A
    "season": 2024,
}

if not API_KEY:
    print("ERRO: API_FOOTBALL_KEY não foi configurada.")
    raise SystemExit(1)

headers = {
    "x-apisports-key": API_KEY
}

try:
    resposta = requests.get(
        URL,
        headers=headers,
        params=PARAMETROS,
        timeout=30
    )

    print("Código HTTP:", resposta.status_code)
    resposta.raise_for_status()

    dados = resposta.json()

    if dados.get("errors"):
        print("Erros informados pela API:", dados["errors"])
        raise SystemExit(1)

    partidas = dados.get("response", [])

    print("Total de partidas retornadas:", len(partidas))

    for item in partidas[:10]:
        partida = item["fixture"]
        equipes = item["teams"]
        gols = item["goals"]

        print("\nData:", partida["date"])
        print("Mandante:", equipes["home"]["name"])
        print("Visitante:", equipes["away"]["name"])
        print("Placar:", gols["home"], "x", gols["away"])
        print("Status:", partida["status"]["short"])

except requests.exceptions.RequestException as erro:
    print("Falha na comunicação com a API:", erro)