"""
Fase 16 — Visualizações.

Gera os gráficos que documentam os dados e o comportamento dos
modelos: distribuição de resultados, distribuição de gols,
importância de variáveis, comparação de métricas, calibração e
evolução do ELO. Todos salvos como PNG em `resultados/`.

Não recalcula nada -- reaproveita o que já existe em
train_model.py (métricas, calibração) e form.py (colunas de ELO).
"""

import matplotlib
matplotlib.use("Agg")  # não exige um display gráfico para rodar

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CAMINHO_RESULTADOS


def _preparar_pasta():
    CAMINHO_RESULTADOS.mkdir(parents=True, exist_ok=True)


def plotar_distribuicao_resultados(df):
    """
    Item 61 -- barras com a contagem de vitórias do mandante,
    empates e vitórias do visitante.
    """

    _preparar_pasta()

    contagem = df["resultado"].value_counts().reindex(
        ["Mandante", "Empate", "Visitante"]
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        contagem.index,
        contagem.values,
        color=["#2E7D32", "#F9A825", "#C62828"]
    )
    ax.set_title("Distribuição dos resultados")
    ax.set_ylabel("Quantidade de jogos")

    for i, valor in enumerate(contagem.values):
        ax.text(
            i, valor + 60,
            f"{valor}\n({valor / contagem.sum():.1%})",
            ha="center"
        )

    fig.tight_layout()
    caminho = CAMINHO_RESULTADOS / "01_distribuicao_resultados.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


def plotar_distribuicao_gols(df):
    """
    Item 62 -- histograma de gols totais por partida
    (mandante + visitante).
    """

    _preparar_pasta()

    gols_totais = df["mandante_Placar"] + df["visitante_Placar"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        gols_totais,
        bins=range(0, gols_totais.max() + 2),
        color="#1565C0",
        edgecolor="white",
        align="left"
    )
    ax.set_title("Distribuição de gols por partida (total)")
    ax.set_xlabel("Gols na partida (mandante + visitante)")
    ax.set_ylabel("Quantidade de jogos")
    ax.set_xticks(range(0, gols_totais.max() + 1))

    fig.tight_layout()
    caminho = CAMINHO_RESULTADOS / "02_distribuicao_gols.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


def plotar_importancia_variaveis(modelo_rf, features, top_n=10):
    """
    Item 63 -- importância das variáveis do Random Forest.

    Parâmetros
    ----------
    modelo_rf : sklearn.ensemble.RandomForestClassifier
        O modelo treinado (resultado_treino["modelo_rf"]).
    features : list[str]
        Lista de nomes das variáveis, na mesma ordem usada no
        treino (resultado_treino["features"]).
    """

    _preparar_pasta()

    importancias = pd.Series(
        modelo_rf.feature_importances_, index=features
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importancias.index, importancias.values, color="#6A1B9A")
    ax.set_title(f"Top {top_n} variáveis mais importantes (Random Forest)")
    ax.set_xlabel("Importância")

    fig.tight_layout()
    caminho = CAMINHO_RESULTADOS / "03_importancia_variaveis.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


def plotar_comparacao_modelos(metricas):
    """
    Item 64 -- Acurácia, Log Loss e Brier Score lado a lado,
    comparando Baseline, Random Forest e Regressão Logística.

    Parâmetros
    ----------
    metricas : dict
        resultado_treino["metricas"], devolvido por
        treinar_modelo().
    """

    _preparar_pasta()

    modelos = ["baseline", "random_forest", "regressao_logistica"]
    nomes = ["Baseline", "Random\nForest", "Regressão\nLogística"]

    fig, eixos = plt.subplots(1, 3, figsize=(13, 4))

    valores_acc = [metricas[m]["acuracia"] for m in modelos]
    eixos[0].bar(nomes, valores_acc, color="#00897B")
    eixos[0].set_title("Acurácia (maior = melhor)")

    valores_logloss = [metricas[m]["log_loss"] for m in modelos]
    eixos[1].bar(nomes, valores_logloss, color="#E65100")
    eixos[1].set_title("Log Loss (menor = melhor)")

    valores_brier = [metricas[m]["brier"] for m in modelos]
    eixos[2].bar(nomes, valores_brier, color="#AD1457")
    eixos[2].set_title("Brier Score (menor = melhor)")

    fig.suptitle("Comparação dos modelos")
    fig.tight_layout()
    caminho = CAMINHO_RESULTADOS / "04_comparacao_modelos.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


def plotar_calibracao(tabelas_calibracao):
    """
    Item 65 -- curva de calibração (reliability diagram):
    probabilidade prevista x frequência observada, para a classe
    "Vitória do Mandante", comparando os modelos.

    Parâmetros
    ----------
    tabelas_calibracao : dict[str, pandas.DataFrame]
        resultado_treino["calibracao_mandante"], devolvido por
        treinar_modelo() (saída de curva_calibracao() por modelo).
    """

    _preparar_pasta()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="gray", label="Calibração perfeita"
    )

    cores = {
        "Baseline": "#616161",
        "Random Forest": "#2E7D32",
        "Regressão Logística": "#1565C0",
    }

    for nome_modelo, tabela in tabelas_calibracao.items():
        if tabela.empty:
            continue

        ax.plot(
            tabela["probabilidade_media_prevista"],
            tabela["frequencia_observada"],
            marker="o",
            label=nome_modelo,
            color=cores.get(nome_modelo),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Calibração — Vitória do Mandante")
    ax.legend()

    fig.tight_layout()
    caminho = CAMINHO_RESULTADOS / "05_calibracao.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


def plotar_evolucao_elo(df, times, ano_inicio=None):
    """
    Item 66 -- evolução do rating ELO de alguns times ao longo
    do tempo.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Base já processada por calcular_elo() (precisa das
        colunas elo_mandante_antes / elo_visitante_antes).
    times : list[str]
        Nomes dos times a plotar.
    ano_inicio : int ou None
        Se informado, mostra só a partir dessa temporada.
    """

    _preparar_pasta()

    df = df.sort_values(["data", "hora", "ID"])

    if ano_inicio is not None:
        df = df[df["ano"] >= ano_inicio]

    fig, ax = plt.subplots(figsize=(9, 5))

    for time in times:
        jogos_casa = df[df["mandante"] == time][
            ["data", "elo_mandante_antes"]
        ].rename(columns={"elo_mandante_antes": "elo"})

        jogos_fora = df[df["visitante"] == time][
            ["data", "elo_visitante_antes"]
        ].rename(columns={"elo_visitante_antes": "elo"})

        historico = pd.concat([jogos_casa, jogos_fora]).sort_values("data")

        if historico.empty:
            continue

        ax.plot(historico["data"], historico["elo"], label=time)

    ax.set_title("Evolução do ELO Rating")
    ax.set_xlabel("Data")
    ax.set_ylabel("ELO")
    ax.legend()
    fig.autofmt_xdate()

    fig.tight_layout()
    caminho = CAMINHO_RESULTADOS / "06_evolucao_elo.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

    return caminho


def gerar_todas_visualizacoes(df, resultado_treino, times_elo=None, ano_inicio_elo=2018):
    """
    Gera e salva os 6 gráficos da Fase 16 de uma vez.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Base processada (mesma usada em treinar_modelo()).
    resultado_treino : dict
        O que treinar_modelo() devolve.
    times_elo : list[str] ou None
        Times a mostrar na evolução do ELO. Se None, usa os 5
        times com mais partidas como mandante na base.
    ano_inicio_elo : int
        Primeira temporada mostrada no gráfico de ELO.

    Retorna
    -------
    list[pathlib.Path]
        Caminhos dos 6 arquivos PNG gerados.
    """

    caminhos = [
        plotar_distribuicao_resultados(df),
        plotar_distribuicao_gols(df),
        plotar_importancia_variaveis(
            resultado_treino["modelo_rf"], resultado_treino["features"]
        ),
        plotar_comparacao_modelos(resultado_treino["metricas"]),
        plotar_calibracao(resultado_treino["calibracao_mandante"]),
    ]

    if times_elo is None:
        times_elo = df["mandante"].value_counts().head(5).index.tolist()

    caminhos.append(plotar_evolucao_elo(df, times_elo, ano_inicio=ano_inicio_elo))

    return caminhos


if __name__ == "__main__":

    from src.data.loader import carregar_dados
    from src.data.cleaner import limpar_dados
    from src.features.form import (
        calcular_forma_recente,
        calcular_media_gols,
        calcular_desempenho_casa_fora,
        calcular_elo,
    )
    from src.models.train_model import treinar_modelo

    print("Carregando e preparando dados para as visualizações...")

    df = carregar_dados()
    df = limpar_dados(df)
    df = calcular_forma_recente(df)
    df = calcular_media_gols(df)
    df = calcular_desempenho_casa_fora(df)
    df = calcular_elo(df)

    resultado_treino = treinar_modelo(df, salvar_modelo=False)

    print("\nGerando gráficos...")
    caminhos = gerar_todas_visualizacoes(df, resultado_treino)

    print("\nGráficos salvos em:")
    for caminho in caminhos:
        print(f"  {caminho}")