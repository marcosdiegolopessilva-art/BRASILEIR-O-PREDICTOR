"""
Métricas de avaliação probabilística para os modelos do
Brasileirão Predictor.

Diferente de acurácia/F1 (que só avaliam se a classe prevista
está certa), as métricas aqui avaliam a QUALIDADE das
PROBABILIDADES previstas -- o que é o objetivo central deste
projeto, já que o produto final é "Flamengo 40% / Empate 43,5%
/ Palmeiras 16,5%", não só "Flamengo vence".
"""

import numpy as np
import pandas as pd

# 0 = Empate, 1 = Mandante, 2 = Visitante
CLASSES = [0, 1, 2]


def calcular_brier_multiclasse(y_real, probabilidades):
    """
    Calcula o Brier Score multiclasse.

    O Brier Score mede o erro quadrático médio entre a
    probabilidade prevista e o resultado real (1 para a classe
    que ocorreu, 0 para as demais). Quanto MENOR, melhor:

    - 0.0   -> previsões perfeitas (100% de confiança na classe certa)
    - ~0.67 -> equivalente a "chutar" 33%/33%/33% sempre
    - 2.0   -> pior caso possível (100% de confiança na classe errada)

    Parâmetros
    ----------
    y_real : array-like
        Rótulos reais, com valores em {0, 1, 2}.
    probabilidades : array-like, shape (n_amostras, 3)
        Probabilidades previstas para as classes 0, 1 e 2,
        nessa ordem de colunas.

    Retorna
    -------
    float
        Brier Score médio sobre todas as amostras.

    Levanta
    -------
    ValueError
        Se o formato de `probabilidades` não bater com o
        número de amostras de `y_real` x 3 classes.
    """

    resultados_reais = pd.get_dummies(
        pd.Series(y_real).reset_index(drop=True)
    ).reindex(
        columns=CLASSES,
        fill_value=0
    ).values

    probabilidades = np.asarray(probabilidades)

    if probabilidades.shape != resultados_reais.shape:
        raise ValueError(
            "Formato incompatível: esperava "
            f"{resultados_reais.shape} (n_amostras x 3 classes), "
            f"recebi {probabilidades.shape}. Confira se "
            "`probabilidades` tem uma coluna por classe (0, 1, 2)."
        )

    erro_quadratico = (probabilidades - resultados_reais) ** 2

    return erro_quadratico.sum(axis=1).mean()


def comparar_brier(y_real, probabilidades_por_modelo):
    """
    Compara o Brier Score de vários modelos de uma só vez.

    Parâmetros
    ----------
    y_real : array-like
        Rótulos reais do conjunto de teste.
    probabilidades_por_modelo : dict[str, array-like]
        Dicionário no formato {"Nome do modelo": probabilidades}.
        Ex.: {"Baseline": ..., "Random Forest": ..., "Regressão Logística": ...}

    Retorna
    -------
    pandas.DataFrame
        Tabela com uma linha por modelo, ordenada do melhor
        (menor Brier Score) para o pior.
    """

    linhas = []

    for nome_modelo, probabilidades in probabilidades_por_modelo.items():
        try:
            brier = calcular_brier_multiclasse(y_real, probabilidades)
        except ValueError as erro:
            raise ValueError(
                f"Erro ao calcular Brier Score para '{nome_modelo}': {erro}"
            ) from erro

        linhas.append({"modelo": nome_modelo, "brier_score": brier})

    tabela = (
        pd.DataFrame(linhas)
        .sort_values("brier_score")
        .reset_index(drop=True)
    )

    return tabela


def curva_calibracao(y_real, probabilidades, classe, n_bins=10):
    """
    Calcula os dados de uma curva de calibração (reliability
    diagram) para UMA classe específica.

    A lógica: agrupamos as previsões em faixas (bins) de acordo
    com a probabilidade que o modelo previu para `classe`, e
    comparamos, dentro de cada faixa:

    - probabilidade_media_prevista: o que o modelo "disse"
    - frequencia_observada: o que realmente aconteceu

    Um modelo bem calibrado tem essas duas colunas próximas em
    todas as faixas.

    Parâmetros
    ----------
    y_real : array-like
        Rótulos reais (valores em {0, 1, 2}).
    probabilidades : array-like, shape (n_amostras, 3)
        Probabilidades previstas para as 3 classes.
    classe : int
        Classe a avaliar (0=Empate, 1=Mandante, 2=Visitante).
    n_bins : int
        Número de faixas de probabilidade (padrão 10 = decis).

    Retorna
    -------
    pandas.DataFrame
        Uma linha por faixa com dados suficientes, ordenada da
        faixa de menor para a de maior probabilidade prevista.
        Faixas sem nenhuma amostra são omitidas.
    """

    y_real = pd.Series(y_real).reset_index(drop=True)
    probabilidades = np.asarray(probabilidades)

    if classe not in CLASSES:
        raise ValueError(
            f"Classe inválida: {classe}. Use um valor em {CLASSES}."
        )

    prob_classe = probabilidades[:, classe]
    ocorreu = (y_real == classe).astype(int).values

    bordas = np.linspace(0, 1, n_bins + 1)
    indices_bin = np.clip(
        np.digitize(prob_classe, bordas) - 1, 0, n_bins - 1
    )

    linhas = []

    for i in range(n_bins):
        mascara = indices_bin == i
        quantidade = int(mascara.sum())

        if quantidade == 0:
            continue

        linhas.append({
            "faixa": f"{bordas[i]:.0%}-{bordas[i + 1]:.0%}",
            "probabilidade_media_prevista": prob_classe[mascara].mean(),
            "frequencia_observada": ocorreu[mascara].mean(),
            "quantidade_amostras": quantidade,
        })

    return pd.DataFrame(linhas)