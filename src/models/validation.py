"""
Walk-Forward Validation para o Brasileirão Predictor.

Diferente do corte único 80/20 usado em train_model.py, aqui
simulamos o uso real do modelo ao longo do tempo: para cada
temporada X, treinamos com TODAS as temporadas anteriores e
testamos apenas na temporada X. Depois avançamos: X+1 entra no
treino, e testamos em X+2. E assim por diante.

Isso responde uma pergunta que um único split não responde: o
modelo funciona de forma consistente, ou só "acertou" naquele
recorte específico de teste?
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

from src.models.metrics import calcular_brier_multiclasse
from src.models.train_model import construir_features, FEATURES


def walk_forward_validation(df, temporada_inicial_teste=None, class_weight="balanced"):
    """
    Executa a validação Walk-Forward, temporada a temporada.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Base já processada (mesma entrada que treinar_modelo()
        espera: saída de limpar_dados() + calcular_forma_recente()
        + calcular_media_gols() + calcular_desempenho_casa_fora()
        + calcular_elo()).
    temporada_inicial_teste : int ou None
        Primeira temporada a ser usada como teste. As temporadas
        anteriores a ela formam o treino inicial. Se None, usa a
        6ª temporada disponível na base (dá tempo do ELO e das
        médias recentes se estabilizarem antes de avaliar).
    class_weight : str ou None
        Passado direto para RandomForestClassifier e
        LogisticRegression -- "balanced" ou None.

    Retorna
    -------
    pandas.DataFrame
        Uma linha por temporada testada, com acurácia, Log Loss
        e Brier Score do Baseline, Random Forest e Regressão
        Logística.
    """

    df = construir_features(df)
    df = df.sort_values(["data", "hora", "ID"]).reset_index(drop=True)

    temporadas = sorted(df["ano"].unique())

    if len(temporadas) < 2:
        raise ValueError(
            "A base precisa ter pelo menos 2 temporadas para "
            "rodar o Walk-Forward Validation."
        )

    if temporada_inicial_teste is None:
        indice_inicial = min(5, len(temporadas) - 1)
        temporada_inicial_teste = temporadas[indice_inicial]

    resultados = []

    for temporada_teste in temporadas:

        if temporada_teste < temporada_inicial_teste:
            continue

        treino = df[df["ano"] < temporada_teste]
        teste = df[df["ano"] == temporada_teste]

        if treino.empty or teste.empty:
            continue

        X_treino = treino[FEATURES]
        y_treino = treino["resultado_encoded"]
        X_teste = teste[FEATURES]
        y_teste = teste["resultado_encoded"]

        # ---- Baseline ----

        frequencias = y_treino.value_counts(normalize=True)
        classe_baseline = frequencias.idxmax()

        probs_baseline = pd.DataFrame(
            {c: frequencias.get(c, 0) for c in [0, 1, 2]},
            index=y_teste.index,
        )[[0, 1, 2]].values

        preds_baseline = pd.Series(classe_baseline, index=y_teste.index)

        acuracia_baseline = accuracy_score(y_teste, preds_baseline)
        logloss_baseline = log_loss(y_teste, probs_baseline, labels=[0, 1, 2])
        brier_baseline = calcular_brier_multiclasse(y_teste, probs_baseline)

        # ---- Random Forest ----

        modelo_rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        )
        modelo_rf.fit(X_treino, y_treino)

        probs_rf = modelo_rf.predict_proba(X_teste)
        preds_rf = modelo_rf.predict(X_teste)

        acuracia_rf = accuracy_score(y_teste, preds_rf)
        logloss_rf = log_loss(y_teste, probs_rf, labels=modelo_rf.classes_)
        brier_rf = calcular_brier_multiclasse(y_teste, probs_rf)

        # ---- Regressão Logística ----

        scaler = StandardScaler()
        X_treino_escalado = scaler.fit_transform(X_treino)
        X_teste_escalado = scaler.transform(X_teste)

        modelo_logistico = LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=42,
        )
        modelo_logistico.fit(X_treino_escalado, y_treino)

        probs_logistico = modelo_logistico.predict_proba(X_teste_escalado)
        preds_logistico = modelo_logistico.predict(X_teste_escalado)

        acuracia_logistico = accuracy_score(y_teste, preds_logistico)
        logloss_logistico = log_loss(
            y_teste, probs_logistico, labels=modelo_logistico.classes_
        )
        brier_logistico = calcular_brier_multiclasse(y_teste, probs_logistico)

        resultados.append({
            "temporada_teste": temporada_teste,
            "jogos_treino": len(treino),
            "jogos_teste": len(teste),
            "brier_baseline": brier_baseline,
            "brier_rf": brier_rf,
            "brier_logistica": brier_logistico,
            "acuracia_baseline": acuracia_baseline,
            "acuracia_rf": acuracia_rf,
            "acuracia_logistica": acuracia_logistico,
            "logloss_baseline": logloss_baseline,
            "logloss_rf": logloss_rf,
            "logloss_logistica": logloss_logistico,
        })

        print(
            f"Temporada {temporada_teste}: "
            f"treino={len(treino)} jogos, teste={len(teste)} jogos "
            f"-> Brier RF={brier_rf:.4f} | Baseline={brier_baseline:.4f}"
        )

    return pd.DataFrame(resultados)


def resumir_walk_forward(tabela):
    """
    Resume os resultados do Walk-Forward: média e desvio-padrão
    do Brier Score e da acurácia de cada modelo entre as
    temporadas testadas.

    O desvio-padrão importa tanto quanto a média: um modelo com
    média boa mas desvio alto é instável -- funciona bem em
    algumas temporadas e mal em outras, o que é mais arriscado
    do que uma média discreta porém consistente.

    Parâmetros
    ----------
    tabela : pandas.DataFrame
        Saída de walk_forward_validation().

    Retorna
    -------
    pandas.DataFrame
        Uma linha por modelo, ordenada do melhor (menor Brier
        médio) para o pior.
    """

    linhas = []

    modelos = [
        ("Baseline", "baseline"),
        ("Random Forest", "rf"),
        ("Regressão Logística", "logistica"),
    ]

    for nome_modelo, sufixo in modelos:
        linhas.append({
            "modelo": nome_modelo,
            "brier_medio": tabela[f"brier_{sufixo}"].mean(),
            "brier_desvio": tabela[f"brier_{sufixo}"].std(),
            "acuracia_media": tabela[f"acuracia_{sufixo}"].mean(),
            "acuracia_desvio": tabela[f"acuracia_{sufixo}"].std(),
        })

    return (
        pd.DataFrame(linhas)
        .sort_values("brier_medio")
        .reset_index(drop=True)
    )


def comparar_class_weight(df, temporada_inicial_teste=None):
    """
    Roda o Walk-Forward duas vezes -- uma com
    class_weight="balanced" (configuração atual) e outra com
    class_weight=None (sem balanceamento) -- e compara os
    resumos lado a lado.

    Existe porque a curva de calibração (Fase 9) mostrou os
    dois modelos SUBESTIMANDO sistematicamente a classe
    "Mandante" (a majoritária), e balanceamento de classes é o
    principal suspeito: ele força o modelo a tratar as 3 classes
    como se fossem igualmente prováveis durante o treino, o que
    distorce a probabilidade justamente da classe mais frequente.

    Retorna
    -------
    dict
        {
            "balanced": (tabela_por_temporada, resumo),
            "sem_balanceamento": (tabela_por_temporada, resumo),
        }
    """

    print("\n" + "#" * 70)
    print("# RODADA 1: class_weight='balanced' (configuração atual)")
    print("#" * 70)
    tabela_balanced = walk_forward_validation(
        df, temporada_inicial_teste, class_weight="balanced"
    )
    resumo_balanced = resumir_walk_forward(tabela_balanced)

    print("\n" + "#" * 70)
    print("# RODADA 2: class_weight=None (sem balanceamento)")
    print("#" * 70)
    tabela_sem_balanceamento = walk_forward_validation(
        df, temporada_inicial_teste, class_weight=None
    )
    resumo_sem_balanceamento = resumir_walk_forward(tabela_sem_balanceamento)

    print("\n" + "=" * 70)
    print("COMPARAÇÃO: BALANCEADO vs SEM BALANCEAMENTO")
    print("=" * 70)

    print("\ncom class_weight='balanced':")
    print(resumo_balanced.to_string(index=False))

    print("\ncom class_weight=None:")
    print(resumo_sem_balanceamento.to_string(index=False))

    return {
        "balanced": (tabela_balanced, resumo_balanced),
        "sem_balanceamento": (tabela_sem_balanceamento, resumo_sem_balanceamento),
    }


if __name__ == "__main__":

    from src.data.loader import carregar_dados
    from src.data.cleaner import limpar_dados
    from src.features.form import (
        calcular_forma_recente,
        calcular_media_gols,
        calcular_desempenho_casa_fora,
        calcular_elo,
    )

    print("Carregando e preparando dados para o Walk-Forward Validation...")

    df = carregar_dados()
    df = limpar_dados(df)
    df = calcular_forma_recente(df)
    df = calcular_media_gols(df)
    df = calcular_desempenho_casa_fora(df)
    df = calcular_elo(df)

    comparar_class_weight(df)