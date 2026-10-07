import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

from src.models.metrics import calcular_brier_multiclasse, comparar_brier, curva_calibracao
from src.config import (
    PROPORCAO_TREINO,
    RF_N_ESTIMATORS,
    RF_CLASS_WEIGHT,
    RF_RANDOM_STATE,
    LOGISTICA_MAX_ITER,
    LOGISTICA_CLASS_WEIGHT,
    LOGISTICA_RANDOM_STATE,
    CAMINHO_MODELOS,
)


# Variáveis usadas pelos modelos. Fica aqui, em nível de módulo,
# para que outros arquivos (como validation.py) usem exatamente
# a mesma lista -- evitando o tipo de divergência treino/inferência
# que já corrigimos antes em predict.py.
FEATURES = [
    "pontos_mandante_antes",
    "pontos_visitante_antes",
    "vitorias_recentes_mandante",
    "empates_recentes_mandante",
    "derrotas_recentes_mandante",
    "vitorias_recentes_visitante",
    "empates_recentes_visitante",
    "derrotas_recentes_visitante",
    "media_gols_marcados_mandante",
    "media_gols_sofridos_mandante",
    "media_gols_marcados_visitante",
    "media_gols_sofridos_visitante",
    "vitorias_casa",
    "empates_casa",
    "derrotas_casa",
    "vitorias_fora",
    "empates_fora",
    "derrotas_fora",
    "diferenca_elo",
    "dif_pontos",
    "dif_vitorias_recentes",
    "dif_empates_recentes",
    "dif_derrotas_recentes",
    "dif_gols_marcados",
    "dif_gols_sofridos",
    "dif_vitorias_casa_fora",
    "dif_empates_casa_fora",
    "dif_derrotas_casa_fora",
]


def construir_features(df):
    """
    Cria as variáveis de diferença (dif_*) usadas pelos modelos,
    a partir das colunas já calculadas em cleaner.py e form.py.

    Extraída como função própria para que treinar_modelo() e o
    Walk-Forward Validation (validation.py) usem exatamente a
    mesma lógica de construção de atributos.
    """

    df = df.copy()

    df["dif_pontos"] = (
        df["pontos_mandante_antes"] - df["pontos_visitante_antes"]
    )
    df["dif_vitorias_recentes"] = (
        df["vitorias_recentes_mandante"] - df["vitorias_recentes_visitante"]
    )
    df["dif_empates_recentes"] = (
        df["empates_recentes_mandante"] - df["empates_recentes_visitante"]
    )
    df["dif_derrotas_recentes"] = (
        df["derrotas_recentes_mandante"] - df["derrotas_recentes_visitante"]
    )
    df["dif_gols_marcados"] = (
        df["media_gols_marcados_mandante"] - df["media_gols_marcados_visitante"]
    )
    df["dif_gols_sofridos"] = (
        df["media_gols_sofridos_mandante"] - df["media_gols_sofridos_visitante"]
    )
    df["dif_vitorias_casa_fora"] = df["vitorias_casa"] - df["vitorias_fora"]
    df["dif_empates_casa_fora"] = df["empates_casa"] - df["empates_fora"]
    df["dif_derrotas_casa_fora"] = df["derrotas_casa"] - df["derrotas_fora"]

    return df


def treinar_modelo(df, salvar_modelo=True):

    print("\nTreinando modelo...")

    # ==========================================
    # ENGENHARIA DE ATRIBUTOS
    # ==========================================

    df = construir_features(df)
    features = FEATURES

    # ==========================================
    # ORDENAR CRONOLOGICAMENTE E DIVIDIR
    # ==========================================

    df = df.sort_values(["data", "hora", "ID"]).reset_index(drop=True)

    X = df[features]
    y = df["resultado_encoded"]

    ponto_corte = int(len(df) * PROPORCAO_TREINO)

    X_treino = X.iloc[:ponto_corte]
    X_teste = X.iloc[ponto_corte:]
    y_treino = y.iloc[:ponto_corte]
    y_teste = y.iloc[ponto_corte:]

    print("\nDivisão temporal:")
    print(f"Treino: {len(X_treino)} jogos")
    print(f"Teste: {len(X_teste)} jogos")

    # ==========================================
    # BASELINE PROBABILÍSTICO
    # (calculado ANTES dos modelos, pois RF e Regressão
    # Logística são comparados contra ele)
    # ==========================================

    frequencias = y_treino.value_counts(normalize=True)

    prob_baseline_empate = frequencias.get(0, 0)
    prob_baseline_mandante = frequencias.get(1, 0)
    prob_baseline_visitante = frequencias.get(2, 0)

    classe_baseline = frequencias.idxmax()

    previsoes_baseline = pd.Series(classe_baseline, index=y_teste.index)

    acuracia_baseline = accuracy_score(y_teste, previsoes_baseline)

    probabilidades_baseline = pd.DataFrame({
        0: prob_baseline_empate,
        1: prob_baseline_mandante,
        2: prob_baseline_visitante
    }, index=y_teste.index)

    logloss_baseline = log_loss(
        y_teste, probabilidades_baseline, labels=[0, 1, 2]
    )

    brier_baseline = calcular_brier_multiclasse(
        y_teste, probabilidades_baseline.values
    )

    print("\n" + "=" * 50)
    print("BASELINE PROBABILÍSTICO")
    print("=" * 50)
    print("\nProbabilidades baseadas no conjunto de treino:")
    print(f"  Empate: {prob_baseline_empate:.2%}")
    print(f"  Mandante: {prob_baseline_mandante:.2%}")
    print(f"  Visitante: {prob_baseline_visitante:.2%}")
    print(f"\nAcurácia: {acuracia_baseline:.2%}")
    print(f"Log Loss: {logloss_baseline:.4f}")
    print(f"Brier Score: {brier_baseline:.4f}")

    # ==========================================
    # RANDOM FOREST
    # ==========================================

    modelo = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1
    )

    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    probabilidades = modelo.predict_proba(X_teste)

    acuracia = accuracy_score(y_teste, previsoes)
    logloss = log_loss(y_teste, probabilidades, labels=modelo.classes_)
    brier_random_forest = calcular_brier_multiclasse(y_teste, probabilidades)

    # ==========================================
    # REGRESSÃO LOGÍSTICA
    # ==========================================

    scaler = StandardScaler()

    X_treino_logistico = scaler.fit_transform(X_treino)
    X_teste_logistico = scaler.transform(X_teste)

    # class_weight=None (sem balanceamento): ver comentário em
    # src/config.py -- validado no Walk-Forward Validation.
    modelo_logistico = LogisticRegression(
        max_iter=LOGISTICA_MAX_ITER,
        class_weight=LOGISTICA_CLASS_WEIGHT,
        random_state=LOGISTICA_RANDOM_STATE
    )

    modelo_logistico.fit(X_treino_logistico, y_treino)

    previsoes_logistico = modelo_logistico.predict(X_teste_logistico)
    probabilidades_logistico = modelo_logistico.predict_proba(X_teste_logistico)

    acuracia_logistico = accuracy_score(y_teste, previsoes_logistico)
    f1_logistico = f1_score(y_teste, previsoes_logistico, average="macro")
    logloss_logistico = log_loss(
        y_teste, probabilidades_logistico, labels=modelo_logistico.classes_
    )
    brier_logistico = calcular_brier_multiclasse(
        y_teste, probabilidades_logistico
    )

    # ==========================================
    # COMPARAÇÃO DOS MODELOS
    # ==========================================

    print("\n" + "=" * 50)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 50)

    print("\nRandom Forest:")
    print(f"  Acurácia: {acuracia:.2%}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier_random_forest:.4f}")

    print("\nRegressão Logística:")
    print(f"  Acurácia: {acuracia_logistico:.2%}")
    print(f"  F1 Macro: {f1_logistico:.2f}")
    print(f"  Log Loss: {logloss_logistico:.4f}")
    print(f"  Brier Score: {brier_logistico:.4f}")

    # ==========================================
    # PROBABILIDADES DOS 5 PRIMEIROS JOGOS
    # ==========================================

    print("\nProbabilidades dos 5 primeiros jogos (%):")

    probabilidades_percentual = (probabilidades[:5] * 100).round(1)

    probabilidades_df = pd.DataFrame(
        probabilidades_percentual,
        columns=["Prob_Classe_0", "Prob_Classe_1", "Prob_Classe_2"]
    )

    print(probabilidades_df.map(lambda x: f"{x:.1f}%"))

    # ==========================================
    # IMPORTÂNCIA DAS VARIÁVEIS
    # ==========================================

    importancias = pd.DataFrame({
        "Feature": features,
        "Importancia": modelo.feature_importances_
    }).sort_values("Importancia", ascending=False)

    print("\nTop 10 Variáveis Mais Importantes:")
    print(importancias.head(10).to_string(index=False))

    # ==========================================
    # MATRIZ DE CONFUSÃO E RELATÓRIO
    # ==========================================

    matriz = confusion_matrix(y_teste, previsoes)
    print("\nMatriz de Confusão:")
    print(matriz)

    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, previsoes, zero_division=0))

    # ==========================================
    # RANKING POR BRIER SCORE (novo, via src/models/metrics.py)
    # ==========================================

    ranking_brier = comparar_brier(
        y_teste,
        {
            "Baseline": probabilidades_baseline.values,
            "Random Forest": probabilidades,
            "Regressão Logística": probabilidades_logistico,
        }
    )

    print("\n" + "=" * 50)
    print("RANKING POR BRIER SCORE (menor = melhor)")
    print("=" * 50)
    print(ranking_brier.to_string(index=False))

    # ==========================================
    # CALIBRAÇÃO — VITÓRIA DO MANDANTE (classe 1)
    # ==========================================
    # Focamos na classe "Mandante" por ser a mais frequente e
    # a mais relevante na hora de comunicar a previsão. A mesma
    # função (curva_calibracao) serve para as outras classes.

    tabelas_calibracao = {}

    print("\n" + "=" * 50)
    print("CALIBRAÇÃO — VITÓRIA DO MANDANTE (classe 1)")
    print("=" * 50)

    for nome_modelo, probs in [
        ("Baseline", probabilidades_baseline.values),
        ("Random Forest", probabilidades),
        ("Regressão Logística", probabilidades_logistico),
    ]:
        tabela = curva_calibracao(y_teste, probs, classe=1)
        tabelas_calibracao[nome_modelo] = tabela

        print(f"\n{nome_modelo}:")
        print(tabela.to_string(index=False))

    # ==========================================
    # PERSISTÊNCIA DOS MODELOS (novo)
    # ==========================================

    if salvar_modelo:
        CAMINHO_MODELOS.mkdir(parents=True, exist_ok=True)

        joblib.dump(modelo, CAMINHO_MODELOS / "random_forest.joblib")
        joblib.dump(modelo_logistico, CAMINHO_MODELOS / "regressao_logistica.joblib")
        joblib.dump(scaler, CAMINHO_MODELOS / "scaler.joblib")

        print(f"\nModelos salvos em: {CAMINHO_MODELOS.resolve()}")

    # ==========================================
    # RETORNO (agora devolve tudo, não só o modelo RF)
    # ==========================================

    return {
        "modelo_rf": modelo,
        "modelo_logistico": modelo_logistico,
        "scaler": scaler,
        "features": features,
        "calibracao_mandante": tabelas_calibracao,
        "metricas": {
            "baseline": {
                "acuracia": acuracia_baseline,
                "log_loss": logloss_baseline,
                "brier": brier_baseline,
            },
            "random_forest": {
                "acuracia": acuracia,
                "log_loss": logloss,
                "brier": brier_random_forest,
            },
            "regressao_logistica": {
                "acuracia": acuracia_logistico,
                "f1_macro": f1_logistico,
                "log_loss": logloss_logistico,
                "brier": brier_logistico,
            },
        },
    }


def treinar_modelo_producao(df):
    """
    Treina o modelo FINAL (Regressão Logística) usando TODA a
    base disponível, sem reservar conjunto de teste.

    Diferença em relação a treinar_modelo(): aquela função
    reserva 20% dos dados para medir Brier Score, calibração
    etc. -- correto para VALIDAR a metodologia, o que já fizemos
    nas Fases 9 a 13. Mas para o modelo que efetivamente faz
    previsões reais (usado na interface, Fase 17), não faz
    sentido deixar de fora justamente os jogos mais recentes --
    eles são os mais relevantes para refletir a forma atual dos
    times. Depois que a metodologia está validada, o modelo de
    produção deve usar 100% dos dados.

    Parâmetros
    ----------
    df : pandas.DataFrame
        Base processada pelo pipeline padrão (mesma entrada de
        treinar_modelo()).

    Retorna
    -------
    dict
        {"modelo_logistico": ..., "scaler": ..., "features": ...}
    """

    print("\nTreinando modelo de PRODUÇÃO (100% dos dados, sem split de teste)...")

    df = construir_features(df)
    features = FEATURES

    df = df.sort_values(["data", "hora", "ID"]).reset_index(drop=True)

    X = df[features]
    y = df["resultado_encoded"]

    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    # Mesma configuração validada nas Fases 11-13: sem
    # balanceamento de classes (ver src/config.py).
    modelo_logistico = LogisticRegression(
        max_iter=LOGISTICA_MAX_ITER,
        class_weight=LOGISTICA_CLASS_WEIGHT,
        random_state=LOGISTICA_RANDOM_STATE,
    )
    modelo_logistico.fit(X_escalado, y)

    print(f"Modelo de produção treinado com {len(df)} jogos (100% da base).")

    CAMINHO_MODELOS.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        modelo_logistico, CAMINHO_MODELOS / "regressao_logistica_producao.joblib"
    )
    joblib.dump(scaler, CAMINHO_MODELOS / "scaler_producao.joblib")

    print(f"Modelo de produção salvo em: {CAMINHO_MODELOS.resolve()}")

    return {
        "modelo_logistico": modelo_logistico,
        "scaler": scaler,
        "features": features,
    }