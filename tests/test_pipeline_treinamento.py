import pandas as pd

from src.data.loader import carregar_dados
from src.data.cleaner import limpar_dados
from src.features.form import (
    calcular_forma_recente,
    calcular_media_gols,
    calcular_desempenho_casa_fora,
    calcular_elo,
)
from src.models.train_model import (
    construir_features,
    treinar_modelo,
    FEATURES,
)


def test_pipeline_completo_treinamento():
    """
    Testa a integração do pipeline completo de treinamento:

    1. Carregamento dos dados
    2. Limpeza
    3. Engenharia de atributos
    4. Construção das features
    5. Treinamento dos modelos
    6. Geração de probabilidades
    """

    # ==========================================================
    # 1. CARREGAMENTO
    # ==========================================================

    df = carregar_dados()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # ==========================================================
    # 2. LIMPEZA
    # ==========================================================

    df = limpar_dados(df)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # ==========================================================
    # 3. ENGENHARIA DE ATRIBUTOS
    # ==========================================================

    df = calcular_forma_recente(df)

    df = calcular_media_gols(df)

    df = calcular_desempenho_casa_fora(df)

    df = calcular_elo(df)

    # Verifica se as principais features foram criadas
    colunas_forma = [
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
        "elo_mandante_antes",
        "elo_visitante_antes",
        "diferenca_elo",
    ]

    for coluna in colunas_forma:
        assert coluna in df.columns

    # ==========================================================
    # 4. CONSTRUÇÃO DAS FEATURES DOS MODELOS
    # ==========================================================

    df = construir_features(df)

    # Verifica se todas as 28 features existem
    for feature in FEATURES:
        assert feature in df.columns

    # ==========================================================
    # 5. VERIFICAÇÃO DO TARGET
    # ==========================================================

    assert "resultado_encoded" in df.columns

    valores_target = set(df["resultado_encoded"].unique())

    assert valores_target.issubset({0, 1, 2})

    # Precisamos das três classes para treinar o modelo
    assert valores_target == {0, 1, 2}

    # ==========================================================
    # 6. TREINAMENTO
    # ==========================================================

    resultado = treinar_modelo(
        df,
        salvar_modelo=False,
    )

    # ==========================================================
    # 7. VERIFICAR RETORNO DO TREINAMENTO
    # ==========================================================

    assert isinstance(resultado, dict)

    assert "modelo_rf" in resultado
    assert "modelo_logistico" in resultado
    assert "scaler" in resultado
    assert "features" in resultado
    assert "metricas" in resultado

    # ==========================================================
    # 8. VERIFICAR MODELO RANDOM FOREST
    # ==========================================================

    modelo_rf = resultado["modelo_rf"]

    assert hasattr(modelo_rf, "predict")
    assert hasattr(modelo_rf, "predict_proba")

    assert len(modelo_rf.classes_) == 3

    # ==========================================================
    # 9. VERIFICAR REGRESSÃO LOGÍSTICA
    # ==========================================================

    modelo_logistico = resultado["modelo_logistico"]

    assert hasattr(modelo_logistico, "predict")
    assert hasattr(modelo_logistico, "predict_proba")

    assert len(modelo_logistico.classes_) == 3

    # ==========================================================
    # 10. VERIFICAR SCALER
    # ==========================================================

    scaler = resultado["scaler"]

    assert hasattr(scaler, "transform")
    assert hasattr(scaler, "mean_")

    # ==========================================================
    # 11. VERIFICAR AS MÉTRICAS
    # ==========================================================

    metricas = resultado["metricas"]

    assert "baseline" in metricas
    assert "random_forest" in metricas
    assert "regressao_logistica" in metricas

    # Cada modelo deve possuir Brier Score
    assert "brier" in metricas["baseline"]
    assert "brier" in metricas["random_forest"]
    assert "brier" in metricas["regressao_logistica"]