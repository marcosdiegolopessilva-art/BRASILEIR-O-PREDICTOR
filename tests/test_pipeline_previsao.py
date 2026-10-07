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
)

from src.models.predict import calcular_previsao


def test_pipeline_completo_previsao():
    """
    Testa a integração completa do pipeline de previsão:

    1. Carregamento dos dados
    2. Limpeza
    3. Engenharia de atributos
    4. Construção das features
    5. Treinamento do modelo
    6. Seleção de dois times válidos
    7. Geração da previsão
    8. Integração Logistic Regression + Poisson + Monte Carlo
    """

    # 1. Carregar dados
    df = carregar_dados()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # 2. Limpar dados
    df = limpar_dados(df)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # 3. Engenharia de atributos
    df = calcular_forma_recente(df)
    df = calcular_media_gols(df)
    df = calcular_desempenho_casa_fora(df)
    df = calcular_elo(df)

    # 4. Construir features
    df = construir_features(df)

    # 5. Treinar modelo
    resultado_treinamento = treinar_modelo(
        df,
        salvar_modelo=False
    )

    modelo = resultado_treinamento["modelo_logistico"]
    scaler = resultado_treinamento["scaler"]

    assert hasattr(modelo, "predict_proba")
    assert hasattr(scaler, "transform")

    # 6. Selecionar dois times existentes na base
    times = sorted(
        set(df["mandante"].unique())
        | set(df["visitante"].unique())
    )

    assert len(times) >= 2

    mandante = times[0]
    visitante = times[1]

    assert mandante != visitante

    # 7. Gerar previsão
    resultado = calcular_previsao(
        modelo=modelo,
        df=df,
        mandante=mandante,
        visitante=visitante,
        scaler=scaler,
    )

    # 8. Verificar estrutura do resultado
    assert isinstance(resultado, dict)

    chaves_esperadas = [
        "prob_mandante",
        "prob_empate",
        "prob_visitante",
        "prob_mandante_poisson",
        "prob_empate_poisson",
        "prob_visitante_poisson",
        "prob_mandante_final",
        "prob_empate_final",
        "prob_visitante_final",
        "previsao",
        "confianca",
        "gols_esperados_mandante",
        "gols_esperados_visitante",
        "placares",
    ]

    for chave in chaves_esperadas:
        assert chave in resultado

    # 9. Verificar probabilidades da Regressão Logística
    probabilidades_logistica = [
        resultado["prob_mandante"],
        resultado["prob_empate"],
        resultado["prob_visitante"],
    ]

    for probabilidade in probabilidades_logistica:
        assert 0 <= probabilidade <= 1

    assert abs(
        sum(probabilidades_logistica) - 1
    ) < 1e-6

    # 10. Verificar probabilidades do Poisson
    probabilidades_poisson = [
        resultado["prob_mandante_poisson"],
        resultado["prob_empate_poisson"],
        resultado["prob_visitante_poisson"],
    ]

    for probabilidade in probabilidades_poisson:
        assert 0 <= probabilidade <= 1

    assert abs(
        sum(probabilidades_poisson) - 1
    ) < 1e-4

    # 11. Verificar probabilidades finais do ensemble
    probabilidades_finais = [
        resultado["prob_mandante_final"],
        resultado["prob_empate_final"],
        resultado["prob_visitante_final"],
    ]

    for probabilidade in probabilidades_finais:
        assert 0 <= probabilidade <= 1

    assert abs(
        sum(probabilidades_finais) - 1
    ) < 1e-4

    # 12. Verificar previsão textual
    assert isinstance(
        resultado["previsao"],
        str
    )

    assert resultado["previsao"] != ""

    # 13. Verificar confiança
    assert 0 <= resultado["confianca"] <= 1

    # A confiança deve corresponder à maior probabilidade final
    assert abs(
        resultado["confianca"]
        - max(probabilidades_finais)
    ) < 1e-6

    # 14. Verificar gols esperados
    assert (
        resultado["gols_esperados_mandante"] >= 0
    )

    assert (
        resultado["gols_esperados_visitante"] >= 0
    )

    # 15. Verificar Monte Carlo
    assert isinstance(
        resultado["placares"],
        list
    )

    assert len(resultado["placares"]) > 0

    for placar, quantidade in resultado["placares"]:
        assert isinstance(placar, tuple)
        assert len(placar) == 2

        assert placar[0] >= 0
        assert placar[1] >= 0

        assert quantidade >= 0