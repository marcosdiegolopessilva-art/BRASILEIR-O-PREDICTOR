import pandas as pd

from src.data.loader import carregar_dados
from src.data.cleaner import limpar_dados


def test_pipeline_completo_dados():
    """
    Testa a integração completa do pipeline de dados:

    1. Carregamento do CSV
    2. Limpeza dos dados
    3. Criação das variáveis derivadas
    4. Retorno de um DataFrame válido
    """

    # 1. Carregar os dados brutos
    df_bruto = carregar_dados()

    # Verificar se o carregamento retornou um DataFrame
    assert isinstance(df_bruto, pd.DataFrame)

    # Verificar se a base não está vazia
    assert not df_bruto.empty

    # Verificar algumas colunas essenciais da base original
    colunas_esperadas = [
        "data",
        "mandante",
        "visitante",
        "mandante_Placar",
        "visitante_Placar",
    ]

    for coluna in colunas_esperadas:
        assert coluna in df_bruto.columns

    # 2. Executar a limpeza
    df_processado = limpar_dados(df_bruto.copy())

    # 3. Verificar se o resultado continua sendo um DataFrame
    assert isinstance(df_processado, pd.DataFrame)

    # Verificar se o DataFrame processado não está vazio
    assert not df_processado.empty

    # 4. Verificar variáveis criadas pela limpeza
    colunas_criadas = [
        "resultado",
        "resultado_encoded",
        "saldo_gols",
        "ano",
        "mes",
        "dia_semana",
        "pontos_mandante_antes",
        "pontos_visitante_antes",
    ]

    for coluna in colunas_criadas:
        assert coluna in df_processado.columns

    # 5. Verificar se não existem datas inválidas
    assert pd.api.types.is_datetime64_any_dtype(
        df_processado["data"]
    )

    # 6. Verificar se o resultado possui apenas as três categorias esperadas
    resultados_validos = {
        "Mandante",
        "Visitante",
        "Empate",
    }

    assert set(df_processado["resultado"].unique()).issubset(
        resultados_validos
    )

    # 7. Verificar codificação dos resultados
    assert set(df_processado["resultado_encoded"].unique()).issubset(
        {0, 1, 2}
    )

    # 8. Verificar se o saldo de gols foi calculado corretamente
    saldo_calculado = (
        df_processado["mandante_Placar"]
        - df_processado["visitante_Placar"]
    )

    assert (
        df_processado["saldo_gols"] == saldo_calculado
    ).all()

    # 9. Verificar se os pontos históricos não são negativos
    assert (
        df_processado["pontos_mandante_antes"] >= 0
    ).all()

    assert (
        df_processado["pontos_visitante_antes"] >= 0
    ).all()