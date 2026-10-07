from src.data.loader import carregar_dados
from src.data.cleaner import limpar_dados
from src.features.form import calcular_forma_recente
from src.features.form import calcular_media_gols
from src.features.form import calcular_desempenho_casa_fora
from src.features.form import calcular_elo
from src.models.train_model import treinar_modelo, treinar_modelo_producao
from src.models.predict import prever_partida


def mostrar_cabecalho():
    print("=" * 50)
    print("        BRASILEIRÃO PREDICTOR")
    print("=" * 50)


def iniciar_projeto():
    print("Projeto iniciado com sucesso!")
    print()


def main():

    # Exibe o cabeçalho
    mostrar_cabecalho()
    iniciar_projeto()

    # ==========================================
    # CARREGAMENTO DOS DADOS
    # ==========================================

    df = carregar_dados()

    # ==========================================
    # LIMPEZA DOS DADOS
    # ==========================================

    df = limpar_dados(df)

    print("\n==========================================")
    print("DISTRIBUIÇÃO DOS RESULTADOS")
    print("==========================================")

    print(df["resultado"].value_counts())

    print("\nPercentual:")
    print(
        (df["resultado"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # ==========================================
    # FORMA RECENTE DOS TIMES
    # ==========================================

    df = calcular_forma_recente(df)

    # ==========================================
    # MÉDIA DE GOLS RECENTES
    # ==========================================

    df = calcular_media_gols(df)

    # ==========================================
    # DESEMPENHO CASA/FORA
    # ==========================================

    df = calcular_desempenho_casa_fora(df)

    df = calcular_elo(df)

    # treinar_modelo() valida a metodologia (split 80/20,
    # métricas, calibração) -- já rodado e confirmado nas Fases
    # 9-13. Mantido aqui só para exibir o relatório de validação.
    resultado_treino = treinar_modelo(df)

    # O modelo que efetivamente prevê a partida usa 100% dos
    # dados (produção), não o split de validação -- ver
    # docstring de treinar_modelo_producao() em train_model.py.
    resultado_producao = treinar_modelo_producao(df)
    modelo = resultado_producao["modelo_logistico"]
    scaler = resultado_producao["scaler"]

    # ==========================================
    # TESTE DE PREVISÃO
    # ==========================================

    prever_partida(
        modelo,
        df,
        "Flamengo",
        "Palmeiras",
        scaler=scaler,
        nome_modelo="Regressão Logística"
    )

    print("\nBase carregada e processada com sucesso!")

    # ==========================================
    # PRIMEIRAS LINHAS
    # ==========================================

    print("\nPrimeiras linhas da base:")
    print(df.head())

    # ==========================================
    # INFORMAÇÕES DA BASE
    # ==========================================

    print("\n" + "=" * 50)
    print("INFORMAÇÕES DA BASE")
    print("=" * 50)

    print(f"\nDimensões: {df.shape}")

    print("\nColunas:")
    print(df.columns)

    print("\nTipos das colunas:")
    print(df.dtypes)

    print("\nValores ausentes:")
    print(df.isnull().sum())

    print("\nLinhas duplicadas:")
    print(df.duplicated().sum())

    print("\nInformações gerais:")
    df.info()

    # ==========================================
    # RELATÓRIO DE LIMPEZA
    # ==========================================

    print("\n" + "=" * 50)
    print("RELATÓRIO DE LIMPEZA")
    print("=" * 50)

    print(f"Total de linhas: {df.shape[0]}")
    print(f"Total de colunas: {df.shape[1]}")
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    print(
        f"Total de valores ausentes: "
        f"{df.isnull().sum().sum()}"
    )


if __name__ == "__main__":
    main()