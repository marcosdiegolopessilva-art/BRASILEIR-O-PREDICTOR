"""
Módulo responsável pelo carregamento dos dados do Brasileirão.
"""

import pandas as pd

from src.config import CAMINHO_DADOS_RAW


def carregar_dados():
    """
    Lê a base de dados do Brasileirão e retorna um DataFrame.

    Levanta
    -------
    FileNotFoundError
        Com uma mensagem indicando o caminho esperado, caso o
        arquivo CSV não seja encontrado (item 59 do roadmap).
    """

    if not CAMINHO_DADOS_RAW.exists():
        raise FileNotFoundError(
            "Não encontrei o arquivo de dados em:\n"
            f"  {CAMINHO_DADOS_RAW}\n\n"
            "Confira se o CSV está na pasta dados/raw/ na raiz "
            f"do projeto, com o nome '{CAMINHO_DADOS_RAW.name}'."
        )

    df = pd.read_csv(CAMINHO_DADOS_RAW)

    return df