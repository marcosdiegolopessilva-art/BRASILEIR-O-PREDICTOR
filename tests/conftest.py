"""
Fixtures compartilhadas entre os testes (Fase 18 do roadmap).

Os testes NÃO dependem do CSV real (8.785 jogos) -- usam uma base
sintética pequena, gerada aqui, para rodar rápido e sem depender
de um arquivo externo.
"""

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isola_arquivos_gerados(tmp_path, monkeypatch):
    """
    Evita que os testes escrevam em cima dos arquivos reais do
    projeto (modelos/, resultados/, dados/processed/) -- redireciona
    tudo para uma pasta temporária, criada e destruída a cada teste.

    Cada módulo importou as constantes de config.py com
    `from src.config import X`, então o monkeypatch precisa ser
    feito em cada módulo separadamente (from-import copia a
    referência, não reexporta o nome original).
    """

    import src.models.train_model as train_model_module

    monkeypatch.setattr(train_model_module, "CAMINHO_MODELOS", tmp_path / "modelos")

    try:
        import src.visualizations as visualizations_module
        monkeypatch.setattr(
            visualizations_module, "CAMINHO_RESULTADOS", tmp_path / "resultados"
        )
    except ImportError:
        pass

    try:
        import src.data.cleaner as cleaner_module
        monkeypatch.setattr(
            cleaner_module, "CAMINHO_DADOS_PROCESSED",
            tmp_path / "dados_processados.csv"
        )
    except ImportError:
        pass


@pytest.fixture
def df_bruto_exemplo():
    """
    Base sintética no formato bruto (como vem do CSV), com 4 times
    e 2 temporadas -- suficiente para exercitar pontos, forma
    recente, casa/fora e ELO sem precisar do CSV real.
    """

    jogos = []
    id_jogo = 1
    times = ["Time A", "Time B", "Time C", "Time D"]

    placares = [
        (2, 1), (1, 1), (0, 2), (3, 0), (1, 0), (2, 2),
        (0, 0), (1, 2), (2, 0), (1, 1), (3, 1), (0, 1),
    ]
    indice_placar = 0

    for ano in (2020, 2021):
        for mandante in times:
            for visitante in times:
                if mandante == visitante:
                    continue

                gols_m, gols_v = placares[indice_placar % len(placares)]
                indice_placar += 1

                jogos.append({
                    "ID": id_jogo,
                    "rodata": (id_jogo % 10) + 1,
                    "data": f"{(id_jogo % 28) + 1:02d}/0{(id_jogo % 8) + 1}/{ano}",
                    "hora": "16:00",
                    "mandante": mandante,
                    "visitante": visitante,
                    "formacao_mandante": "4-4-2",
                    "formacao_visitante": "4-3-3",
                    "tecnico_mandante": "Técnico X",
                    "tecnico_visitante": "Técnico Y",
                    "vencedor": (
                        mandante if gols_m > gols_v
                        else visitante if gols_v > gols_m
                        else "-"
                    ),
                    "arena": "Estádio Exemplo",
                    "mandante_Placar": gols_m,
                    "visitante_Placar": gols_v,
                    "mandante_Estado": "SP",
                    "visitante_Estado": "RJ",
                })

                id_jogo += 1

    return pd.DataFrame(jogos)


@pytest.fixture
def df_processado_exemplo(df_bruto_exemplo):
    """
    Base sintética já passada pelo pipeline completo (limpeza +
    todas as features) -- pronta para testar train_model.py e
    predict.py.
    """

    from src.data.cleaner import limpar_dados
    from src.features.form import (
        calcular_forma_recente,
        calcular_media_gols,
        calcular_desempenho_casa_fora,
        calcular_elo,
    )

    df = limpar_dados(df_bruto_exemplo.copy())
    df = calcular_forma_recente(df)
    df = calcular_media_gols(df)
    df = calcular_desempenho_casa_fora(df)
    df = calcular_elo(df)

    return df