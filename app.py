"""
Fase 17 — Interface Streamlit.

Tela única: escolhe mandante e visitante, aperta "Prever
partida" e mostra a probabilidade final (Ensemble), o
detalhamento por modelo, gols esperados e os placares mais
prováveis. Não recalcula nada por conta própria -- usa
calcular_previsao() de src/models/predict.py, a mesma função
que alimenta o main.py no terminal.
"""

import pandas as pd
import streamlit as st

from src.data.loader import carregar_dados
from src.data.cleaner import limpar_dados
from src.features.form import (
    calcular_forma_recente,
    calcular_media_gols,
    calcular_desempenho_casa_fora,
    calcular_elo,
)
from src.models.train_model import treinar_modelo_producao
from src.models.predict import calcular_previsao
from src.config import N_SIMULACOES_MONTE_CARLO


st.set_page_config(page_title="Brasileirão Predictor", page_icon="⚽")


@st.cache_data(show_spinner="Carregando e preparando os dados...")
def preparar_dados():
    """
    Roda o pipeline de dados uma única vez por sessão (cache do
    Streamlit) -- o mesmo pipeline usado em main.py.
    """

    df = carregar_dados()
    df = limpar_dados(df)
    df = calcular_forma_recente(df)
    df = calcular_media_gols(df)
    df = calcular_desempenho_casa_fora(df)
    df = calcular_elo(df)

    return df


@st.cache_resource(show_spinner="Treinando o modelo...")
def preparar_modelo(df):
    """
    Treina o modelo de PRODUÇÃO (100% dos dados) uma única vez
    por sessão -- ver treinar_modelo_producao() em train_model.py.
    """

    resultado = treinar_modelo_producao(df)

    return resultado["modelo_logistico"], resultado["scaler"]


def main():

    st.title("⚽ Brasileirão Predictor")
    st.caption(
        "Previsão probabilística de partidas do Campeonato "
        "Brasileiro, combinando Regressão Logística e Poisson "
        "em uma média das probabilidades dos dois modelos."
    )

    df = preparar_dados()
    modelo, scaler = preparar_modelo(df)

    times = sorted(set(df["mandante"]) | set(df["visitante"]))

    coluna_mandante, coluna_visitante = st.columns(2)

    with coluna_mandante:
        indice_padrao = times.index("Flamengo") if "Flamengo" in times else 0
        mandante = st.selectbox("Mandante", times, index=indice_padrao)

    with coluna_visitante:
        opcoes_visitante = [time for time in times if time != mandante]
        indice_padrao_visitante = (
            opcoes_visitante.index("Palmeiras")
            if "Palmeiras" in opcoes_visitante else 0
        )
        visitante = st.selectbox(
            "Visitante", opcoes_visitante, index=indice_padrao_visitante
        )

    if not st.button("Prever partida", type="primary"):
        return

    try:
        resultado = calcular_previsao(modelo, df, mandante, visitante, scaler=scaler)
    except ValueError as erro:
        st.error(str(erro))
        return

    # ==========================================
    # RESULTADO FINAL (item 70)
    # ==========================================

    st.subheader("Probabilidade final (Ensemble)")

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric(f"Vitória {mandante}", f"{resultado['prob_mandante_final']:.1%}")
    coluna2.metric("Empate", f"{resultado['prob_empate_final']:.1%}")
    coluna3.metric(f"Vitória {visitante}", f"{resultado['prob_visitante_final']:.1%}")

    grafico_probabilidades = pd.DataFrame(
        {
            "Probabilidade": [
                resultado["prob_mandante_final"],
                resultado["prob_empate_final"],
                resultado["prob_visitante_final"],
            ]
        },
        index=[f"Vitória {mandante}", "Empate", f"Vitória {visitante}"],
    )
    st.bar_chart(grafico_probabilidades)

    st.write(
        f"**Previsão:** {resultado['previsao']}  \n"
        f"**Confiança:** {resultado['confianca']:.1%}"
    )

    with st.expander("Ver probabilidades por modelo"):

        st.markdown("**Regressão Logística**")
        st.write(
            f"- Vitória {mandante}: {resultado['prob_mandante']:.1%}  \n"
            f"- Empate: {resultado['prob_empate']:.1%}  \n"
            f"- Vitória {visitante}: {resultado['prob_visitante']:.1%}"
        )

        st.markdown("**Poisson**")
        st.write(
            f"- Vitória {mandante}: {resultado['prob_mandante_poisson']:.1%}  \n"
            f"- Empate: {resultado['prob_empate_poisson']:.1%}  \n"
            f"- Vitória {visitante}: {resultado['prob_visitante_poisson']:.1%}"
        )

    # ==========================================
    # GOLS ESPERADOS
    # ==========================================

    st.subheader("Gols esperados")

    coluna1, coluna2 = st.columns(2)
    coluna1.metric(mandante, f"{resultado['gols_esperados_mandante']:.2f}")
    coluna2.metric(visitante, f"{resultado['gols_esperados_visitante']:.2f}")

    # ==========================================
    # PLACARES MAIS PROVÁVEIS (itens 71, 72, 73)
    # ==========================================

    st.subheader("Placares mais prováveis")
    st.caption(f"{N_SIMULACOES_MONTE_CARLO:,} simulações".replace(",", "."))

    placares = resultado["placares"][:10]

    grafico_placares = pd.DataFrame(
        {
            "Probabilidade": [
                quantidade / N_SIMULACOES_MONTE_CARLO for _, quantidade in placares
            ]
        },
        index=[f"{gm} x {gv}" for (gm, gv), _ in placares],
    )
    st.bar_chart(grafico_placares)


if __name__ == "__main__":
    main()