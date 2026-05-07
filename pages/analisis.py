
import streamlit as st
import pandas as pd
import plotly.express as px

from core.modelo_dl import detect_anomalies

def render_analisis():

    st.title("🚨 Detección Inteligente de Incumplimientos")

    uploaded_file = st.file_uploader(
        "Cargar archivo CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Debe cargar un dataset")
        return

    df = pd.read_csv(uploaded_file)

    st.success("Dataset cargado correctamente")

    st.subheader("Vista preliminar")
    st.dataframe(df.head())

    threshold = st.slider(
        "Sensibilidad del modelo",
        70,
        99,
        90
    )

    ejecutar = st.button(
        "🚀 Ejecutar Detección Deep Learning"
    )

    # ========================================
    # IMPORTANTE:
    # SOLO EJECUTA AL PRESIONAR
    # ========================================

    if ejecutar:

        with st.spinner("Procesando modelos..."):

            results = detect_anomalies(
                df,
                threshold_percentile=threshold
            )

        total = len(results)

        anomalies = len(
            results[results["estado"] == "INCUMPLIMIENTO"]
        )

        critical = len(
            results[results["riesgo"] == "RIESGO CRITICO"]
        )

        medium = len(
            results[results["riesgo"] == "RIESGO MEDIO"]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Registros", total)
        c2.metric("Anomalías", anomalies)
        c3.metric("Riesgo Crítico", critical)
        c4.metric("Riesgo Medio", medium)

        st.subheader("Distribución de riesgos")

        fig = px.pie(
            results,
            names="riesgo"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Scores de anomalía")

        hist = px.histogram(
            results,
            x="anomaly_score",
            color="estado"
        )

        st.plotly_chart(
            hist,
            use_container_width=True
        )

        st.subheader("Incumplimientos detectados")

        anomalies_df = results[
            results["estado"] == "INCUMPLIMIENTO"
        ].sort_values(
            by="anomaly_score",
            ascending=False
        )

        st.dataframe(anomalies_df)

        csv = anomalies_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Descargar resultados",
            csv,
            "incumplimientos.csv",
            "text/csv"
        )
