
import streamlit as st
from pages.inicio import render_inicio
from pages.analisis import render_analisis
from pages.modelos import render_modelos
from pages.metodologia import render_metodologia
from pages.historial import render_historial

st.set_page_config(
    page_title="DeepReq AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #071021;
    color: white;
}

.block-container {
    padding-top: 1rem;
}

div[data-testid="metric-container"] {
    background-color: #0E1B31;
    border-radius: 15px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🧠 DeepReq AI")

menu = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "Análisis",
        "Modelos",
        "Metodología",
        "Historial"
    ]
)

if menu == "Inicio":
    render_inicio()

elif menu == "Análisis":
    render_analisis()

elif menu == "Modelos":
    render_modelos()

elif menu == "Metodología":
    render_metodologia()

elif menu == "Historial":
    render_historial()
