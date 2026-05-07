
import streamlit as st

def render_modelos():

    st.title("🧠 Arquitectura Deep Learning")

    st.markdown("""
## Modelos utilizados

### Dense Autoencoder
Detecta:
- Outliers
- Relaciones anómalas
- Valores extremos

### Isolation Forest
Detecta:
- Registros sospechosos
- Operaciones poco probables

### Ensemble AI
Fusiona múltiples scores para mejorar precisión.
""")
