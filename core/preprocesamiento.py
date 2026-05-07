"""
core/preprocesamiento.py
========================
Preprocesamiento de datos para el detector de incumplimiento de requisitos.
Ajustado al dominio de Comercio Exterior e Inteligencia Comercial.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

# Columnas típicas de comercio exterior que contienen fechas
COLUMNAS_FECHA_CE = [
    "fecha", "date", "fecha_despacho", "fecha_embarque", "fecha_llegada",
    "fecha_numeracion", "fecha_levante", "fecha_emision", "fecha_vencimiento",
    "fecha_registro", "time", "timestamp",
]

# Columnas típicas del dominio de comercio exterior (para sugerencias)
COLUMNAS_CE_NUMERICAS = [
    "valor_fob", "valor_cif", "valor_seguro", "flete", "arancel",
    "igv", "ipm", "percepcion", "peso_neto", "peso_bruto",
    "cantidad_unidades", "tipo_cambio", "valor_aduana", "precio_unitario",
    "dias_almacenaje", "plazo_entrega", "dias_demora", "score_riesgo",
    "n_documentos_faltantes", "n_observaciones", "monto_multa",
]


def preparar_datos(df: pd.DataFrame):
    """
    Extrae columnas numéricas, imputa nulos y aplica escalado robusto.

    Returns:
        X_original   (DataFrame) — datos numéricos sin escalar
        X_scaled     (ndarray)   — datos normalizados
        columnas     (list)      — nombres de columnas usadas
        col_fecha    (str|None)  — nombre de columna de fecha detectada
    """
    col_fecha = _detectar_columna_fecha(df)
    X = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).copy()

    # Excluir columnas internas de fecha ya descompuestas
    cols_excluir = ["anio", "mes", "dia", "año"] + (
        [col_fecha] if col_fecha and col_fecha in X.columns else []
    )
    X = X.drop(columns=[c for c in cols_excluir if c in X.columns], errors="ignore")

    if X.shape[1] == 0:
        X = pd.DataFrame({"indice": range(len(df))})

    # Imputar con mediana (robusto a outliers)
    X = X.fillna(X.median(numeric_only=True))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, list(X.columns), col_fecha


def _detectar_columna_fecha(df: pd.DataFrame) -> str | None:
    """Detecta y parsea la primera columna de fecha encontrada."""
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
    }
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in COLUMNAS_FECHA_CE):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                if df[col].notna().sum() > len(df) * 0.3:
                    df["anio"] = df[col].dt.year
                    df["mes"] = df[col].dt.month
                    df["dia"] = df[col].dt.day
                    df["mes_nombre"] = df["mes"].map(meses)
                    return col
            except Exception:
                continue
    return None


def leer_csv(archivo) -> pd.DataFrame:
    """Intenta múltiples combinaciones de separador y encoding."""
    intentos = [
        (None, "utf-8"),
        (";", "utf-8"),
        (",", "latin1"),
        (";", "latin1"),
        ("\t", "utf-8"),
    ]
    for sep, enc in intentos:
        try:
            kwargs = dict(engine="python", encoding=enc, on_bad_lines="skip")
            kwargs["sep"] = sep if sep else None
            return pd.read_csv(archivo, **kwargs)
        except Exception:
            continue
    raise ValueError("No se pudo leer el CSV. Verifica separador y encoding.")
