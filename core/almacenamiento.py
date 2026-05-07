"""
core/almacenamiento.py
======================
Gestión del histórico de análisis de incumplimiento en CSV local.
"""

import pandas as pd
import os
import uuid
from datetime import datetime
import pytz

ARCHIVO_HISTORICO = "historico_requisitos.csv"
ARCHIVO_LOG       = "log_incumplimientos.csv"
ZONA_PERU         = pytz.timezone("America/Lima")


def timestamp_peru() -> str:
    return datetime.now(ZONA_PERU).strftime("%Y-%m-%d %H:%M:%S")


def nuevo_id() -> str:
    return str(uuid.uuid4())[:8].upper()


def guardar_resultado(df: pd.DataFrame, nombre_archivo: str, id_carga: str, modelos: str):
    df_g = df.copy()
    df_g["id_carga"]          = id_carga
    df_g["archivo_nombre"]    = nombre_archivo
    df_g["fecha_evaluacion"]  = timestamp_peru()
    df_g["modelos_dl"]        = modelos

    if os.path.exists(ARCHIVO_HISTORICO):
        df_old = pd.read_csv(ARCHIVO_HISTORICO)
        df_total = pd.concat([df_old, df_g], ignore_index=True)
    else:
        df_total = df_g
    df_total.to_csv(ARCHIVO_HISTORICO, index=False)
    return df_g


def registrar_log(id_carga: str, nombre_archivo: str, n_registros: int,
                  n_incumplimientos: int, modelos: list):
    entrada = {
        "id_carga":         id_carga,
        "archivo":          nombre_archivo,
        "fecha":            timestamp_peru(),
        "registros":        n_registros,
        "incumplimientos":  n_incumplimientos,
        "pct_incumplimiento": round(n_incumplimientos / n_registros * 100, 2) if n_registros else 0,
        "modelos_dl":       ", ".join(modelos),
    }
    df_e = pd.DataFrame([entrada])
    if os.path.exists(ARCHIVO_LOG):
        df_log = pd.read_csv(ARCHIVO_LOG)
        df_log = pd.concat([df_log, df_e], ignore_index=True)
    else:
        df_log = df_e
    df_log.to_csv(ARCHIVO_LOG, index=False)


def cargar_historico() -> pd.DataFrame | None:
    if os.path.exists(ARCHIVO_HISTORICO):
        return pd.read_csv(ARCHIVO_HISTORICO)
    return None


def cargar_log() -> pd.DataFrame | None:
    if os.path.exists(ARCHIVO_LOG):
        return pd.read_csv(ARCHIVO_LOG)
    return None


def borrar_historico():
    for f in [ARCHIVO_HISTORICO, ARCHIVO_LOG]:
        if os.path.exists(f):
            os.remove(f)
