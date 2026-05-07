"""
core/metricas.py
================
Cálculo de métricas estadísticas y comparativas para el detector DL
de incumplimiento de requisitos en Comercio Exterior.
"""

import numpy as np
import pandas as pd
from scipy import stats


def resumen_estadistico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    X = df[columnas] if all(c in df.columns for c in columnas) else df.select_dtypes(include="number")
    return pd.DataFrame({
        "Media":     X.mean(),
        "Mediana":   X.median(),
        "Std":       X.std(),
        "Min":       X.min(),
        "Max":       X.max(),
        "Asimetría": X.skew(),
        "Curtosis":  X.kurtosis(),
        "Nulos":     X.isnull().sum(),
        "% Nulos":   (X.isnull().mean() * 100).round(2),
    }).round(4)


def calcular_correlaciones(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    X = df[columnas] if all(c in df.columns for c in columnas) else df.select_dtypes(include="number")
    return X.corr().round(3)


def comparar_resultados(resultados: dict, n_total: int) -> pd.DataFrame:
    filas = []
    for nombre, r in resultados.items():
        if nombre.startswith("_") or "error" in r:
            continue
        pct = (r["n_anomalias"] / n_total * 100) if n_total > 0 else 0
        filas.append({
            "Modelo DL":               r.get("icono", "") + " " + r["nombre"],
            "Incumplimientos detectados": r["n_anomalias"],
            "% del total":             f"{pct:.1f}%",
            "Descripción":             r.get("descripcion", "—"),
        })
    return pd.DataFrame(filas)


def perfil_incumplimiento(df: pd.DataFrame, mascara: np.ndarray, columnas: list) -> pd.DataFrame:
    X = df[columnas] if all(c in df.columns for c in columnas) else df.select_dtypes(include="number")
    normales = X[~mascara]
    anomalos = X[mascara]

    pvalores = []
    for c in X.columns:
        n_clean = normales[c].dropna()
        a_clean = anomalos[c].dropna()
        if len(a_clean) > 1 and len(n_clean) > 1:
            _, p = stats.ttest_ind(n_clean, a_clean, equal_var=False)
            pvalores.append(round(p, 4))
        else:
            pvalores.append(None)

    perfil = pd.DataFrame({
        "Media (conforme)":      normales.mean(),
        "Media (incumplimiento)": anomalos.mean(),
        "Diferencia %":          ((anomalos.mean() - normales.mean()) /
                                  (normales.mean().abs() + 1e-9) * 100).round(1),
        "p-valor":               pvalores,
    }, index=X.columns)

    perfil["Significativo"] = perfil["p-valor"].apply(
        lambda p: "✅ Sí" if p is not None and p < 0.05 else "❌ No"
    )
    return perfil.round(4)


def distribucion_scores(scores: np.ndarray) -> dict:
    return {
        "media":   float(np.mean(scores)),
        "max":     float(np.max(scores)),
        "p95":     float(np.percentile(scores, 95)),
        "p99":     float(np.percentile(scores, 99)),
        "cv":      float(np.std(scores) / (np.mean(scores) + 1e-9)),
    }
