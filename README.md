# 🛃 DeepReq CE — Detector de Incumplimiento de Requisitos con Deep Learning

Sistema de detección de **incumplimiento de requisitos** para empresas de
**Inteligencia Comercial en el sector Comercio Exterior**, implementado
**exclusivamente con modelos de Deep Learning** usando PyTorch.

---

## 🧠 Modelos de Deep Learning incluidos

| Modelo | Técnica | Uso recomendado |
|--------|---------|-----------------|
| **Autoencoder Denso** | MLP encoder-decoder | Datos tabulares generales |
| **Autoencoder LSTM** | LSTM encoder-decoder | Variables con dependencias secuenciales |
| **VAE** | Variational Autoencoder | Modelado probabilístico |
| **Deep SVDD** | One-class classification | Frontera compacta de datos conformes |
| **Ensemble DL** | Votación mayoritaria | Consenso entre los modelos |

---

## 🚀 Instalación y ejecución

### 1. Requisitos

- Python 3.10 o superior
- pip

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` puede tardar en descargar (~800MB). Si tienes GPU CUDA disponible,
> instala la versión CUDA de PyTorch desde https://pytorch.org/get-started/locally/

### 3. Ejecutar

```bash
streamlit run app.py
```

Se abrirá automáticamente en: http://localhost:8501

---

## 📂 Estructura del proyecto

```
DeepReq-ComercioExterior/
├── app.py                    # Aplicación principal (Streamlit)
├── requirements.txt          # Dependencias
├── core/
│   ├── modelo_dl.py          # 4 modelos de Deep Learning (PyTorch)
│   ├── preprocesamiento.py   # Limpieza y normalización de datos
│   ├── metricas.py           # Estadísticas y comparación de modelos
│   └── almacenamiento.py     # Historial en CSV local
└── pages/
    ├── inicio.py             # Página de bienvenida
    ├── analisis.py           # Análisis con DL
    ├── modelos.py            # Explicación de modelos DL
    ├── historial.py          # Historial de análisis
    └── metodologia.py        # Marco metodológico académico
```

---

## 📊 Formato del CSV de entrada

El sistema acepta cualquier CSV con variables numéricas.
Variables típicas del dominio CE:

```
valor_fob, valor_cif, flete, seguro, arancel, igv, percepcion,
peso_neto, peso_bruto, cantidad_unidades, tipo_cambio,
dias_almacenaje, dias_demora, plazo_entrega,
n_documentos_faltantes, n_observaciones, score_riesgo, monto_multa,
fecha_despacho, fecha_embarque, fecha_llegada
```

---

## 🔍 Tipos de incumplimiento detectables

- **Valoración aduanera**: subvaluación / sobrevaluación
- **Documental**: documentos faltantes o inconsistentes
- **Temporal**: demoras o plazos incumplidos
- **Clasificación arancelaria**: partidas atípicas
- **Volumétrico**: peso o cantidad inconsistente

---

## ⚙️ Parámetros configurables (sidebar)

| Parámetro | Rango | Default | Descripción |
|-----------|-------|---------|-------------|
| Percentil | 90–99 | 97 | Top N% más anómalo = incumplimiento |
| Épocas | 50–500 | 100 | Iteraciones de entrenamiento DL |
| Learning rate | 1e-4 – 1e-2 | 1e-3 | Tasa de aprendizaje AdamW |
| Modelos DL | 1–4 | 3 | Modelos a ejecutar en paralelo |

---

## 📚 Referencias

- Goodfellow et al. (2016). *Deep Learning*. MIT Press.
- Kingma & Welling (2014). *Auto-Encoding Variational Bayes*. ICLR.
- Ruff et al. (2018). *Deep One-Class Classification*. ICML.
- Paszke et al. (2019). *PyTorch*. NeurIPS.

---

> Sistema desarrollado para Inteligencia Comercial en Comercio Exterior.
> Usando **únicamente Deep Learning** (sin algoritmos de ML tradicional).
