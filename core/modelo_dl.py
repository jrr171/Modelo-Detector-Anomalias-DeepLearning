
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def preprocess_data(df):

    numeric = df.select_dtypes(include=np.number).fillna(0)

    scaler = MinMaxScaler()

    X = scaler.fit_transform(numeric)

    return numeric, X

def build_autoencoder(input_dim):

    inputs = Input(shape=(input_dim,))

    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    encoded = Dense(16, activation='relu')(x)

    x = Dense(32, activation='relu')(encoded)
    x = Dense(64, activation='relu')(x)

    outputs = Dense(input_dim, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(0.001),
        loss='mse'
    )

    return model

def detect_anomalies(df, threshold_percentile=90):

    numeric, X = preprocess_data(df)

    model = build_autoencoder(X.shape[1])

    model.fit(
        X,
        X,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    reconstructed = model.predict(X, verbose=0)

    mse = np.mean(np.power(X - reconstructed, 2), axis=1)

    iso = IsolationForest(
        contamination=0.15,
        random_state=42
    )

    iso.fit(X)

    iso_score = -iso.score_samples(X)

    mse_norm = (mse - mse.min()) / (mse.max() - mse.min())
    iso_norm = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min())

    ensemble = (mse_norm * 0.7) + (iso_norm * 0.3)

    threshold = np.percentile(
        ensemble,
        threshold_percentile
    )

    anomalies = ensemble >= threshold

    results = df.copy()

    results["anomaly_score"] = ensemble

    results["estado"] = np.where(
        anomalies,
        "INCUMPLIMIENTO",
        "NORMAL"
    )

    results["riesgo"] = np.select(
        [
            ensemble <= 0.3,
            ensemble <= 0.6,
            ensemble <= 0.8,
            ensemble > 0.8
        ],
        [
            "NORMAL",
            "RIESGO BAJO",
            "RIESGO MEDIO",
            "RIESGO CRITICO"
        ]
    )

    return results
