import joblib
import numpy as np


def load_model1(model_path: str):
    """
    Carga el artefacto del Modelo 1.
    En tu caso es un dict con:
      - parametros: {alpha, beta}
      - transformador: {mean, std}
      - metricas: {...}
    """
    return joblib.load(model_path)


def predict_model1_from_export(model_obj, price: float):
    """
    Predice total_revenue usando el export del Modelo 1 (alpha, beta, mean, std).

    IMPORTANTE:
    El Modelo 1 (según su export) fue entrenado con UNA sola feature escalada:
      X = discounted_price (o price) estandarizado

    Aquí asumo que están usando PRICE como input principal.
    Si entrenaron con discounted_price, pásale ese valor como 'price' desde Streamlit.
    """
    if not isinstance(model_obj, dict):
        raise TypeError("Modelo 1 esperado como dict exportado (joblib).")

    params = model_obj.get("parametros", {})
    transf = model_obj.get("transformador", {})

    alpha = float(params.get("alpha"))
    beta = float(params.get("beta"))
    mean = float(transf.get("mean"))
    std = float(transf.get("std"))

    # Escalado igual que en entrenamiento
    x_scaled = (price - mean) / std

    # Predicción lineal
    y_pred = alpha + beta * x_scaled

    return float(y_pred)


