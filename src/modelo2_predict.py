import numpy as np
import arviz as az
import joblib


def load_model2(trace_path: str, scaler_path: str):
    """
    Carga los artefactos del Modelo 2:
    - trace.nc: posterior bayesiana (betas)
    - scaler.joblib: escalador para transformar inputs
    """
    trace = az.from_netcdf(trace_path)
    scaler = joblib.load(scaler_path)

    # Aplanamos muestras posteriores para predicción rápida
    b0 = trace.posterior["beta_0"].values.reshape(-1)
    b1 = trace.posterior["beta_rating"].values.reshape(-1)
    b2 = trace.posterior["beta_price"].values.reshape(-1)

    return trace, scaler, (b0, b1, b2)


def predict_bestseller_proba(
    rating: float,
    discounted_price: float,
    scaler,
    betas,
    ci_low: float = 0.05,
    ci_high: float = 0.95,
):
    """
    Predice P(Best Seller) usando la posterior bayesiana:
    - rating, discounted_price en escala original
    - devuelve probabilidad media y un intervalo creíble (por defecto 5%-95%)
    """
    b0, b1, b2 = betas

    # 1) Construimos input y escalamos (igual que en entrenamiento)
    X = np.array([[rating, discounted_price]], dtype=float)
    Xs = scaler.transform(X)

    # 2) Calculamos logits para cada muestra posterior
    logits = b0 + b1 * Xs[0, 0] + b2 * Xs[0, 1]

    # 3) Convertimos a probabilidad con sigmoide
    p_samples = 1 / (1 + np.exp(-logits))

    # 4) Resumen: media e intervalo creíble
    p_mean = float(p_samples.mean())
    p_ci_low = float(np.quantile(p_samples, ci_low))
    p_ci_high = float(np.quantile(p_samples, ci_high))

    return p_mean, p_ci_low, p_ci_high
