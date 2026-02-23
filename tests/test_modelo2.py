import pytest
import numpy as np
from src.modelo2_predict import predict_bestseller_proba

class MockScaler:
    def transform(self, X):
        return X / 10.0  # Simulación simple: divide por 10

def test_predict_model2_logic():
    n_samples = 1000
    # Creamos una posterior con pequeña variabilidad para el intervalo creíble
    betas = (
        np.random.normal(0.0, 0.01, n_samples),  # b0
        np.random.normal(1.0, 0.01, n_samples),  # b1 (rating)
        np.random.normal(-1.0, 0.01, n_samples) # b2 (price)
    )
    
    scaler = MockScaler()
    
    # Rating 40, Precio 20 -> Escalados: 4.0 y 2.0
    # Logit ≈ 0 + (1 * 4) + (-1 * 2) = 2.0
    p_mean, p_low, p_high = predict_bestseller_proba(40.0, 20.0, scaler, betas)

    prob_esperada = 1 / (1 + np.exp(-2.0))
    
    assert np.isclose(p_mean, prob_esperada, atol=0.05)
    assert p_low < p_mean < p_high
    assert 0 <= p_mean <= 1

def test_logit_negativo_probabilidad_baja():
    # Si el precio es altisimo y la beta es negativa, la prob debe ser casi 0
    betas = (np.array([0.0]), np.array([0.0]), np.array([-10.0]))
    scaler = MockScaler()
    p_mean, _, _ = predict_bestseller_proba(1.0, 100.0, scaler, betas)
    assert p_mean < 0.1