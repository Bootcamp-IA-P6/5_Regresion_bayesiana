import pytest
import numpy as np
import xarray as xr
from src.modelo3_predict import predict_model3

def test_predict_model3_logic():
    # 1. SIMULAMOS el objeto 'post' que devolvería ArviZ
    # Creamos un dataset con los parámetros que tu función espera
    mock_post = xr.Dataset({
        "a_cat": (["a_cat_dim_0"], [1.0, 2.0]), # Dos categorías con interceptos diferentes
        "b_p": 0.5,
        "b_r": 0.1
    }, coords={"a_cat_dim_0": ["electronica", "ropa"]})

    # 2. DEFINIMOS entradas de prueba
    cat_test = "electronica"
    p_scaled = 1.0
    r_scaled = 0.0
    
    # 3. EJECUTAMOS la función
    # log_rev = a_val(1.0) + (0.5 * 1.0) + (0.1 * 0.0) = 1.5
    log_rev, rev = predict_model3(mock_post, cat_test, p_scaled, r_scaled)

    # 4. ASERCIONES (Los checks de seguridad)
    assert np.isclose(log_rev, 1.5), f"Cálculo de log_revenue fallido: {log_rev}"
    assert rev == np.exp(1.5), "La conversión de exp() no coincide"
    assert isinstance(rev, float), "El resultado debe ser un número flotante"

def test_predict_model3_diferentes_categorias():
    # Test para asegurar que el modelo distingue entre categorías
    mock_post = xr.Dataset({
        "a_cat": (["a_cat_dim_0"], [1.0, 5.0]), 
        "b_p": 0.0, "b_r": 0.0
    }, coords={"a_cat_dim_0": ["cat_baja", "cat_alta"]})

    _, rev_baja = predict_model3(mock_post, "cat_baja", 0, 0)
    _, rev_alta = predict_model3(mock_post, "cat_alta", 0, 0)

    assert rev_alta > rev_baja, "El modelo no está diferenciando interceptos por categoría"