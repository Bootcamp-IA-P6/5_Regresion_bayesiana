import pytest
import numpy as np
from src.modelo1_predict import predict_model1_from_export

def test_predict_model1_logic():
    # 1. ARRANGE: Creamos el "diccionario falso" con valores fáciles
    # Simulamos que el modelo tiene alpha=10, beta=2
    # Y que el entrenamiento tenía media=100 y desviación=10
    mock_model_obj = {
        "parametros": {
            "alpha": 10.0,
            "beta": 2.0
        },
        "transformador": {
            "mean": 100.0,
            "std": 10.0
        }
    }

    # 2. ACT: Definimos un precio de entrada
    # Si metemos precio 110, el escalado será: (110 - 100) / 10 = 1.0
    # La predicción será: 10 + 2 * (1.0) = 12.0
    precio_test = 110.0
    resultado = predict_model1_from_export(mock_model_obj, precio_test)

    # 3. ASSERT: Verificamos las matemáticas
    assert isinstance(resultado, float), "El resultado debe ser un número flotante"
    assert np.isclose(resultado, 12.0), f"Error en el cálculo: se esperaba 12.0 y dio {resultado}"

def test_predict_model1_error_tipo():
    # Test de seguridad: ¿Qué pasa si le pasamos algo que no es un dict?
    with pytest.raises(TypeError):
        predict_model1_from_export("no_soy_un_dict", 100.0)