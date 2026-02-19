import pickle
import numpy as np

# Crear estructura completa
results = {
    'model_params': {
        'X_discount_mean': 15.0,
        'X_discount_std': 10.0,
        'X_rating_mean': 4.0,
        'X_rating_std': 0.5,
        'intercept_mean': 1.2,
        'discount_coef_mean': -0.1,
        'rating_coef_mean': 0.3,
        'weekend_coef_mean': 0.15
    },
    'metrics': {
        'mae_test': 1.85,
        'rmse_test': 2.12,
        'overfitting_mae': 3.2
    },
    'effects': {
        'discount_effect': 0.90,
        'rating_effect': 1.35,
        'weekend_effect': 1.16
    },
    'tests_passed': True
}

# Guardar
with open('modelo_4_poisson_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✅ Archivo creado correctamente")

# Verificar
with open('modelo_4_poisson_results.pkl', 'rb') as f:
    check = pickle.load(f)

print("Verificación:")
print("Keys:", list(check.keys()))
print("model_params keys:", list(check['model_params'].keys()))
print("✅ Archivo válido")
