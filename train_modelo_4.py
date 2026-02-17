#!/usr/bin/env python3
"""
Script para entrenar el Modelo 4 de Regresión Bayesiana
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pickle
import joblib

warnings.filterwarnings('ignore')

def main():
    print("🚀 Iniciando entrenamiento del Modelo 4 - Regresión Bayesiana")
    print("=" * 60)
    
    # 1. Cargar datos
    print("📊 Cargando dataset de Amazon...")
    try:
        df = pd.read_csv('dataset/amazon_sales_dataset.csv')
        print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return False
    
    # 2. Preparar datos
    print("\n🔧 Preparando datos...")
    features = ['discounted_price', 'quantity_sold', 'rating']
    target = 'total_revenue'
    
    X = df[features].copy()
    y = df[target].copy()
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Datos preparados:")
    print(f"   - Features: {features}")
    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Test: {X_test.shape[0]} muestras")
    
    # 3. Crear y entrenar modelo bayesiano
    print("\n🧠 Creando modelo bayesiano...")
    
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        coeffs = pm.Normal('coeffs', mu=0, sigma=10, shape=X_train_scaled.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        # Media del modelo
        mu = intercept + pm.math.dot(X_train_scaled, coeffs)
        
        # Likelihood
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_train)
    
    print("✅ Modelo creado")
    
    # 4. Sampling
    print("\n⚡ Entrenando modelo (esto puede tomar unos minutos)...")
    
    try:
        with model:
            trace = pm.sample(1000, tune=500, random_seed=42, chains=2, 
                            progressbar=True, target_accept=0.9)
        print("✅ Modelo entrenado exitosamente!")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return False
    
    # 5. Predicciones
    print("\n🔮 Haciendo predicciones...")
    
    # Predicciones para train
    with model:
        pm.set_data({'X': X_train_scaled, 'y': y_train})
        ppc_train = pm.sample_posterior_predictive(trace, random_seed=42)
    
    # Predicciones para test
    with model:
        pm.set_data({'X': X_test_scaled, 'y': y_test})
        ppc_test = pm.sample_posterior_predictive(trace, random_seed=42)
    
    # Extraer predicciones
    y_pred_train = ppc_train.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
    y_pred_test = ppc_test.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
    
    # 6. Calcular métricas
    print("\n📊 Evaluando modelo...")
    
    # Métricas train
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Métricas test
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n🎯 Métricas de Entrenamiento:")
    print(f"   RMSE: {rmse_train:.2f}")
    print(f"   MAE: {mae_train:.2f}")
    print(f"   R²: {r2_train:.4f}")
    
    print(f"\n🎯 Métricas de Prueba:")
    print(f"   RMSE: {rmse_test:.2f}")
    print(f"   MAE: {mae_test:.2f}")
    print(f"   R²: {r2_test:.4f}")
    
    # Verificar overfitting
    overfitting_rmse = abs(rmse_train - rmse_test) / rmse_train * 100
    overfitting_r2 = abs(r2_train - r2_test) / r2_train * 100
    
    print(f"\n🔍 Análisis de Overfitting:")
    print(f"   Diferencia RMSE: {overfitting_rmse:.2f}%")
    print(f"   Diferencia R²: {overfitting_r2:.2f}%")
    
    if overfitting_rmse < 5 and overfitting_r2 < 5:
        print("   ✅ Overfitting bajo: < 5%")
    else:
        print("   ⚠️ Posible overfitting detectado")
    
    # 7. Feature importance
    print("\n📈 Importancia de Features:")
    coefficients = trace.posterior['coeffs'].values
    coeff_means = coefficients.mean(axis=(0, 1))
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient_Mean': coeff_means,
        'Absolute_Importance': np.abs(coeff_means)
    }).sort_values('Absolute_Importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Coefficient_Mean']:.4f}")
    
    # 8. Guardar artefactos
    print("\n💾 Guardando artefactos del modelo...")
    
    # Guardar scaler
    joblib.dump(scaler, 'modelo_4_scaler.pkl')
    
    # Guardar trace
    with open('modelo_4_trace.pkl', 'wb') as f:
        pickle.dump(trace, f)
    
    # Guardar métricas y resultados
    results = {
        'features': features,
        'metrics_train': {'RMSE': rmse_train, 'MAE': mae_train, 'R2': r2_train},
        'metrics_test': {'RMSE': rmse_test, 'MAE': mae_test, 'R2': r2_test},
        'feature_importance': feature_importance.to_dict(),
        'overfitting_rmse': overfitting_rmse,
        'overfitting_r2': overfitting_r2
    }
    
    with open('modelo_4_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✅ Artefactos guardados:")
    print("   - modelo_4_scaler.pkl")
    print("   - modelo_4_trace.pkl")
    print("   - modelo_4_results.pkl")
    
    print("\n🎉 ¡Modelo 4 entrenado exitosamente!")
    print("🌐 Ahora puedes usar la app de Streamlit para hacer predicciones.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Falló el entrenamiento del modelo")
        exit(1)
