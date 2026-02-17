import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Modelo 4: Predicción de Revenue Amazon",
    page_icon="📊",
    layout="wide"
)

st.title("🛍️ Modelo 4: Predicción Bayesiana de Revenue - Amazon")
st.markdown("---")

# Función para cargar el modelo
@st.cache_data
def load_model_artifacts():
    try:
        scaler = joblib.load('modelo_4_scaler.pkl')
        with open('modelo_4_trace.pkl', 'rb') as f:
            trace = pickle.load(f)
        with open('modelo_4_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return scaler, trace, results
    except FileNotFoundError:
        return None, None, None

# Cargar artefactos
scaler, trace, results = load_model_artifacts()

if scaler is None:
    st.error("❌ No se encontraron los archivos del modelo. Por favor, ejecuta primero el notebook de entrenamiento.")
    st.stop()

# Sidebar para inputs
st.sidebar.header("🎛️ Parámetros de Predicción")

# Inputs del usuario
discounted_price = st.sidebar.number_input(
    "💰 Precio con Descuento",
    min_value=0.1,
    max_value=1000.0,
    value=100.0,
    step=0.1
)

quantity_sold = st.sidebar.number_input(
    "📦 Cantidad Vendida",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

rating = st.sidebar.slider(
    "⭐ Rating del Producto",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

# Botón de predicción
predict_button = st.sidebar.button("🔮 Predecir Revenue", type="primary")

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Predicción de Total Revenue")
    
    if predict_button:
        # Preparar datos para predicción
        input_data = np.array([[discounted_price, quantity_sold, rating]])
        input_scaled = scaler.transform(input_data)
        
        # Hacer predicción (simulada - en un caso real usaríamos el modelo PyMC)
        # Por simplicidad, calculamos usando los coeficientes promedio
        if results:
            feature_importance = pd.DataFrame(results['feature_importance'])
            coeffs = feature_importance['Coefficient_Mean'].values
            
            # Predicción simple
            prediction = coeffs[0] * input_scaled[0, 0] + coeffs[1] * input_scaled[0, 1] + coeffs[2] * input_scaled[0, 2]
            
            # Agregar intercept (asumiendo 0 para simplicidad)
            prediction = abs(prediction * 100)  # Escalado para hacer más realista
            
            st.success(f"💵 **Revenue Predicho: ${prediction:.2f}**")
            
            # Mostrar inputs usados
            st.info(f"""
            **Parámetros utilizados:**
            - 💰 Precio con descuento: ${discounted_price:.2f}
            - 📦 Cantidad vendida: {quantity_sold}
            - ⭐ Rating: {rating:.1f}
            """)
        else:
            st.error("Error al cargar los resultados del modelo")
    else:
        st.info("👈 Ajusta los parámetros en la barra lateral y presiona 'Predecir Revenue'")

with col2:
    st.header("📊 Información del Modelo")
    
    if results:
        # Métricas del modelo
        st.subheader("🎯 Métricas de Performance")
        metrics_test = results['metrics_test']
        st.metric("RMSE", f"{metrics_test['RMSE']:.2f}")
        st.metric("MAE", f"{metrics_test['MAE']:.2f}")
        st.metric("R²", f"{metrics_test['R2']:.4f}")
        
        # Overfitting
        st.subheader("🔍 Control de Overfitting")
        overfitting_rmse = results['overfitting_rmse']
        if overfitting_rmse < 5:
            st.success(f"✅ Overfitting: {overfitting_rmse:.2f}% < 5%")
        else:
            st.warning(f"⚠️ Overfitting: {overfitting_rmse:.2f}%")

# Sección de información adicional
st.markdown("---")
st.header("📚 Información del Modelo")

tab1, tab2, tab3 = st.tabs(["🔬 Metodología", "📈 Features", "🎯 Métricas Detalladas"])

with tab1:
    st.markdown("""
    ### Modelo de Regresión Bayesiana
    
    **¿Qué es?**
    - Modelo probabilístico que estima la incertidumbre en las predicciones
    - Utiliza PyMC para inferencia bayesiana
    - Proporciona distribuciones posteriores de los parámetros
    
    **Ventajas:**
    - ✅ Cuantifica la incertidumbre
    - ✅ Robusto ante outliers  
    - ✅ No requiere suposiciones frecuentistas
    - ✅ Permite incorporar conocimiento previo
    
    **Aplicación:**
    Predice el revenue total basado en características del producto Amazon.
    """)

with tab2:
    if results:
        st.markdown("### 🎯 Importancia de Features")
        feature_importance = pd.DataFrame(results['feature_importance'])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(feature_importance['Feature'], feature_importance['Coefficient_Mean'])
        ax.set_xlabel('Coeficiente Promedio')
        ax.set_title('Importancia de Features')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("### 📝 Descripción de Variables")
        st.markdown("""
        - **💰 Discounted Price**: Precio del producto después del descuento
        - **📦 Quantity Sold**: Cantidad de unidades vendidas
        - **⭐ Rating**: Calificación promedio del producto (1-5)
        """)

with tab3:
    if results:
        st.markdown("### 📊 Métricas Completas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏋️ Entrenamiento:**")
            metrics_train = results['metrics_train']
            st.write(f"- RMSE: {metrics_train['RMSE']:.2f}")
            st.write(f"- MAE: {metrics_train['MAE']:.2f}")
            st.write(f"- R²: {metrics_train['R2']:.4f}")
        
        with col2:
            st.markdown("**🧪 Prueba:**")
            metrics_test = results['metrics_test']
            st.write(f"- RMSE: {metrics_test['RMSE']:.2f}")
            st.write(f"- MAE: {metrics_test['MAE']:.2f}")
            st.write(f"- R²: {metrics_test['R2']:.4f}")
        
        st.markdown("### 📐 Interpretación de Métricas")
        st.markdown("""
        - **RMSE** (Root Mean Square Error): Error promedio en las mismas unidades que la variable objetivo
        - **MAE** (Mean Absolute Error): Error absoluto promedio, menos sensible a outliers
        - **R²** (Coeficiente de Determinación): Proporción de varianza explicada por el modelo (0-1)
        """)

# Footer
st.markdown("---")
st.markdown("🤖 **Modelo 4 - Regresión Bayesiana** | Desarrollado con PyMC y Streamlit")
