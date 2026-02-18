import streamlit as st
import numpy as np
import pickle

# Configuración de la página
st.set_page_config(
    page_title="Modelo 4: Predicción Poisson",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Modelo 4: Predicción de Cantidad Vendida")
st.markdown("**Modelo Poisson Bayesiano** - Predice quantity_sold basado en descuento, rating y weekend")

# Cargar modelo
@st.cache_data
def load_model():
    try:
        with open('modelo_4_poisson_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        return None

# Función de predicción
def predict_quantity(discount, rating, is_weekend, model_params):
    # Estandarizar inputs
    discount_scaled = (discount - model_params['X_discount_mean']) / model_params['X_discount_std']
    rating_scaled = (rating - model_params['X_rating_mean']) / model_params['X_rating_std']
    
    # Calcular predicción
    log_mu = (model_params['intercept'] + 
              model_params['beta_discount'] * discount_scaled + 
              model_params['beta_rating'] * rating_scaled + 
              model_params['beta_weekend'] * is_weekend)
    
    return np.exp(log_mu)

# Cargar modelo
results = load_model()

if results is None:
    st.error("❌ Modelo no encontrado. Ejecuta primero: `04_Modelo_Poisson_Bayesiano.ipynb`")
    st.stop()

st.success("✅ Modelo cargado exitosamente!")

# Sidebar para inputs
st.sidebar.header("🎛️ Parámetros del Producto")

discount = st.sidebar.slider(
    "💰 Descuento (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)

rating = st.sidebar.slider(
    "⭐ Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

is_weekend = st.sidebar.selectbox(
    "📅 Día de la semana",
    options=[0, 1],
    format_func=lambda x: "🏝️ Weekend" if x == 1 else "📊 Weekday",
    index=0
)

# Botón de predicción
if st.sidebar.button("🔮 Predecir Cantidad", type="primary"):
    # Hacer predicción
    model_params = results['model_params']
    predicted_qty = predict_quantity(discount, rating, is_weekend, model_params)
    
    # Mostrar resultado
    st.markdown("### 📊 Resultado de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📦 Cantidad Predicha",
            value=f"{predicted_qty:.1f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🎯 Redondeo",
            value=f"{int(round(predicted_qty))} unidades",
            delta=None
        )
    
    with col3:
        confidence = "Alta" if 1 <= predicted_qty <= 6 else "Media"
        st.metric(
            label="🔍 Confianza",
            value=confidence,
            delta=None
        )
    
    # Mostrar parámetros usados
    st.markdown("#### 📋 Parámetros utilizados:")
    st.write(f"• **Descuento**: {discount}%")
    st.write(f"• **Rating**: {rating:.1f} estrellas")
    st.write(f"• **Tipo de día**: {'Weekend' if is_weekend else 'Weekday'}")

# Información del modelo
st.markdown("---")
st.markdown("### 📈 Información del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📊 Métricas:**")
    if 'metrics' in results:
        metrics = results['metrics']
        st.write(f"• MAE Test: {metrics['mae_test']:.3f}")
        st.write(f"• RMSE Test: {metrics['rmse_test']:.3f}")
        st.write(f"• Overfitting: {metrics['overfitting_mae']:.1f}%")

with col2:
    st.markdown("**🎯 Efectos:**")
    if 'effects' in results:
        effects = results['effects']
        st.write(f"• Descuento: {(effects['discount_effect']-1)*100:+.1f}%")
        st.write(f"• Rating: {(effects['rating_effect']-1)*100:+.1f}%")
        st.write(f"• Weekend: {(effects['weekend_effect']-1)*100:+.1f}%")

# Tests status
if 'tests_passed' in results:
    status = "✅ Todos los tests pasaron" if results['tests_passed'] else "⚠️ Algunos tests fallaron"
    st.markdown(f"**🧪 Tests**: {status}")

# Footer
st.markdown("---")
st.markdown("🤖 **Modelo 4 - Regresión Poisson Bayesiana** | Desarrollado con PyMC + Streamlit")
