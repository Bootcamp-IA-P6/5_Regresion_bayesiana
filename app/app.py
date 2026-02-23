import sys
import os
import random
import pandas as pd
import streamlit as st
from datetime import datetime

# Añadimos el path para imports locales
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importamos las herramientas de DB
from database.mongo_client import registrar_prediccion, db, obtener_estadisticas_precios

# Importación de funciones de predicción
from src.modelo1_predict import load_model1, predict_model1_from_export
from src.modelo2_predict import load_model2, predict_bestseller_proba
from src.modelo3_predict import load_model3, predict_model3

# -----------------------------
# Configuración general UI
# -----------------------------
st.set_page_config(page_title="Amazon Sales - Modelos Bayesianos", layout="centered")

st.title("Amazon Sales — Modelos Bayesianos (MVP)")
st.write("Sistemas de entrenamiento y despliegue automático (MLOps) con Shonos IA.")

# Rutas de artefactos
MODEL1_PATH = "models/modelo1/modelo_ingresos_bayesian.joblib"
TRACE_PATH = "models/modelo2/modelo2_trace.nc"
SCALER_PATH = "models/modelo2/modelo2_scaler.joblib"
MODEL3_PATH = "models/modelo3/modelo_jerarquico.nc" 

@st.cache_resource
def _cached_load_model1(): return load_model1(MODEL1_PATH)
@st.cache_resource
def _cached_load_model2(): return load_model2(TRACE_PATH, SCALER_PATH)
@st.cache_resource
def _cached_load_model3(): return load_model3(MODEL3_PATH)

# -----------------------------
# Estructura de Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Modelo 1 (Regresión)", "⭐ Modelo 2 (Best Seller)", "🌍 Modelo 3 (Jerárquico)"])

# --- TAB 1: MODELO 1 (INGRESOS + A/B TESTING) ---
with tab1:
    st.subheader("📈 Modelo 1 — Regresión (A/B Testing)")
    if not os.path.exists(MODEL1_PATH):
        st.error(f"No se encontró el artefacto en: `{MODEL1_PATH}`")
    else:
        model1 = _cached_load_model1()
        with st.form("form_modelo1"):
            x_input = st.number_input("Precio (price/discounted_price)", min_value=0.01, value=200.0, step=1.0)
            submitted1 = st.form_submit_button("Predecir ingreso")

        if submitted1:
            version_id = random.choice(["v1_control", "v2_experimental"])
            y_pred = predict_model1_from_export(model1, price=x_input)
            if version_id == "v2_experimental": y_pred = y_pred * 1.05 
            
            st.metric(f"💰 Ingreso estimado ({version_id})", f"{y_pred:,.2f}")
            registrar_prediccion("Modelo 1 - Regresión", {"price": x_input}, {"ingreso": y_pred}, version=version_id)

# --- TAB 2: MODELO 2 (BEST SELLER) ---
with tab2:
    st.subheader("⭐ Modelo 2 — Probabilidad de Best Seller")
    if not (os.path.exists(TRACE_PATH) and os.path.exists(SCALER_PATH)):
        st.error("No se encontraron los artefactos del Modelo 2.")
    else:
        trace, scaler, betas = _cached_load_model2()
        with st.form("form_modelo2"):
            rating = st.slider("Rating (0 a 5)", 0.0, 5.0, 4.0, 0.1)
            discounted_price = st.number_input("Discounted Price", min_value=0.01, value=200.0, step=1.0, key="m2_price")
            submitted2 = st.form_submit_button("Predecir Probabilidad")

        if submitted2:
            p_mean, p_low, p_high = predict_bestseller_proba(rating, discounted_price, scaler, betas)
            st.metric("Probabilidad media", f"{p_mean*100:.2f}%")
            st.write(f"**Intervalo creíble**: {p_low*100:.2f}% — {p_high*100:.2f}%")
            registrar_prediccion("Modelo 2 - Best Seller", {"rating": rating, "price": discounted_price}, {"p_mean": p_mean})

# --- TAB 3: MODELO 3 (JERÁRQUICO) ---
with tab3:
    st.subheader("🌍 Modelo 3 — Regresión Jerárquica")
    if not os.path.exists(MODEL3_PATH):
        st.error("No se encontró el artefacto del Modelo 3.")
    else:
        post3, cat_names = _cached_load_model3()
        with st.form("form_modelo3"):
            categoria_sel = st.selectbox("Categoría de Producto", options=cat_names)
            p_scaled = st.number_input("Precio Escalado", value=0.0, step=0.1)
            r_scaled = st.number_input("Rating Escalado", value=0.0, step=0.1)
            submitted3 = st.form_submit_button("Calcular Predicción Jerárquica")

        if submitted3:
            y_log, y_real = predict_model3(post3, categoria_sel, p_scaled, r_scaled)
            st.metric("💰 Ingreso Estimado", f"${y_real:,.2f}")
            registrar_prediccion("Modelo 3 - Jerárquico", {"cat": categoria_sel, "p": p_scaled}, {"y_real": y_real})

# -----------------------------
# MONITOREO Y HISTORIAL (MLOps)
# -----------------------------
st.divider()
st.subheader("🕵️ Panel de Monitoreo MLOps (Data Drift)")

PRECIO_MEDIO_ENTRENAMIENTO = 149.0 
precio_actual_it = obtener_estadisticas_precios(model1, "precio")

col1, col2 = st.columns(2)
with col1:
    st.metric("Precio Promedio Usuarios", f"{precio_actual_it:.2f} €")
with col2:
    drift = ((precio_actual_it - PRECIO_MEDIO_ENTRENAMIENTO) / PRECIO_MEDIO_ENTRENAMIENTO) * 100
    st.metric("Data Drift (Precio)", f"{drift:.2f} %", delta_color="inverse")

if abs(drift) > 20:
    st.warning("⚠️ ¡ALERTA DE DATA DRIFT!")
else:
    st.success("✅ Estabilidad de Datos.")

st.divider()
st.subheader("📜 Historial de Predicciones")
if st.button("🔄 Actualizar Historial"):
    cursor = db.historial_predicciones.find().sort("_id", -1).limit(10)
    df_hist = pd.DataFrame(list(cursor))
    if not df_hist.empty:
        if '_id' in df_hist.columns: df_hist.drop(columns=['_id'], inplace=True)
        st.dataframe(df_hist, use_container_width=True)
        
# NUEVA SECCIÓN: AUTO-REEMPLAZO ()
# ---------------------------------------------------------
st.divider()
st.subheader("🤖 Decisión de Auto-reemplazo (Model Promotion)")

# Simulamos métricas de precisión obtenidas de una validación automática
acc_v1 = 0.85  # Precisión del modelo actual (v1_control)
acc_v2 = 0.89  # Precisión del nuevo modelo (v2_experimental)

col_acc1, col_acc2 = st.columns(2)
col_acc1.write(f"📊 Precisión v1 (Control): **{acc_v1*100:.1f}%**")
col_acc2.write(f"📊 Precisión v2 (Experimental): **{acc_v2*100:.1f}%**")

# LÓGICA DE MLOps: Solo reemplaza si v2 es mejor por un margen del 2%
if acc_v2 > (acc_v1 + 0.02):
    st.success(f"✅ **CRITERIO CUMPLIDO**: La v2 supera a la v1 por {(acc_v2-acc_v1)*100:.1f}%. El sistema recomienda promover v2 a Producción.")
    
    # Este botón simula la acción de CI/CD (Despliegue automático)
    if st.button("🚀 Ejecutar Auto-reemplazo Automático"):
        st.balloons()
        st.info("Iniciando Pipeline de despliegue... Generando nueva imagen de Docker con Modelo v2.")
else:
    st.error("❌ **CRITERIO NO CUMPLIDO**: La mejora no es significativa. Se mantiene el Modelo v1 para evitar riesgos.")