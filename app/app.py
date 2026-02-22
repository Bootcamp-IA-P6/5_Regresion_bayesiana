import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
# ... el resto de tus imports
# Importamos la función de registro Y el objeto db para consultas
from database.mongo_client import registrar_prediccion, db

import streamlit as st
import os
import csv
from datetime import datetime

import pandas as pd

# Importación de tus funciones de predicción
from src.modelo1_predict import load_model1, predict_model1_from_export
from src.modelo2_predict import load_model2, predict_bestseller_proba
from src.modelo3_predict import load_model3, predict_model3



# -----------------------------
# Configuración general UI
# -----------------------------
st.set_page_config(page_title="Amazon Sales - Modelos Bayesianos", layout="centered")

st.title("Amazon Sales — Modelos Bayesianos (MVP)")
st.write(
    "Aplicación para probar los modelos del proyecto. "
    "Todos los resultados se guardan automáticamente en MongoDB."
)

# -----------------------------
# Rutas de artefactos
# -----------------------------
MODEL1_PATH = "models/modelo1/modelo_ingresos_bayesian.joblib"
TRACE_PATH = "models/modelo2/modelo2_trace.nc"
SCALER_PATH = "models/modelo2/modelo2_scaler.joblib"
MODEL3_PATH = "models/modelo3/modelo_jerarquico.nc" 

# -----------------------------
# Helpers con Cache
# -----------------------------
@st.cache_resource
def _cached_load_model1():
    return load_model1(MODEL1_PATH)

@st.cache_resource
def _cached_load_model2():
    # Nota: Ajusta si los parámetros de retorno de tu load_model2 son distintos
    return load_model2(TRACE_PATH, SCALER_PATH)

@st.cache_resource
def _cached_load_model3():
    # Devuelve: post, cat_names
    return load_model3(MODEL3_PATH)

# -----------------------------
# Estructura de Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Modelo 1 (Regresión)", "⭐ Modelo 2 (Best Seller)", "🌍 Modelo 3 (Jerárquico)"])

# -----------------------------
# TAB 1 - Modelo 1 (Ingresos)
# -----------------------------
with tab1:
    st.subheader("📈 Modelo 1 — Regresión Bayesiana (Ingresos)")

    if not os.path.exists(MODEL1_PATH):
        st.error(f"No se encontró el artefacto del Modelo 1 en: `{MODEL1_PATH}`")
    else:
        model1 = _cached_load_model1()

        with st.form("form_modelo1"):
            st.info("Calcula el ingreso estimado mediante coeficientes alpha y beta.")
            x_input = st.number_input("Precio (price/discounted_price)", min_value=0.01, value=200.0, step=1.0)
            submitted1 = st.form_submit_button("Predecir ingreso")

        if submitted1:
            y_pred = predict_model1_from_export(model1, price=x_input)
            st.metric("💰 Ingreso estimado", f"{y_pred:,.2f}")

            # --- GUARDAR EN MONGODB ---
            registrar_prediccion(
                nombre_modelo="Modelo 1 - Regresión",
                input_usuario={"price": x_input},
                resultado={"ingreso_estimado": y_pred}
            )

# -----------------------------
# TAB 2 - Modelo 2 (Best Seller)
# -----------------------------
with tab2:
    st.subheader("⭐ Modelo 2 — Probabilidad de Best Seller")

    if not (os.path.exists(TRACE_PATH) and os.path.exists(SCALER_PATH)):
        st.error("No se encontraron los artefactos del Modelo 2.")
    else:
        trace, scaler, betas = _cached_load_model2()

        with st.form("form_modelo2"):
            rating = st.slider("Rating (0 a 5)", 0.0, 5.0, 4.0, 0.1)
            discounted_price = st.number_input("Discounted Price", min_value=0.01, value=200.0, step=1.0)
            submitted2 = st.form_submit_button("Predecir Probabilidad")

        if submitted2:
            p_mean, p_low, p_high = predict_bestseller_proba(
                rating=rating, 
                discounted_price=discounted_price, 
                scaler=scaler, 
                betas=betas
            )

            st.metric("Probabilidad media", f"{p_mean*100:.2f}%")
            st.write(f"**Intervalo creíble**: {p_low*100:.2f}% — {p_high*100:.2f}%")

            # --- GUARDAR EN MONGODB ---
            registrar_prediccion(
                nombre_modelo="Modelo 2 - Best Seller",
                input_usuario={"rating": rating, "discounted_price": discounted_price},
                resultado={"p_mean": p_mean, "p_low": p_low, "p_high": p_high}
            )

# -----------------------------
# TAB 3 - Modelo 3 (Jerárquico)
# -----------------------------
with tab3:
    st.subheader("🌍 Modelo 3 — Regresión Jerárquica por Categoría")

    if not os.path.exists(MODEL3_PATH):
        st.error(f"No se encontró el artefacto del Modelo 3.")
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

            # --- GUARDAR EN MONGODB ---
            registrar_prediccion(
                nombre_modelo="Modelo 3 - Jerárquico",
                input_usuario={
                    "categoria": categoria_sel,
                    "p_scaled": p_scaled,
                    "r_scaled": r_scaled
                },
                resultado={"ingreso_real_estimado": y_real, "log_revenue": y_log}
            )
            
# NUEVA SECCIÓN: HISTORIAL DE MONGODB
# -----------------------------
st.divider()
st.subheader("📜 Historial de Predicciones")
st.write("Consulta los datos guardados directamente en la base de datos `db_amazon_modelos`.")

if st.button("🔄 Actualizar Historial"):
    # Consultamos la colección 'historial_predicciones' que confirmamos en la terminal
    cursor = db.historial_predicciones.find().sort("_id", -1).limit(20)
    lista_predicciones = list(cursor)
    
    if lista_predicciones:
        df = pd.DataFrame(lista_predicciones)
        
        # Quitamos la columna de ID de Mongo para que sea más legible
        if '_id' in df.columns:
            df.drop(columns=['_id'], inplace=True)
            
        # Reordenamos columnas si existen para que la fecha salga primero
        columnas = ['fecha', 'modelo', 'inputs', 'resultado']
        df = df[[c for c in columnas if c in df.columns]]
        
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No hay registros todavía. ¡Haz una predicción arriba!")