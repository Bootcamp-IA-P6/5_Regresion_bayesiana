import os
import csv
from datetime import datetime

import streamlit as st

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.modelo2_predict import load_model2, predict_bestseller_proba


from src.modelo2_predict import load_model2, predict_bestseller_proba

from src.modelo1_predict import load_model1

from src.modelo3_predict import load_model3, predict_model3

from src.modelo4_predict import load_model4, predict_model4_quantity



# -----------------------------
# Configuración general UI
# -----------------------------
st.set_page_config(page_title="Amazon Sales - Modelos Bayesianos", layout="centered")

st.title("Amazon Sales — Modelos Bayesianos (MVP)")
st.write(
    "Aplicación para probar los modelos del proyecto. "
    "Actualmente: Modelo 2 (Best Seller) funcional. "
    "Modelo 1 y 3 quedan listos para integrar."
)

# Rutas de artefactos (ajusta si tu estructura difiere)
TRACE_PATH = "models/modelo2/modelo2_trace.nc"
SCALER_PATH = "models/modelo2/modelo2_scaler.joblib"

LOG_PATH = "reports/predictions_log.csv"

MODEL1_PATH = "models/modelo1/modelo_ingresos_bayesian.joblib"   # AJUSTA este nombre al archivo real

MODEL4_TRACE_PATH = "models/modelo4/modelo4_trace.nc"
MODEL4_SCALER_PATH = "models/modelo4/modelo4_scaler.joblib"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def _cached_load_model1():
    return load_model1(MODEL1_PATH)


@st.cache_resource
def _cached_load_model2():
    return load_model2(TRACE_PATH, SCALER_PATH)


def log_prediction(row: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

@st.cache_resource
def _cached_load_model3():
    """
    Carga el modelo jerárquico desde el archivo .nc.
    Devuelve:
    - post: El objeto con las medias de los parámetros (interceptos y pendientes).
    - cat_names: La lista de nombres de las categorías.
    """
    # Llamamos a la función que está en src/modelo3_predict.py
    post, cat_names = load_model3(MODEL3_PATH)
    return post, cat_names

@st.cache_resource
def _cached_load_model4():
    return load_model4(MODEL4_TRACE_PATH, MODEL4_SCALER_PATH)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Modelo 1 (Regresión)", "⭐ Modelo 2 (Best Seller)", "🌍 Modelo 3 (Jerárquico)", "📊 Modelo 4 (Poisson)"])


# -----------------------------
# TAB 1 - Modelo 1
# -----------------------------
with tab1:
    st.subheader("📈 Modelo 1 — Regresión Bayesiana (Ingresos)")

    if not os.path.exists(MODEL1_PATH):
        st.error(
            "No se encontró el artefacto del Modelo 1.\n\n"
            f"Ruta esperada: `{MODEL1_PATH}`"
        )
    else:
        model1 = _cached_load_model1()

        # -------- FORMULARIO --------
        with st.form("form_modelo1"):

            st.info(
                "Este modelo utiliza una Regresión Lineal Bayesiana. "
                "La variable de entrada se estandariza utilizando los parámetros "
                "del entrenamiento y luego se calcula el ingreso estimado "
                "mediante los coeficientes alpha y beta aprendidos."
            )

            x_input = st.number_input(
                "Precio usado por el modelo (price/discounted_price)",
                min_value=0.01,
                value=200.0,
                step=1.0
            )

            submitted1 = st.form_submit_button("Predecir ingreso")

        # -------- RESULTADOS --------
        if submitted1:
            from src.modelo1_predict import predict_model1_from_export

            y_pred = predict_model1_from_export(model1, price=x_input)

            st.metric("💰 Ingreso estimado (predicción)", f"{y_pred:,.2f}")

            # Mostrar fórmula
            st.caption("Fórmula utilizada:")
            st.latex(r"y = \alpha + \beta \cdot \frac{(x - \mu)}{\sigma}")

            # Mostrar métricas guardadas
            metricas = model1.get("metricas", {})
            if metricas:
                st.write("📌 Métricas guardadas en el artefacto:")
                st.json(metricas)

            # -------- SIMULACIÓN --------
            st.divider()
            st.subheader("🔎 Simulación de escenario")

            if st.checkbox("Simular aumento del 10% en el precio"):
                nuevo_precio = x_input * 1.10
                nuevo_pred = predict_model1_from_export(model1, price=nuevo_precio)

                st.metric(
                    "Ingreso estimado con +10% precio",
                    f"{nuevo_pred:,.2f}",
                    delta=f"{nuevo_pred - y_pred:,.2f}"
                )


# -----------------------------w
# TAB 2 - Modelo 2 funcional
# -----------------------------
with tab2:
    st.subheader("⭐ Modelo 2 — Probabilidad de Best Seller (Bayesiano)")

    # Validación de artefactos
    if not (os.path.exists(TRACE_PATH) and os.path.exists(SCALER_PATH)):
        st.error(
            "No se encontraron los artefactos del Modelo 2.\n\n"
            f"- Trace esperado: `{TRACE_PATH}`\n"
            f"- Scaler esperado: `{SCALER_PATH}`\n\n"
            "Verifica que existen y que la ruta sea correcta."
        )
    else:
        trace, scaler, betas = _cached_load_model2()

        with st.form("form_modelo2"):
            rating = st.slider("Rating (0 a 5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            discounted_price = st.number_input("Discounted Price", min_value=0.01, value=200.0, step=1.0)

            submitted = st.form_submit_button("Predecir")

        if submitted:
            p_mean, p_low, p_high = predict_bestseller_proba(
                rating=rating,
                discounted_price=discounted_price,
                scaler=scaler,
                betas=betas
            )

            st.metric("Probabilidad media de Best Seller", f"{p_mean*100:.2f}%")
            st.write(f"**Intervalo creíble (5%–95%)**: {p_low*100:.2f}% — {p_high*100:.2f}%")

            # Interpretación simple
            if p_mean >= 0.7:
                st.success("Alta probabilidad de ser Best Seller ✅")
            elif p_mean >= 0.4:
                st.warning("Probabilidad moderada ⚠️")
            else:
                st.info("Probabilidad baja ❌")

            # Logging (feedback simple)
            log_prediction({
                "timestamp": datetime.utcnow().isoformat(),
                "rating": rating,
                "discounted_price": discounted_price,
                "p_mean": p_mean,
                "p_low": p_low,
                "p_high": p_high,
            })

            st.caption(f"✅ Predicción guardada en `{LOG_PATH}` (para feedback/monitorización).")


# -----------------------------
# TAB 3 - Placeholder Modelo 3
# -----------------------------
# -----------------------------
# TAB 3 - Modelo 3 Jerárquico
# -----------------------------
# -----------------------------
# TAB 3 - Modelo 3 Jerárquico
# -----------------------------
with tab3:
    st.subheader("🌍 Modelo 3 — Regresión Jerárquica por Categoría")

    # 1. Definir la ruta del archivo .nc
    MODEL3_PATH = "models/modelo3/modelo_jerarquico.nc" 

    # 2. Verificar si existe el archivo antes de intentar cargarlo
    if not os.path.exists(MODEL3_PATH):
        st.error(f"No se encontró el artefacto del Modelo 3 en: `{MODEL3_PATH}`")
        st.info("Asegúrate de que el archivo 'modelo_jerarquico.nc' esté en la carpeta 'models/modelo3/'.")
    else:
        # 3. CARGA: Obtenemos el objeto 'post3' (para cálculos) y 'cat_names' (para el selectbox)
        post3, cat_names = _cached_load_model3()

        with st.form("form_modelo3"):
            st.info("Este modelo predice ingresos considerando el comportamiento específico (shrinkage) de cada categoría.")
            
            # Selector de categoría (La parte Jerárquica)
            categoria_sel = st.selectbox("Selecciona la Categoría de Producto", options=cat_names)
            
            col1, col2 = st.columns(2)
            with col1:
                p_scaled = st.number_input("Precio Escalado (price_scaled)", value=0.0, step=0.1, help="Valor del precio tras el StandardScaler")
            with col2:
                r_scaled = st.number_input("Rating Escalado (rating_scaled)", value=0.0, step=0.1, help="Valor del rating tras el StandardScaler")

            submitted3 = st.form_submit_button("Calcular Predicción Jerárquica")

        # 4. Lógica de resultados
        if submitted3:
            # CORRECCIÓN: Pasamos 'post3' como primer argumento para que la función tenga los datos
            y_log, y_real = predict_model3(post3, categoria_sel, p_scaled, r_scaled)

            # Mostrar resultado principal
            st.metric("💰 Ingreso Estimado", f"${y_real:,.2f}")
            
            # Detalles técnicos para transparencia del modelo
            with st.expander("Ver detalles del ajuste bayesiano"):
                st.write(f"**Log-Revenue (Predicción base):** `{y_log:.4f}`")
                
                # Renderizado de la fórmula matemática
                st.latex(rf"y_{{log}} = \alpha_{{{categoria_sel}}} + \beta_p \cdot x_p + \beta_r \cdot x_r")
                
                # CORRECCIÓN: Accedemos al intercepto usando el objeto 'post3' cargado
                intercepto_cat = float(post3['a_cat'].sel(a_cat_dim_0=categoria_sel))
                st.write(f"El intercepto calculado para **{categoria_sel}** es: `{intercepto_cat:.4f}`")
                st.caption("Nota: Este valor incluye el efecto de 'shrinkage', ajustando la categoría hacia la media global si hay pocos datos.")

# -----------------------------
# TAB 4 - Modelo 4 (Poisson)
# -----------------------------
with tab4:
    st.subheader("📊 Modelo 4 — Regresión Poisson Bayesiana (Cantidad Vendida)")
    
    # Validación de artefactos
    if not (os.path.exists(MODEL4_TRACE_PATH) and os.path.exists(MODEL4_SCALER_PATH)):
        st.error(
            "No se encontraron los artefactos del Modelo 4.\n\n"
            f"- Trace esperado: `{MODEL4_TRACE_PATH}`\n"
            f"- Scaler esperado: `{MODEL4_SCALER_PATH}`\n\n"
            "Ejecuta el notebook 04_modelo4_poisson.ipynb para generar los artefactos."
        )
    else:
        trace, scaler_data, params = _cached_load_model4()

        with st.form("form_modelo4"):
            st.info("Este modelo utiliza una **Regresión Poisson Bayesiana** para predecir la cantidad vendida.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                discount_percent = st.slider("Descuento (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
                rating = st.slider("Rating del producto", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            
            with col2:
                is_weekend = st.selectbox(
                    "Día de la semana", 
                    options=[0, 1],
                    format_func=lambda x: "Fin de semana" if x == 1 else "Día laboral", 
                    index=0
                )

            submitted4 = st.form_submit_button("Predecir cantidad vendida")

        if submitted4:
            pred_quantity = predict_model4_quantity(
                discount_percent, rating, is_weekend, scaler_data, params
            )
            
            st.metric("📦 Cantidad vendida estimada", f"{pred_quantity:.2f} unidades")

            # Mostrar fórmula
            st.caption("Fórmula utilizada:")
            st.latex(r"\lambda = \exp(\alpha + \beta_1 \cdot discount_{scaled} + \beta_2 \cdot rating_{scaled} + \beta_3 \cdot weekend)")

            # Mostrar métricas guardadas  
            metrics = scaler_data.get("metrics", {})
            if metrics:
                st.write("📌 Métricas del modelo:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Entrenamiento:**")
                    st.write(f"- MAE: {metrics.get('mae_train', 0):.3f}")
                    st.write(f"- RMSE: {metrics.get('rmse_train', 0):.3f}")
                
                with col2:
                    st.write("**Prueba:**")
                    st.write(f"- MAE: {metrics.get('mae_test', 0):.3f}")
                    st.write(f"- RMSE: {metrics.get('rmse_test', 0):.3f}")

            # Mostrar efectos si están disponibles
            effects = scaler_data.get("effects", {})
            if effects:
                st.divider()
                st.subheader("🔍 Efectos Multiplicativos")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    discount_effect = effects.get('discount_effect', 1.0)
                    st.metric("Efecto Descuento", f"×{discount_effect:.3f}", 
                             delta=f"{(discount_effect-1)*100:+.1f}%")
                
                with col2:
                    rating_effect = effects.get('rating_effect', 1.0)
                    st.metric("Efecto Rating", f"×{rating_effect:.3f}", 
                             delta=f"{(rating_effect-1)*100:+.1f}%")
                
                with col3:
                    weekend_effect = effects.get('weekend_effect', 1.0)
                    st.metric("Efecto Weekend", f"×{weekend_effect:.3f}", 
                             delta=f"{(weekend_effect-1)*100:+.1f}%")