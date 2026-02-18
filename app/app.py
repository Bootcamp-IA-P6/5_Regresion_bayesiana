import os
import csv
from datetime import datetime

import streamlit as st

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.modelo2_predict import load_model2, predict_bestseller_proba


from src.modelo2_predict import load_model2, predict_bestseller_proba



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


# -----------------------------
# Helpers
# -----------------------------
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


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Modelo 1 (Regresión)", "⭐ Modelo 2 (Best Seller)", "🌍 Modelo 3 (Jerárquico)"])


# -----------------------------
# TAB 1 - Placeholder Modelo 1
# -----------------------------
with tab1:
    st.subheader("📈 Modelo 1 — Regresión Lineal Bayesiana (Pendiente de integración)")
    st.info(
        "Este tab está preparado para integrar el Modelo 1.\n\n"
        "Recomendación: exportar el modelo (joblib/pkl) y crear una función `predict_modelo1()` en `src/`."
    )


# -----------------------------
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
with tab3:
    st.subheader("🌍 Modelo 3 — Jerárquico por Región (Pendiente de integración)")
    st.info(
        "Este tab está preparado para integrar el Modelo 3.\n\n"
        "Idea: seleccionar `customer_region` y mostrar estimaciones ajustadas (shrinkage). "
        "Cuando Naizabyth exporte artefactos, los cargamos aquí."
    )
