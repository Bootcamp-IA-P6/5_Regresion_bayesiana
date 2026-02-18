import streamlit as st
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción Bayesiana", layout="wide")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Cargamos el archivo NetCDF generado por ArviZ
        # Asegúrate de que el archivo esté en la carpeta 'app' relativa a donde ejecutas el comando
        model = az.from_netcdf("./app/modelo_jerarquico.nc")
        return model
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        return None

trace = load_model()

# --- INTERFAZ DE USUARIO ---
st.title("📊 Dashboard de Regresión Bayesiana")
st.markdown("---")

if trace is not None:
    # --- SECCIÓN 1: INSPECCIÓN (Visualización de parámetros) ---
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.header("📥 Inspección de Parámetros")
        # Obtenemos las variables disponibles en el posterior (mu_a, a_cat, b_p, etc.)
        var_names = list(trace.posterior.data_vars)
        selected_var = st.selectbox("Selecciona una variable del modelo:", var_names)
        st.info(f"Mostrando información de: {selected_var}")

    with col_result:
        st.header("📈 Distribución Posterior")
        # Gráfico de densidad posterior y HDI real
        fig, ax = plt.subplots(figsize=(8, 4))
        az.plot_posterior(trace, var_names=[selected_var], ax=ax) 
        st.pyplot(fig)
        
        # Tabla de resumen estadístico (Media, SD, HDI)
        summary = az.summary(trace, var_names=[selected_var]) 
        st.table(summary)

    # --- SECCIÓN 2: SIMULADOR (Barras y Predicción) ---
    st.markdown("---")
    st.header("🔮 Simulador de Predicción en Tiempo Real")
    st.write("Ajusta los valores para calcular el resultado basado en el modelo entrenado.")

    # Extraemos las medias de los parámetros para el cálculo lineal
    post_means = trace.posterior.mean(dim=["chain", "draw"])

    c1, c2, c3 = st.columns(3)

    with c1:
        # Slider para la variable con pendiente b_p
        val_p = st.slider("Valor de Variable P (ej. Precio):", 0.0, 100.0, 50.0)
    
    with c2:
        # Slider para la variable con pendiente b_r
        val_r = st.slider("Valor de Variable R (ej. Rating):", 0.0, 10.0, 5.0)

    with c3:
        # Selector para elegir el intercepto específico de la categoría (a_cat)
        n_cats = len(post_means["a_cat"])
        cat_idx = st.selectbox("Categoría de Grupo:", range(n_cats))

    # --- CÁLCULO MATEMÁTICO ---
    # Usamos la lógica de regresión: Intercepto + (Beta1 * X1) + (Beta2 * X2)
    intercepto = post_means["a_cat"][cat_idx].values
    beta_p = post_means["b_p"].values
    beta_r = post_means["b_r"].values

    prediccion = intercepto + (beta_p * val_p) + (beta_r * val_r)

    # --- MOSTRAR RESULTADO (CORREGIDO SIN ERRORES) ---
    st.markdown("---")
    
    # Diseño visual del resultado final
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:30px; border-radius:15px; text-align:center; border: 1px solid #d1d5db;">
        <h2 style="color:#1f77b4; margin-bottom:10px;">Predicción Estimada</h2>
        <h1 style="font-size:60px; color:#111827; margin:0;">{prediccion:.2f}</h1>
        <p style="color:#6b7280; margin-top:10px;">Cálculo basado en las medias posteriores de <b>{n_cats}</b> categorías.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("No se pudo cargar el archivo 'modelo_jerarquico.nc'. Verifica que la ruta './app/modelo_jerarquico.nc' sea correcta.")