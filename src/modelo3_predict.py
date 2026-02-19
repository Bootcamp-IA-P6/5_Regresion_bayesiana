import arviz as az
import numpy as np

def load_model3(trace_path):
    """Carga el .nc y extrae parámetros clave"""
    idata = az.from_netcdf(trace_path)
    
    # Extraemos las medias de la distribución posterior
    post = idata.posterior.mean(dim=["chain", "draw"])
    
    # Extraemos los nombres de las categorías (coordenadas del modelo)
    # Si no guardaste coordenadas, usaremos el índice numérico
    cat_names = idata.posterior.a_cat_dim_0.values
    
    return post, cat_names

def predict_model3(post, cat_name, price_scaled, rating_scaled):
    """Fórmula: y = a_cat[i] + b_p * price + b_r * rating"""
    # 1. Obtener intercepto de la categoría seleccionada
    # .sel busca por el nombre de la categoría si está en las coordenadas
    a_val = float(post["a_cat"].sel(a_cat_dim_0=cat_name))
    
    # 2. Obtener pendientes globales
    b_p = float(post["b_p"])
    b_r = float(post["b_r"])
    
    # 3. Calcular predicción (está en escala logarítmica según tu código)
    log_revenue = a_val + (b_p * price_scaled) + (b_r * rating_scaled)
    
    # 4. Revertir el logaritmo para mostrar ingresos reales (opcional)
    revenue = np.exp(log_revenue)
    
    return log_revenue, revenue