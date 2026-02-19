# 5_Regresion_bayesiana


# Modelo 2 — Clasificador de Éxito (Regresión Logística Bayesiana)

## Objetivo
Construir un modelo probabilístico que estime la **probabilidad de que un producto sea Best Seller (venta alta)** utilizando como predictores:

- `rating`
- `discounted_price`

La variable objetivo se construye a partir de:

- `quantity_sold`

---

## Definición de *Best Seller*
Se define como aquellos productos que pertenecen al **Top 25%** en `quantity_sold` (percentil 75).

Esto transforma el problema en clasificación binaria:

- `1` → Best Seller  
- `0` → No Best Seller

---

## Enfoque metodológico
Se utiliza una **Regresión Logística Bayesiana** implementada en PyMC.

A diferencia de modelos clásicos:

- los coeficientes no son valores fijos,  
- sino **distribuciones posteriores**.

Esto permite obtener:

- probabilidades  
- intervalos creíbles  
- estimaciones de incertidumbre  

---

## Optimización para entorno notebook
El muestreo bayesiano completo (MCMC) puede ser costoso en tiempo de cómputo.

Para garantizar ejecución rápida y reproducible se aplican:

- **Submuestra estratificada** (balance por clase)  
- **ADVI (Variational Inference)** para aproximar la posterior  

El pipeline queda preparado para escalar a entrenamiento completo en entornos con compilación optimizada.

---

## Pipeline del modelo

1. Carga de datos con **Polars**
2. Limpieza y validación
3. Construcción del target `best_seller`
4. Submuestreo estratificado
5. Split train/test
6. Escalado de variables
7. Entrenamiento bayesiano
8. Predicción probabilística
9. Evaluación
10. Visualización de incertidumbre
11. Export de artefactos

---

## Métricas reportadas
- Accuracy  
- ROC AUC  
- Matriz de confusión  

---

## Entregable principal
Curva:

**P(Best Seller) vs Rating**  
(manteniendo el precio fijo)

con **intervalo creíble 5–95%**.

Esto permite visualizar no solo la predicción sino la **confianza del modelo**.

---

## Artefactos generados

En la carpeta `models/`:

- `modelo2_scaler.joblib` → transformaciones necesarias para nuevas predicciones  
- `modelo2_trace.nc` → posterior bayesiana del modelo  

Estos archivos permiten reutilizar el modelo en aplicaciones como APIs o Streamlit.

---

## ▶️ Cómo ejecutar el notebook

### 1) Crear entorno
```bash
python -m venv .venv
```

### 2) Activar entorno

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 3) Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4) Ejecutar notebook
Abrir:

```
notebooks/02_modelo2_logistica_bayesiana.ipynb
```

y seleccionar **Run All**.

---

## Interpretación de resultados

- `beta_rating > 0` → mejores calificaciones aumentan probabilidad de éxito.  
- `beta_price < 0` → precios mayores reducen probabilidad (si el modelo aprende ese efecto).  
- Las bandas de incertidumbre muestran dónde el modelo es menos confiable.

---

## Limitaciones

- Entrenamiento sobre submuestra para rapidez.  
- ADVI es aproximación, no MCMC exacto.  
- Resultados dependen de la definición de Best Seller.

---

## Próximos pasos

- Entrenamiento completo con NUTS en entorno optimizado.  
- Incorporar más variables.  
- Ajustar umbral de clasificación.  
- Despliegue en aplicación interactiva.
```

