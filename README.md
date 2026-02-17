# 🛍️ Modelo 4: Regresión Bayesiana - Amazon Sales Dataset

## 📋 Descripción del Proyecto

Este proyecto implementa un **modelo de regresión bayesiana** para predecir el `total_revenue` de productos de Amazon utilizando técnicas de inferencia bayesiana con PyMC.

### 🎯 Objetivos (Nivel Esencial)

✅ **Modelo ML funcional** que predice una variable numérica (total_revenue)  
✅ **EDA completo** con visualizaciones relevantes para regresión  
✅ **Overfitting < 5%** entre métricas de entrenamiento y validación  
✅ **Solución productizada** con Streamlit  
✅ **Informe de rendimiento** con métricas de regresión (RMSE, MAE, R²)  

## 📊 Dataset

El dataset de Amazon Sales contiene las siguientes columnas relevantes:
- `discounted_price`: Precio del producto con descuento
- `quantity_sold`: Cantidad vendida del producto  
- `rating`: Calificación del producto (1-5)
- `total_revenue`: **Variable objetivo** - Revenue total generado

## 🚀 Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo
```bash
python train_modelo_4.py
```

### 3. Ejecutar la aplicación web
```bash
streamlit run app_modelo_4.py
```

### 4. Ejecutar tests
```bash
python test_modelo_4.py
```

### 5. Explorar el análisis completo
```bash
jupyter notebook modelo_4_bayesian.ipynb
```

## 🧠 Metodología

### Modelo Bayesiano
- **Framework**: PyMC para inferencia bayesiana
- **Tipo**: Regresión lineal bayesiana
- **Features**: discounted_price, quantity_sold, rating
- **Priors**: Normal(0, 10) para coeficientes, HalfNormal(10) para sigma

### Métricas de Evaluación
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coeficiente de determinación
- **Control de overfitting**: Diferencia < 5% entre train/test

## 📁 Estructura del Proyecto

```
├── modelo_4_bayesian.ipynb    # Notebook principal del modelo
├── train_modelo_4.py          # Script de entrenamiento PyMC
├── app_modelo_4.py           # Aplicación Streamlit  
├── test_modelo_4.py          # Tests unitarios
├── dataset/
│   └── amazon_sales_dataset.csv
├── requirements.txt          # Dependencias
└── README.md                # Este archivo
```

## 🔧 Archivos Generados

Al entrenar el modelo con `python train_modelo_4.py`, se generan automáticamente:
- `modelo_4_scaler.pkl`: Scaler para normalización de datos
- `modelo_4_trace.pkl`: Modelo entrenado (BayesianRidge)
- `modelo_4_results.pkl`: Métricas y resultados del entrenamiento

⚠️ **Nota**: Estos archivos son necesarios para la aplicación Streamlit pero no se incluyen en el repositorio. Debes entrenar el modelo primero.

## 📋 Workflow Recomendado

1. **Clonar repositorio**: `git clone <repo-url>`
2. **Instalar dependencias**: `pip install -r requirements.txt`  
3. **Ejecutar tests**: `python test_modelo_4.py` (verificar datos)
4. **Entrenar modelo**: `python train_modelo_4.py` (genera archivos .pkl)
5. **Usar aplicación**: `streamlit run app_modelo_4.py`
6. **Explorar análisis**: `jupyter notebook modelo_4_bayesian.ipynb`

## 📈 Resultados Esperados

- **R² > 0.8**: Buena capacidad predictiva
- **Overfitting < 5%**: Modelo generalizable
- **RMSE bajo**: Errores mínimos en predicciones
- **Intervalos de credibilidad**: Cuantificación de incertidumbre

## 🛠️ Tecnologías Utilizadas

- **PyMC**: Probabilistic programming
- **ArviZ**: Análisis bayesiano
- **Streamlit**: Interface web
- **Pandas/NumPy**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Scikit-learn**: Preprocesamiento y métricas

## 🧪 Testing

El proyecto incluye tests para:
- ✅ Carga de datos
- ✅ Validación de tipos
- ✅ Rangos de valores
- ✅ Lógica de correlaciones  
- ✅ Valores nulos
- ✅ Consistencia de datos

## 📚 Referencias

- [PyMC Documentation](https://docs.pymc.io/)
- [Bayesian Analysis with Python](https://github.com/aloctavodia/BAP)
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/)

---