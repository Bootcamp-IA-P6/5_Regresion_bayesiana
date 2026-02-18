# � Modelo 4: Regresión Poisson Bayesiana - Amazon Sales Dataset

## 📋 Descripción del Proyecto

Este proyecto implementa un **modelo de regresión Poisson bayesiana** para predecir `quantity_sold` (cantidad vendida) de productos Amazon usando variables temporales y de descuento.

### 🎯 Objetivos (Nivel Esencial)

✅ **Modelo ML funcional** que predice cantidad vendida (distribución Poisson)  
✅ **EDA completo** con análisis de sobredispersión y patrones temporales  
✅ **Overfitting < 5%** entre métricas de entrenamiento y validación  
✅ **Solución productizada** con Streamlit minimalista  
✅ **Informe de rendimiento** con MAE, RMSE e interpretación bayesiana  

## 📊 Dataset y Variables

**Variable objetivo**: `quantity_sold` (distribución Poisson)

**Variables predictoras**:
- `discount_percent`: Porcentaje de descuento aplicado
- `rating`: Calificación del producto (1-5)
- `is_weekend`: Si la venta ocurrió en fin de semana (0/1)
- `day_of_week`: Día de la semana (0=Lunes, 6=Domingo)
- `month`: Mes para capturar estacionalidad
- `product_category`: Categorías convertidas a dummies

## 🚀 Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el modelo completo
```bash
jupyter notebook 04_Modelo_Poisson_Bayesiano.ipynb
```

### 3. Usar la aplicación web
```bash
streamlit run app_simple_modelo_4.py
```

## 🧠 Metodología

### Modelo Poisson Bayesiano
- **Distribución**: Poisson (ideal para conteos como quantity_sold)
- **Framework**: PyMC para inferencia bayesiana
- **Variables**: discount_percent, rating, is_weekend + ingeniería temporal
- **Priors**: Normal(0, 1) para coeficientes, intercept centrado en log(media)
- **Función de enlace**: Log-link para garantizar predicciones positivas

### Preparación de Datos
- **Limpieza**: quantity_sold como entero ≥ 0
- **Ingeniería temporal**: day_of_week, is_weekend, month desde order_date
- **Codificación**: product_category a variables dummy
- **Estandarización**: Variables continuas normalizadas

### Métricas de Evaluación
- **MAE**: Mean Absolute Error (fácil interpretación para conteos)
- **RMSE**: Root Mean Square Error
- **Análisis de sobredispersión**: Ratio varianza/media
- **Control de overfitting**: Diferencia < 5% entre train/test

## 📁 Estructura del Proyecto

```
├── 04_Modelo_Poisson_Bayesiano.ipynb    # Notebook principal (TODO incluido)
├── app_simple_modelo_4.py               # Aplicación Streamlit minimalista
├── dataset/
│   └── amazon_sales_dataset.csv
├── requirements.txt                     # Dependencias
└── README.md                           # Este archivo
```

## 🔧 Archivos Generados

Al ejecutar el notebook completo, se generan automáticamente:
- `modelo_4_poisson_results.pkl`: Parámetros del modelo y métricas
- `modelo_4_poisson_trace.pkl`: Trace completo de PyMC para análisis avanzado

⚠️ **Nota**: Estos archivos son necesarios para la aplicación Streamlit pero no se incluyen en el repositorio.

## 📋 Workflow Recomendado

1. **Clonar repositorio**: `git clone <repo-url>`
2. **Instalar dependencias**: `pip install -r requirements.txt`  
3. **Ejecutar notebook**: `jupyter notebook 04_Modelo_Poisson_Bayesiano.ipynb`
4. **Usar aplicación**: `streamlit run app_simple_modelo_4.py`

## 🎯 Características del Modelo

### Análisis Exploratorio (EDA)
- ✅ Histograma de quantity_sold (forma Poisson típica)
- ✅ Análisis media/varianza para detectar sobredispersión  
- ✅ Gráficos weekend vs quantity_sold
- ✅ Correlaciones con descuentos y ratings

### Modelo Bayesiano
- ✅ Distribución Poisson para conteos
- ✅ Variables temporales (weekend, día, mes)
- ✅ Interpretación de coeficientes (efectos multiplicativos)
- ✅ Diagnósticos de convergencia (R-hat)

### Tests Integrados
- ✅ Validación de tipos de datos
- ✅ Verificación de variables temporales
- ✅ Control de convergencia del modelo
- ✅ Análisis de overfitting < 5%
- ✅ Validación de predicciones

## 📈 Resultados Esperados

- **MAE < 2.0**: Error promedio menor a 2 unidades
- **Overfitting < 5%**: Modelo generalizable
- **R-hat < 1.1**: Convergencia bayesiana adecuada
- **Interpretabilidad**: Efectos claros de descuento y weekend

## 🛠️ Tecnologías Utilizadas

- **PyMC**: Programación probabilística
- **ArviZ**: Análisis bayesiano y diagnósticos
- **Streamlit**: Interface web minimalista
- **Pandas/Polars**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización

## 🧪 Testing

Tests integrados en el notebook principal:
- ✅ Carga de datos correcta
- ✅ Creación de variables temporales
- ✅ Validación de quantity_sold (entero ≥ 0)
- ✅ Convergencia del modelo (R-hat < 1.1)
- ✅ Control de overfitting (< 5%)
- ✅ Predicciones razonables (MAE < 2.0)

## 📚 Referencias

- [PyMC Documentation](https://docs.pymc.io/)
- [Bayesian Analysis with Python](https://github.com/aloctavodia/BAP)
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/)

---