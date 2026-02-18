# Modelo 4: Regresion Poisson Bayesiana - Amazon Sales Dataset

## Descripcion del Proyecto

Este proyecto implementa un **modelo de regresion Poisson bayesiana** para predecir `quantity_sold` (cantidad vendida) de productos Amazon usando variables temporales y de descuento.

### Objetivos (Nivel Esencial)4: Regresión Poisson Bayesiana - Amazon Sales Dataset

## Descripción del Proyecto️ Modelo 4: Regresión Poisson Bayesiana - Amazon Sales Dataset � Modelo 4: Regresión Poisson Bayesiana - Amazon Sales Dataset

## 📋 Descripción del Proyecto

Este proyecto implementa un **modelo de regresión Poisson bayesiana** para predecir `quantity_sold` (cantidad vendida) de productos Amazon usando variables temporales y de descuento.

### 🎯 Objetivos (Nivel Esencial)

- [x] **Modelo ML funcional** que predice cantidad vendida (distribucion Poisson)  
- [x] **EDA completo** con analisis de sobredispersion y patrones temporales  
- [x] **Overfitting < 5%** entre metricas de entrenamiento y validacion  
- [x] **Solucion productizada** con Streamlit minimalista  
- [x] **Informe de rendimiento** con MAE, RMSE e interpretacion bayesiana  

## Dataset y Variables

**Variable objetivo**: `quantity_sold` (distribucion Poisson)

**Variables predictoras**:
- `discount_percent`: Porcentaje de descuento aplicado
- `rating`: Calificacion del producto (1-5)
- `is_weekend`: Si la venta ocurrio en fin de semana (0/1)
- `day_of_week`: Dia de la semana (0=Lunes, 6=Domingo)
- `month`: Mes para capturar estacionalidad
- `product_category`: Categorias convertidas a dummies

## Instalacion y Uso

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

## Metodologia

### Modelo Poisson Bayesiano
- **Distribucion**: Poisson (ideal para conteos como quantity_sold)
- **Framework**: PyMC para inferencia bayesiana
- **Variables**: discount_percent, rating, is_weekend + ingenieria temporal
- **Priors**: Normal(0, 1) para coeficientes, intercept centrado en log(media)
- **Funcion de enlace**: Log-link para garantizar predicciones positivas

### Preparacion de Datos
- **Limpieza**: quantity_sold como entero ≥ 0
- **Ingenieria temporal**: day_of_week, is_weekend, month desde order_date
- **Codificacion**: product_category a variables dummy
- **Estandarizacion**: Variables continuas normalizadas

### Metricas de Evaluacion
- **MAE**: Mean Absolute Error (facil interpretacion para conteos)
- **RMSE**: Root Mean Square Error
- **Analisis de sobredispersion**: Ratio varianza/media
- **Control de overfitting**: Diferencia < 5% entre train/test

## Estructura del Proyecto

```
├── 04_Modelo_Poisson_Bayesiano.ipynb    # Notebook principal (TODO incluido)
├── app_simple_modelo_4.py               # Aplicación Streamlit minimalista
├── dataset/
│   └── amazon_sales_dataset.csv
├── requirements.txt                     # Dependencias
└── README.md                           # Este archivo
```

## Archivos Generados

Al ejecutar el notebook completo, se generan automaticamente:
- `modelo_4_poisson_results.pkl`: Parametros del modelo y metricas
- `modelo_4_poisson_trace.pkl`: Trace completo de PyMC para analisis avanzado

**Nota**: Estos archivos son necesarios para la aplicacion Streamlit pero no se incluyen en el repositorio.

## Workflow Recomendado

1. **Clonar repositorio**: `git clone <repo-url>`
2. **Instalar dependencias**: `pip install -r requirements.txt`  
3. **Ejecutar notebook**: `jupyter notebook 04_Modelo_Poisson_Bayesiano.ipynb`
4. **Usar aplicación**: `streamlit run app_simple_modelo_4.py`

## Caracteristicas del Modelo

### Analisis Exploratorio (EDA)
- [x] Histograma de quantity_sold (forma Poisson tipica)
- [x] Analisis media/varianza para detectar sobredispersion  
- [x] Graficos weekend vs quantity_sold
- [x] Correlaciones con descuentos y ratings

### Modelo Bayesiano
- [x] Distribucion Poisson para conteos
- [x] Variables temporales (weekend, dia, mes)
- [x] Interpretacion de coeficientes (efectos multiplicativos)
- [x] Diagnosticos de convergencia (R-hat)

### Tests Integrados
- [x] Validacion de tipos de datos
- [x] Verificacion de variables temporales
- [x] Control de convergencia del modelo
- [x] Analisis de overfitting < 5%
- [x] Validacion de predicciones

## Resultados Esperados

- **MAE < 2.0**: Error promedio menor a 2 unidades
- **Overfitting < 5%**: Modelo generalizable
- **R-hat < 1.1**: Convergencia bayesiana adecuada
- **Interpretabilidad**: Efectos claros de descuento y weekend

## Tecnologias Utilizadas

- **PyMC**: Programación probabilística
- **ArviZ**: Análisis bayesiano y diagnósticos
- **Streamlit**: Interface web minimalista
- **Pandas/Polars**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización

## Testing

Tests integrados en el notebook principal:
- [x] Carga de datos correcta
- [x] Creacion de variables temporales
- [x] Validacion de quantity_sold (entero ≥ 0)
- [x] Convergencia del modelo (R-hat < 1.1)
- [x] Control de overfitting (< 5%)
- [x] Predicciones razonables (MAE < 2.0)

## Referencias

- [PyMC Documentation](https://docs.pymc.io/)
- [Bayesian Analysis with Python](https://github.com/aloctavodia/BAP)
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/)

---