


# 📊 Proyecto de Modelado Bayesiano con PyMC
Este repositorio contiene una implementación avanzada de modelos estadísticos bajo el enfoque bayesiano, utilizando PyMC para el muestreo y ArviZ para el análisis de diagnósticos y visualización de resultados.

El proyecto abarca tres arquitecturas fundamentales:

Regresión Lineal Bayesiana: Para entender relaciones continuas.

Regresión Logística Bayesiana: Para problemas de clasificación y probabilidades.

Modelo Jerárquico (Multinivel): Para capturar la variabilidad en diferentes niveles de agrupación de los datos, permitiendo el "intercambio de información" entre grupos.

🚀 Guía de Inicio Rápido
Sigue estos pasos para replicar el entorno de desarrollo y ejecutar los modelos.

1. **Preparación del Entorno**
Es fundamental aislar las dependencias para evitar conflictos de versiones.
###  Crear el entorno virtual
python -m venv venv

### Activar el entorno (Windows)
.\venv\Scripts\activate

### Activar el entorno (Linux/Mac)
source venv/bin/activate  

### 2. Instalación de Dependencias
Utilizamos librerías de alto rendimiento para el manejo de datos y computación científica:

- Polars: Para un procesamiento de datos ultra rápido (alternativa eficiente a Pandas).

- PyMC: Nuestro motor de inferencia bayesiana.

- ArviZ: Herramienta esencial para diagnósticos de cadenas MCMC y visualización.

- Joblib: Para la persistencia de modelos y paralelización.

- pip install pymc arviz polars joblib matplotlib seaborn

## 🛠️ Flujo de Trabajo del Proyecto

El desarrollo se dividió en las siguientes fases técnicas:Carga de Datos: Implementada con polars para garantizar eficiencia en la lectura y preprocesamiento.Definición del Prior: Selección de distribuciones a priori (Normal, Half-Cauchy, etc.) basadas en conocimiento experto o criterios no informativos.Muestreo (Inferencia): Ejecución del algoritmo NUTS (No-U-Turn Sampler) para obtener las distribuciones posteriores.Validación: Uso de arviz para verificar la convergencia mediante el indicador $\hat{R}$ (R-hat) y el tamaño efectivo de la muestra (ESS).Serialización: Guardado de los trazos y modelos resultantes mediante joblib para su posterior uso sin necesidad de re-entrenar.




📊 Proyecto de Modelado Bayesiano con PyMC
Este repositorio contiene una implementación avanzada de modelos estadísticos bajo el enfoque bayesiano, utilizando PyMC para el muestreo y ArviZ para el análisis de diagnósticos y visualización de resultados.

El proyecto abarca tres arquitecturas fundamentales:

Regresión Lineal Bayesiana: Para entender relaciones continuas.

Regresión Logística Bayesiana: Para problemas de clasificación y probabilidades.

Modelo Jerárquico (Multinivel): Para capturar la variabilidad en diferentes niveles de agrupación de los datos, permitiendo el "intercambio de información" entre grupos.

🚀 Guía de Inicio Rápido


## Estructura de Carpetas 
![Estructura del modelo](https://github.com/Bootcamp-IA-P6/5_Regresion_bayesiana/blob/develop/img/Estructura.png?raw=true)



1. Preparación del Entorno
Es fundamental aislar las dependencias para evitar conflictos de versiones.

Bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno (Windows)
.\venv\Scripts\activate

# Activar el entorno (Linux/Mac)
source venv/bin/activate
2. Instalación de Dependencias
Utilizamos librerías de alto rendimiento para el manejo de datos y computación científica:

Polars: Para un procesamiento de datos ultra rápido (alternativa eficiente a Pandas).

PyMC: Nuestro motor de inferencia bayesiana.

ArviZ: Herramienta esencial para diagnósticos de cadenas MCMC y visualización.

Joblib: Para la persistencia de modelos y paralelización.

Bash
pip install pymc arviz polars joblib matplotlib seaborn
🛠️ Flujo de Trabajo del Proyecto
El desarrollo se dividió en las siguientes fases técnicas:

Carga de Datos: Implementada con polars para garantizar eficiencia en la lectura y preprocesamiento.

Definición del Prior: Selección de distribuciones a priori (Normal, Half-Cauchy, etc.) basadas en conocimiento experto o criterios no informativos.

Muestreo (Inferencia): Ejecución del algoritmo NUTS (No-U-Turn Sampler) para obtener las distribuciones posteriores.

Validación: Uso de arviz para verificar la convergencia mediante el indicador  
R
^
  (R-hat) y el tamaño efectivo de la muestra (ESS).

Serialización: Guardado de los trazos y modelos resultantes mediante joblib para su posterior uso sin necesidad de re-entrenar.

### 📈 Resumen de Modelos
Modelo	Uso Principal	Características
Lineal	Predicción de valores continuos.	Relación directa entre variables independientes y dependientes.
Logístico	Clasificación binaria.	Uso de función de enlace logit para modelar probabilidades.
Jerárquico	Datos agrupados o anidados.	Estima parámetros globales y locales simultáneamente, ideal para datos con estructura de grupos.


## Modelo Bayesiano Lineal 

![Modelo Bayesiano Lineal 1](https://github.com/Bootcamp-IA-P6/5_Regresion_bayesiana/blob/develop/img/ModeloLinealBayesiano1.png?raw=true)

📈 Regresión Lineal Bayesiana: Interpretación de Resultados
Un Modelo Bayesiano Lineal estima la relación entre una variable dependiente (Ingreso Total) y una independiente (Precio Descontado) utilizando distribuciones de probabilidad. A diferencia de la regresión tradicional que te da una sola línea "fija", aquí obtenemos todo un rango de posibilidades que cuantifican nuestra incertidumbre.


Línea Roja (Media de la Regresión): Representa el valor más probable de la relación. Indica que a medida que el precio descontado (estandarizado) aumenta, el ingreso total tiende a subir siguiendo esta trayectoria central.

Haces de Líneas Azules (Muestras de la Posterior): Cada línea azul es una hipótesis válida generada por el modelo. Al haber muchas líneas cerca de la roja, confirmamos que el modelo tiene una dirección clara, aunque la dispersión en los valores altos muestra dónde hay mayor incertidumbre.

Intervalo de Credibilidad (94%): El sombreado gris (HDI) define el rango donde, con un 94% de certeza, se encuentra la verdadera relación. Es la herramienta clave para la toma de decisiones basada en riesgos.

Estandarización: El eje X está estandarizado (centrado en 0), lo que facilita que el algoritmo de PyMC converja más rápido y que el intercepto sea más fácil de interpretar.


## Modelo Bayesiano Logistico 
![Modelo Bayesiano Logistico 2 ]([img/ModeloLogistico2.(https://github.com/Bootcamp-IA-P6/5_Regresion_bayesiana/blob/develop/img/modeloLogistico2.png?raw=true))

Eje Y - P(Best Seller): Representa la probabilidad de ser un "Súper Ventas". El valor varía de 0 a 1 (0% a 100%).

Línea Azul Central: Es la media posterior. Nos indica la tendencia promedio. Curiosamente, en tu gráfico la línea es casi plana cerca del 0.5 (50%), lo que sugiere que, para este modelo en particular, el rating por sí solo no es un predictor extremadamente fuerte para cambiar la probabilidad de ser Best Seller.

Área Sombreada Azul (Incertidumbre 5-95%): Este es el Intervalo de Credibilidad. Es la parte más importante del análisis bayesiano:

Incertidumbre Alta: Al ser un área muy ancha (que va desde casi 0.1 hasta 0.9), el modelo nos está diciendo: "No tengo datos suficientes o el rating es muy ruidoso para asegurar si un producto será Best Seller".

Si tuviéramos miles de datos muy claros, esa banda sombreada sería muy delgadita alrededor de la línea central.

A diferencia de los modelos tradicionales, el uso de PyMC nos permite visualizar no solo la probabilidad media, sino el grado de incertidumbre (Intervalo de Credibilidad del 90%). En la gráfica se observa que el modelo mantiene una postura cautelosa debido a la dispersión de los datos, lo cual es vital para evitar decisiones basadas en falsas certezas.


## Modelo Jerarquico Bayesiano 
![Modelo Bayesiano Jerarquico 3 ](https://github.com/Bootcamp-IA-P6/5_Regresion_bayesiana/blob/develop/img/ModeloJerarquico3.png?raw=true))

🏛️ Modelo Bayesiano Jerárquico (Multinivel)
El objetivo de este modelo es capturar la estructura anidada de los datos. En lugar de asumir que todas las categorías se comportan igual, permitimos que cada una tenga su propio intercepto, pero compartiendo una distribución común ("hiperprior").

Beneficios clave:

Intercambio de información: Los grupos con mucha información ayudan a estabilizar las estimaciones de los grupos con pocos datos.

Robustez: Reduce el riesgo de sobreajuste en categorías pequeñas.

Análisis Comparativo: Como se observa en el gráfico de intervalos (HDI), podemos comparar directamente si las diferencias entre categorías (p. ej., a_cat[1] vs a_cat[3]) son estadísticamente significativas si sus intervalos no se solapan.





