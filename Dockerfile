# 1. Imagen base ligera de Python
FROM python:3.9-slim

# 2. Definimos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos el archivo de requisitos desde tu raíz al contenedor
# Al estar en la raíz, Docker lo encuentra directamente
COPY requirements.txt .

# 2. Variables de entorno para Python
#Le dice a Python que no genere carpetas __pycache__ ni archivos .pyc (archivos compilados).
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Instalamos las librerías
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos el CONTENIDO de tu carpeta local 'app' al contenedor
# Esto buscará la carpeta 'app' en tu raíz y copiará app.py (y lo que haya dentro)
COPY app/app.py .

# 5. Copiar carpetas necesarias para que la app funcione
# Copiamos las carpetas de datos, imágenes, modelos y código fuente
COPY data/ ./data/
COPY img/ ./img/
COPY models/ ./models/
COPY src/ ./src/

# 6. Abrimos el puerto de Streamlit
EXPOSE 8501

# 7. Ejecutamos la aplicación
# Como copiamos el contenido de 'app' a '.', app.py está en la raíz del WORKDIR
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]