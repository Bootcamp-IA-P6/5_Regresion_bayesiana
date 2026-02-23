FROM python:3.11-slim

# 1. Instalamos dependencias del sistema necesarias para compilar modelos bayesianos
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copiamos e instalamos requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copiamos el resto del proyecto
COPY . ..

# 3. EJECUCIÓN
# Como tu app.py está dentro de una carpeta llamada 'app', 
# la ruta dentro del contenedor será /app/app/app.py
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]