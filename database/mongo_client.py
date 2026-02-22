import os
from pymongo import MongoClient
from datetime import datetime

# Configuración de conexión para Docker
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
client = MongoClient(MONGO_URI)
db = client["db_amazon_modelos"]  # Nombre de tu DB
collection = db["historial_predicciones"]

def registrar_prediccion(nombre_modelo, input_usuario, resultado):
    """
    Guarda cualquier predicción en MongoDB.
    - nombre_modelo: str (ej: "Modelo 1 - Revenue")
    - input_usuario: dict (los datos que metió el usuario)
    - resultado: float o dict (lo que devolvió el modelo)
    """
    documento = {
        "fecha": datetime.utcnow(),
        "modelo": nombre_modelo,
        "inputs": input_usuario,
        "resultado": resultado
    }
    
    try:
        res = collection.insert_one(documento)
        print(f"✅ [{nombre_modelo}] guardado con éxito.")
        return res.inserted_id
    except Exception as e:
        print(f"❌ Error al guardar en DB: {e}")
        return None