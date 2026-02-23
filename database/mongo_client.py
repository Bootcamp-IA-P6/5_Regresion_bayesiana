import os
from pymongo import MongoClient
from datetime import datetime

# Configuración de conexión para Docker
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
client = MongoClient(MONGO_URI)
db = client["db_amazon_modelos"]  # Nombre de tu DB
collection = db["historial_predicciones"]

def registrar_prediccion(nombre_modelo, input_usuario, resultado, version="v1"):
    """
    Guarda cualquier predicción en MongoDB.
    - nombre_modelo: str (ej: "Modelo 1 - Revenue")
    - input_usuario: dict (los datos que metió el usuario)
    - resultado: float o dict (lo que devolvió el modelo)
    """
    documento = {
        "fecha": datetime.utcnow(),
        "modelo": nombre_modelo,
        "version": version,  
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
    
    # Aqui se guardara el proceso de datos de los 3 modelo, se puede agregar otro modelo sin necesidad de cambiar 
    # este fragmento de código 
def obtener_estadisticas_precios(nombre_modelo, campo_input):
    # Usamos f-string para construir la ruta del campo en el documento de Mongo
    campo_ruta = f"inputs.{campo_input}" 
    
    pipeline = [
        { "$match": { "modelo": nombre_modelo, campo_ruta: { "$exists": True } } },
        { "$group": { "_id": None, "promedio": { "$avg": f"${campo_ruta}" } } }
    ]
    
    try:
        resultado = list(collection.aggregate(pipeline))
        if resultado and resultado[0]["promedio"] is not None:
            return resultado[0]["promedio"]
        return 0.0
    except Exception as e:
        print(f"Error en MongoDB: {e}")
        return 0.0