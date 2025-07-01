# PerfectOCR/core/reception/recibing_results_server.py - SERVIDOR DE RECEPCIÓN
import os
import yaml
import time
import logging
import uvicorn  # ✅ AÑADIDO: Para ejecutar FastAPI
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List

# Importar el worker de recepción
from core.reception.reception_worker import reception_worker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ruta al archivo de configuración
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "master_config.yaml")

def load_reception_config():
    """Carga la configuración del YAML."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get("enhancement_service_api", {})  # Mantener clave por compatibilidad
    except Exception as e:
        logging.error(f"Error cargando configuración: {e}")
        return {}

def is_reception_enabled():
    """Verifica si el servicio de recepción está habilitado."""
    cfg = load_reception_config()
    return cfg.get("enabled", False)

def get_api_hash():
    """Obtiene el hash de autenticación."""
    cfg = load_reception_config()
    return cfg.get("api_hash", "")

# Usar la configuración en el servidor
reception_config = load_reception_config()

# =========================
# SERVIDOR FASTAPI DE RECEPCIÓN  ✅ CORREGIDO: Cambié comentario
# =========================

app = FastAPI()

# Modelos Pydantic
class PreprocessedImagesRequest(BaseModel):
    document_id: str
    ocr_images: Dict[str, str]  # base64 images
    noise_regions: List[List[int]]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    service: str
    message: str

@app.get('/health', response_model=HealthResponse)
async def health():
    """Endpoint de salud del servicio de recepción."""
    return HealthResponse(
        status="healthy",
        service="Image_Reception_Server",
        message="Servidor de recepción funcionando correctamente"
    )

@app.post('/receive_preprocessed')
async def receive_preprocessed(request: Request):
    """
    Recibe imágenes preprocesadas de sistemas externos.
    Solo guarda las imágenes - NO hace OCR.
    """
    try:
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail="No se recibieron datos JSON.")
        
        # Validación de hash
        api_hash = get_api_hash()
        received_hash = data.get("metadata", {}).get("api_hash")
        
        if received_hash != api_hash:
            logger.warning(f"⚠️ Hash inválido recibido: {received_hash}")
            raise HTTPException(status_code=403, detail="Hash de autenticación inválido.")
        
        doc_id = data.get("document_id", "sin_id")
        ocr_images_b64 = data.get("ocr_images", {})
        noise_regions = data.get("noise_regions", [])
        
        logger.info(f"📨 Recibiendo imágenes preprocesadas para: {doc_id}")
        
        # Solo recibir y guardar imágenes - SIN OCR
        result = reception_worker.receive_and_save_images(doc_id, ocr_images_b64, noise_regions)
        
        if "error" not in result:
            # ✅ SIMPLIFICADO: Sin JSONResponse
            return {
                "status": "received",
                "document_id": doc_id,
                "saved_images": len(result.get("saved_images", {})),
                "message": "Imágenes preprocesadas recibidas y guardadas correctamente"
            }
        else:
            logger.error(f"❌ Error recibiendo imágenes para {doc_id}: {result['error']}")
            raise HTTPException(status_code=500, detail=result.get("error"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en endpoint receive_preprocessed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/activate')
async def activate():
    """Endpoint para activar el servicio de recepción."""
    # ✅ SIMPLIFICADO: Sin JSONResponse
    return {
        "status": "activated",
        "message": "Servidor de recepción listo para recibir imágenes"
    }

def main():
    """Función principal del servidor de recepción."""
    if not is_reception_enabled():
        logger.info("❌ Servicio de recepción desactivado por configuración.")
        return
    
    logger.info("✅ Servidor de Recepción de Imágenes habilitado")
    logger.info("🚀 Iniciando servidor de recepción en puerto 8000...")
    logger.info("📋 Funciones: Recibe imágenes preprocesadas - NO hace OCR")
    
    # ✅ CORREGIDO: Flask app.run() → FastAPI uvicorn.run()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()