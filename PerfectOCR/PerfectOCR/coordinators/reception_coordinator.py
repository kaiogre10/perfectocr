import subprocess
import sys
import os
import time
import logging

logger = logging.getLogger(__name__)

class ReceptionCoordinator:
    """
    Coordinador del módulo de recepción de imágenes preprocesadas.
    Solo se encarga de iniciar/detener el servidor receptor.
    NO hace OCR - esa responsabilidad es exclusiva del OCREngineCoordinator.
    """
    def __init__(self, script_path, wait_time=2):
        self.script_path = script_path
        self.process = None
        self.wait_time = wait_time

    def start_reception_server(self):
        """Inicia el servidor de recepción de imágenes preprocesadas."""
        if self.process is None or self.process.poll() is not None:
            logger.info("🔄 Iniciando servidor de recepción de imágenes preprocesadas...")
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(self.wait_time)  # Espera a que el servidor arranque
            logger.info("✅ Servidor de recepción activo")

    def stop_reception_server(self):
        """Detiene el servidor de recepción."""
        if self.process and self.process.poll() is None:
            logger.info("🔄 Deteniendo servidor de recepción")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                logger.info("✅ Servidor de recepción detenido")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Servidor de recepción no respondió, forzando cierre")
                self.process.kill()
            self.process = None