# PerfectOCR/domain/correction_plan.py
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class CorrectionPlanJob:
    """
    Job especializado para el plan de corrección.
    Es interno entre InputValidation y Preprocessing.
    No sale de estos dos coordinadores.
    """
    # === IDENTIFICACIÓN ===
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    # === DATOS DE ENTRADA ===
    image_array: np.ndarray | None = None
    source_uri: str = ""
    
    # === RESULTADOS DE VALIDACIÓN ===
    observations: List[str] = field(default_factory=list)
    correction_plans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_time: float = 0.0
    
    # === ESTADO ===
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message: str | None = None
    
    # === METADATOS ===
    enabled_engines: Dict[str, bool] = field(default_factory=dict)
    processing_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # === MÉTODOS ===
    def add_processing_step(self, step_name: str, details: Dict[str, Any] = None):
        """Agrega un paso al log de procesamiento."""
        step = {
            "step": step_name,
            "timestamp": datetime.utcnow(),
            "status": self.status
        }
        if details:
            step.update(details)
        self.processing_steps.append(step)
    
    def update_status(self, new_status: str, error_message: str = None):
        """Actualiza el estado del job."""
        self.status = new_status
        if error_message:
            self.error_message = error_message
        if new_status in ["COMPLETED", "FAILED"]:
            self.completed_at = datetime.utcnow()
    
    def has_failed(self) -> bool:
        """Verifica si el job ha fallado."""
        return self.status == "FAILED"
    
    def is_completed(self) -> bool:
        """Verifica si el job está completado."""
        return self.status == "COMPLETED"
    
    def get_summary(self) -> Dict[str, Any]:
        """Genera un resumen del job."""
        return {
            "job_id": self.job_id,
            "source_uri": self.source_uri,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "validation_time": self.validation_time,
            "engines_processed": len(self.correction_plans),
            "observations_count": len(self.observations),
            "error_message": self.error_message
        }
    
    def cleanup(self):
        """Limpia los datos después de ser usado por preprocessing."""
        self.image_array = None
        self.correction_plans = {}
        self.observations = []
        self.processing_steps = []
        self.status = "CLEANED"
        self.completed_at = datetime.utcnow()