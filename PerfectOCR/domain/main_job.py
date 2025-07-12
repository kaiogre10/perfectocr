#PerfectOCR/domain/main_job
import uuid
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class ProcessingJob:
    source_uri: str  # La ruta del archivo original, URL, etc.
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
    image_data: np.ndarray | None = None
    processing_steps: list[dict] = field(default_factory=list) # Log de cada paso
    final_result: dict | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    has_correction_plan: bool = False