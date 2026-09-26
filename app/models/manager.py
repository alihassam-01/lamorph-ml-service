import logging
from pathlib import Path
import onnxruntime as ort
from app.core.config import settings

logger = logging.getLogger("ml_service.models")

class ModelManager:
    def __init__(self):
        self.wrinkle_session = None
        self.eyebag_session = None

    def load_models(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.ort_intra_op_threads
        opts.inter_op_num_threads = settings.ort_inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load Wrinkle model
        w_path = settings.wrinkle_model_path
        if w_path.exists():
            try:
                self.wrinkle_session = ort.InferenceSession(str(w_path), opts)
                logger.info(f"Loaded wrinkle model from {w_path}")
            except Exception as e:
                logger.error(f"Failed to load wrinkle model: {e}")
        else:
            logger.warning(f"Wrinkle model not found at {w_path}")

        # Load Eye-bag model
        e_path = settings.eyebag_model_path
        if e_path.exists():
            try:
                self.eyebag_session = ort.InferenceSession(str(e_path), opts)
                logger.info(f"Loaded eye-bag model from {e_path}")
            except Exception as e:
                logger.error(f"Failed to load eye-bag model: {e}")
        else:
            logger.warning(f"Eye-bag model not found at {e_path}")

model_manager = ModelManager()
