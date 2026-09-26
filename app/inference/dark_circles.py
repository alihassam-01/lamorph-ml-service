import math
import numpy as np
from app.models.manager import model_manager

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def run_eyebag_inference(input_tensor: np.ndarray) -> dict:
    session = model_manager.eyebag_session
    if not session:
        raise RuntimeError("Eye-bag model not loaded")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    presence_tensor = outputs[0]
    severity_logits = outputs[1]

    presence_logit = float(np.ravel(presence_tensor)[0])
    severity_list = np.ravel(severity_logits).tolist()

    # Process severity
    if len(severity_list) >= 4:
        # Original model: 4 binary severity logits
        severity_probs = [sigmoid(v) for v in severity_list[:4]]
        grade = sum(1 for p in severity_probs if p > 0.5)
        confidence = sum(2.0 * abs(p - 0.5) for p in severity_probs) / len(severity_probs)
    elif len(severity_list) > 0:
        # Quantized model: single severity logit
        prob = sigmoid(severity_list[0])
        grade = min(4, max(0, round(prob * 4.0)))
        confidence = min(1.0, max(0.0, 2.0 * abs(prob - 0.5)))
    else:
        grade = 0
        confidence = 0.0

    presence_prob = sigmoid(presence_logit)

    return {
        "grade": grade,
        "presence_prob": presence_prob,
        "confidence": confidence,
    }
