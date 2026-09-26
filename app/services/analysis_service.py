import io
import time
import json
from typing import Dict, Any, List
from PIL import Image

from app.schemas.analysis import ScanAnalysisResult, DetectedIssue, SkinObservation
from app.preprocessing.pipeline import prepare_eyebag_input
from app.inference.dark_circles import run_eyebag_inference
from app.core.config import settings
import logging

logger = logging.getLogger("ml_service.analysis")

def get_eyebag_severity(grade: int) -> str:
    if grade >= 4: return 'Severe'
    if grade >= 2: return 'Moderate'
    return 'Mild'

def process_analysis_request(image_bytes: bytes, metadata_json: str) -> ScanAnalysisResult:
    start_time = time.time()

    # 1. Decode inputs
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    metadata = json.loads(metadata_json)

    issues: List[DetectedIssue] = []

    # 2. Extract crops based on metadata (sent from frontend face detector)
    left_eye_box = metadata.get("left_eye_box")
    right_eye_box = metadata.get("right_eye_box")

    # Process Left Eye
    if left_eye_box:
        try:
            tensor = prepare_eyebag_input(image, left_eye_box)
            res = run_eyebag_inference(tensor)
            if res["grade"] > 0:
                issues.append(DetectedIssue(
                    issueType="Left Eye Bag",
                    severity=get_eyebag_severity(res["grade"]),
                    confidence=res["confidence"],
                    x=left_eye_box["x"] / image.width,
                    y=left_eye_box["y"] / image.height,
                    width=left_eye_box["width"] / image.width,
                    height=left_eye_box["height"] / image.height,
                ))
        except Exception as e:
            logger.error(f"Error processing left eye: {e}")

    # Process Right Eye
    if right_eye_box:
        try:
            tensor = prepare_eyebag_input(image, right_eye_box)
            res = run_eyebag_inference(tensor)
            if res["grade"] > 0:
                issues.append(DetectedIssue(
                    issueType="Right Eye Bag",
                    severity=get_eyebag_severity(res["grade"]),
                    confidence=res["confidence"],
                    x=right_eye_box["x"] / image.width,
                    y=right_eye_box["y"] / image.height,
                    width=right_eye_box["width"] / image.width,
                    height=right_eye_box["height"] / image.height,
                ))
        except Exception as e:
            logger.error(f"Error processing right eye: {e}")

    # Build Observations (placeholder for full integration)
    observations = [
        SkinObservation(
            concernId="dark_circles",
            measurementStatus="measured" if (left_eye_box or right_eye_box) else "unavailable",
            confidenceType="unavailable"
        )
    ]

    # Overall score basic logic
    penalty = sum(24 if i.severity in ['Severe', 'Pronounced'] else 16 if i.severity == 'Moderate' else 8 for i in issues)
    overall_score = max(45.0, min(96.0, 92 - penalty))

    return ScanAnalysisResult(
        overallScore=overall_score,
        issues=issues,
        observations=observations,
        notes=f"Analysis completed in {(time.time() - start_time) * 1000:.0f}ms",
        scannerVersion=settings.scanner_version,
        modelBundleVersion=settings.model_bundle_version,
        appVersion=settings.app_version,
        observationSchemaVersion=settings.observation_schema_version,
        modelManifest={}
    )
