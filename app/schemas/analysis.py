from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DetectedIssue(BaseModel):
    issueType: str
    severity: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

class SkinObservation(BaseModel):
    concernId: str
    measurementStatus: str
    severity: Optional[float] = None
    severitySource: Optional[str] = None
    confidence: Optional[float] = None
    confidenceType: str
    confidenceIsCalibrated: bool = False
    region: Optional[str] = None
    spread: Optional[float] = None
    modelArtifact: Optional[str] = None
    modelVersion: Optional[str] = None
    modelHash: Optional[str] = None
    preprocessingVersion: Optional[str] = None
    postprocessingVersion: Optional[str] = None
    calibrationVersion: Optional[str] = None
    rawMetrics: Dict[str, Any] = {}

class ScanAnalysisResult(BaseModel):
    overallScore: float
    issues: List[DetectedIssue]
    observations: List[SkinObservation]
    notes: str = ""
    wrinkleNotes: str = ""
    eyeBagNotes: str = ""

    # We omit the raw byte arrays (like maskImageBytes) to keep the JSON light,
    # but they can be sent as base64 or multipart if really needed.

    scannerVersion: str
    modelBundleVersion: str
    appVersion: str
    observationSchemaVersion: str
    modelManifest: Dict[str, Any]
