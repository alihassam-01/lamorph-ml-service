from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.core.security import verify_internal_token
from app.schemas.analysis import ScanAnalysisResult
from app.services.analysis_service import process_analysis_request

router = APIRouter()

@router.post("/analyze", response_model=ScanAnalysisResult, dependencies=[Depends(verify_internal_token)])
async def analyze_image(
    file: UploadFile = File(...),
    metadata: str = Form(...)  # JSON string containing face contours & bounding boxes
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        return process_analysis_request(contents, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
