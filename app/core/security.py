from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)

async def verify_internal_token(api_key_header: str = Security(api_key_header)):
    if settings.internal_api_token and api_key_header != settings.internal_api_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return api_key_header
