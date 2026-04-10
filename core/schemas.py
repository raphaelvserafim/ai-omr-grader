from pydantic import BaseModel
from typing import Dict, Optional, Any

class OMRRequest(BaseModel):
    image_base64: str

class OMRResponse(BaseModel):
    status: str
    gabarito: Optional[Dict[int, Any]] = None
    erro: Optional[str] = None