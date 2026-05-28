from pydantic import BaseModel, Field
from typing import Optional, List

class RequestPayload(BaseModel):
    prompt: str
    height: int = Field(..., ge=256, le=2048)
    width: int = Field(..., ge=256, le=2048)
    input_image: Optional[str] = None
    reference_images: Optional[List[str]] = None

class ResponsePayload(BaseModel):
    image: str