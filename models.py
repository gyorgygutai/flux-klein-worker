from pydantic import BaseModel, Field

class RequestPayload(BaseModel):
    prompt: str
    height: int = Field(..., ge=256, le=2048)
    width: int = Field(..., ge=256, le=2048)

class ResponsePayload(BaseModel):
    image: str