from pydantic import BaseModel, Field

class NewsInput(BaseModel):
    title: str = Field(..., min_length=5, max_length=500, example="Scientists discover new vaccine")
    text: str = Field(..., min_length=10, max_length=5000, example="Researchers at MIT have developed...")

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float