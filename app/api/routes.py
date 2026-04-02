from fastapi import APIRouter, HTTPException
from app.schemas.prediction import NewsInput, PredictionOutput
from app.core.model import news_model

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
def predict(news: NewsInput):
    try:
        result = news_model.predict(news.title, news.text)
        return PredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
def model_info():
    return {
        "model": "DistilBERT",
        "dataset": "WELFake (72K samples)",
        "labels": ["FAKE", "REAL"],
        "max_input_length": 512
    }
