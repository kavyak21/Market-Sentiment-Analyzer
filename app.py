# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from text_cleaner import clean_text
from sentiment_model import predict_sentiment
from database import save_sentiment

app = FastAPI(title="Pulse Sentiment API")

class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: SentimentRequest):
    cleaned = clean_text(payload.text)
    sentiment, confidence = predict_sentiment(cleaned)

    save_sentiment(payload.text, "app", sentiment, confidence)

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
