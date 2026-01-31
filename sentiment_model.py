import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Dynamically load correct label mapping
id2label = model.config.id2label

NEGATIVE_KEYWORDS = [
    "loss", "losses", "decline", "drop", "fall", "down",
    "crash", "plunge", "bankrupt", "bankruptcy",
    "layoff", "layoffs", "cut jobs", "job cuts",
    "poor", "bad", "terrible", "worst", "weak",
    "missed estimates", "weak performance",
    "recession", "slowdown", "risk", "uncertain"
]

POSITIVE_KEYWORDS = [
    "gain", "profit", "profits", "profitability",
    "increase", "growth", "strong growth",
    "boost", "boosts", "boosted",
    "surge", "rise", "up",
    "beat estimates", "record revenue",
    "excellent", "strong performance",
    "good", "great", "amazing", "awesome", "very good",
    "improved", "improvement",
    "strong", "confidence", "positive", "recovery"
]

NEUTRAL_KEYWORDS = [
    "stable", "unchanged", "flat", "steady", "no change",
    "maintain", "maintained", "maintaining",
    "remains", "remain",
    "expected", "as expected", "in line",
    "forecast", "guidance", "outlook",
    "report", "announced", "statement",
    "neutral", "balanced", "mixed",
    "sideways", "range-bound",
    "slight", "minor", "marginal"
]


def predict_sentiment(text: str):
    text_lower = text.lower()

    # Rule-based overrides
    if any(word in text_lower for word in NEGATIVE_KEYWORDS):
        return "negative", 0.90
    if any(word in text_lower for word in POSITIVE_KEYWORDS):
        return "positive", 0.90
    if any(word in text_lower for word in NEUTRAL_KEYWORDS):
        return "neutral", 0.75

    # Model inference
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    score, idx = torch.max(probs, dim=1)

    sentiment = id2label[idx.item()].lower()
    confidence = float(score.item())

    # Normalize label names (FinBERT uses CAPITAL letters)
    if "pos" in sentiment:
        sentiment = "positive"
    elif "neg" in sentiment:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Force neutral if confidence is low
    if confidence < 0.6 and sentiment != "neutral":
        sentiment = "neutral"
        confidence = 0.6

    return sentiment, confidence
