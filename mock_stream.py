# ingestion/mock_stream.py

import time
import random
from database import save_sentiment
from sentiment_model import predict_sentiment

SAMPLE_TWEETS = [
    # Negative
    "Company posted huge losses this quarter",
    "Market reacts negatively to earnings miss",
    "Worst experience ever with this stock",
    "Company announces massive layoffs",
    "Revenue decline worries investors",

    # Positive
    "Strong revenue growth boosts investor confidence",
    "Shares surge after positive guidance",
    "Company reports record profits this year",
    "Strong performance continues across all segments",
    "Revenue growth exceeds analyst estimates",
    "Company reports best quarter ever",
    "Investors optimistic about future expansion",
    "Stock prices surge after record profits",
    "Company beats earnings expectations",
    "Strong growth outlook boosts investor confidence",
    "Market rallies on positive economic data",
    "Shares jump after successful product launch",

    # Neutral
    "Stock performance is stable",
    "Investors are waiting for the official earnings report",
    "Market remains steady with no major changes",
    "Company released its annual statement today",
    "Outlook remains in line with expectations",

    # Mixed
    "Slight increase in revenue but costs remain high",
    "Growth reported, but investors remain cautious"
]


def stream_mock_data():
    print("📡 Starting mock market sentiment stream...\n")

    while True:
        texts = SAMPLE_TWEETS.copy()
        random.shuffle(texts)        # <-- SHUFFLE HERE

        for text in texts:
            sentiment, confidence = predict_sentiment(text)

            save_sentiment(
                text=text,
                source="mock",
                sentiment=sentiment,
                confidence=confidence
            )

            print(f"📝 Ingested: {text}")
            print(f"👉 Sentiment: {sentiment} | Confidence: {confidence:.2f}\n")

            time.sleep(5)


if __name__ == "__main__":
    stream_mock_data()
