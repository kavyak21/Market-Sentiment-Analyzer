# 📊 Market Sentiment Analyzer (Real-Time)

A real-time **Market Sentiment Analysis system** that processes streaming financial text and classifies market sentiment as **Positive, Negative, or Neutral** using a hybrid approach combining **FinBERT (Transformer-based NLP)** and **rule-based financial keyword analysis**.  
The system visualizes sentiment trends through an interactive **Streamlit dashboard**.

---

## 🚀 Features
- Real-time sentiment inference on financial text
- Finance-domain transformer model (**FinBERT**)
- Hybrid rule-based + ML-based sentiment detection
- Confidence score for each prediction
- Live dashboard for sentiment monitoring
- Modular and scalable architecture

---

## 🏗️ Project Structure
Market-Sentiment-Analyzer/
   - app.py             # FastAPI backend
   - sentiment_model.py # FinBERT-based sentiment inference logic
   - text_cleaner.py    # Text preprocessing & cleaning
   - mock_stream.py     # Simulated real-time data stream
   - dashboard.py       # Streamlit dashboard
   - requirements.txt   # Project dependencies
   - README.md          # Project documentation

---

## 🧠 Sentiment Analysis Model
- **Model Used:** `ProsusAI/finbert`
- **Type:** Transformer-based (BERT architecture)
- **Domain:** Financial news and market-related text

FinBERT is a pre-trained model fine-tuned specifically on financial datasets, making it highly suitable for analyzing:
- Market news
- Earnings reports
- Financial announcements
- Economic statements

The model outputs sentiment labels (**Positive, Neutral, Negative**) along with probability-based confidence scores.

---

## 🔍 Sentiment Decision Logic
The system uses a **hybrid approach**:

### 1️⃣ Rule-Based Override
Strong financial keywords trigger immediate sentiment decisions:
- **Positive / Negative:** Confidence = 0.90
- **Neutral:** Confidence = 0.75

This improves reliability for critical market signals such as *profits, losses, layoffs, growth,* etc.

### 2️⃣ Model-Based Inference
If no keyword match is found:
- Text is passed to FinBERT
- Softmax probabilities are computed
- The highest probability determines sentiment and confidence

### 3️⃣ Confidence Normalization
- Low-confidence predictions are normalized to **Neutral**
- Reduces false positives in uncertain cases

---

## 📈 Confidence Score Methodology
- Rule-based predictions use fixed confidence values
- Model-based predictions use softmax probabilities
- Confidence scores help interpret model reliability in real-time analytics

---

## ⚙️ How to Run the Project

### 1️⃣ Create & activate virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Start FastAPI backend
```
uvicorn app:app --reload
```

API runs at:  
http://127.0.0.1:8000

### 4️⃣ Start mock data stream
```
python mock_stream.py
```

### 5️⃣ Run Streamlit dashboard
```
streamlit run dashboard.py
```

Dashboard opens at:  
http://localhost:8501

## 📊 Dashboard Overview

The Streamlit dashboard provides:
- Real-time sentiment updates
- Distribution of Positive / Negative / Neutral sentiment
- Confidence-aware sentiment visualization
- Continuous monitoring of market mood
