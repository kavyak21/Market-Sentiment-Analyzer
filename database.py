# storage/database.py
import sqlite3
from datetime import datetime

conn = sqlite3.connect("pulse.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sentiment_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    source TEXT,
    sentiment TEXT,
    confidence REAL,
    timestamp TEXT
)
""")
conn.commit()

def save_raw_text(text, source):
    cursor.execute("""
    INSERT INTO sentiment_data (text, source, timestamp)
    VALUES (?, ?, ?)
    """, (text, source, datetime.now().isoformat()))
    conn.commit()

def save_sentiment(text, source, sentiment, confidence):
    cursor.execute("""
    INSERT INTO sentiment_data
    (text, source, sentiment, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (text, source, sentiment, confidence, datetime.now().isoformat()))
    conn.commit()

def fetch_sentiments():
    cursor.execute("SELECT sentiment, timestamp FROM sentiment_data")
    return cursor.fetchall()
