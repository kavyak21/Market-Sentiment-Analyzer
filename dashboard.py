import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Pulse – Market Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st_autorefresh(interval=5000, key="auto_refresh")

# ----------------------------
# Custom CSS (Dark Cards)
# ----------------------------
st.markdown("""
<style>
.kpi-card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
}
.kpi-title {
    color: #A0AEC0;
    font-size: 14px;
}
.kpi-value {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<h1 style='text-align:center;'>📊 Pulse – Real-Time Market Sentiment Analyzer</h1>
<p style='text-align:center; color:#A0AEC0;'>
Dark-mode analytics dashboard for live market sentiment
</p>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------
# Load Data
# ----------------------------
conn = sqlite3.connect("pulse.db")
df = pd.read_sql("SELECT * FROM sentiment_data", conn)

if df.empty:
    st.info("No sentiment data available yet.")
    st.stop()

# Convert timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Remove rows without sentiment
df = df.dropna(subset=["sentiment"])

if df.empty:
    st.info("Waiting for predictions… (ingestion may not have started yet)")
    st.stop()

# ----------------------------
# KPI CARDS
# ----------------------------
counts = df["sentiment"].value_counts()
total = len(df)

pos = counts.get("positive", 0)
neu = counts.get("neutral", 0)
neg = counts.get("negative", 0)

# Average Confidence
avg_conf = round(df["confidence"].mean() * 100, 2)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Total Mentions</div>
        <div class='kpi-value'>{total}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Positive</div>
        <div class='kpi-value' style='color:#4CAF50'>{pos}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Neutral</div>
        <div class='kpi-value' style='color:#FFC107'>{neu}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Negative</div>
        <div class='kpi-value' style='color:#F44336'>{neg}</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Average Model Confidence</div>
        <div class='kpi-value' style='color:#00BFFF'>{avg_conf}%</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ----------------------------
# Charts
# ----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📊 Sentiment Distribution")

    if total > 0:
        fig, ax = plt.subplots()
        ax.pie(
            [pos, neu, neg],
            labels=["Positive", "Neutral", "Negative"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4CAF50", "#FFC107", "#F44336"],
            wedgeprops={"width": 0.4}
        )
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.write("No data to display")

with right:
    st.subheader("📈 Sentiment Trend (1-minute window)")

    trend = (
        df.groupby([pd.Grouper(key="timestamp", freq="1min"), "sentiment"])
        .size()
        .reset_index(name="count")
    )

    trend_pivot = trend.pivot(index="timestamp", columns="sentiment", values="count").fillna(0)

    if not trend_pivot.empty:
        st.line_chart(trend_pivot)
    else:
        st.write("Trend data not available yet")

st.divider()

# ----------------------------
# Recent Mentions
# ----------------------------
st.subheader("📰 Latest Mentions")

recent = df.sort_values("timestamp", ascending=False).head(10)

st.dataframe(
    recent[["text", "sentiment", "confidence", "timestamp"]],
    use_container_width=True,
    hide_index=True
)

# ----------------------------
# Market Alert
# ----------------------------
pos_pct = round((pos / total) * 100, 1) if total else 0
neu_pct = round((neu / total) * 100, 1) if total else 0
neg_pct = round((neg / total) * 100, 1) if total else 0

if neg_pct >= 60:
    st.error("🚨 Market sentiment is strongly negative")
elif neg_pct >= 40:
    st.warning("📉 Market sentiment is leaning negative")
elif pos_pct >= 60:
    st.success("🚀 Market sentiment is strongly positive")
elif pos_pct >= 40:
    st.info("📈 Market sentiment is leaning positive")
else:
    st.warning("⚠️ Market sentiment is mixed / uncertain")
