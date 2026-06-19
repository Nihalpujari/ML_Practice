"""
AI CEO: Strategic Intelligence Agent — Lufthansa
Executive Dashboard (Streamlit) — multi-page via sidebar navigation.

Run from this folder:  streamlit run app.py
Loads saved artifacts (no live LLM calls) so it's fast + reliable:
  lufthansa_labeled.json -> docs with category + sentiment
  recommendations.json   -> 5 strategic recommendations
  ceo_briefing.json      -> executive summary
"""

import json
import os
from collections import Counter
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="AI CEO — Lufthansa", page_icon="🛫", layout="wide")


# ----------------------------------------------------------------------
# Load data once (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    docs     = json.load(open("lufthansa_labeled.json", encoding="utf-8"))
    recs     = json.load(open("recommendations.json", encoding="utf-8"))
    briefing = json.load(open("ceo_briefing.json", encoding="utf-8"))
    return docs, recs, briefing

docs, recommendations, briefing = load_data()
last_update = datetime.fromtimestamp(
    os.path.getmtime("lufthansa_data.json")).strftime("%d %b %Y, %H:%M")

# small look-up tables for coloured labels
PRIORITY  = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
SENTIMENT = {"positive": "🟢 positive", "neutral": "⚪ neutral", "negative": "🔴 negative"}


# ----------------------------------------------------------------------
# Sidebar navigation (each radio choice = one "page")
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🛫 AI CEO")
    st.caption("Lufthansa · Strategic Intelligence")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📰 Market Intelligence", "🚀 Opportunities",
         "⚠️ Risks", "📊 Sentiment", "🎯 Recommendations", "📋 CEO Briefing"],
    )
    st.divider()
    st.metric("Documents", len(docs))
    st.metric("Data sources", len(set(d["source"] for d in docs)))
    st.caption(f"🕒 Updated {last_update}")


# ======================================================================
# PAGE: Overview
# ======================================================================
if page == "🏠 Overview":
    # Full-screen background image — ONLY on this page.
    # Save your image as background.jpg in this folder (JPEG). Falls back to the banner.
    import base64
    if os.path.exists("background.jpg"):
        with open("background.jpg", "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(2,16,58,0.55), rgba(2,16,58,0.72)),
                              url("data:image/jpeg;base64,{_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<h1 style='color:white;margin-bottom:0'>🛫 AI CEO — Lufthansa</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#dfe6f5'>Executive Intelligence Dashboard · open-source · evidence-based</p>",
                    unsafe_allow_html=True)
    else:
        if os.path.exists("lufthansa.png"):
            st.image("lufthansa.png", use_container_width=True)
        st.title("🛫 AI CEO — Lufthansa")
        st.caption("Executive Intelligence Dashboard · open-source · evidence-based")
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("✈️ About Lufthansa")
            st.write(
                "Deutsche Lufthansa AG is Germany's flagship carrier and one of Europe's "
                "largest airline groups. With major hubs in **Frankfurt** and **Munich** and "
                "as a founding member of the **Star Alliance**, the Lufthansa Group spans "
                "passenger aviation, cargo (Lufthansa Cargo) and aircraft maintenance "
                "(Lufthansa Technik) — operating one of the world's largest fleets."
            )
    with col2:
        with st.container(border=True):
            st.subheader("💡 Why I chose Lufthansa")
            st.write(
                "I love travelling — mostly by aeroplane. I'm fascinated by the **discipline "
                "and planning** behind how an airline runs, especially how it manages its "
                "aircraft and flight schedules. And one day, I'd love to see the cockpit."
            )

    st.write("")
    with st.container(border=True):
        a, b, c, d = st.columns(4)
        a.metric("Company", "Lufthansa")
        b.metric("Industry", "Aviation")
        c.metric("Documents", len(docs))
        d.metric("Data sources", len(set(x["source"] for x in docs)))


# ======================================================================
# PAGE: Market Intelligence
# ======================================================================
elif page == "📰 Market Intelligence":
    st.title("📰 Market Intelligence")
    tab_news, tab_comp = st.tabs(["Recent news", "Competitor activity"])

    with tab_news:
        for x in [d for d in docs if d["source"] == "news"][:8]:
            with st.container(border=True):
                st.write(x["text"][:220] + "…")
                st.caption(f"{SENTIMENT.get(x['sentiment'], x['sentiment'])} · [source]({x['url']})")

    with tab_comp:
        for x in [d for d in docs if d["source"] == "competitor"][:8]:
            with st.container(border=True):
                st.write(x["text"][:220] + "…")
                st.caption(f"[source]({x['url']})")


# ======================================================================
# PAGE: Opportunities
# ======================================================================
elif page == "🚀 Opportunities":
    st.title("🚀 Opportunity Monitor")
    opps = [d for d in docs if d["category"] == "opportunity"]
    st.caption(f"{len(opps)} documents classified as opportunities")
    for x in opps:
        with st.container(border=True):
            st.write(x["text"])
            st.caption(f"{SENTIMENT.get(x['sentiment'], x['sentiment'])} · source: {x['source']} · [evidence]({x['url']})")


# ======================================================================
# PAGE: Risks
# ======================================================================
elif page == "⚠️ Risks":
    st.title("⚠️ Risk Monitor")
    risks = [d for d in docs if d["category"] == "risk"]
    st.caption(f"{len(risks)} documents classified as risks")
    for x in risks[:40]:
        with st.container(border=True):
            st.write(x["text"])
            st.caption(f"{SENTIMENT.get(x['sentiment'], x['sentiment'])} · source: {x['source']} · [evidence]({x['url']})")


# ======================================================================
# PAGE: Sentiment
# ======================================================================
elif page == "📊 Sentiment":
    st.title("📊 Sentiment Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall")
        order  = ["negative", "neutral", "positive"]
        counts = Counter(d["sentiment"] for d in docs)
        vals   = [counts.get(k, 0) for k in order]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(vals, labels=order, autopct="%1.0f%%",
               colors=["#e74c3c", "#bdc3c7", "#2ecc71"])
        ax.axis("equal")
        st.pyplot(fig)

    with col2:
        st.subheader("By source (news vs public vs company)")
        df = pd.DataFrame([{"source": d["source"], "sentiment": d["sentiment"]} for d in docs])
        pivot = df.pivot_table(index="source", columns="sentiment", aggfunc=len, fill_value=0)
        st.bar_chart(pivot)


# ======================================================================
# PAGE: Recommendations
# ======================================================================
elif page == "🎯 Recommendations":
    st.title("🎯 Strategic Recommendations")
    for rec in recommendations:
        with st.container(border=True):
            st.markdown(f"### {rec['recommendation']}")
            st.markdown(
                f"{PRIORITY.get(rec.get('priority'), rec.get('priority'))} priority "
                f"· **Risk:** {rec.get('risk_level', '—')}"
            )
            st.markdown(f"**Why it matters:** {rec.get('justification', '')}")
            st.markdown(f"**Expected impact:** {rec.get('expected_impact', '')}")
            with st.expander("📎 Evidence & sources"):
                for e in rec.get("supporting_evidence", []):
                    st.markdown(f"- {e}")
                st.caption("Sources:")
                for u in rec.get("sources", []):
                    st.markdown(f"- {u}")


# ======================================================================
# PAGE: CEO Briefing
# ======================================================================
elif page == "📋 CEO Briefing":
    st.title("📋 CEO Briefing")
    st.caption("One-page executive summary")

    with st.container(border=True):
        st.subheader("📌 What happened?")
        st.write(briefing.get("what_happened", ""))
    with st.container(border=True):
        st.subheader("💡 Why does it matter?")
        st.write(briefing.get("why_it_matters", ""))
    with st.container(border=True):
        st.subheader("✅ What should management do next?")
        st.write(briefing.get("what_to_do_next", ""))
