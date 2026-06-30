"""
AI CEO: Strategic Intelligence Agent — Lufthansa
Executive Dashboard (Streamlit) — multi-page, chat-enabled.

Run from this folder:  streamlit run app.py
Needs Ollama running for the live chat (floating 💬 button, bottom-right of every page).
"""

import json, os, base64, html
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import chromadb, ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agent import run_agent          # the full AI agent (Plan→Retrieve→Analyze→Decide→Recommend→Validate)

st.set_page_config(page_title="AI CEO — Lufthansa", page_icon="🛫", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — Lufthansa brand (navy #05164D + yellow #F9BA00)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── sidebar: always dark navy, regardless of Streamlit theme ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #02103a 0%, #05164D 70%, #091d5e 100%) !important;
    border-right: 3px solid #F9BA00 !important;
}
/* Only target text/label elements inside sidebar, not every div */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #dde4ff !important; }
[data-testid="stSidebar"] h1   { color: white !important; letter-spacing:1px; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #F9BA00 !important; font-weight:700; }

/* ── metric cards: yellow accent + subtle frame (theme-safe, works on the bg image) ── */
.main [data-testid="stMetric"] {
    border: 1px solid rgba(5,22,77,.18);
    border-left: 5px solid #F9BA00;
    border-radius: 12px;
    padding: 14px 18px;
    background: rgba(13,27,62,.55);          /* navy glass: readable over the hero image */
    box-shadow: 0 3px 15px rgba(5,22,77,.18);
}
.main [data-testid="stMetric"] [data-testid="stMetricLabel"] p { color:#b8c8ff !important; }
.main [data-testid="stMetric"] [data-testid="stMetricValue"] { color:#ffffff !important; font-weight:800; }

/* ── headings: only inside main content, not sidebar ── */
.main h1 { font-weight:800; letter-spacing:-.5px; }
.main h2, .main h3 { font-weight:700; }

/* ── larger, more readable body text in the main content ── */
.main [data-testid="stMarkdownContainer"] p,
.main [data-testid="stMarkdownContainer"] li { font-size: 1.07rem; line-height: 1.65; }
/* keep the chat popover text at normal size (it's compact) */
[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p,
[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] li { font-size: .95rem; line-height: 1.5; }

/* ── bordered containers (cards) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 3px 14px rgba(5,22,77,.08) !important;
    border-color: rgba(5,22,77,0.2) !important;
    /* No background override — lets Streamlit's theme (light/dark) control it */
}

/* ── tabs ── */
[data-baseweb="tab-list"]  { border-radius:10px; padding:4px; }
[data-baseweb="tab"]       { border-radius:8px !important; }
[aria-selected="true"]     { background: #05164D !important; color:white !important; }

/* ── FLOATING CHAT WIDGET (st.popover pinned to bottom-right corner) ── */
/* pin the popover wrapper itself; width:auto shrinks it to the button so */
/* right/bottom anchor it to the true corner (not the full-width row).    */
[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    left: auto !important;
    top: auto !important;
    width: auto !important;
    max-width: 80px !important;
    z-index: 1000001 !important;
}
/* the trigger -> a round 💬 bubble (like Air India's AI.g button) */
[data-testid="stPopoverButton"] {
    border-radius: 50% !important;
    width: 64px !important;
    height: 64px !important;
    min-height: 64px !important;
    font-size: 28px !important;
    background: #05164D !important;
    color: white !important;
    border: 3px solid #F9BA00 !important;
    box-shadow: 0 6px 24px rgba(5,22,77,.5) !important;
    padding: 0 !important;
    transition: transform .15s ease;
}
[data-testid="stPopoverButton"]:hover { transform: scale(1.08); }
/* hide the little dropdown chevron so it's a clean bubble */
[data-testid="stPopoverButton"] svg { display: none !important; }
/* the popup panel that opens above the button */
[data-testid="stPopoverBody"] {
    width: 380px !important;
    max-height: 72vh !important;
}

/* ── success / warning banners ── */
[data-testid="stSuccess"] { border-radius:10px; }
[data-testid="stWarning"] { border-radius:10px; }

/* ── divider ── */
hr { border-color: rgba(5,22,77,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    docs     = json.load(open("data/lufthansa_labeled.json", encoding="utf-8"))
    recs     = json.load(open("data/recommendations.json",   encoding="utf-8"))
    briefing = json.load(open("data/ceo_briefing.json",      encoding="utf-8"))
    return docs, recs, briefing

@st.cache_resource
def load_retrieval():
    client     = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("lufthansa")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    texts      = [d["text"] for d in json.load(open("data/lufthansa_labeled.json", encoding="utf-8"))]
    doc_emb    = model.encode(texts)
    return collection, model, texts, doc_emb

docs, recommendations, briefing   = load_data()
collection, emb_model, texts, doc_emb = load_retrieval()

PRIORITY  = {"High":"🔴 High",      "Medium":"🟠 Medium", "Low":"🟢 Low"}
SENTIMENT = {"positive":"🟢 Positive","neutral":"⚪ Neutral","negative":"🔴 Negative"}


def _txt(v):
    """Flatten any JSON value (str / dict / list) to readable text.
    The LLM sometimes returns nested objects instead of plain strings — this keeps the UI from crashing."""
    if isinstance(v, dict):
        return " — ".join(_txt(x) for x in v.values())
    if isinstance(v, list):
        return "; ".join(_txt(x) for x in v)
    return str(v)

# ─────────────────────────────────────────────────────────────────────────────
# CHAT SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("images/lufthansa.png"):
        st.image("images/lufthansa.png", use_container_width=True)
    else:
        st.title("🛫 AI CEO")
    st.caption("Lufthansa · Strategic Intelligence Agent")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview", "📰 Market Intelligence", "🚀 Opportunities",
        "⚠️ Risks", "📈 Trends", "📊 Sentiment", "🎯 Recommendations",
        "📋 CEO Briefing",
    ])
    st.divider()

    # ── LIVE clock — runs in the browser (JS), ticks every second ──
    components.html(
        """
        <div id="clock" style="font-family:'Source Sans Pro',sans-serif;
             color:#dde4ff; font-size:0.9rem; letter-spacing:.3px;"></div>
        <script>
        function tick(){
            const now = new Date();
            const d = now.toLocaleDateString([], {day:'2-digit', month:'short', year:'numeric'});
            const t = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
            document.getElementById('clock').textContent = '🕒 ' + d + ' · ' + t;
        }
        tick(); setInterval(tick, 1000);
        </script>
        <style>body{margin:0;background:transparent;overflow:hidden;}</style>
        """,
        height=26,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: page hero banner
# ─────────────────────────────────────────────────────────────────────────────
def hero(icon, title, subtitle, color="#05164D"):
    """Editorial masthead header — same design language as the magazine cover."""
    st.markdown(f"""
    <div style="background:linear-gradient(120deg,{color} 0%,#0a2570 100%);
                padding:22px 32px 26px; border-radius:16px; margin-bottom:26px;
                box-shadow:0 8px 30px rgba(5,22,77,.28); overflow:hidden">
      <div style="display:flex;justify-content:space-between;align-items:center;
           border-bottom:2px solid #F9BA00;padding-bottom:10px;margin-bottom:16px">
        <span style="color:#F9BA00;font-weight:700;letter-spacing:3px;font-size:.72rem;text-transform:uppercase">
          Strategic Intelligence Briefing</span>
        <span style="color:#9fb0d6;letter-spacing:2px;font-size:.72rem"></span>
      </div>
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
        <span style="font-size:2.4rem;line-height:1">{icon}</span>
        <span style="color:#fff;font-weight:800;font-size:2.2rem;letter-spacing:-.5px">{title}</span>
      </div>
      <p style="color:#b8c8ff;margin:10px 0 0;font-size:1.06rem">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# nice human label for each raw source tag
SOURCE_LABEL = {"news": "📰 News", "reddit": "💬 Reddit", "competitor": "🏢 Competitor",
                "company": "✈️ Company"}

# left-border accent colour per category (instant visual scanning)
ACCENT = {"opportunity": "#2ecc71", "risk": "#e74c3c", "trend": "#3498db"}

def doc_card(x, show_sentiment=True, show_source=True, accent=None):
    """Render one document as a rich, full-text card with a coloured accent border.
    Built as escaped HTML so it stays readable on any Streamlit theme."""
    colour = accent or ACCENT.get(x.get("category"), "#05164D")
    text   = html.escape(x["text"])                      # FULL text — no truncation
    url    = html.escape(x["url"], quote=True)            # safe inside href='...'
    meta   = []
    if show_sentiment and x.get("sentiment"):
        meta.append(SENTIMENT.get(x["sentiment"], x["sentiment"]))
    if show_source:
        meta.append(SOURCE_LABEL.get(x["source"], x["source"]))
    meta_html = " &nbsp;·&nbsp; ".join(meta)
    st.markdown(
        f"<div style='border:1px solid rgba(128,140,180,.25); border-left:5px solid {colour};"
        f"border-radius:10px; padding:14px 18px; margin-bottom:14px; background:rgba(128,140,180,.05)'>"
        f"<div style='font-size:1.05rem; line-height:1.6'>{text}</div>"
        f"<div style='margin-top:10px; font-size:.85rem; opacity:.8'>{meta_html}"
        f"{' &nbsp;·&nbsp; ' if meta_html else ''}"
        f"<a href='{url}' target='_blank' style='color:{colour}; text-decoration:none'>🔗 Open source</a>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def category_legend():
    """Small legend explaining the doc-card left-border colour = category."""
    dot = "display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:middle"
    st.markdown(
        "<div style='display:flex;gap:18px;flex-wrap:wrap;align-items:center;"
        "margin:-4px 0 14px;font-size:.85rem;color:#9aa6c4'>"
        "<span style='font-weight:600'>Card border = category</span>"
        f"<span><span style='{dot};background:#2ecc71'></span>Opportunity</span>"
        f"<span><span style='{dot};background:#e74c3c'></span>Risk</span>"
        f"<span><span style='{dot};background:#3498db'></span>Trend</span>"
        "</div>", unsafe_allow_html=True)


def opportunity_card(x):
    """Opportunity Monitor card — title · impact · evidence · confidence (brief Section 3).
    impact = LLM-rated (saved in the notebook); category_score = zero-shot confidence (Task 4)."""
    text   = x["text"]
    title  = text.split(". ", 1)[0]
    body   = text[len(title):].lstrip(". ").strip() or "—"
    impact = x.get("impact", "—")
    conf   = x.get("category_score")
    conf_s = f"{round(conf*100)}%" if isinstance(conf, (int, float)) else "—"
    icol   = {"High": "#2ecc71", "Medium": "#e6a700", "Low": "#8a93a8"}.get(impact, "#8a93a8")
    st.markdown(
        "<div style='border:1px solid rgba(128,140,180,.25);border-left:5px solid #2ecc71;"
        "border-radius:10px;padding:16px 18px;margin-bottom:14px;background:rgba(128,140,180,.05)'>"
        f"<div style='font-size:1.12rem;font-weight:700;line-height:1.35'>{html.escape(title)}</div>"
        "<div style='margin:9px 0;font-size:.86rem'>"
        f"<span style='background:{icol};color:#fff;padding:2px 11px;border-radius:11px;font-weight:600'>Impact: {impact}</span>"
        f"&nbsp;&nbsp;<span style='color:#7f8db0'>Confidence: <b>{conf_s}</b></span></div>"
        f"<div style='font-size:1.0rem;line-height:1.55;opacity:.92'>{html.escape(body)}</div>"
        "<div style='margin-top:9px;font-size:.85rem'>"
        f"<a href='{html.escape(x['url'], quote=True)}' target='_blank' style='color:#2ecc71;text-decoration:none'>🔗 Open source (evidence)</a>"
        "</div></div>", unsafe_allow_html=True)


def risk_card(x):
    """Risk Monitor card — title · category · severity · evidence · confidence (brief Section 4).
    severity = zero-shot rated (saved in the notebook); category_score = zero-shot confidence (Task 4)."""
    text     = x["text"]
    title    = text.split(". ", 1)[0]
    body     = text[len(title):].lstrip(". ").strip() or "—"
    severity = x.get("severity", "—")
    conf     = x.get("category_score")
    conf_s   = f"{round(conf*100)}%" if isinstance(conf, (int, float)) else "—"
    scol     = {"High": "#e74c3c", "Medium": "#e6a700", "Low": "#8a93a8"}.get(severity, "#8a93a8")
    st.markdown(
        "<div style='border:1px solid rgba(128,140,180,.25);border-left:5px solid #e74c3c;"
        "border-radius:10px;padding:16px 18px;margin-bottom:14px;background:rgba(128,140,180,.05)'>"
        f"<div style='font-size:1.12rem;font-weight:700;line-height:1.35'>{html.escape(title)}</div>"
        "<div style='margin:9px 0;font-size:.86rem'>"
        "<span style='background:#e74c3c;color:#fff;padding:2px 11px;border-radius:11px;font-weight:600'>Category: Risk</span>"
        f"&nbsp;&nbsp;<span style='background:{scol};color:#fff;padding:2px 11px;border-radius:11px;font-weight:600'>Severity: {severity}</span>"
        f"&nbsp;&nbsp;<span style='color:#7f8db0'>Confidence: <b>{conf_s}</b></span></div>"
        f"<div style='font-size:1.0rem;line-height:1.55;opacity:.92'>{html.escape(body)}</div>"
        "<div style='margin-top:9px;font-size:.85rem'>"
        f"<a href='{html.escape(x['url'], quote=True)}' target='_blank' style='color:#e74c3c;text-decoration:none'>🔗 Open source (evidence)</a>"
        "</div></div>", unsafe_allow_html=True)


def sentiment_filter(items, key):
    """Radio filter by sentiment + live count. Returns the filtered list.
    Used on Trends/Opportunities, where 'positive vs negative' is meaningful."""
    choice = st.radio("Filter by sentiment",
                      ["All", "🟢 Positive", "⚪ Neutral", "🔴 Negative"],
                      horizontal=True, key=key)
    smap = {"🟢 Positive": "positive", "⚪ Neutral": "neutral", "🔴 Negative": "negative"}
    shown = items if choice == "All" else [d for d in items if d.get("sentiment") == smap[choice]]
    st.caption(f"Showing **{len(shown)}** of {len(items)} documents")
    return shown


def source_filter(items, key):
    """Radio filter by source ('where is this coming from?') + live count.
    Used on the Risk page — 'positive risk' is nonsense, but 'where is the risk
    coming from?' (news / competitor / community) is the useful question.
    Only offers sources that actually appear in this list."""
    present = [s for s in ["news", "competitor", "reddit", "company"]
               if any(d.get("source") == s for d in items)]
    labels  = ["All"] + [SOURCE_LABEL[s] for s in present]
    choice  = st.radio("Filter by source", labels, horizontal=True, key=key)
    if choice == "All":
        shown = items
    else:
        rev   = {v: k for k, v in SOURCE_LABEL.items()}   # label → raw source tag
        shown = [d for d in items if d.get("source") == rev[choice]]
    st.caption(f"Showing **{len(shown)}** of {len(items)} documents")
    return shown


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    # magazine "paper": solid dark navy behind the whole page
    st.markdown("<style>.stApp{background:#0a1330 !important;}</style>", unsafe_allow_html=True)

    # cover image (jpg preferred, png fallback, gradient if neither)
    cover_bg = "linear-gradient(135deg,#05164D,#0a2570)"
    for fname, mime in [("images/background.jpg", "jpeg"), ("images/lufthansa.png", "png")]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            cover_bg = f"url('data:image/{mime};base64,{b64}')"
            break

    opp   = sum(d["category"] == "opportunity" for d in docs)
    risk  = sum(d["category"] == "risk"        for d in docs)
    trend = sum(d["category"] == "trend"       for d in docs)
    n_src = len(set(d["source"] for d in docs))

    coverlines = [
        f"🚀 &nbsp;{opp} opportunities the board can’t ignore",
        f"⚠️ &nbsp;Inside the {risk} risks shaping the year ahead",
        f"📈 &nbsp;{trend} trends to watch — SAF, A350s &amp; beyond",
        "💬 &nbsp;Ask the AI CEO — live, anything you want",
    ]
    lines_html = "".join(
        "<div style='display:flex;align-items:center;gap:12px;margin:9px 0'>"
        "<span style='width:4px;height:24px;background:#F9BA00;border-radius:2px'></span>"
        f"<span style='color:#fff;font-size:1.06rem;font-weight:600;"
        f"text-shadow:0 2px 10px rgba(0,0,0,.75)'>{t}</span></div>"
        for t in coverlines
    )

    st.markdown(f"""
    <div style="position:relative;min-height:80vh;border-radius:18px;overflow:hidden;
         background-image:linear-gradient(180deg,rgba(3,12,40,.30) 0%,rgba(3,12,40,.55) 50%,rgba(3,12,40,.93) 100%),{cover_bg};
         background-size:cover;background-position:center;
         display:flex;flex-direction:column;justify-content:space-between;
         padding:28px 36px;box-shadow:0 16px 50px rgba(0,0,0,.5)">

      <div style="display:flex;justify-content:space-between;align-items:center;
           border-bottom:2px solid #F9BA00;padding-bottom:10px">
        <span style="color:#F9BA00;font-weight:700;letter-spacing:3px;font-size:.78rem;text-transform:uppercase">
          Strategic Intelligence Briefing</span>
        <span style="color:#dfe6f5;letter-spacing:2px;font-size:.78rem"></span>
      </div>

      <div>
        <div style="font-size:5.5rem;font-weight:900;color:#fff;line-height:.92;letter-spacing:-2px;
             text-shadow:0 4px 24px rgba(0,0,0,.55)">AI&nbsp;CEO</div>
        <div style="color:#F9BA00;font-weight:700;letter-spacing:6px;font-size:1.05rem;
             text-transform:uppercase;margin-top:8px">The Lufthansa Edition</div>
        <div style="color:#eef2ff;font-size:1.25rem;font-style:italic;margin-top:16px;max-width:560px;
             text-shadow:0 2px 12px rgba(0,0,0,.65)">
          “If you were the CEO today — what would you do next, and why?”</div>
      </div>

      <div>
        {lines_html}
        <div style="margin-top:16px;border-top:1px solid rgba(255,255,255,.25);padding-top:12px;
             color:#cbd6f0;letter-spacing:2px;font-size:.76rem;text-transform:uppercase">
          </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── company facts (the brief's required Overview fields) ──
    last_update = datetime.fromtimestamp(os.path.getmtime("data/lufthansa_data.json")).strftime("%d %b %Y, %H:%M")
    facts = [
        ("Company", "Lufthansa"), ("Industry", "Aviation"),
        ("Documents", str(len(docs))), ("Data sources", str(n_src)),
        ("Last updated", last_update),
    ]
    facts_html = "".join(
        "<div style='flex:1;min-width:150px;background:rgba(13,27,62,.55);border:1px solid rgba(255,255,255,.14);"
        "border-left:5px solid #F9BA00;border-radius:12px;padding:14px 18px'>"
        f"<div style='color:#F9BA00;font-size:.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase'>{lbl}</div>"
        f"<div style='color:#fff;font-size:1.25rem;font-weight:800;margin-top:4px'>{val}</div></div>"
        for lbl, val in facts
    )
    st.markdown(f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:22px'>{facts_html}</div>",
                unsafe_allow_html=True)

    # ── FROM THE EDITOR (About + Why, editorial style) ──
    st.markdown("""
    <div style="margin-top:26px">
      <div style="color:#F9BA00;font-weight:700;letter-spacing:4px;text-transform:uppercase;font-size:.82rem;
           border-bottom:2px solid rgba(249,186,0,.4);padding-bottom:8px;margin-bottom:16px">From the Editor</div>
      <div style="display:flex;gap:36px;flex-wrap:wrap">
        <div style="flex:1;min-width:280px;font-family:Georgia,'Times New Roman',serif;color:#dfe6f5;
             font-size:1.06rem;line-height:1.78;text-align:justify">
          <span style="float:left;font-family:Georgia,serif;font-size:3.4rem;line-height:.8;color:#F9BA00;
               font-weight:700;margin:6px 10px 0 0">D</span>eutsche Lufthansa AG is Germany’s flagship
          carrier and one of Europe’s largest airline groups. With major hubs in
          <strong style="color:#fff">Frankfurt</strong> and <strong style="color:#fff">Munich</strong>,
          and as a founding member of the <strong style="color:#fff">Star Alliance</strong>, the group
          spans passenger aviation, cargo (Lufthansa Cargo) and aircraft maintenance (Lufthansa Technik) —
          operating one of the world’s largest fleets.
        </div>
        <div style="flex:1;min-width:280px;font-family:Georgia,'Times New Roman',serif;color:#dfe6f5;
             font-size:1.06rem;line-height:1.78;text-align:justify">
          <div style="font-family:sans-serif;color:#F9BA00;font-weight:700;font-size:.78rem;letter-spacing:2px;
               text-transform:uppercase;margin-bottom:8px">Why I chose Lufthansa</div>
          I love travelling — mostly by aeroplane. I’m fascinated by the
          <strong style="color:#fff">discipline and planning</strong> behind how an airline runs: managing
          thousands of flights, crew and aircraft schedules every single day. And one day, I’d love to see
          the cockpit. ✈️
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Market Intelligence
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📰 Market Intelligence":
    hero("📰", "Market Intelligence",
         "Recent news · competitor activity · emerging technologies · company announcements")
    category_legend()

    news_docs    = [d for d in docs if d["source"] == "news"]
    comp_docs    = [d for d in docs if d["source"] == "competitor"]
    company_docs = [d for d in docs if d["source"] == "company"]
    reddit_docs  = [d for d in docs if d["source"] == "reddit"]
    tech_docs    = [d for d in docs if d.get("category") == "trend"]   # emerging tech / trends

    tab_news, tab_comp, tab_tech, tab_company, tab_comm = st.tabs([
        f"📰 news ({len(news_docs)})",
        f"🏢 Competitor activity ({len(comp_docs)})",
        f"📈 Emerging tech ({len(tech_docs)})",
        f"✈️ Company announcements ({len(company_docs)})",
        f"💬 Community ({len(reddit_docs)})",
    ])

    with tab_news:
        if news_docs:
            for x in news_docs:
                doc_card(x, show_source=False)
        else:
            st.info("No news documents in the current corpus.")

    with tab_comp:
        if comp_docs:
            for x in comp_docs:
                doc_card(x, show_sentiment=False, show_source=False)
        else:
            st.info("No competitor documents in the current corpus.")

    with tab_tech:
        if tech_docs:
            st.caption("Documents flagged as **emerging trends / technologies to monitor** "
                       "(category = trend), pulled across all sources.")
            for x in tech_docs:
                doc_card(x)                          # show source — trend docs span all origins
        else:
            st.info("No trend / technology documents in the current corpus.")

    with tab_company:
        if company_docs:
            for x in company_docs:
                doc_card(x, show_source=False)
        else:
            st.info("No company documents in the current corpus.")

    with tab_comm:
        if reddit_docs:
            for x in reddit_docs:
                doc_card(x, show_source=False)
        else:
            st.info("No community documents in the current corpus.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Opportunities
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚀 Opportunities":
    opps = [d for d in docs if d["category"] == "opportunity"]
    hero("🚀", "Opportunity Monitor",
         f"{len(opps)} opportunities — title · impact · evidence · confidence", "#0a5c2e")
    for x in sentiment_filter(opps, key="opp_filter"):
        opportunity_card(x)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Risks
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚠️ Risks":
    risks = [d for d in docs if d["category"] == "risk"]
    hero("⚠️", "Risk Monitor",
         f"{len(risks)} risks — title · category · severity · evidence · confidence", "#7a1c1c")
    for x in source_filter(risks, key="risk_filter"):
        risk_card(x)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Trends
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Trends":
    trends = [d for d in docs if d["category"] == "trend"]
    hero("📈", "Trend Monitor",
         f"{len(trends)} documents — technologies & market shifts to watch", "#1d4e7a")
    for x in sentiment_filter(trends, key="trend_filter"):
        doc_card(x)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Sentiment
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Sentiment":
    hero("📊", "Sentiment Analysis", "How the market feels about Lufthansa")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Overall sentiment")
            order  = ["Negative", "Neutral", "Positive"]
            counts = Counter(d["sentiment"] for d in docs)
            vals   = [counts.get(k.lower(), 0) for k in order]
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#05164D")        # navy panel = readable on any theme
            ax.set_facecolor("#05164D")
            wedges, labels_t, autop = ax.pie(
                vals, labels=order, autopct="%1.0f%%",
                colors=["#e74c3c", "#95a5a6", "#2ecc71"],
                startangle=140, wedgeprops={"edgecolor": "#05164D", "linewidth": 2},
                textprops={"color": "white", "fontsize": 11})
            for a in autop:                            # percentage text
                a.set_color("white"); a.set_fontweight("bold")
            ax.axis("equal")
            st.pyplot(fig)

    with col2:
        with st.container(border=True):
            st.subheader("Sentiment by source")
            df    = pd.DataFrame([{"source": d["source"], "sentiment": d["sentiment"]} for d in docs])
            pivot = df.pivot_table(index="source", columns="sentiment", aggfunc=len, fill_value=0)
            st.bar_chart(pivot)

    # ── News sentiment vs Public sentiment (brief Section 5) ──
    st.write("")
    with st.container(border=True):
        st.subheader("📰 News vs 💬 Public sentiment")
        news_items   = [d for d in docs if d["source"] == "news"]
        public_items = [d for d in docs if d["source"] == "reddit"]   # community = public voice
        order  = ["negative", "neutral", "positive"]
        news_c = Counter(d["sentiment"] for d in news_items)
        pub_c  = Counter(d["sentiment"] for d in public_items)
        news_v = [news_c.get(k, 0) for k in order]
        pub_v  = [pub_c.get(k, 0)  for k in order]

        x = np.arange(3); w = 0.38
        fig, ax = plt.subplots(figsize=(6, 3.0))
        fig.patch.set_facecolor("#05164D"); ax.set_facecolor("#05164D")
        b1 = ax.bar(x - w/2, news_v, w, label="News",   color="#3498db")
        b2 = ax.bar(x + w/2, pub_v,  w, label="Public", color="#F9BA00")
        ax.set_xticks(x); ax.set_xticklabels(["Negative", "Neutral", "Positive"])
        ax.tick_params(colors="white")
        ax.set_ylabel("documents", color="white")
        for sp in ax.spines.values(): sp.set_color("#33406b")
        ax.bar_label(b1, color="white", fontsize=9, padding=2)
        ax.bar_label(b2, color="white", fontsize=9, padding=2)
        ax.legend(facecolor="#05164D", edgecolor="#33406b", labelcolor="white")
        st.pyplot(fig)

        def _dom(counter, n):
            if not n:
                return "—"
            mx   = max(counter.get(k, 0) for k in order)
            tied = [k for k in order if counter.get(k, 0) == mx]
            return tied[0] if len(tied) == 1 else " & ".join(tied) + " (tied)"
        st.caption(f"News ({len(news_items)} docs): leads with **{_dom(news_c, len(news_items))}**  ·  "
                   f"Public ({len(public_items)} docs): leads with **{_dom(pub_c, len(public_items))}**")

    st.write("")
    with st.container(border=True):
        st.subheader("📋 Raw counts")
        c1, c2, c3 = st.columns(3)
        cnt = Counter(d["sentiment"] for d in docs)
        c1.metric("🔴 Negative", cnt.get("negative", 0))
        c2.metric("⚪ Neutral",  cnt.get("neutral",  0))
        c3.metric("🟢 Positive", cnt.get("positive", 0))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Recommendations
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Recommendations":
    hero("🎯", "Strategic Recommendations", "Evidence-based actions for Lufthansa leadership")
    PRIO_COL = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#2ecc71"}
    for i, rec in enumerate(recommendations, 1):
        pr   = rec.get("priority", "—")
        col  = PRIO_COL.get(pr, "#05164D")
        head = html.escape(_txt(rec.get("recommendation", "")))
        why  = html.escape(_txt(rec.get("justification", "")))
        imp  = html.escape(_txt(rec.get("expected_impact", "")))
        risk = html.escape(_txt(rec.get("risk_level", "—")))
        ev   = "".join(f"<li style='margin:5px 0'>{html.escape(_txt(e))}</li>"
                       for e in rec.get("supporting_evidence", []))
        st.markdown(f"""
        <div style="border:1px solid rgba(128,140,180,.25);border-left:6px solid {col};
             border-radius:14px;padding:22px 28px;margin-bottom:6px;background:rgba(128,140,180,.05)">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px">
            <span style="background:{col};color:#fff;font-weight:800;border-radius:50%;width:30px;height:30px;
                 display:inline-flex;align-items:center;justify-content:center;font-size:.9rem">{i}</span>
            <span style="color:{col};font-weight:700;letter-spacing:2px;text-transform:uppercase;font-size:.72rem">
              Recommendation</span>
            <span style="margin-left:auto;color:#9aa6c4;font-size:.78rem;letter-spacing:.5px">
              {PRIORITY.get(pr, pr)} PRIORITY &nbsp;·&nbsp; RISK: {risk}</span>
          </div>
          <div style="font-family:Georgia,'Times New Roman',serif;font-size:1.55rem;font-weight:700;
               line-height:1.25;margin-bottom:16px">{head}</div>
          <div style="border-left:3px solid {col};padding:2px 0 2px 16px;margin-bottom:18px;
               font-style:italic;font-size:1.08rem;line-height:1.6;opacity:.92">“{why}”</div>
          <div style="color:#8493b8;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
               font-size:.7rem;margin-bottom:4px">📈 Expected impact</div>
          <div style="line-height:1.6;margin-bottom:18px">{imp}</div>
          <div style="color:#8493b8;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
               font-size:.7rem;margin-bottom:2px">📎 Supporting evidence</div>
          <ul style="margin:0;padding-left:20px;line-height:1.55">{ev}</ul>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔗 Sources"):
            for u in rec.get("sources", []):
                st.markdown(f"- {u}")
        st.write("")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CEO Briefing
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 CEO Briefing":
    hero("📋", "CEO Briefing", "One-page executive summary for leadership")

    raw_wh  = briefing.get("what_happened", "")
    cap     = html.escape(raw_wh[:1])               # drop-cap letter
    wh_rest = html.escape(raw_wh[1:])
    wm      = html.escape(briefing.get("why_it_matters", ""))
    nx      = html.escape(briefing.get("what_to_do_next", ""))

    SERIF = "font-family:Georgia,'Times New Roman',serif;font-size:1.1rem;line-height:1.8;text-align:justify"
    KICK  = ("color:#F9BA00;font-weight:700;letter-spacing:4px;text-transform:uppercase;font-size:.78rem;"
             "border-bottom:2px solid rgba(249,186,0,.4);padding-bottom:8px;margin:26px 0 16px")

    st.markdown(f"""
    <div style="border:1px solid rgba(128,140,180,.25);border-radius:14px;padding:28px 36px;
         background:rgba(128,140,180,.05)">
      <div style="{KICK};margin-top:0">📌 What happened</div>
      <div style="{SERIF}">
        <span style="float:left;font-family:Georgia,serif;font-size:3.6rem;line-height:.78;color:#F9BA00;
             font-weight:700;margin:6px 12px 0 0">{cap}</span>{wh_rest}</div>

      <div style="{KICK}">💡 Why it matters</div>
      <div style="{SERIF}">{wm}</div>

      <div style="{KICK}">✅ What to do next</div>
      <div style="{SERIF}">{nx}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FLOATING CHAT WIDGET  — round 💬 button bottom-right; click to open the panel
# (CSS above pins st.popover to the corner and makes the trigger a circle)
# ─────────────────────────────────────────────────────────────────────────────
with st.popover("💬", use_container_width=False):
    # ── header (like "Hello I'm AI.g") ──
    st.markdown(
        "<div style='background:linear-gradient(120deg,#05164D,#0a2570);"
        "margin:-1rem -1rem 0.5rem -1rem;padding:18px 20px;border-radius:12px 12px 0 0'>"
        "<h3 style='color:white;margin:0'>🛫 Hello, I'm the AI CEO</h3>"
        "<p style='color:#b8c8ff;margin:4px 0 0;font-size:.85rem'>"
        "Ask me anything about Lufthansa's strategy</p></div>",
        unsafe_allow_html=True,
    )

    # ── conversation history ──
    if not st.session_state.chat_history:
        st.caption("👋 Try: *\"What are Lufthansa's biggest risks right now?\"*")
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant", avatar="🛫"):
            st.markdown(f"**{turn['recommendation']}**")
            st.caption(f"{PRIORITY.get(turn.get('priority'), '—')} · Risk: {turn.get('risk_level','—')}")
            st.write(turn.get("justification", ""))
            with st.expander("📎 Evidence & sources"):
                for e in turn.get("supporting_evidence", []):
                    st.markdown(f"- {e}")
                for u in turn.get("sources", []):
                    st.caption(u)

    # ── input box (form = stays open + clears after sending) ──
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="Type your question here…",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("Ask  ➤", use_container_width=True)

    if sent and question:
        # =====================================================================
        #  THE 7-STEP AGENT RUNS HERE. run_agent() (in agent.py) executes:
        #    STEP 1 RESOLVE → STEP 2 PLAN → STEP 3 RETRIEVE → STEP 4 ANALYZE
        #    → STEP 5 DECIDE (loop) → STEP 6 RECOMMEND → STEP 7 VALIDATE
        #  history = short-term memory; max_tries=1 = one self-correction (fast live chat)
        # =====================================================================
        with st.spinner("🧠 Planning → retrieving → analyzing → deciding → recommending → validating… (a few min on CPU)"):
            rec = run_agent(question, history=st.session_state.chat_history, max_tries=1)
        st.session_state.chat_history.append(rec)
        st.rerun()
