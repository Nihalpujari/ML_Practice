# streamlit_app.py — DEPLOYMENT ENTRY POINT (Streamlit Community Cloud)
# =====================================================================
# ADDITIVE ONLY. This file does NOT modify app.py / agent.py / retrieval.py.
# Your exam architecture stays 100% unchanged.
#
#   • LOCAL exam / demo :  run  `streamlit run app.py`   → uses local OLLAMA (unchanged)
#   • PUBLIC deploy     :  Streamlit Cloud runs THIS file → it transparently redirects
#                          the agent's  ollama.chat(...)  calls to GROQ's free hosted
#                          Llama 3.1 8B, then runs the EXACT same dashboard (app.py).
#
# How it works: the agent calls `ollama.chat(...)`. Here we swap that one function
# for a Groq-backed version *at runtime* (monkey-patch), so not a single line of the
# original code changes — deployment is just a thin adapter wrapped around app.py.
# =====================================================================
import os
import streamlit as st
import ollama   # we redirect ollama.chat -> Groq just below


# --- 1) read the Groq API key (set in the Streamlit Cloud "Secrets" box) -----
def _groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]          # Streamlit Cloud secret
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")  # or a local env var, if testing


# --- 2) a drop-in replacement for ollama.chat that calls Groq instead --------
#     returns the SAME shape the existing code expects: {"message": {"content": ...}}
_client = None
def _groq_chat(model=None, messages=None, format=None, **kwargs):
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=_groq_key())
    # the agent's prompts already say "Return JSON", so json mode is safe
    extra = {"response_format": {"type": "json_object"}} if format == "json" else {}
    resp = _client.chat.completions.create(
        model="llama-3.1-8b-instant",   # Groq's free, hosted Llama 3.1 8B
        messages=messages,
        **extra,
    )
    return {"message": {"content": resp.choices[0].message.content}}


# --- 3) install the redirect, THEN run the UNCHANGED dashboard ---------------
ollama.chat = _groq_chat   # the only "change" — done at runtime, not in your files

with open("app.py", encoding="utf-8") as f:
    exec(compile(f.read(), "app.py", "exec"), {"__name__": "__main__"})
