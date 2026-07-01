# Deploying the AI CEO dashboard (with a working chatbot)

This deploys the **exact same dashboard** to a free public URL, with the live chatbot
working — without changing your exam architecture.

- **Local (exam/demo):** `streamlit run app.py` → uses **local Ollama** (unchanged).
- **Public (deployed):** Streamlit Cloud runs **`streamlit_app.py`**, which redirects the
  agent's `ollama.chat(...)` calls to **Groq's free hosted Llama 3.1 8B**, then runs `app.py`.

Nothing in `app.py` / `agent.py` / `retrieval.py` is modified. Deployment is an **add-on layer**.

---

## Why this is needed
Free hosting can't run a local 8B model (no Ollama, not enough RAM). So the *deployed*
chatbot uses **Groq** — which hosts the **same Llama 3.1 8B** model, free, via an API.
Your local version keeps using Ollama, so the "local, no paid API" exam story is intact.

---

## Step 1 — Get a free Groq API key
1. Go to <https://console.groq.com> and sign in (free).
2. Open **API Keys** → **Create API Key** → copy it (looks like `gsk_...`).
   *(You enter this; never paste it into code or commit it.)*

## Step 2 — Put the project on GitHub
Make sure these are **committed** (the deployed app needs them at runtime):
- `app.py`, `agent.py`, `retrieval.py`, `streamlit_app.py`, `requirements.txt`
- `data/` (the JSON files)
- `chroma_db/` (the vector store) ← important, don't gitignore it
- `images/`

```bash
git add -A
git commit -m "Add Groq deployment entry point (streamlit_app.py)"
git push
```

## Step 3 — Deploy on Streamlit Community Cloud (free)
1. Go to <https://share.streamlit.io> → sign in with GitHub → **New app**.
2. Pick your **repo** and **branch**.
3. **Main file path:** `streamlit_app.py`  ← (NOT app.py — this is the deploy entry point)
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**.

First load takes a few minutes (it installs deps and downloads the MiniLM embedder,
then builds the BM25 index). After that the dashboard and the floating 💬 chat work.

---

## Test it locally as Groq (optional)
You can run the deployed configuration on your own machine:
```bash
pip install groq
# PowerShell:
$env:GROQ_API_KEY = "gsk_your_key_here"
streamlit run streamlit_app.py
```
Without `GROQ_API_KEY`, just run `streamlit run app.py` for the normal Ollama version.

---

## Notes & gotchas
- **Keep `chroma_db/` and `data/` in the repo** — retrieval loads them at startup.
- **Secrets stay in Streamlit Cloud**, never in the code or git. Don't commit a key.
- **Cold start is slow** on the free tier (downloads the embedder, builds the index). Normal.
- **Groq free tier** has rate limits — fine for a demo, not heavy traffic.
- **Optional speed-up:** `transformers` + `torch` are only needed by the *notebooks*
  (Task 4 classification), not the dashboard. Trimming them would make the deploy build
  faster — but that edits `requirements.txt`, so it's optional and left to you.

---

## What to say in the oral
> "The system runs locally on Ollama by design — no paid APIs, private, offline. For a
> public deployment I added a thin entry point, `streamlit_app.py`, that redirects the
> agent's LLM calls to Groq's free hosted Llama 3.1 8B and runs the same dashboard. The
> core architecture is untouched; only where the LLM runs changes."
