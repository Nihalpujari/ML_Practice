# 🔎 RAG Deep Dive

A hands-on deep dive into **Retrieval-Augmented Generation (RAG)** and the **LangChain / LangGraph**
ecosystem — building RAG from first principles, then with a framework.

> Goes beyond "just use a library": each notebook unpacks *how* RAG actually works —
> tokenization, retrieval, prompting, generation — before wiring it together with LangChain.

---

## 📁 What's inside

```
RAG deep dive/
└── Langchain/
    ├── tokenlization.ipynb            # how text becomes tokens (BPE / WordPiece)
    ├── rag_from_scratch_1_to_4.ipynb  # RAG step by step: indexing → retrieval → generation
    └── Langchain_basics.ipynb         # LangChain fundamentals: Runnables, LCEL, chains
```

---

## 📚 Topics covered

- **Tokenization** — how text is split into tokens before it reaches a model
- **RAG from scratch** — the full loop: load → split (chunk) → embed → store → retrieve → augment → generate
- **LangChain basics** — the `Runnable` abstraction, the LCEL pipe (`|`), prompt templates, models, output parsers
- **Toward LangGraph** — moving from linear chains to looping, stateful agent graphs

---

## 🧠 Key ideas

- **RAG = Retrieve → Augment → Generate** — ground an LLM in real documents so it answers from
  evidence instead of stale training memory.
- **LangChain** packages the pieces (loaders, splitters, vector stores, retrievers, models, parsers)
  and connects them with the `|` pipe — the output of one step feeds the next.
- **LangGraph** extends chains into **graphs with loops and branches** — what you need for agents that
  plan, decide, and self-correct (a straight chain can't loop back).

---

## ▶️ Running locally (no API key)

The notebooks use OpenAI, but they run fully **free and local** by swapping in **Ollama**:

```python
# instead of:  from langchain_openai import ChatOpenAI ; model = ChatOpenAI()
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.1:8b")
```

```bash
pip install langchain langchain-community langchain-ollama chromadb beautifulsoup4
# and once: install Ollama (https://ollama.com) + `ollama pull llama3.1:8b`
```

---

Part of the [ML Practice](../README.md) learning lab.
