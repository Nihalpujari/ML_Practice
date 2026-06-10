# 🤖 LLM Agents

> An **agent** is an LLM that can **perceive**, **reason** and **act** — retrieving information, analyzing it, and making decisions in a loop.

---

## 📂 Contents

| File | Topic |
|------|-------|
| `Agents.ipynb` | Three agent demos: research (Wikipedia + summarizer), news analyst (DDGS + sentiment), customer support (zero-shot intent detection) |
| `CheatSheet.ipynb` | Interactive reference — agent types, model selection guide, and a "choose your agent" selector |
| `LangChain agent.ipynb` | Building agents with the **LangChain** framework |
| `LangGraph.ipynb` | Multi-step agent workflows with **LangGraph** |
| `ZeroDependencyAgent.ipynb` | A minimal agent built with no frameworks — just Python + Hugging Face |

---

## 🧠 Agent Types Covered

| Agent Type | What it does | Example |
|------------|-------------|---------|
| **Research Agent** | Search + summarize | "Summarize renewable energy in Germany" |
| **Analyst Agent** | Retrieve + evaluate | "What's the sentiment of AI news?" |
| **Support Agent** | Classify intent + respond | "Handle a customer complaint" |
| **Fact-Checker** | Search + verify | "Did solar surpass coal in 2023?" |

---

## ⚡ The Agent Loop

```
Perceive (retrieve info)  →  Reason (analyze / classify)  →  Act (respond / decide)
                ↑                                                       |
                └───────────────────────────────────────────────────────┘
```

---

## 🛠️ Requirements

```bash
pip install transformers torch wikipedia ddgs langchain langgraph
```

---

## ▶️ Run

```bash
jupyter notebook Agents.ipynb
jupyter notebook CheatSheet.ipynb
```

---

## 💡 Tips

- Start with `Agents.ipynb` for hands-on demos before diving into frameworks.
- `CheatSheet.ipynb` is a great quick reference for picking the right model for your agent's task.
- LangChain and LangGraph add structure but also complexity — use them when you need tool orchestration or multi-step workflows.

---

## 🔁 Related

- [`../RAG/`](../RAG/) — retrieval-augmented generation (a common agent tool)
- [`../Dense_Retrival/`](../Dense_Retrival/) — the retrieval backend agents use
