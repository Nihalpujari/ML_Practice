# retrieval.py — the agent's 3 retrieval tools (semantic / BM25 / hybrid)
import json, numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

#  SETUP (runs once on import) 
docs  = json.load(open("lufthansa_labeled.json", encoding="utf-8"))
texts = [d["text"] for d in docs]

client     = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("lufthansa")      # semantic store

model   = SentenceTransformer("all-MiniLM-L6-v2")    # embedder (for hybrid)
doc_emb = model.encode(texts)                        # all docs → vectors
bm25    = BM25Okapi([t.lower().split() for t in texts])  # keyword index

#  THE 3 TOOLS (run per query, fast) 
def semantic_search(query, k=5):
    res = collection.query(query_texts=[query], n_results=k)
    return [{"text": d, "url": m["url"], "source": m["source"]}
            for d, m in zip(res["documents"][0], res["metadatas"][0])]

def bm25_search(query, k=5):
    scores = bm25.get_scores(query.lower().split())
    top    = np.argsort(scores)[::-1][:k]
    return [{"text": texts[i], "url": docs[i]["url"], "source": docs[i]["source"]} for i in top]

def hybrid_search(query, k=5, alpha=0.5):
    bm25_scores  = bm25.get_scores(query.lower().split())
    dense_scores = cosine_similarity(model.encode([query]), doc_emb)[0]   
    norm = lambda x: (np.array(x, float) - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
    combined = alpha*norm(dense_scores) + (1-alpha)*norm(bm25_scores)
    top = np.argsort(combined)[::-1][:k]
    return [{"text": texts[i], "url": docs[i]["url"], "source": docs[i]["source"]} for i in top]