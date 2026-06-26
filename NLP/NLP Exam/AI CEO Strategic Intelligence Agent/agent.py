# agent.py — the AI CEO agent (Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate)
# Extracted from CEO Agent.ipynb so the dashboard (app.py) can reuse run_agent().
import ollama, json
from collections import Counter
from retrieval import semantic_search, bm25_search, hybrid_search, docs as labeled_docs

# lookup table: URL -> full labeled doc (used by the Analyze step)
label_by_url = {d["url"]: d for d in labeled_docs}


def retrieve_evidence(query, k_each=5, final_k=5):
    """Run all 3 retrievers, pool results, dedup by URL, keep the best by consensus."""
    methods = {
        "semantic": semantic_search(query, k_each),
        "bm25":     bm25_search(query, k_each),
        "hybrid":   hybrid_search(query, k_each),
    }
    seen = {}
    for docs_list in methods.values():
        for rank, d in enumerate(docs_list):
            key = d["url"]
            if key not in seen:
                seen[key] = {"doc": d, "hits": 0, "rank_sum": 0}
            seen[key]["hits"]     += 1
            seen[key]["rank_sum"] += rank
    ranked = sorted(seen.values(), key=lambda x: (-x["hits"], x["rank_sum"]))
    return [x["doc"] for x in ranked[:final_k]]


def make_plan(goal):
    """Break the abstract goal into specific, searchable sub-questions."""
    system = (
        "You are a research planner for a strategic intelligence agent about Lufthansa. "
        "Break the user's question into 2-4 SPECIFIC, keyword-rich sub-questions "
        "that will retrieve good evidence from a news database. "
        'Return JSON exactly like: {"steps": ["...", "...", "..."]}'
    )
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": goal}],
        format="json",
    )
    return json.loads(res["message"]["content"])["steps"]


def gather_evidence(goal, final_per_step=4):
    """Plan the goal, retrieve for each sub-question, pool + dedup the evidence."""
    plan = make_plan(goal)
    print(f"🧭 PLANNING: {goal}")
    for s in plan:
        print("   -", s)
    pooled = []
    for sub_q in plan:
        docs = retrieve_evidence(sub_q, final_k=final_per_step)
        print(f"   🔎 {sub_q[:45]}... → {len(docs)} docs")
        pooled.extend(docs)
    unique = list({d["url"]: d for d in pooled}.values())
    print(f"📊 Pooled {len(pooled)} → {len(unique)} unique evidence docs")
    return unique


def analyze(evidence):
    """Attach each doc's Task-4 label (by URL) and summarize what we found."""
    for d in evidence:
        full = label_by_url.get(d["url"], {})
        d["category"]  = full.get("category", "unknown")
        d["sentiment"] = full.get("sentiment", "unknown")
        d["severity"]  = full.get("severity")
    counts = Counter(d["category"] for d in evidence)
    print("📊 ANALYSIS:", dict(counts))
    return evidence, counts


def decide_enough(goal, evidence, counts):
    """Ask the LLM whether the evidence found is enough to answer the goal."""
    summary = ", ".join(f"{n} {cat}" for cat, n in counts.items())
    sample  = " | ".join(d["text"][:70] for d in evidence[:4])
    system_prompt = (
        "You are the decision step of a research agent about Lufthansa. "
        "Decide if there is AT LEAST SOME relevant evidence to give a reasonable answer. "
        "You do NOT need perfect or exhaustive coverage — if several documents relate to the "
        "question, that is ENOUGH. Only answer false if the evidence is clearly off-topic or nearly empty. "
        'Return JSON: {"sufficient": true or false, "reason": "one short sentence"}'
    )
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Question: {goal}\nEvidence: {summary}\nSamples: {sample}"}],
        format="json",
    )
    return json.loads(res["message"]["content"])


def reformulate(goal, reason):
    """Rewrite the question to be more specific when the evidence was insufficient."""
    system = (
        "The previous search for this question did not find specific enough evidence. "
        "Rewrite it into ONE more specific, focused question that targets the missing information. "
        'Return JSON: {"new_goal": "..."}'
    )
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Question: {goal}\nWhy it failed: {reason}"}],
        format="json",
    )
    return json.loads(res["message"]["content"])["new_goal"]


def recommend(goal, evidence):
    """Generate the structured recommendation from the analyzed evidence."""
    context = "\n\n".join(
        f"[{d['source']} · {d.get('category','?')}] {d['text']}" for d in evidence
    )
    system_prompt = """You are a strategic advisor to the CEO of Lufthansa.
Use ONLY the evidence provided — do not invent facts.
Return a JSON object with EXACTLY these keys:
- "recommendation": one clear strategic action (string)
- "justification": 1-2 sentences explaining WHY this recommendation follows from the evidence
- "supporting_evidence": list of 2-3 short evidence points from the context
- "expected_impact": expected business impact (string)
- "risk_level": one of "High", "Medium", "Low"
- "priority": one of "High", "Medium", "Low"
"""
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {goal}"}],
        format="json",
    )
    rec = json.loads(res["message"]["content"])
    rec["question"] = goal
    rec["sources"]  = [d["url"] for d in evidence]
    return rec


def validate(rec, evidence):
    """Check the recommendation is grounded in the evidence (no invented facts)."""
    ev = " | ".join(d["text"][:500] for d in evidence)
    system = (
        "You are the validation step of a research agent about Lufthansa. "
        "Check whether the recommendation and its justification are SUPPORTED by the evidence. "
        "If any claim is not backed by the evidence, it is invalid. "
        'Return JSON: {"valid": true or false, "reason": "one short sentence"}'
    )
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content":
                      f"Recommendation: {rec['recommendation']}\n"
                      f"Justification: {rec['justification']}\n"
                      f"Evidence: {ev}"}],
        format="json",
    )
    return json.loads(res["message"]["content"])


def resolve_goal(goal, history):
    """Use conversation history to rewrite a follow-up into a standalone question."""
    if not history:
        return goal
    convo = "\n".join(f"Q: {t['question']}\nA: {t['recommendation']}" for t in history)
    system = (
        "Rewrite the user's latest question into a STANDALONE question that includes any context "
        "it refers to from the conversation (resolve words like 'that', 'it', 'this'). "
        "If it is already standalone, return it unchanged. "
        'Return JSON: {"goal": "..."}'
    )
    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Conversation so far:\n{convo}\n\nLatest question: {goal}"}],
        format="json",
    )
    return json.loads(res["message"]["content"])["goal"]


def run_agent(goal, history=None, max_tries=2):
    """The full agent: resolve → plan → retrieve → analyze → decide (loop) → recommend → validate."""
    goal = resolve_goal(goal, history)
    print(f"\n🎯 GOAL: {goal}\n")

    evidence = gather_evidence(goal)
    analyzed, counts = analyze(evidence)
    verdict = decide_enough(goal, analyzed, counts)
    print("🤔 DECIDE:", verdict["sufficient"], "—", verdict["reason"])

    tries = 0
    while not verdict["sufficient"] and tries < max_tries:
        print(f"\n🔄 Not enough — reformulating (try {tries+1}/{max_tries})")
        sharper = reformulate(goal, verdict["reason"])
        evidence += gather_evidence(sharper)
        evidence = list({d["url"]: d for d in evidence}.values())
        analyzed, counts = analyze(evidence)
        verdict = decide_enough(goal, analyzed, counts)
        print("🤔 DECIDE:", verdict["sufficient"], "—", verdict["reason"])
        tries += 1

    print(f"\n📝 RECOMMENDING on {len(evidence)} docs...")
    rec = recommend(goal, analyzed)

    check = validate(rec, analyzed)
    print("🛡️ VALIDATE:", check)
    if not check["valid"]:
        print("   ↻ regenerating...")
        rec = recommend(goal, analyzed)

    rec["evidence_sufficient"] = verdict["sufficient"]
    return rec
