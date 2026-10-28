"""Step 1.6: the triage graph.

summarize (retry up to 3x) -> retrieve -> suggest_related | flag_new

Run:
  uv run python triage_graph.py --show-graph
  uv run python triage_graph.py "your own question"
  uv run python triage_graph.py --demo
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

from summarize_chain import TriageSummary, build_chain

EMBED_MODEL = "nomic-embed-text"   # frozen in step 1.4
ISSUES_PATH = Path("data/issues.json")
INDEX_PATH = Path("data/index_nomic.npz")
LOG_PATH = Path("logs/failures.jsonl")
MAX_CHARS = 2000
MAX_ATTEMPTS = 3
SIMILARITY_THRESHOLD = 0.60        # starting guess; we tune it from the demo scores
TOP_K = 3


def clean(text: str) -> str:
    """Same prep as step 1.4: strip #numbers, cut long bodies."""
    return re.sub(r"#\d+", "", text or "")[:MAX_CHARS]


# ---------- Retrieval: nomic vectors for all issues, cached on disk ----------
class Retriever:
    def __init__(self):
        data = json.loads(ISSUES_PATH.read_text())
        self.issues = list(data.values()) if isinstance(data, dict) else data
        self.embedder = OllamaEmbeddings(model=EMBED_MODEL)
        numbers = np.array([i["number"] for i in self.issues])

        if INDEX_PATH.exists():
            cached = np.load(INDEX_PATH)
            if np.array_equal(cached["numbers"], numbers):
                self.vecs = cached["vecs"]
                return
        print(f"Building index for {len(self.issues)} issues (one time, ~30 s)...")
        texts = [clean(f"{i['title']}\n\n{i.get('body') or ''}") for i in self.issues]
        #vecs = np.array(self.embedder.embed_documents(texts), dtype=np.float32)
        vecs = []
        for start in range(0, len(texts), 32):
            vecs.extend(self.embedder.embed_documents(texts[start:start + 32]))
        vecs = np.array(vecs, dtype=np.float32)
        self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        np.savez(INDEX_PATH, vecs=self.vecs, numbers=numbers)

    def search(self, text: str, k: int = TOP_K) -> list[dict]:
        q = np.array(self.embedder.embed_query(clean(text)), dtype=np.float32)
        scores = self.vecs @ (q / np.linalg.norm(q))   # cosine similarity
        top = np.argsort(-scores)[:k]
        return [
            {
                "number": self.issues[i]["number"],
                "title": self.issues[i]["title"],
                "url": self.issues[i]["url"],
                "score": float(scores[i]),
            }
            for i in top
        ]


# ---------- State: the shared dictionary every node reads and writes ----------
class TriageState(TypedDict, total=False):
    question: str
    summary: Optional[TriageSummary]
    attempts: int
    error: Optional[str]
    matches: list
    route: str
    reply: str


def log_failure(state: TriageState) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question_start": state["question"][:200],
        "attempts": state.get("attempts"),
        "error": state.get("error"),
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------- The graph ----------
def build_graph(chain=None, retriever=None, threshold: float = SIMILARITY_THRESHOLD):
    """chain and retriever can be swapped for fakes in tests (step 1.7)."""
    chain = chain or build_chain()
    holder = {"retriever": retriever}  # built lazily, so --show-graph needs no Ollama

    def summarize(state: TriageState) -> dict:
        attempts = state.get("attempts", 0) + 1
        try:
            summary = chain.invoke({"question": state["question"]})
            return {"summary": summary, "attempts": attempts, "error": None}
        except Exception as e:  # broken JSON / failed validation
            return {"summary": None, "attempts": attempts,
                    "error": f"{type(e).__name__}: {e}"[:500]}

    def after_summarize(state: TriageState) -> str:
        if state.get("summary") is not None:
            return "ok"
        return "retry" if state["attempts"] < MAX_ATTEMPTS else "give_up"

    def retrieve(state: TriageState) -> dict:
        if holder["retriever"] is None:
            holder["retriever"] = Retriever()
        return {"matches": holder["retriever"].search(state["question"])}

    def after_retrieve(state: TriageState) -> str:
        m = state.get("matches") or []
        return "suggest" if m and m[0]["score"] >= threshold else "new"


    def suggest_related(state: TriageState) -> dict:
        lines = ["This might be related to existing issues:"]
        for m in state["matches"]:
            if m["score"] >= threshold:
                lines.append(f"  #{m['number']} {m['title']} ({m['score']:.2f})\n    {m['url']}")
        lines.append("If none of these match, a maintainer will take a look.")
        s = state["summary"]
        lines.append(f"Summary for maintainers: {s.pain_point} | area: {s.area} | kind: {s.kind}")
        return {"route": "suggest_related", "reply": "\n".join(lines)}


    def flag_new(state: TriageState) -> dict:
        s = state.get("summary")
        if s is None:
            log_failure(state)
            reply = (f"Flagged for maintainers. Summary failed after {state['attempts']} "
                     "tries, so the original question is attached as-is.")
            return {"route": "flag_new", "reply": reply}
        lines = [
            "Flagged as new for maintainers:",
            f"  pain_point: {s.pain_point}",
            f"  area: {s.area} | kind: {s.kind}",
            f"  key_terms: {', '.join(s.key_terms)}",
        ]
        if state.get("matches"):
            top = state["matches"][0]
            lines.append(f"  closest existing: #{top['number']} ({top['score']:.2f}, below threshold)")
        return {"route": "flag_new", "reply": "\n".join(lines)}

    g = StateGraph(TriageState)
    g.add_node("summarize", summarize)
    g.add_node("retrieve", retrieve)
    g.add_node("suggest_related", suggest_related)
    g.add_node("flag_new", flag_new)

    g.add_edge(START, "summarize")
    g.add_conditional_edges("summarize", after_summarize,
                            {"ok": "retrieve", "retry": "summarize", "give_up": "flag_new"})
    g.add_conditional_edges("retrieve", after_retrieve,
                            {"suggest": "suggest_related", "new": "flag_new"})
    g.add_edge("suggest_related", END)
    g.add_edge("flag_new", END)
    return g.compile()


DEMO_QUESTIONS = [
    "Ollama won't use my GPU on Ubuntu 24.04. I have an RTX 3060 and the driver is "
    "installed, but `ollama ps` shows 100% CPU and llama3 is really slow.",
    "Can Ollama handle several requests at the same time? When two users send prompts, "
    "the second one waits until the first finishes.",
    "When I send two images to gemma3 in one chat message on my Mac I get an EOF error.",
    "ollama run gemma4 just fails to start",
    "What's the best recipe for sourdough bread with a crunchy crust?",  # unrelated
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--show-graph", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    graph = build_graph()
    if args.show_graph:
        print(graph.get_graph().draw_mermaid())
        raise SystemExit

    questions = DEMO_QUESTIONS if args.demo else [args.question or DEMO_QUESTIONS[0]]
    for q in questions:
        start = time.time()
        out = graph.invoke({"question": q, "attempts": 0})
        top = out.get("matches", [{}])[0].get("score")
        print("=" * 70)
        print("Q:", q[:90])
        print(f"route={out['route']}  attempts={out['attempts']}  "
              f"top_score={top if top is None else round(top, 3)}  {time.time() - start:.1f}s")
        print(out["reply"])