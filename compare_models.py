"""Step 2.2: compare 3 small open chat models on the same triage task.

Only the chat model changes. Embedder (nomic), threshold (0.60), prompt,
schema and graph stay fixed. Temperature 0 comes from build_chain().

Run:  uv run python compare_models.py
Cut:  uv run python compare_models.py --n 10 --models qwen3.5:4b gemma4:e4b-it-qat
"""
import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from summarize_chain import SAMPLE_QUESTION, build_chain
from triage_graph import TOP_K, Retriever, build_graph, clean

MODELS = ["qwen3.5:4b", "gemma4:e4b-it-qat", "nemotron-3-nano:4b"]
PAIRS_PATH = Path("data/pairs.json")
CSV_PATH = Path("results/compare.csv")
SUMMARIES_PATH = Path("data/compare_summaries.jsonl")  # model text about issues: stays out of git


class PastOnlyRetriever(Retriever):
    """Same nomic search as the bot, but only over issues created before the question.
    Without this, a duplicate would match itself at 1.000 (data leakage)."""

    def __init__(self):
        super().__init__()
        self.created = np.array([i["created_at"] for i in self.issues])
        self.cutoff = None  # set before each question

    def search(self, text, k=TOP_K):
        q = np.array(self.embedder.embed_query(clean(text)), dtype=np.float32)
        scores = self.vecs @ (q / np.linalg.norm(q))
        if self.cutoff is not None:
            scores = np.where(self.created < self.cutoff, scores, -np.inf)
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


def load_pairs():
    """Returns [(duplicate_number, original_number), ...] whatever the key names are."""
    raw = json.loads(PAIRS_PATH.read_text())
    pairs = []
    for p in raw:
        try:
            dup = next(v for k, v in p.items() if "dup" in k.lower())
            orig = next(v for k, v in p.items() if "orig" in k.lower())
        except StopIteration:
            raise SystemExit(f"Can't find duplicate/original keys in pairs.json: {list(p)}")
        dup = dup["number"] if isinstance(dup, dict) else dup
        orig = orig["number"] if isinstance(orig, dict) else orig
        pairs.append((int(dup), int(orig)))
    return sorted(pairs)


def ollama_stop(model):
    subprocess.run(["ollama", "stop", model], capture_output=True)


def ollama_memory(model):
    """SIZE and CONTEXT columns of `ollama ps`, e.g. '4.1 GB ctx=4096'.
    Note: under-reports llama.cpp models (Gemma); cross-check with wired memory."""
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0] == model:
            return f"{parts[2]} {parts[3]} ctx={parts[6]}"
    return "n/a"


def run_model(model, pairs, retriever, issues_by_num, summaries_file):
    for m in MODELS:  # one chat model in memory at a time
        ollama_stop(m)

    chain = build_chain(model)
    t = time.perf_counter()
    try:
        chain.invoke({"question": SAMPLE_QUESTION})  # warm-up: loads the model, not timed
    except Exception:
        pass
    warmup_s = round(time.perf_counter() - t, 1)
    print(f"\n== {model} (warm-up {warmup_s} s) ==")

    graph = build_graph(chain=chain, retriever=retriever)
    rows = []
    for n, (dup, orig) in enumerate(pairs, 1):
        issue = issues_by_num[dup]
        retriever.cutoff = issue["created_at"]
        question = clean(f"{issue['title']}\n\n{issue.get('body') or ''}")

        t = time.perf_counter()
        out = graph.invoke({"question": question})
        secs = time.perf_counter() - t

        s = out.get("summary")
        attempts = out.get("attempts", 0)
        found = [m["number"] for m in out.get("matches") or []]
        rows.append({
            "model": model,
            "duplicate": dup,
            "original": orig,
            "valid_first_try": int(s is not None and attempts == 1),
            "attempts": attempts,
            "gave_up": int(s is None),
            "seconds": round(secs, 2),
            "route": out.get("route", ""),
            "original_in_top3": int(orig in found),
            "area": s.area if s else "",
            "kind": s.kind if s else "",
        })
        if s:
            summaries_file.write(json.dumps({"model": model, "duplicate": dup,
                                             **s.model_dump()}) + "\n")
        print(f"  {n:2}/{len(pairs)} #{dup}: attempts={attempts} "
              f"{secs:5.1f}s route={rows[-1]['route']}")

    memory = ollama_memory(model)  # read while the model is still loaded
    for r in rows:
        r["memory"] = memory
        r["warmup_s"] = warmup_s
    ollama_stop(model)
    return rows


def print_summary(rows, models):
    print("\nmodel | valid 1st try | gave up | avg attempts | median s | route=suggest | orig in top3 | memory")
    for model in models:
        r = [x for x in rows if x["model"] == model]
        if not r:
            continue
        n = len(r)
        print(f"{model} | {sum(x['valid_first_try'] for x in r)}/{n} | "
              f"{sum(x['gave_up'] for x in r)}/{n} | "
              f"{statistics.mean(x['attempts'] for x in r):.2f} | "
              f"{statistics.median(x['seconds'] for x in r):.1f} | "
              f"{sum(x['route'].startswith('suggest') for x in r)}/{n} | "
              f"{sum(x['original_in_top3'] for x in r)}/{n} | {r[0]['memory']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n", type=int, default=20, help="number of duplicate questions")
    args = ap.parse_args()

    retriever = PastOnlyRetriever()
    issues_by_num = {i["number"]: i for i in retriever.issues}
    pairs = load_pairs()[: args.n]
    print(f"{len(pairs)} questions x {len(args.models)} models")

    rows = []
    CSV_PATH.parent.mkdir(exist_ok=True)
    with SUMMARIES_PATH.open("w") as summaries_file:
        for model in args.models:
            rows += run_model(model, pairs, retriever, issues_by_num, summaries_file)

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, args.models)
    print(f"\nSaved {CSV_PATH} ({len(rows)} rows) and {SUMMARIES_PATH} (not committed)")


if __name__ == "__main__":
    main()