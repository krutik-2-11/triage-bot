"""
Step 1.4: Embeddings and search, with a comparison of embedding models.

What this script does:
  1. Loads every issue (data/issues.json) and the answer key (data/pairs.json).
  2. For EACH embedding model, turns every issue into a vector (once, then cached).
  3. For each known duplicate, ranks ONLY the issues created before it
     by cosine similarity.
  4. Scores the ranking against the answer key: Recall@1, Recall@5, MRR.
  5. Prints one comparison table, then the misses of the best model
     (by MRR) for a hand audit.

Before running, pull the models once:
  ollama pull nomic-embed-text
  ollama pull qwen3-embedding:0.6b

Run:  uv run python build_index.py                     (compares the default models)
      uv run python build_index.py nomic-embed-text    (just one model)
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings

# ---------------- Settings ----------------

# Models to compare. Command-line arguments override this list.
DEFAULT_MODELS = ["nomic-embed-text", "qwen3-embedding:0.6b"]

DATA = Path("data")

# Many issues contain huge log dumps. Keeping only the first 2,000 characters
# (title + start of body) keeps the vector focused on the actual problem.
# Same limit for every model, so the comparison is fair.
MAX_CHARS = 2000

# How many issues to send to Ollama in one request.
BATCH_SIZE = 32


# ---------------- Preparing the text ----------------

def clean(text):
    """Remove '#1234'-style references so an issue number can't leak the answer."""
    return re.sub(r"#\d+", "", text or "")


def issue_text(issue):
    """The text we embed for one issue: title, blank line, body, trimmed."""
    return clean(f"{issue['title']}\n\n{issue.get('body') or ''}")[:MAX_CHARS]


def load():
    """Read the issues and the answer key that fetch_issues.py saved."""
    issues = json.loads((DATA / "issues.json").read_text())
    pairs = json.loads((DATA / "pairs.json").read_text())
    return issues, pairs


# ---------------- Embedding ----------------

def vec_file(model):
    """One cache file per model, so switching models never mixes up vectors."""
    return DATA / f"vectors_{model.replace(':', '_').replace('/', '_')}.npy"


def embed_all(issues, model):
    """Return (vectors, seconds). Embeds once per model, then reuses the cache.

    seconds is None when the vectors came from the cache (no fresh timing).
    """
    path = vec_file(model)
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(issues):
            print(f"[{model}] loaded cached vectors from {path}")
            return vecs, None

    emb = OllamaEmbeddings(model=model)
    texts = [issue_text(i) for i in issues]

    # Send the texts in batches and collect one vector per issue, in order.
    vecs = []
    t0 = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        vecs.extend(emb.embed_documents(texts[start:start + BATCH_SIZE]))
        print(f"[{model}] embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    seconds = time.perf_counter() - t0   # speed is part of the trade-off

    vecs = np.array(vecs, dtype=np.float32)
    np.save(path, vecs)   # data/ is in .gitignore, so this is never committed
    return vecs, seconds


# ---------------- Scoring logic ----------------

def cosine_scores(query_vec, matrix):
    """One similarity score per row of `matrix`.

    cosine = dot(a, b) / (length(a) * length(b))
    1 = same direction (same meaning), 0 = unrelated, -1 = opposite.
    It ignores length, so a long issue and a short issue can still match.
    """
    dots = matrix @ query_vec                                   # all dot products at once
    lengths = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    return dots / np.maximum(lengths, 1e-12)                    # avoid dividing by zero


def rank_candidates(dup_idx, issues, vecs):
    """Issue numbers ranked best match first, using ONLY issues created earlier.

    This is the most important rule in the evaluation: when a question is posted,
    only older issues exist, so the bot must never see "future" issues.
    The strict '<' also excludes the duplicate itself.
    """
    cutoff = issues[dup_idx]["created_at"]       # ISO strings compare correctly with <
    cand_idx = [i for i, iss in enumerate(issues) if iss["created_at"] < cutoff]
    if not cand_idx:
        return []

    scores = cosine_scores(vecs[dup_idx], vecs[cand_idx])
    order = np.argsort(-scores)                  # minus sign = highest score first
    return [issues[cand_idx[j]]["number"] for j in order]


def rank_of(original_number, ranked_numbers):
    """1-based position of the original in the ranked list, or None if absent."""
    try:
        return ranked_numbers.index(original_number) + 1   # +1: Python counts from 0
    except ValueError:
        return None


def metrics(ranks):
    """Recall@1, Recall@5 and MRR over all pairs.

    ranks: one entry per pair, the original's rank (int) or None if not found.
    Recall@k = share of pairs whose original is in the top k.
    MRR      = average of 1/rank (a not-found pair adds 0).
    """
    n = len(ranks)
    return {
        "recall@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
    }


# ---------------- One model, end to end ----------------

def evaluate(model, issues, pairs):
    """Embed with one model and score it. Returns metrics, seconds and misses."""
    vecs, seconds = embed_all(issues, model)
    index_of = {iss["number"]: i for i, iss in enumerate(issues)}

    ranks, misses = [], []
    for p in pairs:
        dup, orig = p["duplicate"], p["original"]   # KEYS: rename if pairs.json differs
        ranked = rank_candidates(index_of[dup], issues, vecs)
        r = rank_of(orig, ranked)
        ranks.append(r)
        if r != 1:
            # Keep what we need for the hand audit.
            misses.append((dup, orig, r, ranked[0] if ranked else None))

    return metrics(ranks), seconds, misses


# ---------------- Main ----------------

def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    issues, pairs = load()
    title_of = {iss["number"]: iss["title"] for iss in issues}

    # Same issues, same pairs, same text for every model: a fair comparison.
    results = {m: evaluate(m, issues, pairs) for m in models}

    # The strict score: only the maintainer's label counts as correct.
    print(f"\nComparison on {len(pairs)} pairs, {len(issues)} issues")
    print(f"{'model':<26}{'Recall@1':>10}{'Recall@5':>10}{'MRR':>8}{'embed s':>10}")
    for model, (m, seconds, _) in results.items():
        secs = f"{seconds:.1f}" if seconds is not None else "cached"
        print(f"{model:<26}{m['recall@1']:>10.3f}{m['recall@5']:>10.3f}"
              f"{m['mrr']:>8.3f}{secs:>10}")
    print("Note: 1 pair = ~0.02. A gap of 2-3 pairs or less is a tie.")

    # Hand audit for the best model only: is the bot's #1 actually related?
    best = max(results, key=lambda mdl: results[mdl][0]["mrr"])
    misses = results[best][2]
    print(f"\nBest by MRR: {best}. Not ranked #1 ({len(misses)} pairs), for the hand audit:")
    for dup, orig, r, top in misses:
        print(f"\n#{dup} (original #{orig}, found at rank {r})")
        print(f"  dup:      {title_of[dup]}")
        print(f"  original: {title_of[orig]}")
        print(f"  bot's #1: #{top} {title_of.get(top, '?')}")


if __name__ == "__main__":
    main()