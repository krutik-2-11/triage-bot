"""Step 1.6b: pick the similarity threshold from labeled data, not a guess.

Run: uv run python calibrate_threshold.py
"""
import json

import numpy as np

from triage_graph import Retriever

PAIRS_PATH = "data/pairs.json"
DUP_KEYS = ("duplicate", "dup", "duplicate_number", "dup_number")
ORIG_KEYS = ("original", "orig", "original_number", "orig_number")


def pick(p: dict, keys: tuple) -> int:
    for k in keys:
        if k in p:
            v = p[k]
            return int(v["number"] if isinstance(v, dict) else v)
    raise KeyError(f"None of {keys} in pair keys {list(p)}. Tell Claude the key names.")


def read_pair(p) -> tuple[int, int]:
    if isinstance(p, dict):
        return pick(p, DUP_KEYS), pick(p, ORIG_KEYS)
    return int(p[0]), int(p[1])


def main():
    r = Retriever()  # reuses the cached index from step 1.6
    idx = {issue["number"]: k for k, issue in enumerate(r.issues)}
    created = np.array([issue["created_at"] for issue in r.issues])
    sims = r.vecs @ r.vecs.T  # cosine, since vectors are normalized

    raw = json.load(open(PAIRS_PATH))
    pairs = [read_pair(p) for p in (raw.values() if isinstance(raw, dict) else raw)]

    # Positives: duplicate vs its labeled original (bot SHOULD suggest)
    pos = np.array([sims[idx[d], idx[o]] for d, o in pairs if d in idx and o in idx])

    # Negatives: normal issue vs its closest OLDER issue (bot should NOT suggest)
    in_pairs = {n for pair in pairs for n in pair}
    neg = []
    for k, issue in enumerate(r.issues):
        if issue["number"] in in_pairs:
            continue
        older = created < created[k]
        if older.any():
            neg.append(sims[k][older].max())
    neg = np.array(neg)

    print(f"Positives (dup vs original): {len(pos)}   Negatives (normal issues): {len(neg)}")
    print(f"Positive scores: min {pos.min():.3f} | 10th pct {np.percentile(pos, 10):.3f} | median {np.median(pos):.3f}")
    print(f"Negative scores: median {np.median(neg):.3f} | 90th pct {np.percentile(neg, 90):.3f} | max {neg.max():.3f}")
    print()
    print("threshold  caught  false_suggest  caught-false")
    for t in np.arange(0.60, 0.861, 0.02):
        c, f = (pos >= t).mean(), (neg >= t).mean()
        print(f"  {t:.2f}     {c:5.1%}     {f:5.1%}        {c - f:+.3f}")

    grid = np.arange(0.50, 0.951, 0.01)
    gaps = [(pos >= t).mean() - (neg >= t).mean() for t in grid]
    best = grid[int(np.argmax(gaps))]
    print()
    print(f"Best threshold by caught-false: {best:.2f} "
          f"(caught {(pos >= best).mean():.1%}, false-suggest {(neg >= best).mean():.1%})")


if __name__ == "__main__":
    main()