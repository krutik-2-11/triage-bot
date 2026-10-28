# NOTES: what I assumed, what surprised me, what I got wrong

A running log, written as I built. Small test set: 49 duplicate pairs from one repo (ollama/ollama).

## Phase 1: building the bot

### Data and leakage (steps 1.2–1.3)
- Expected: duplicate posts would mention their original, which would make the test too easy.
- Checked: 0 of 49 duplicate bodies mention the original's number. The "Duplicate of #N" text lives in maintainer comments, which I excluded.
- Learned: maintainers don't label every duplicate, so some "misses" are real related issues. That's why I hand-audit misses.

### Embeddings (step 1.4)
- Expected: the bigger embedder (qwen3-embedding:0.6b) would win.
- Got wrong: quality was a tie (1–2 pairs on 49), and nomic-embed-text was ~3x faster. Bigger isn't automatically better on your task.
- Result: Recall@1 0.796 (audited 0.878), Recall@5 0.898, MRR 0.843.

### Structured output (step 1.5)
- Got wrong: I thought the chain returns JSON text. It returns a validated Python object (TriageSummary); broken JSON raises an error instead.

### Threshold (step 1.6)
- Expected: a cosine cutoff (~0.75) would separate duplicates from non-duplicates.
- Surprised: normal issues scored higher than true duplicates (negative median 0.783 vs positive median 0.759). In a one-product repo every issue has a same-topic neighbor.
- Learned: ranking works, absolute scores don't. 0.60 is now an on-topic gate, not a duplicate detector, and a human confirms.

### Search limits (step 1.8 demo)
- Surprised: a GPT-5.5 question matched 3 "model not loading" issues (~0.73), but the real answer wasn't in any issue, and the summary dropped the model name.
- Learned: search finds the topic, not the answer.

## Phase 2: comparing open models

### Picking models (step 2.1)
- Expected: disk size ≈ memory used. Wrong: gemma4:e4b-it-qat is 6.1 GB on disk but showed 3.1 GB in `ollama ps`.
- Surprised: Ollama chose different runners (MLX for Qwen and Nemotron, llama.cpp for Gemma), so speed comparisons mix model and runtime.
- Got wrong: I'd have trusted the first `ollama ps` reading (261 MB for gemma). Measuring twice caught it.
- Learned: "open-weight" doesn't mean one license. Two use Apache 2.0, Nemotron uses NVIDIA's own open model license.