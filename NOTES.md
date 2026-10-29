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

- Expected: disk size ≈ memory used. Wrong: gemma4:e4b-it-qat is 6.1 GB on disk but added ≈4.1 GB of wired memory at load.
- Surprised: Ollama chose different runners (MLX for Qwen and Nemotron, llama.cpp for Gemma), so speed comparisons mix model and runtime.
- Learned: "open-weight" doesn't mean one license. Two use Apache 2.0, Nemotron uses NVIDIA's own open model license.

### Comparing models (steps 2.2–2.3)

- Expected: small models would break JSON and need retries. Got wrong: all three gave valid JSON on the first try, 49/49, and the retry loop never fired. With a JSON schema, temperature 0 and reasoning off, structure wasn't the hard part.
- Got wrong: I trusted `ollama ps` for memory. It showed Gemma at 261 MB, impossible for a 4b model. Cross-checking macOS wired memory gave ≈4.1 GB, and showed the two MLX models growing from ~3–4 GB to 10–11 GB after 49 questions (cause unproven). Memory after many requests matters more than memory at load.
- Got wrong: on 20 questions, one model looked clearly fastest. On 49, the order flipped. Medians ended within 0.5 s, so speed is a tie.
- Got wrong: I first hand-checked only the issues where models disagreed on labels. That made Nemotron look terrible on `kind` (0/8). A random sample showed a real but smaller gap (5/9 vs 8/9 and 9/9). Picking cases by disagreement biases the result against the odd one out.
- Learned: the models didn't differ in filling the form. They differed in the labels, and product would count demand by those labels. My label schema was also too coarse for a one-product repo (8 of 10 random issues landed in `model_support`).