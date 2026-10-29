# Models compared

MacBook Air M5, 16 GB. Context 4K (Ollama default). Checked 2026-10-28.

## 1. The three models (step 2.1)

| Model | Family | Disk | Memory at load | Runner | License | Released |
|---|---|---|---|---|---|---|
| qwen3.5:4b | Qwen (Alibaba) | 4.0 GB | 4.1 GB (`ollama ps`), ≈3.9 GB (wired Δ) | MLX | Apache 2.0 | 2026 |
| gemma4:e4b-it-qat | Gemma (Google) | 6.1 GB | ≈4.1 GB (wired Δ); `ollama ps` shows 261 MB | llama.cpp | Apache 2.0 | Apr 2026 (QAT build Jun 2026) |
| nemotron-3-nano:4b | Nemotron (NVIDIA) | 2.9 GB | 3.1 GB (`ollama ps`), ≈3.0 GB (wired Δ) | MLX | NVIDIA Nemotron Open Model License | 16 Mar 2026 |

- Ollama picked the runner per model (MLX = Apple's framework, llama.cpp = cross-platform engine). Speed reflects model + runtime together.
- Licenses checked on each official Hugging Face page.
- Skipped: gemma4:e4b (~9.6 GB, over budget), nemotron-3.5-bolt:e4b (~7.2–7.9 GB, over budget), nemotron-mini (2 years old).

**Memory correction (step 2.2):** in step 2.1 one `ollama ps` reading showed gemma at 3.1 GB. Later readings consistently showed 261 MB, which is too low for a 4b model. So I cross-checked with macOS wired memory (`top -l 1 | grep PhysMem`, before vs after loading). Wired matches `ollama ps` for the two MLX models, but not for gemma. I trust wired memory for gemma. Likely cause (not verified): llama.cpp memory-maps the weights and `ollama ps` doesn't count them.

## 2. Comparison run (step 2.2)

Setup: 49 maintainer-labeled duplicate pairs from ollama/ollama, one question each (title + body, #numbers stripped, max 2,000 chars). Fixed: embedder nomic-embed-text, threshold 0.60 (on-topic gate), prompt and schema, temperature 0, reasoning=False. Only the chat model changes. One model in memory at a time, with an untimed warm-up call. Retrieval searches only issues created before each question (no self-match). Raw rows: [compare.csv](compare.csv) (147 rows). Script: [compare_models.py](../compare_models.py).

| Model | Valid JSON 1st try | Gave up | Avg attempts | Median s/question | Route suggest | Original in top 3 | Memory after 49 (`ollama ps`) |
|---|---|---|---|---|---|---|---|
| qwen3.5:4b | 49/49 | 0/49 | 1.00 | 2.9 | 47/49 | 43/49 | 10 GB |
| gemma4:e4b-it-qat | 49/49 | 0/49 | 1.00 | 3.4 | 47/49 | 43/49 | 261 MB (under-reported) |
| nemotron-3-nano:4b | 49/49 | 0/49 | 1.00 | 3.1 | 47/49 | 43/49 | 11 GB |

## 3. Labels (area, kind)

All 3 models agree: area 30/49, kind 20/49.

| Model | bug | question | feature_request | Hand audit kind (random 9) | Hand audit area (random 10) |
|---|---|---|---|---|---|
| qwen3.5:4b | 25 | 11 | 13 | 8/9 | 9/10 |
| gemma4:e4b-it-qat | 28 | 7 | 14 | 9/9 | 10/10 |
| nemotron-3-nano:4b | 10 | 26 | 13 | 5/9 | 9/10 |

Rules, disagreement sample and schema limits: [audit_2.3.md](audit_2.3.md).

## 4. What it shows (trade-offs, not a winner)

- **JSON reliability does not separate these models.** 147/147 valid on the first try; the retry loop never fired.
- **Retrieval stayed fixed.** Route and top-3 results are identical across models, as designed.
- **Speed is a tie.** The ~0.5 s gap flipped order between a 20-question and a 49-question run.
- **Memory needs two tools, and grows under load.** `ollama ps` under-reports the llama.cpp model. The MLX models grew from ~3–4 GB to 10–11 GB over 49 requests at the same 4K context (cause not proven).
- **Labels are where models differ.** Nemotron labels many failures (e.g. pull errors) as "question", which would change a demand count by kind.

## 5. Limits

- 49 pairs from one repo; no "X% better" claims.
- Speed includes the runner (MLX vs llama.cpp), not the model alone.
- Gemma memory after 49 requests is unknown (`ollama ps` unreliable for it; wired memory measured at load only).
- Label audit: one labeler, titles only, 17 issues; labels proposed by Claude, confirmed or overridden by me.
- `model_support` bucket is too broad (schema limit, not a model fault).