# Models compared (step 2.1)

Checked on 2026-10-28. Memory = `ollama ps` SIZE right after one short prompt, 4K context (Ollama default), MacBook Air M5, 16 GB.

| Model | Family | Disk | Memory (ollama ps) | Runner | License | Released |
|---|---|---|---|---|---|---|
| qwen3.5:4b | Qwen (Alibaba) | 4.0 GB | 4.1 GB | MLX | Apache 2.0 | 2026 |
| gemma4:e4b-it-qat | Gemma (Google) | 6.1 GB | 3.1 GB | llama.cpp | Apache 2.0 | Apr 2026 (QAT build Jun 2026) |
| nemotron-3-nano:4b | Nemotron (NVIDIA) | 2.9 GB | 3.1 GB | MLX | NVIDIA Nemotron Open Model License | 16 Mar 2026 |

Notes
- Ollama picked the runner per model (MLX = Apple's framework, llama.cpp = cross-platform engine). Speed results reflect model + runtime together, not the model alone.
- Process RSS isn't comparable across runners (MLX memory lives in Metal), so we use `ollama ps` for all three.
- First gemma reading was 261 MB (likely mid-load); a second reading gave 3.1 GB. Why gemma's memory is below its disk size is not verified.
- Licenses checked on each official Hugging Face page.
- Skipped: gemma4:e4b (~9.6 GB, over budget), nemotron-3.5-bolt:e4b (~7.2–7.9 GB, over budget), nemotron-mini (2 years old).