# Step 2.3 hand audit: kind and area labels

Labels are from issue titles only, decided by Krutik (some proposed by Claude, all confirmed or overridden by Krutik).

Rule used:
- bug = something should work and doesn't
- question = asks how to do something; nothing is broken
- feature_request = asks for something new or better (including clearer error messages)

An error message alone does not make a bug: a setup problem is not a bug; Ollama behaving wrongly is.
Borderline items are marked. Ambiguous items are left out of the tally.

## Sample A: disagreement sample (first 8 issues where the 3 models disagreed on kind)
Caveat: chosen because the models disagreed, so the odd-one-out model looks worse than it is.

| Issue | kind | area | gemma kind/area | qwen kind/area | nemotron kind/area |
|---|---|---|---|---|---|
| #85 | bug | (all agree) | ✓ / – | ✓ / – | ✗ / – |
| #832 | bug (borderline) | gpu | ✓ / ✗ | ✗ / ✓ | ✗ / ✓ |
| #947 | feature_request (asks for a clearer error) | none fits (auth) | ✗ / – | ✗ / – | ✗ / – |
| #1366 | feature_request | api | ✓ / ✓ | ✓ / ✓ | ✗ / ✗ |
| #2683 | bug | model_support | ✓ / ✓ | ✓ / ✗ | ✗ / ✓ |
| #2971 | bug (borderline: maintainer linked it to a root cause) | api | ✓ / ✓ | ✓ / ✗ | ✗ / ✓ |
| #3164 | bug | gpu | ✓ / ✗ | ✓ / ✗ | ✗ / ✓ |
| #3170 | feature_request | (all agree) | ✓ / – | ✓ / – | ✗ / – |

kind (8): gemma 7/8, qwen 6/8, nemotron 0/8
area (5 judged): gemma 3/5, qwen 2/5, nemotron 4/5

## Sample B: random sample (seed 42, 10 of the remaining 41 pairs)

| Issue | kind | area | gemma kind/area | qwen kind/area | nemotron kind/area |
|---|---|---|---|---|---|
| #1557 | ambiguous (no baseline for "slow"), not counted | performance | – / ✓ | – / ✓ | – / ✗ |
| #3191 | bug | model_support | ✓ / ✓ | ✓ / ✓ | ✗ / ✓ |
| #3212 | bug | model_support | ✓ / ✓ | ✓ / ✓ | ✗ / ✓ |
| #3322 | bug | model_support | ✓ / ✓ | ✓ / ✓ | ✗ / ✓ |
| #3326 | bug | model_support | ✓ / ✓ | ✓ / ✓ | ✗ / ✓ |
| #5346 | bug | model_support | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| #5845 | feature_request | model_support | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| #6387 | feature_request | model_support | ✓ / ✓ | ✗ / ✓ | ✓ / ✓ |
| #10004 | feature_request | model_support | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| #18836 | bug | model_support | ✓ / ✓ | ✓ / ✗ | ✓ / ✓ |

kind (9 judged): gemma 9/9, qwen 8/9, nemotron 5/9
area (10): gemma 10/10, qwen 9/10, nemotron 9/10

## What this shows
- Nemotron tends to label failures as "question" (e.g. pull errors). The skew is real but smaller on the random sample (5/9) than on the disagreement sample (0/8).
- Area labels are reliable for all three, but 8 of 10 random issues are model_support: the bucket is too broad (schema limit, not a model fault).
- Limits: 17 issues, one labeler, titles only. Not enough to claim "X% better".

## Schema limits found during the audit

The models choose from fixed lists (set with `Literal` in `TriageSummary`, step 1.5):
- area: install, gpu, api, model_support, performance, cli, other
- kind: bug, feature_request, question

**model_support is too broad.** 8 of 10 random issues landed there, but they are different problems:
- model download/pull fails: #3191, #3212
- request for a new model: #5845, #6387
- bad model output: #5346
- vision model not working: #3322

In a single-product repo almost every issue mentions a model, so model_support was a fair answer from the models. The fault is the category list, not the models. A count of "33 model_support issues" doesn't tell a product team whether downloads are broken or users want new models.

**Missing categories:**
- area: nothing fits auth/registry errors (#947) or off-topic posts (Phase 1 sourdough test, labeled `cli`)
- kind: no option for "improve something that exists" (#947, clearer error message) or "not enough information" (#1557)

**Next version (not done, to keep prompt and schema fixed across the model comparison):**
- split model_support into model_request, model_download, model_output
- add area `not_about_ollama`
- add kind `enhancement` and `needs_info`
- re-run all 3 models after the change