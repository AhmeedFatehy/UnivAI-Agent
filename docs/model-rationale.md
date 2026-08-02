# Model selection rationale

This document records why specific model families are used for each capability.
It exists so the selection is a decision that can be revisited, not an accident.

## Grounding contract

Every answer must be grounded in retrieved course material. That constraint
drives every model choice below: the expensive budget goes into deciding which
passages are relevant, not into producing prose.

## Routed LLM

| Capability | Chosen model | Why |
| --- | --- | --- |
| Main assistant | `qwen3:4b-instruct` | Balances Arabic/English instruction following with latency acceptable for interactive chat; runs on a single consumer GPU. |
| Plan and lecture generation | Same routed LLM | Generation inherits retrieval-grounding; a larger model did not measurably improve plan quality on the held-out rubric. |
| Judge (evaluation) | Same routed LLM, invoked by a strict prompt | The judge validates faithfulness and relevance from the same context the answer used; a separate judge model would only add a second untrusted opinion, not evidence. |
| Fallback | `qwen3:0.6b-instruct` | Small enough to start even when the primary is unavailable; used only for availability, never for novel content. The fallback is bounded and recorded as such in trace metadata. |

## Embeddings and reranking

| Chosen model | Why |
| --- | --- |
| Dense `jinaai/jina-embeddings-v2-base-en` | Long-context dense embeddings (8192 tokens) that capture whole lecture sections without windowed chunking. |
| Sparse `Qdrant/bm25` | Exact-term recall for course-specific jargon (catalog numbers, tool names) that dense embeddings flatten. |
| Reranker `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder relevance is the strongest single predictor of retrieval quality; applied to the top candidates only, keeping it cheap. |

Selection was validated against a grounded RAG dataset; the acceptance
thresholds (`context_recall >= 0.60`, `faithfulness >= 0.70`) live in
`evaluation/report.py` and are enforced by the evaluation runner.

## Deliberate rejections

- **GPT-class hosted APIs**: refused because course material is a private corpus;
  a self-hosted router keeps every query and passage on this machine.
- **A larger instruction model for the judge**: rejected; a strict prompt and a
  strict validator (`evaluation/metrics.py`) make a malformed verdict a visible
  failure rather than a hidden wrong number.
- **Zero-shot generation without grounding**: rejected; it is exactly the
  behaviour the evaluation gate exists to catch.

## Trade-offs accepted

- Embeddings are English-first; low-resource-language content is served by the
  sparse model and cross-encoder reranking, at the cost of weaker semantic recall.
- The fallback model may be underpowered for a task; it is only ever invoked
  when the primary is unavailable, and every such serving is tagged
  `fallback_used` so quality regressions are attributable.
