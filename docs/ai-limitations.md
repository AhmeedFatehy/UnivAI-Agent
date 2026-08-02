# Known limitations and guardrail boundaries

This document is an honest inventory of what the system does not guarantee. It
is kept next to the code so that claims never outrun behaviour.

## Guardrails are deterministic, not adversarial

Prompt-injection screening in `guardrails/input.py` is a rule-based classifier.
It catches the common shapes (instruction overriding, role reassignment,
exfiltration requests, injected delimiters) with no model in the loop. It is
**not** an adversary-proof jailbreak defense:

- Novel or obfuscated attack strings can pass screening; this is mitigated by
  the grounding gate — retrieved source text is treated as data, never as
  instructions — not by perfect classification.
- Raw retrieval results retain flagged source text for operator inspection and
  mark it with `source_injection_flagged`. The grounded tool excludes flagged
  passages from model evidence; if every matching passage is flagged, it
  returns an explicit refusal.

## The fallback buys availability, not correctness

`resilience/fallback.py` retries each backend exactly once (bounded fallback).
The fallback model is small and may answer badly. Every fallback serving is
recorded (`fallback_used`, `fallback_reason`, `attempts`) in trace metadata so a
quality regression caused by a fallback is attributable — but the fallback does
not raise the quality ceiling.

## Judge evaluation is strict but finite

The evaluation judge must return well-formed scores; anything else is an
explicit failure (`JudgeOutputError`, `JudgeUnavailableError`), never a zero.
Acceptance thresholds are fixed at `context_recall >= 0.60` and
`faithfulness >= 0.70`. Limitations:

- A judge model can still be wrong while well-formed; thresholds encode a target
  policy, not ground truth.
- The published evaluation dataset lives in the main repository
  (`tests/fixtures/evaluation/`); CI exercises the runner with self-contained
  fixtures, so the runner is verified even before that dataset exists.

## Trace metadata reflects the provider

`ServingRecord` records only what a provider reports. Token counts and cost stay
`null` when a provider does not report them; the system never invents them, and
cost is never estimated from token counts. This keeps metrics honest at the
price of sometimes-incomplete rows.

## Embeddings are English-first

Dense retrieval (`jinaai/jina-embeddings-v2-base-en`) is optimised for English;
Arabic and other low-resource content relies on exact-term sparse recall and the
cross-encoder reranker. Semantic recall for those languages is weaker, and that
is an accepted, documented trade-off rather than a bug.

## The assistant is a course assistant

The agent is grounded in indexed course material and does not browse the web,
does not claim real-time knowledge, and may be incomplete where the corpus is
incomplete. Injection attempts against the assistant and the retrieval tools are
refused up front, but a non-injection question that the corpus simply cannot
answer is not guaranteed to be refused — it may produce a grounded but partial
answer. Trace metadata records every LLM serving so such gaps are visible rather
than silently absorbed.
