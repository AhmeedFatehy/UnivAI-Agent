# Agent prompt security

## TL;DR

UnivAI does not rely on a jailbreak keyword list. Every dynamic value is
untrusted by default, placed in a length-bounded delimiter it cannot close, and
sent separately from real system-role instructions in production adapters.
Model-accessible tools are read-only and bound to the authenticated tenant.
Structured output rejects unknown fields, extra documents, oversized values,
and malformed repairs.

## Runtime layers

1. Direct learner queries are normalized (Unicode, invisible formatting, HTML
   entities), bounded to 4,000 characters, and screened for compound override,
   role, exfiltration, and delimiter attempts.
2. Book, learner, RAG, metadata, tool-result, and previous-model content is
   HTML-escaped inside `<untrusted-data>` blocks. It remains readable, including
   legitimate lessons about prompt injection.
3. The versioned boundary policy is attached to every catalog prompt. Provider
   adapters split rendered prompts into real system and human messages.
4. Chat receives only `retrieve_grounded_context` and `get_source_location`.
   Their `user_id` is fixed by runtime code and absent from model arguments.
5. Uploaded PDF pages and downstream lecture narration use the same boundary.
   Direct slide and quiz generation now have dedicated catalog prompts instead
   of contradictory ad-hoc contracts.
6. Pydantic LLM schemas forbid extra fields. Exact JSON (or one enclosing JSON
   fence) is required; invalid output gets one bounded, isolated repair.
7. Query rewrites are count/length bounded, screened again, and must retain an
   original scope term. Unsafe rewrites fall back to the original query.
8. Rejected provider text and prompt fragments are not printed to logs.

## Compatibility changes

- `run_agent_stream()` now requires a keyword-only `user_id` so identity is not
  copied from learner text.
- MCP ingestion accepts only existing `.pdf`, `.docx`, `.txt`, or `.md` files
  under `UnivAI/uploads/<user_id>/`; arbitrary local paths fail safely.
- Source passages flagged as instruction-like are retained as bounded data and
  carry risk metadata. They are no longer deleted solely by a keyword rule.
- Chatty JSON, multiple JSON documents, unknown fields, oversized output, and
  duplicate quiz choices now trigger repair or rejection.
- Course identity includes the prompt-boundary and dedicated slide/quiz prompt
  versions, so older generated artifacts are not silently treated as identical.

## Remaining boundary

MCP transport authentication must still be enforced by deployment/network
configuration. Tenant binding prevents the language model from selecting a
different user, but an unauthenticated caller must never be allowed to call the
MCP server directly from an untrusted network.

## Verification

Focused guardrail, graph, retrieval, generation, evaluation, fallback, and
telemetry tests: **212 passed**.
