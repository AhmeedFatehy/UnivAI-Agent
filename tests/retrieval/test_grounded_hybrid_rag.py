"""Hybrid retrieval, multi-book filters, citations and grounded refusal.

Two layers are covered:

* ``retrieval.pipeline`` — the real fusion/dedup/rerank/filter code, with
  ``hybrid_search_rrf`` and the query-transform LLM stubbed out, so no Qdrant
  and no model call is needed;
* ``tools.registry.retrieve_context`` — the grounding gate that turns hits into
  cited passages or an explicit refusal.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from document_processing.metadata import (
    PAYLOAD_BOOK_TITLE,
    PAYLOAD_COLLECTION_ID,
    PAYLOAD_DOCUMENT_ID,
)
from retrieval import pipeline
from retrieval.hybrid_search import _build_filter
from tests.conftest import COLLECTION_ID, USER_ID
from tools.registry import (
    REFUSAL_NO_GROUNDING,
    REFUSAL_NO_HITS,
    REFUSAL_UNCITABLE,
    GroundedContext,
    RetrieveContextInput,
    ToolContext,
    call_tool,
    term_coverage,
)

ANSWERABLE = "How does a hash table handle collisions between keys?"
OFF_CORPUS = "What are the tax filing deadlines for Egyptian sole proprietors?"


def grounded(query: str, retriever, **kwargs) -> GroundedContext:
    result = call_tool(
        "retrieve_context",
        RetrieveContextInput(query=query, user_id=USER_ID, collection_id=COLLECTION_ID, **kwargs),
        ToolContext(retriever=retriever),
    )
    assert isinstance(result, GroundedContext)
    return result


# ── Grounded answers carry citations ──────────────────────────────────


def test_an_answerable_question_returns_cited_passages(retriever):
    context = grounded(ANSWERABLE, retriever)

    assert context.grounded is True
    assert context.refusal is None
    assert context.passages

    for passage in context.passages:
        citation = passage.citation
        assert citation.collection_id == COLLECTION_ID
        assert citation.document_id
        assert citation.book_title
        assert citation.page >= 1
        assert citation.section
        assert citation.label().startswith(citation.book_title)


def test_the_prompt_block_tags_every_passage_with_its_id(retriever):
    context = grounded(ANSWERABLE, retriever)
    block = context.as_prompt_block()

    for passage in context.passages:
        assert f"[{passage.passage_id}]" in block
        assert passage.citation.book_title in block
    assert set(context.by_id()) == {p.passage_id for p in context.passages}


def test_hash_collision_evidence_is_found_in_both_books_that_cover_it(retriever):
    context = grounded(ANSWERABLE, retriever, limit=10)
    books = {passage.citation.book_title for passage in context.passages}
    assert "Foundations of Algorithms" in books
    assert "Database Systems" in books, (
        "hash collisions are covered by both books; source isolation must not hide one"
    )


# ── Grounded refusal ──────────────────────────────────────────────────


def test_a_question_the_books_do_not_cover_is_refused(retriever):
    context = grounded(OFF_CORPUS, retriever)

    assert context.grounded is False
    assert context.passages == []
    assert context.refusal is not None
    assert context.refusal.reason in (REFUSAL_NO_HITS, REFUSAL_NO_GROUNDING)
    assert context.refusal.query == OFF_CORPUS
    assert context.refusal.scope["collection_id"] == COLLECTION_ID


def test_a_refused_context_renders_as_no_evidence_not_as_an_empty_answer(retriever):
    context = grounded(OFF_CORPUS, retriever)
    assert context.as_prompt_block().startswith("NO EVIDENCE AVAILABLE")


def test_a_weakly_matching_passage_is_refused_rather_than_cited(retriever):
    """One incidental shared word is not evidence."""
    context = grounded("Explain the collisions of planetary orbits", retriever)
    assert context.grounded is False
    assert context.refusal.reason == REFUSAL_NO_GROUNDING
    assert context.refusal.candidates_examined > 0


def test_uncitable_hits_are_refused_not_silently_cited():
    def retriever_without_identity(**kwargs):
        return [{"content": "A hash table handles collisions by chaining.", "score": 0.9}]

    context = grounded(ANSWERABLE, retriever_without_identity)
    assert context.grounded is False
    assert context.refusal.reason == REFUSAL_UNCITABLE


def test_a_grounded_context_cannot_be_constructed_without_passages():
    with pytest.raises(ValueError, match="at least one passage"):
        GroundedContext(query="q", grounded=True, passages=[])


def test_a_refusal_cannot_smuggle_passages_through(retriever):
    context = grounded(ANSWERABLE, retriever)
    with pytest.raises(ValueError, match="must not carry passages"):
        GroundedContext(
            query="q", grounded=False, passages=context.passages, refusal=context.refusal
        )


def test_term_coverage_scores_what_the_gate_uses():
    assert term_coverage("hash table collisions", "collisions in a hash table") == 1.0
    assert term_coverage("hash table collisions", "orbital collisions") == pytest.approx(1 / 3)
    assert term_coverage("hash table collisions", "unrelated prose") == 0.0


# ── Multi-book metadata filters ───────────────────────────────────────


def test_filtering_by_document_isolates_one_book(retriever, indexed_books):
    _, records = indexed_books
    algorithms = next(
        record.document_id
        for record in records
        if record.book_title == "Foundations of Algorithms"
    )

    context = grounded(ANSWERABLE, retriever, document_ids=[algorithms], limit=10)

    assert context.grounded is True
    assert {passage.citation.document_id for passage in context.passages} == {algorithms}


def test_filtering_by_book_title_isolates_one_book(retriever):
    context = grounded(ANSWERABLE, retriever, book_titles=["Database Systems"], limit=10)
    assert {passage.citation.book_title for passage in context.passages} == {
        "Database Systems"
    }


def test_a_foreign_collection_returns_nothing(retriever):
    result = call_tool(
        "retrieve_context",
        RetrieveContextInput(
            query=ANSWERABLE, user_id=USER_ID, collection_id="some-other-programme"
        ),
        ToolContext(retriever=retriever),
    )
    assert result.grounded is False


def test_another_user_cannot_reach_this_collection(retriever):
    result = call_tool(
        "retrieve_context",
        RetrieveContextInput(
            query=ANSWERABLE, user_id="a-different-student", collection_id=COLLECTION_ID
        ),
        ToolContext(retriever=retriever),
    )
    assert result.grounded is False


def test_scope_reaches_the_retriever_verbatim(retriever):
    grounded(ANSWERABLE, retriever, document_ids=["doc-1"], book_titles=["Book"])
    call = retriever.calls[-1]
    assert call["user_id"] == USER_ID
    assert call["collection_id"] == COLLECTION_ID
    assert call["document_ids"] == ["doc-1"]
    assert call["book_titles"] == ["Book"]


# ── Resolving a citation back to the book ─────────────────────────────


def locator_over(rows):
    def locate(*, user_id, document_id, chunk_index):
        return [
            row
            for row in rows
            if row["user_id"] == user_id
            and row["document_id"] == document_id
            and (chunk_index is None or row["chunk_index"] == chunk_index)
        ]

    return locate


def test_a_citation_resolves_to_its_book_page_section_and_excerpt(retriever, indexed_books):
    index, records = indexed_books
    target = next(
        record
        for record in records
        if record.book_title == "Database Systems" and record.chunk_index == 2
    )

    result = call_tool(
        "get_source_location",
        {
            "user_id": USER_ID,
            "document_id": target.document_id,
            "chunk_index": 2,
            "collection_id": COLLECTION_ID,
        },
        ToolContext(locator=locator_over(index.rows)),
    )

    assert result.found is True
    assert result.location.book_title == "Database Systems"
    assert result.location.page == target.page
    assert result.location.section == target.section
    assert result.excerpt


def test_an_unknown_document_is_reported_as_not_found(indexed_books):
    index, _ = indexed_books
    result = call_tool(
        "get_source_location",
        {"user_id": USER_ID, "document_id": "no-such-document"},
        ToolContext(locator=locator_over(index.rows)),
    )

    assert result.found is False
    assert result.location is None
    assert "no-such-document" in result.reason


def test_a_document_from_another_collection_is_rejected(indexed_books):
    index, records = indexed_books
    result = call_tool(
        "get_source_location",
        {
            "user_id": USER_ID,
            "document_id": records[0].document_id,
            "collection_id": "a-different-programme",
        },
        ToolContext(locator=locator_over(index.rows)),
    )

    assert result.found is False
    assert "a-different-programme" in result.reason


def test_a_found_result_cannot_omit_the_location():
    from tools.registry import SourceLocationResult

    with pytest.raises(ValueError, match="must include the location"):
        SourceLocationResult(found=True)

    with pytest.raises(ValueError, match="must say why"):
        SourceLocationResult(found=False)


# ── The real pipeline: fusion, filters, transform, rerank ─────────────


def _hit(chunk_id: str, content: str, score: float, **payload) -> dict:
    base = {
        "id": chunk_id,
        "content": content,
        "score": score,
        "document_id": payload.get("document_id", "doc-a"),
        "source_filename": "book.md",
        "chunk_index": payload.get("chunk_index", 0),
        "total_chunks": 4,
        "page_number": payload.get("page_number"),
        "metadata": {
            "collection_id": COLLECTION_ID,
            "book_title": payload.get("book_title", "Book A"),
            "section": payload.get("section", "Some Section"),
            "page_number": str(payload.get("page_number", 3)),
            "page_is_estimated": "True",
        },
    }
    return base


def test_pipeline_expands_document_filters_into_one_search_per_document(monkeypatch):
    seen: list[dict | None] = []

    def fake_search(*, query_text, user_id, limit, collection_name, filters):
        seen.append(filters)
        return [_hit(f"{filters[PAYLOAD_DOCUMENT_ID]}-0", "content about hashing", 0.5)]

    monkeypatch.setattr(pipeline, "hybrid_search_rrf", fake_search)

    pipeline.retrieve(
        "hashing",
        user_id=USER_ID,
        use_reranking=False,
        collection_id=COLLECTION_ID,
        document_ids=["doc-a", "doc-b"],
    )

    assert len(seen) == 2
    assert seen[0][PAYLOAD_DOCUMENT_ID] == "doc-a"
    assert seen[1][PAYLOAD_DOCUMENT_ID] == "doc-b"
    assert all(item[PAYLOAD_COLLECTION_ID] == COLLECTION_ID for item in seen)


def test_pipeline_uses_the_query_transformation_strategy(monkeypatch):
    queries: list[str] = []

    def fake_search(*, query_text, user_id, limit, collection_name, filters):
        queries.append(query_text)
        return [_hit(f"c{len(queries)}", "content about hashing", 0.4)]

    monkeypatch.setattr(pipeline, "hybrid_search_rrf", fake_search)
    monkeypatch.setattr(
        pipeline, "decompose_query", lambda q: [q, "what is a hash function"]
    )

    pipeline.retrieve("hashing", user_id=USER_ID, use_reranking=False, use_query_transform=True)

    assert queries == ["hashing", "what is a hash function"]


def test_pipeline_deduplicates_the_same_chunk_across_sub_queries(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "hybrid_search_rrf",
        lambda **kwargs: [_hit("same-chunk", "hashing content", 0.6)],
    )
    monkeypatch.setattr(pipeline, "decompose_query", lambda q: [q, "second phrasing"])

    results = pipeline.retrieve(
        "hashing", user_id=USER_ID, use_reranking=False, use_query_transform=True
    )
    assert len(results) == 1


def test_pipeline_promotes_book_and_section_into_the_citation_line(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "hybrid_search_rrf",
        lambda **kwargs: [
            _hit("c1", "hashing", 0.7, book_title="Database Systems", section="Hash Indexes", page_number=3)
        ],
    )

    results = pipeline.retrieve("hashing", user_id=USER_ID, use_reranking=False)
    doc = results[0]

    assert doc["book_title"] == "Database Systems"
    assert doc["section"] == "Hash Indexes"
    assert doc["page_number"] == 3
    assert doc["collection_id"] == COLLECTION_ID
    assert "Database Systems" in doc["citation"]
    assert "Page: 3" in doc["citation"]
    assert "Section: Hash Indexes" in doc["citation"]


def test_pipeline_reranks_when_asked(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "hybrid_search_rrf",
        lambda **kwargs: [
            _hit("low", "weak match", 0.10, chunk_index=0),
            _hit("high", "strong match", 0.20, chunk_index=1),
        ],
    )
    reranked_with: list[str] = []

    def fake_rerank(query, documents, top_k):
        reranked_with.append(query)
        return sorted(documents, key=lambda d: -d["score"])[:top_k]

    monkeypatch.setattr(pipeline, "rerank", fake_rerank)

    results = pipeline.retrieve("hashing", user_id=USER_ID, use_reranking=True, limit=1)

    assert reranked_with == ["hashing"]
    assert [doc["id"] for doc in results] == ["high"]


# ── Filter construction is what Qdrant will actually receive ──────────


def test_source_filters_build_the_expected_variants():
    variants = pipeline.build_source_filters(
        {"document_type": "pdf"},
        collection_id=COLLECTION_ID,
        document_ids=["doc-a", "doc-b"],
    )
    assert variants == [
        {"document_type": "pdf", PAYLOAD_COLLECTION_ID: COLLECTION_ID, PAYLOAD_DOCUMENT_ID: "doc-a"},
        {"document_type": "pdf", PAYLOAD_COLLECTION_ID: COLLECTION_ID, PAYLOAD_DOCUMENT_ID: "doc-b"},
    ]


def test_book_title_filters_fall_back_when_no_documents_are_named():
    variants = pipeline.build_source_filters(
        None, collection_id=COLLECTION_ID, book_titles=["Database Systems"]
    )
    assert variants == [
        {PAYLOAD_COLLECTION_ID: COLLECTION_ID, PAYLOAD_BOOK_TITLE: "Database Systems"}
    ]


def test_no_scope_means_no_filter():
    assert pipeline.build_source_filters(None) == [None]


def test_the_qdrant_filter_carries_user_and_collection_conditions():
    query_filter = _build_filter(
        USER_ID, {PAYLOAD_COLLECTION_ID: COLLECTION_ID, PAYLOAD_DOCUMENT_ID: "doc-a"}
    )
    assert isinstance(query_filter, models.Filter)
    keys = [condition.key for condition in query_filter.must]
    assert keys == ["user_id", PAYLOAD_COLLECTION_ID, PAYLOAD_DOCUMENT_ID]
    assert query_filter.must[1].match.value == COLLECTION_ID
