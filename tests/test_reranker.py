"""Regression tests for cross-encoder reranking.

The bug these guard against: ``TextCrossEncoder.rerank`` returns a plain list of
floats in input order, but the reranker read ``.index``/``.score`` off each
result. A float has neither, so ``getattr(..., 0)`` returned 0 every time and
every result collapsed onto ``documents[0]`` with score 0.0.

A test that only counted results would have passed. These assert the two things
that actually broke: which documents come back, and in what order.

The model is faked, so this runs offline with no model download.
"""

from __future__ import annotations

import unittest

from retrieval import reranker


class FakeCrossEncoder:
    """Mimics fastembed: one float per document, in the order given."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents, **kwargs):
        documents = list(documents)
        self.calls.append((query, documents))
        return list(self.scores[: len(documents)])


class ExplodingCrossEncoder:
    def rerank(self, query, documents, **kwargs):
        raise RuntimeError("onnx session died")


class WrongLengthCrossEncoder:
    """Returns fewer scores than documents — the shape we must not trust."""

    def rerank(self, query, documents, **kwargs):
        return [0.5]


def documents() -> list[dict]:
    return [
        {"id": "c1", "content": "the ls command lists files", "score": 0.9},
        {"id": "c2", "content": "grep searches text", "score": 0.8},
        {"id": "c3", "content": "cat prints a file", "score": 0.7},
    ]


class RerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (reranker._reranker, reranker._reranker_available)

    def tearDown(self) -> None:
        reranker._reranker, reranker._reranker_available = self._saved

    def use(self, fake) -> None:
        reranker._reranker = fake
        reranker._reranker_available = True

    # ── The bug ───────────────────────────────────────────────────────

    def test_every_result_is_a_distinct_document(self) -> None:
        """The defect returned documents[0] top_k times."""
        self.use(FakeCrossEncoder([-11.2, -11.4, 0.03]))

        results = reranker.rerank("how do I print a file", documents(), top_k=3)

        self.assertEqual(3, len(results))
        self.assertEqual(["c3", "c1", "c2"], [doc["id"] for doc in results])
        self.assertEqual(3, len({doc["id"] for doc in results}))

    def test_the_best_scoring_document_comes_first(self) -> None:
        """Ordering must follow the cross-encoder, not the input order."""
        self.use(FakeCrossEncoder([-11.2, -11.4, 0.03]))

        results = reranker.rerank("how do I print a file", documents(), top_k=3)

        self.assertEqual("c3", results[0]["id"])
        self.assertEqual("cat prints a file", results[0]["content"])
        scores = [doc["score"] for doc in results]
        self.assertEqual(sorted(scores, reverse=True), scores)

    def test_scores_are_the_real_cross_encoder_scores(self) -> None:
        """The defect reported 0.0 for everything, breaking score thresholds."""
        self.use(FakeCrossEncoder([-11.2, -11.4, 0.03]))

        results = reranker.rerank("how do I print a file", documents(), top_k=3)

        self.assertAlmostEqual(0.03, results[0]["score"])
        self.assertAlmostEqual(0.03, results[0]["rerank_score"])
        self.assertAlmostEqual(0.7, results[0]["original_score"])
        self.assertNotEqual(0.0, results[0]["score"])

    def test_reranking_can_overturn_the_retrieval_order(self) -> None:
        """The document retrieval ranked last is promoted when it scores best."""
        self.use(FakeCrossEncoder([0.1, 0.2, 9.0]))

        results = reranker.rerank("print a file", documents(), top_k=1)

        self.assertEqual(["c3"], [doc["id"] for doc in results])

    # ── Trimming and inputs ───────────────────────────────────────────

    def test_top_k_trims_after_sorting_not_before(self) -> None:
        self.use(FakeCrossEncoder([1.0, 5.0, 3.0]))

        results = reranker.rerank("q", documents(), top_k=2)

        self.assertEqual(["c2", "c3"], [doc["id"] for doc in results])

    def test_top_k_larger_than_the_input_returns_everything(self) -> None:
        self.use(FakeCrossEncoder([1.0, 5.0, 3.0]))
        self.assertEqual(3, len(reranker.rerank("q", documents(), top_k=99)))

    def test_the_model_is_given_the_passage_text(self) -> None:
        fake = FakeCrossEncoder([1.0, 2.0, 3.0])
        self.use(fake)

        reranker.rerank("my query", documents(), top_k=3)

        query, passages = fake.calls[0]
        self.assertEqual("my query", query)
        self.assertEqual(
            ["the ls command lists files", "grep searches text", "cat prints a file"],
            passages,
        )

    def test_the_originals_are_not_mutated(self) -> None:
        self.use(FakeCrossEncoder([1.0, 5.0, 3.0]))
        originals = documents()

        reranker.rerank("q", originals, top_k=3)

        self.assertEqual([0.9, 0.8, 0.7], [doc["score"] for doc in originals])
        self.assertNotIn("rerank_score", originals[0])

    def test_no_documents_means_no_results(self) -> None:
        self.use(FakeCrossEncoder([]))
        self.assertEqual([], reranker.rerank("q", [], top_k=3))

    # ── Failing safe ──────────────────────────────────────────────────

    def test_an_unavailable_reranker_falls_back_to_retrieval_order(self) -> None:
        reranker._reranker = None
        reranker._reranker_available = False

        results = reranker.rerank("q", documents(), top_k=2)

        self.assertEqual(["c1", "c2"], [doc["id"] for doc in results])
        self.assertEqual(0.9, results[0]["score"])

    def test_a_model_error_falls_back_instead_of_raising(self) -> None:
        self.use(ExplodingCrossEncoder())

        results = reranker.rerank("q", documents(), top_k=2)

        self.assertEqual(["c1", "c2"], [doc["id"] for doc in results])

    def test_a_wrong_length_response_is_rejected_not_misapplied(self) -> None:
        """One score for three documents must never be spread across all three."""
        self.use(WrongLengthCrossEncoder())

        results = reranker.rerank("q", documents(), top_k=3)

        self.assertEqual(["c1", "c2", "c3"], [doc["id"] for doc in results])
        self.assertEqual(3, len({doc["id"] for doc in results}))


if __name__ == "__main__":
    unittest.main()
