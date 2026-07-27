from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["UNIVAI_MODE"] = "standalone"

from contracts import validate_course
from runtime import REPOSITORY_ROOT
from standalone_generation import generate_course
from standalone_store import ingest, list_documents, remove, reset, retrieve


class StandaloneAgentTests(unittest.TestCase):
    fixture = REPOSITORY_ROOT / "fixtures" / "sample_course.md"

    def setUp(self) -> None:
        reset()

    def tearDown(self) -> None:
        reset()

    def test_rag_order_and_tenant_isolation(self) -> None:
        result = ingest(str(self.fixture), "student-a")
        self.assertIn("Tenant Isolation", retrieve("tenant isolation learner", "student-a"))
        self.assertEqual([], list_documents("student-b"))
        self.assertEqual("No relevant documents found.", retrieve("tenant", "student-b"))
        self.assertGreater(remove("student-a", result["document_id"]), 0)

    def test_generation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate_course(self.fixture, Path(directory))
            validate_course(output)
            self.assertEqual(4, len(list(output.glob("week-*"))))


if __name__ == "__main__":
    unittest.main()
