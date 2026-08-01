from __future__ import annotations

import json
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

    def test_generation_uses_the_source_section_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "short-course.md"
            source.write_text(
                "# Short course\n\n## First chapter\nFirst material.\n\n"
                "## Second chapter\nSecond material.\n",
                encoding="utf-8",
            )
            output = generate_course(source, root / "output")
            validate_course(output)
            self.assertEqual(2, len(list(output.glob("week-*"))))
            self.assertEqual(
                2,
                json.loads((output / "run.json").read_text("utf-8"))["weeks"],
            )


if __name__ == "__main__":
    unittest.main()
