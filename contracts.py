"""Canonical Agent contracts used by standalone and root integration checks."""

from __future__ import annotations

import json
from pathlib import Path

MCP_TOOLS = {
    "retrieve_context": ["query", "user_id"],
    "ingest_file": ["file_path", "user_id"],
    "list_documents": ["user_id"],
    "remove_document": ["user_id", "document_id"],
    "server_info": [],
}
COURSE_SIZES = {
    "XS": {"slides": 3, "narration": "4-6", "lecture_qs": 8, "self_qs": 2},
    "S": {"slides": 5, "narration": "4-6", "lecture_qs": 10, "self_qs": 2},
    "M": {"slides": 8, "narration": "5-7", "lecture_qs": 14, "self_qs": 3},
    "L": {"slides": 12, "narration": "6-8", "lecture_qs": 18, "self_qs": 4},
    "XL": {"slides": 16, "narration": "6-9", "lecture_qs": 22, "self_qs": 5},
}


def validate_script(data: dict) -> None:
    if not isinstance(data.get("lectureId"), str) or not data["lectureId"]:
        raise ValueError("script.lectureId must be a non-empty string")
    if not isinstance(data.get("title"), str) or not data["title"]:
        raise ValueError("script.title must be a non-empty string")
    if not isinstance(data.get("segments"), list) or not data["segments"]:
        raise ValueError("script.segments must be a non-empty list")
    for segment in data["segments"]:
        if not isinstance(segment.get("slide"), int) or segment["slide"] < 1:
            raise ValueError("segment.slide must be a positive integer")
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise ValueError("segment.text must be non-empty")
        citations = segment.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError("segment.citations must be non-empty")
        if not all(isinstance(item.get("page"), int) and item["page"] > 0 for item in citations):
            raise ValueError("citation.page must be a positive integer")


def validate_quiz(data: dict, expected_week: int | None = None) -> None:
    if not isinstance(data.get("week"), int) or data["week"] < 1:
        raise ValueError("quiz.week must be a positive integer")
    if expected_week is not None and data["week"] != expected_week:
        raise ValueError(f"quiz.week must match week-{expected_week}")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("quiz.questions must be non-empty")
    for question in questions:
        if question.get("type") != "mcq":
            raise ValueError("standalone quiz questions must use type=mcq")
        options = question.get("options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("question.options must contain four items")
        if question.get("correct_option") not in "ABCD":
            raise ValueError("correct_option must be A, B, C, or D")
        if question.get("source") not in {"lecture", "self_study"}:
            raise ValueError("source must be lecture or self_study")


def validate_course(root: Path) -> None:
    folders = sorted(
        (folder for folder in root.glob("week-*") if folder.is_dir()),
        key=lambda folder: int(folder.name.removeprefix("week-")),
    )
    if not folders:
        raise ValueError("course must contain at least one week")
    weeks = [int(folder.name.removeprefix("week-")) for folder in folders]
    if weeks != list(range(1, len(weeks) + 1)):
        raise ValueError("course weeks must be contiguous and start at 1")
    for week, folder in zip(weeks, folders):
        validate_script(json.loads((folder / "script.json").read_text(encoding="utf-8")))
        validate_quiz(
            json.loads((folder / "quiz.json").read_text(encoding="utf-8")), week
        )
        if not (folder / "slides.md").is_file():
            raise ValueError(f"week-{week}/slides.md is missing")
