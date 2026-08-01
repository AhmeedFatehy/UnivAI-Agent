from __future__ import annotations

import json

from generation import lecture_gen


def textbook_pages() -> list[tuple[int, str]]:
    pages = []
    for page in range(1, 61):
        heading = ""
        if page in {1, 21, 41}:
            chapter = ((page - 1) // 20) + 1
            heading = f"Chapter {chapter} Topic {chapter}\n"
        pages.append((page, heading + "Readable textbook material " * 4))
    return pages


def test_upload_generation_uses_chapter_count_instead_of_four():
    plan, weeks = lecture_gen.build_semester_plan(textbook_pages(), "Test book")

    assert plan.week_count == 3
    assert len(weeks) == 3
    assert [[part.chapter_id for part in week.chapters] for week, _ in weeks] == [
        ["C001"],
        ["C002"],
        ["C003"],
    ]


def test_semester_plan_is_saved_for_other_endpoints(tmp_path, monkeypatch):
    plan, _ = lecture_gen.build_semester_plan(textbook_pages(), "Test book")
    monkeypatch.setattr(lecture_gen, "LECTURES_DIR", tmp_path)

    lecture_gen.write_semester_plan("student-1", plan)

    saved = json.loads(
        (tmp_path / "student-1" / "semester-plan.json").read_text(encoding="utf-8")
    )
    assert saved["schema_name"] == "univai.semester.week-plan"
    assert saved["week_count"] == 3
