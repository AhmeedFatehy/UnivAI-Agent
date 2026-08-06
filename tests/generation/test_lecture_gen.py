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
    assert saved["semester_count"] == 1
    assert saved["semesters"][0]["quiz_count"] == 3


def test_minimum_lecture_batches_all_slides_without_an_impossible_tail(monkeypatch):
    pages = [(page, f"Page {page}") for page in range(1, 6)]
    calls: list[tuple[list[int], int, bool]] = []

    def fake_batch(_week, _total, _chapters, batch_pages, *, slides, first, **_kwargs):
        page_numbers = [number for number, _ in batch_pages]
        calls.append((page_numbers, slides, first))
        return {
            "title": "Minimum lecture",
            "intro": "Welcome to the minimum lecture." if first else "",
            "slides": [
                {
                    "heading": f"Slide {index}",
                    "bullets": ["One", "Two"],
                    "narration": "A sufficiently long narration for this generated slide.",
                    "page": page_numbers[0],
                }
                for index in range(slides)
            ],
        }

    monkeypatch.setattr(lecture_gen, "_generate_batch", fake_batch)

    lecture = lecture_gen.generate_week(1, 1, "Chapter 1", pages)

    assert [slides for _, slides, _ in calls] == [10, 10, 9, 9, 9]
    assert [page for batch_pages, _, _ in calls for page in batch_pages] == [1, 2, 3, 4, 5]
    assert [first for _, _, first in calls] == [True, False, False, False, False]
    assert len(lecture["slides"]) == 47
    assert lecture["durationMinutes"] == 45


def test_short_final_batch_accepts_its_requested_size():
    data = {
        "title": "Tail",
        "intro": "",
        "slides": [
            {
                "heading": "Last point",
                "bullets": ["One", "Two"],
                "narration": (
                    "This narration deliberately contains enough spoken words to pass the structural "
                    "validation check for a generated slide."
                ),
                "page": 5,
            }
        ],
    }

    assert lecture_gen.check_lecture(data, expected_slides=1, require_intro=False) is None


def test_quiz_size_is_derived_per_week_including_quiz_only_regeneration(monkeypatch):
    requested: list[int] = []

    def fake_questions(_prompt, count, _source, minimum=None):
        requested.append(count)
        return []

    monkeypatch.setattr(lecture_gen, "ask_questions", fake_questions)

    lecture_gen.generate_quiz("Short", [{"text": "Spoken"}], [(1, "Page")])
    lecture_gen.generate_quiz(
        "Long",
        [{"text": "Spoken"}],
        [(page, "Page") for page in range(1, 61)],
    )

    short = lecture_gen.lecture_shape(1)
    long = lecture_gen.lecture_shape(60)
    assert requested == [
        short["lecture_qs"],
        short["self_qs"],
        long["lecture_qs"],
        long["self_qs"],
    ]
