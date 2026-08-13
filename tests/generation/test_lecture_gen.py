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


def test_semester_plan_is_saved_for_other_endpoints(monkeypatch):
    plan, _ = lecture_gen.build_semester_plan(textbook_pages(), "Test book")
    calls = []
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: calls.append((sql, params)))

    lecture_gen.write_semester_plan("student-1", plan, 7)

    saved = json.loads(calls[0][1][0])
    assert "UPDATE books SET semester_plan" in calls[0][0]
    assert calls[0][1][1:] == (7, "student-1")
    assert saved["schema_name"] == "univai.semester.week-plan"
    assert saved["week_count"] == 3
    assert saved["semester_count"] == 1
    assert saved["semesters"][0]["quiz_count"] == 3


def test_regeneration_removes_only_obsolete_tail_weeks(monkeypatch):
    calls = []
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: calls.append((sql, params)))

    lecture_gen.remove_obsolete_weeks("student-1", 1, 7)

    assert "DELETE FROM lecture_artifacts" in calls[0][0]
    assert calls[0][1] == (7, "student-1", 1)


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
                "layout": "concept",
                "bullets": ["One", "Two"],
                "callout": "",
                "emphasis": [],
                "visual": {},
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


def test_generation_manifest_resumes_only_the_same_source(monkeypatch):
    state = {7: None, 8: None}
    monkeypatch.setattr(
        lecture_gen,
        "fetch_one",
        lambda _sql, params: {"generation_manifest": state[params[0]]},
    )
    def save(_sql, params):
        state[params[1]] = json.loads(params[0])
    monkeypatch.setattr(lecture_gen, "execute", save)

    assert lecture_gen.prepare_generation_manifest("student-1", 7, "a" * 64, 3) is False
    assert lecture_gen.prepare_generation_manifest("student-1", 7, "a" * 64, 3) is True
    assert lecture_gen.prepare_generation_manifest("student-1", 8, "b" * 64, 3) is False


def test_lecture_checkpoint_requires_database_payloads_and_slidev_cache(monkeypatch, tmp_path):
    cache = tmp_path / "opaque-id"
    cache.mkdir()
    (cache / "index.html").write_text("Slidev", encoding="utf-8")
    monkeypatch.setattr(lecture_gen, "_slidev_cache_dir", lambda _artifact_id: cache)
    monkeypatch.setattr(lecture_gen, "lecture_artifact", lambda *_args: {
        "artifact_id": "opaque-id",
        "script_payload": {"segments": [{"text": "Saved narration"}]},
        "lecture_payload": {"slides": [{"heading": "Saved"}]},
        "slides_payload": {"slides": [{"heading": "Saved"}]},
        "book_id": 8,
        "quiz_payload": {
            "schema_version": "learner-assessment-bank-v1",
            "owner_student_id": "student-1",
            "owner_book_id": 8,
            "generation_id": "learner-generation",
            "questions": [{"stem": "Saved question"}],
        },
    })

    assert lecture_gen.valid_lecture_checkpoint("student-1", 1) is True
    assert lecture_gen.valid_quiz_checkpoint("student-1", 1, 8) is True
    assert lecture_gen.valid_slides_checkpoint("student-1", 1) is True


def test_slidev_markdown_is_derived_from_the_database_deck():
    rendered = lecture_gen._slidev_markdown({
        "week": 2,
        "title": "Reliable Systems",
        "slides": [{
            "slide": 2,
            "heading": "Reliability",
            "bullets": ["Tolerate faults", "Keep serving users"],
            "page": 17,
        }],
    })

    assert "theme: default" in rendered
    assert '<h1>Reliable Systems</h1>' in rendered
    assert '<h1 class="ua-heading">Reliability</h1>' in rendered
    assert "Tolerate faults" in rendered
    assert "SOURCE&nbsp;&middot;&nbsp;P.17" in rendered


def test_lecture_write_lets_postgres_generate_the_public_artifact_id(monkeypatch):
    calls = []
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: calls.append((sql, params)))
    lecture = {
        "title": "Opaque identifiers",
        "intro": "The database owns public identifiers.",
        "durationMinutes": 45,
        "slides": [
            {
                "heading": "Database identity",
                "bullets": ["UUID", "Server generated"],
                "narration": "PostgreSQL generates an opaque identifier for this lecture artifact.",
                "page": 4,
            }
        ],
    }

    lecture_gen.write_lecture("student-1", 1, lecture, 7)

    sql, params = calls[0]
    assert "gen_random_uuid()" in sql
    assert "INSERT INTO lecture_artifacts" in sql
    assert params[:4] == (7, "student-1", 1, "Opaque identifiers")
    assert all("week-1" not in str(value) for value in params)
    saved_lecture = json.loads(params[4])
    saved_deck = json.loads(params[6])
    assert saved_lecture["slides"][0]["layout"] == "concept"
    assert saved_deck["slides"][0]["layout"] == "concept"
    assert saved_deck["slides"][0]["visual"] == {}
