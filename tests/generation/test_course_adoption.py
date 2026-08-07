"""A book already taught is adopted, not written again.

The learner-facing rule these tests hold to: identical bytes and an identical
pipeline produce identical teaching, so the second learner gets the first
learner's lectures. Anything that could make the copy wrong — different bytes,
a moved pipeline, a reshaped curriculum, an unfinished donor — falls back to a
real build.
"""

from __future__ import annotations

import json

import pytest

from generation import lecture_gen


PLAN = {"schema_name": "univai.semester.week-plan", "week_count": 2}
FINGERPRINT = "fingerprint-1"


def donor_row(**overrides) -> dict:
    row = {
        "id": 7,
        "student_id": "S-2026-000004",
        "total_weeks": 2,
        "generation_manifest": {"course_fingerprint": FINGERPRINT},
        "semester_plan": PLAN,
    }
    row.update(overrides)
    return row


def install_db(monkeypatch, *, donors, complete_weeks=2):
    """Answer only the queries find_reusable_course actually issues."""

    def fetch_all(sql, params=None):
        if "FROM books b" in sql:
            return donors
        return []

    def fetch_one(sql, params=None):
        if "count(*) AS ready FROM lecture_artifacts" in sql:
            return {"ready": complete_weeks}
        return None

    monkeypatch.setattr(lecture_gen, "fetch_all", fetch_all)
    monkeypatch.setattr(lecture_gen, "fetch_one", fetch_one)


def test_an_identical_book_is_adopted_from_the_learner_who_already_has_it(monkeypatch):
    install_db(monkeypatch, donors=[donor_row()])

    found = lecture_gen.find_reusable_course(
        "S-2026-000005", 8, "sha-abc", FINGERPRINT, PLAN
    )

    assert found is not None
    assert found["id"] == 7


def test_a_finished_course_still_counted_partial_is_adopted(monkeypatch):
    """'partial' tracks narration bookkeeping, not whether the weeks were written.

    Courses built before narration became on-demand sit at 'partial' with every
    week complete; refusing them would regenerate a course that already exists.
    """
    install_db(monkeypatch, donors=[donor_row()], complete_weeks=2)

    found = lecture_gen.find_reusable_course(
        "S-2026-000005", 8, "sha-abc", FINGERPRINT, PLAN
    )

    assert found is not None and found["id"] == 7


def test_a_course_of_unknown_pipeline_is_not_adopted(monkeypatch):
    """A manifest with no fingerprint cannot be shown to match; build instead."""
    install_db(monkeypatch, donors=[donor_row(generation_manifest={})])

    assert (
        lecture_gen.find_reusable_course("S-2026-000005", 8, "sha-abc", FINGERPRINT, PLAN)
        is None
    )


def test_a_moved_pipeline_is_not_adopted(monkeypatch):
    """New prompts or a new lecture shape mean the old course is not this course."""
    install_db(
        monkeypatch,
        donors=[donor_row(generation_manifest={"course_fingerprint": "fingerprint-0"})],
    )

    assert (
        lecture_gen.find_reusable_course("S-2026-000005", 8, "sha-abc", FINGERPRINT, PLAN)
        is None
    )


def test_a_different_semester_plan_is_not_adopted(monkeypatch):
    """Weeks that do not line up would not match the approved curriculum."""
    install_db(monkeypatch, donors=[donor_row(semester_plan={"week_count": 5})])

    assert (
        lecture_gen.find_reusable_course("S-2026-000005", 8, "sha-abc", FINGERPRINT, PLAN)
        is None
    )


def test_an_unfinished_donor_is_not_adopted(monkeypatch):
    """A course missing a week would hand the new learner a hole."""
    install_db(monkeypatch, donors=[donor_row()], complete_weeks=1)

    assert (
        lecture_gen.find_reusable_course("S-2026-000005", 8, "sha-abc", FINGERPRINT, PLAN)
        is None
    )


def test_a_learner_without_a_plan_yet_is_not_adopted(monkeypatch):
    install_db(monkeypatch, donors=[donor_row()])

    assert (
        lecture_gen.find_reusable_course("S-2026-000005", 8, "sha-abc", FINGERPRINT, None)
        is None
    )


def test_an_edited_curriculum_earns_a_real_build(monkeypatch):
    monkeypatch.setattr(
        lecture_gen, "fetch_one", lambda sql, params=None: {"plan_version": 2}
    )

    assert lecture_gen.learner_has_edited_curriculum("S-2026-000005") is True


def test_an_untouched_curriculum_may_adopt(monkeypatch):
    monkeypatch.setattr(
        lecture_gen, "fetch_one", lambda sql, params=None: {"plan_version": 1}
    )

    assert lecture_gen.learner_has_edited_curriculum("S-2026-000005") is False


def test_adoption_copies_the_teaching_and_rekeys_the_identity(monkeypatch):
    """The lecture arrives byte-identical under the adopting learner's own id."""
    donor_week = {
        "artifact_id": "11111111-1111-4111-8111-111111111111",
        "week": 1,
        "title": "Transactions",
        "lecture_payload": {"slides": [{"heading": "Atomicity"}]},
        "script_payload": {
            "title": "Transactions",
            "segments": [{"slide": 1, "text": "A transaction is atomic."}],
            "lectureId": "11111111-1111-4111-8111-111111111111",
        },
        "slides_payload": {"week": 1, "title": "Transactions", "slides": []},
        "quiz_payload": {"questions": [{"prompt": "What is atomicity?"}]},
    }
    writes: list[tuple[str, tuple]] = []
    milestones: list[tuple] = []

    monkeypatch.setattr(
        lecture_gen,
        "fetch_all",
        lambda sql, params=None: [donor_week] if "FROM lecture_artifacts" in sql else [],
    )
    monkeypatch.setattr(
        lecture_gen,
        "fetch_one",
        lambda sql, params=None: {"artifact_id": "22222222-2222-4222-8222-222222222222"},
    )
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: writes.append((sql, params)))
    monkeypatch.setattr(lecture_gen, "register_week_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(lecture_gen, "adopt_section_packs", lambda *a, **k: None)
    monkeypatch.setattr(
        lecture_gen,
        "mark_milestone",
        lambda book, sid, week, stage, status, **kwargs: milestones.append((week, stage, status)),
    )
    # The donor has a built deck, so no Slidev run is needed.
    monkeypatch.setattr(lecture_gen, "_reuse_slidev_cache", lambda donor, target: True)

    adopted = lecture_gen.adopt_course("S-2026-000005", 8, donor_row())

    assert adopted == 1
    insert = next(sql for sql, _ in writes if "INSERT INTO lecture_artifacts" in sql)
    params = next(params for sql, params in writes if "INSERT INTO lecture_artifacts" in sql)
    # The row belongs to the adopting learner's book...
    assert params[0] == 8 and params[1] == "S-2026-000005"
    # ...carries the donor's teaching unchanged...
    assert json.loads(params[4]) == donor_week["lecture_payload"]
    assert json.loads(params[7]) == donor_week["quiz_payload"]
    # ...and never inherits the donor's lecture identity.
    assert "- 'lectureId'" in insert
    assert "gen_random_uuid()" in insert
    assert {stage for _week, stage, _status in milestones} == {
        "lecture",
        "quiz",
        "slides",
        "audio",
    }
    assert all(status == "ready" for _week, _stage, status in milestones)


def test_adoption_never_copies_attendance_or_scores(monkeypatch):
    """Content is shared; the learner's record is theirs alone."""
    monkeypatch.setattr(lecture_gen, "fetch_all", lambda sql, params=None: [])
    monkeypatch.setattr(lecture_gen, "fetch_one", lambda sql, params=None: None)
    written: list[str] = []
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: written.append(sql))
    monkeypatch.setattr(lecture_gen, "adopt_section_packs", lambda *a, **k: None)

    lecture_gen.adopt_course("S-2026-000005", 8, donor_row())

    touched = " ".join(written).lower()
    for private in ("attendance", "exam_attempts", "scores", "grades"):
        assert private not in touched


def test_sections_are_rekeyed_onto_the_adopting_learners_approved_plan(monkeypatch):
    pack = {
        "week": 1,
        "prompt_id": "teaching/section_generation",
        "prompt_version": "1.0.0",
        "payload_hash": "a" * 64,
        "pack_payload": {"title": "Practical: Transactions"},
    }
    writes: list[tuple[str, tuple]] = []

    def fetch_one(sql, params=None):
        if "FROM programmes" in sql:
            return {"id": 9, "plan_version": 1}
        return {"artifact_id": "22222222-2222-4222-8222-222222222222"}

    monkeypatch.setattr(lecture_gen, "fetch_one", fetch_one)
    monkeypatch.setattr(
        lecture_gen,
        "fetch_all",
        lambda sql, params=None: [pack] if "FROM section_packs" in sql else [],
    )
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: writes.append((sql, params)))
    monkeypatch.setattr(lecture_gen, "mark_milestone", lambda *a, **k: None)

    lecture_gen.adopt_section_packs("S-2026-000005", 8, 7)

    params = next(params for sql, params in writes if "INSERT INTO section_packs" in sql)
    # Tenant, programme and course all belong to the adopting learner, or the
    # app's section query would never match the row.
    assert params[0] == "S-2026-000005"
    assert params[1] == "9"
    assert params[2] == "book-8"
    assert params[5] == "9"
    assert json.loads(params[10]) == pack["pack_payload"]


def test_a_section_without_its_lecture_is_skipped(monkeypatch):
    """The pack points at a lecture artifact; without one it would dangle."""
    monkeypatch.setattr(
        lecture_gen,
        "fetch_one",
        lambda sql, params=None: {"id": 9, "plan_version": 1}
        if "FROM programmes" in sql
        else None,
    )
    monkeypatch.setattr(
        lecture_gen,
        "fetch_all",
        lambda sql, params=None: [{"week": 1, "prompt_id": "p", "prompt_version": "1",
                                   "payload_hash": "a" * 64, "pack_payload": {}}],
    )
    writes: list[str] = []
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: writes.append(sql))
    monkeypatch.setattr(lecture_gen, "mark_milestone", lambda *a, **k: None)

    lecture_gen.adopt_section_packs("S-2026-000005", 8, 7)

    assert not any("INSERT INTO section_packs" in sql for sql in writes)


@pytest.mark.parametrize("missing", ["script", "python"])
def test_narration_warm_up_is_best_effort(monkeypatch, tmp_path, missing):
    """A course is finished and usable before the voice cache is warm."""
    monkeypatch.setattr(lecture_gen, "ROOT", tmp_path)
    if missing != "script":
        script = tmp_path / "UnivAI-live"
        script.mkdir(parents=True)
        (script / "prerender_audio.py").write_text("", encoding="utf-8")
    if missing != "python":
        venv = tmp_path / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("", encoding="utf-8")

    assert lecture_gen.warm_narration_cache(8) is False
