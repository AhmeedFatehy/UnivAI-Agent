"""A book already taught is adopted, not written again.

The learner-facing rule these tests hold to: identical bytes and an identical
pipeline produce identical teaching, so the second learner gets the first
learner's lectures. Anything that could make the copy wrong — different bytes,
a moved pipeline, a reshaped curriculum, an unfinished donor — falls back to a
real build.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from generation import lecture_gen


PLAN = {"schema_name": "univai.semester.week-plan", "week_count": 2}
FINGERPRINT = "fingerprint-1"


def donor_row(**overrides) -> dict:
    row = {
        "id": 7,
        "student_id": "S-2026-000004",
        "pages": 320,
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


def test_the_earliest_course_wins_when_a_book_has_been_taught_twice(monkeypatch):
    """A regenerated copy must not become the course new learners inherit.

    Once an admin regenerates one learner's course, two identical-by-fingerprint
    courses exist for the same book. The earliest is the one the most learners
    already hold, so a class converges on a single shared course instead of
    splintering with every regeneration.
    """
    original = donor_row(id=1, student_id="S-2026-000005")
    regenerated = donor_row(id=2, student_id="S-2026-000001")
    # Returned in the query's own order: ORDER BY b.id ASC.
    install_db(monkeypatch, donors=[original, regenerated])

    found = lecture_gen.find_reusable_course(
        "S-2026-000003", 99, "sha-abc", FINGERPRINT, PLAN
    )

    assert found is not None
    assert found["id"] == 1, "a third learner must inherit the original course"


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


def test_a_learner_without_a_plan_reuses_the_finished_donors_plan(monkeypatch):
    install_db(monkeypatch, donors=[donor_row()])

    donor = lecture_gen.find_reusable_course(
        "S-2026-000005", 8, "sha-abc", FINGERPRINT, None
    )

    assert donor is not None
    assert donor["semester_plan"] == PLAN


def test_regeneration_refuses_to_hand_back_an_existing_course(monkeypatch):
    """An admin's "Regenerate course" must write the book again.

    Reuse turned that request into a no-op: the book was adopted from an
    identical course, so regenerating found the same donor and copied it back.
    Observed as "4/4 lectures reused from an identical book" in answer to a
    rebuild.
    """
    monkeypatch.setattr(
        lecture_gen, "fetch_one", lambda sql, params=None: {"plan_version": 1}
    )

    assert lecture_gen.course_reuse_allowed(
        "S-2026-000005", quizzes_only=False, no_reuse=False
    ) is True
    assert lecture_gen.course_reuse_allowed(
        "S-2026-000005", quizzes_only=False, no_reuse=True
    ) is False


def test_an_edited_curriculum_is_never_reused_even_without_the_flag(monkeypatch):
    monkeypatch.setattr(
        lecture_gen, "fetch_one", lambda sql, params=None: {"plan_version": 2}
    )

    assert lecture_gen.course_reuse_allowed(
        "S-2026-000005", quizzes_only=False, no_reuse=False
    ) is False


def test_a_quiz_only_run_never_adopts(monkeypatch):
    """It rewrites question banks in place; there is no course to take."""
    monkeypatch.setattr(
        lecture_gen,
        "fetch_one",
        lambda sql, params=None: pytest.fail("quizzes-only must not query the curriculum"),
    )

    assert lecture_gen.course_reuse_allowed(
        "S-2026-000005", quizzes_only=True, no_reuse=False
    ) is False


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


def test_reused_slidev_cache_is_rekeyed_without_mutating_donor(monkeypatch, tmp_path):
    donor_id = "11111111-1111-4111-8111-111111111111"
    adopted_id = "22222222-2222-4222-8222-222222222222"
    monkeypatch.setattr(
        lecture_gen, "_slidev_cache_dir", lambda artifact_id: tmp_path / artifact_id
    )
    donor = tmp_path / donor_id
    (donor / "assets").mkdir(parents=True)
    donor_index = f'<script src="/api/presentation/{donor_id}/assets/app.js"></script>'
    donor_script = f'const base = "/api/presentation/{donor_id}/";'
    (donor / "index.html").write_text(donor_index, encoding="utf-8")
    (donor / "assets" / "app.js").write_text(donor_script, encoding="utf-8")
    (donor / "assets" / "font.woff2").write_bytes(b"\x00\x01font")

    assert lecture_gen._reuse_slidev_cache(donor_id, adopted_id) is True

    adopted = tmp_path / adopted_id
    assert donor_id not in (adopted / "index.html").read_text(encoding="utf-8")
    assert donor_id not in (adopted / "assets" / "app.js").read_text(encoding="utf-8")
    assert adopted_id in (adopted / "index.html").read_text(encoding="utf-8")
    assert adopted_id in (adopted / "assets" / "app.js").read_text(encoding="utf-8")
    assert (adopted / "assets" / "font.woff2").read_bytes() == b"\x00\x01font"
    assert donor_id in (donor / "index.html").read_text(encoding="utf-8")
    assert donor_id in (donor / "assets" / "app.js").read_text(encoding="utf-8")


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
            return {"id": 9, "name": "My Library Curriculum", "plan_version": 1}
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
    # The teaching survives the copy, and the payload's own identity is rewritten
    # to match the row — Live refuses a pack that disagrees with its session.
    stored = json.loads(params[10])
    assert stored["title"] == "Practical: Transactions"
    assert stored["user_id"] == "S-2026-000005"
    assert stored["course_id"] == "book-8"
    assert stored["topic_id"] == "22222222-2222-4222-8222-222222222222"
    assert stored["programme_title"] == "My Library Curriculum"


def test_an_adopted_pack_describes_the_learner_it_was_copied_to(monkeypatch):
    """Live re-reads the pack's own identity and refuses when it disagrees.

    Copying pack_payload verbatim re-keyed the row but left the payload naming
    the donor, so joining the section failed with section_artifact_unavailable
    — the columns said one learner and the JSON still said another.
    """
    donor_payload = {
        "schema_name": "univai.section.pack",
        "user_id": "S-2026-000004",
        "course_id": "book-7",
        "topic_id": "11111111-1111-4111-8111-111111111111",
        "week_number": 1,
        "programme_title": "Donor Curriculum",
        "plan_version": "1",
        "title": "Practical: Transactions",
    }

    rekeyed = lecture_gen.rekey_section_payload(
        donor_payload,
        sid="S-2026-000005",
        book_id=8,
        topic_id="22222222-2222-4222-8222-222222222222",
        programme_title="My Library Curriculum",
        plan_version=3,
    )

    assert rekeyed["user_id"] == "S-2026-000005"
    assert rekeyed["course_id"] == "book-8"
    assert rekeyed["topic_id"] == "22222222-2222-4222-8222-222222222222"
    assert rekeyed["programme_title"] == "My Library Curriculum"
    assert rekeyed["plan_version"] == "3"
    # The teaching itself is untouched, and the donor's copy is not mutated.
    assert rekeyed["title"] == "Practical: Transactions"
    assert rekeyed["week_number"] == 1
    assert donor_payload["user_id"] == "S-2026-000004"


def test_the_stored_hash_follows_the_rekeyed_payload(monkeypatch):
    written: list[tuple[str, tuple]] = []

    def fetch_one(sql, params=None):
        if "FROM programmes" in sql:
            return {"id": 9, "name": "My Library Curriculum", "plan_version": 1}
        return {"artifact_id": "22222222-2222-4222-8222-222222222222"}

    monkeypatch.setattr(lecture_gen, "fetch_one", fetch_one)
    monkeypatch.setattr(
        lecture_gen,
        "fetch_all",
        lambda sql, params=None: [
            {
                "week": 1,
                "prompt_id": "teaching/section_generation",
                "prompt_version": "1.0.0",
                "payload_hash": "0" * 64,
                "pack_payload": {"user_id": "donor", "course_id": "book-7", "title": "P"},
            }
        ],
    )
    monkeypatch.setattr(lecture_gen, "execute", lambda sql, params: written.append((sql, params)))
    monkeypatch.setattr(lecture_gen, "mark_milestone", lambda *a, **k: None)

    lecture_gen.adopt_section_packs("S-2026-000005", 8, 7)

    params = next(params for sql, params in written if "INSERT INTO section_packs" in sql)
    stored_hash, stored_payload = params[9], params[10]
    assert stored_hash != "0" * 64, "the donor's hash cannot describe a re-keyed payload"
    assert (
        hashlib.sha256(stored_payload.encode("utf-8")).hexdigest() == stored_hash
    )
    assert json.loads(stored_payload)["user_id"] == "S-2026-000005"


def test_a_section_without_its_lecture_is_skipped(monkeypatch):
    """The pack points at a lecture artifact; without one it would dangle."""
    monkeypatch.setattr(
        lecture_gen,
        "fetch_one",
        lambda sql, params=None: {"id": 9, "name": "My Library Curriculum", "plan_version": 1}
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
