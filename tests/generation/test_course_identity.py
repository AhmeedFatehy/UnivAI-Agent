from __future__ import annotations

from generation.course_identity import CourseComponents


def components(**overrides) -> CourseComponents:
    base = dict(
        target_semester_weeks=8,
        max_semester_weeks=12,
        normal_semester_chapters=12,
        max_chapters_per_semester=20,
        max_chapters_per_week=2,
        lecture_minutes_min=45,
        lecture_minutes_max=120,
        minutes_per_page=2.0,
        spoken_words_per_minute=150,
        narration_sentences_per_slide=8,
        slides_per_batch=6,
        min_lecture_questions=15,
        prompt_versions="lecture=1.1.0,quiz=1.1.0",
        generation_model="gemini:gemini-3.5-flash-lite",
    )
    base.update(overrides)
    return CourseComponents(**base)


def test_identical_components_are_the_same_course():
    assert components().fingerprint() == components().fingerprint()


def test_a_planner_change_retires_every_course_built_before_it():
    # The real case this guards: lectures/S-2026-000001 was planned as ONE week
    # by a chapter detector that could not read a bulleted contents page, while
    # the same bytes now plan as four. Reusing on content hash alone would have
    # served that stale course to every later learner.
    before = components(planner_version="1.0.0")
    after = components(planner_version="1.1.0")
    assert before.fingerprint() != after.fingerprint()


def test_a_course_written_by_another_model_is_another_course():
    weak = components(generation_model="ollama:qwen2.5:0.5b")
    strong = components(generation_model="gemini:gemini-3.5-flash-lite")
    assert weak.fingerprint() != strong.fingerprint()


def test_a_reworded_prompt_retires_the_course_it_wrote():
    assert (
        components(prompt_versions="lecture=1.0.0,quiz=1.1.0").fingerprint()
        != components(prompt_versions="lecture=1.1.0,quiz=1.1.0").fingerprint()
    )


def test_every_shape_constant_participates_in_the_key():
    baseline = components().fingerprint()
    for field, changed in (
        ("target_semester_weeks", 6),
        ("max_semester_weeks", 16),
        ("normal_semester_chapters", 10),
        ("max_chapters_per_semester", 24),
        ("max_chapters_per_week", 3),
        ("lecture_minutes_min", 30),
        ("lecture_minutes_max", 90),
        ("minutes_per_page", 1.5),
        ("spoken_words_per_minute", 130),
        ("narration_sentences_per_slide", 6),
        ("slides_per_batch", 8),
        ("min_lecture_questions", 20),
    ):
        assert components(**{field: changed}).fingerprint() != baseline, field
