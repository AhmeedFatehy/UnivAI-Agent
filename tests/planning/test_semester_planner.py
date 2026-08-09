from __future__ import annotations

import pytest
from pydantic import ValidationError

from planning.semester_planner import (
    MAX_CHAPTERS_PER_WEEK,
    MAX_SEMESTER_WEEKS,
    Chapter,
    ChapterInventory,
    ChapterPart,
    ChapterSize,
    SemesterWeek,
    SemesterWeekPlan,
    discover_chapters,
    plan_semester,
)


def chapter(
    number: int,
    start: int,
    end: int,
    size: ChapterSize = ChapterSize.STANDARD,
) -> Chapter:
    return Chapter(
        chapter_id=f"C{number:03d}",
        title=f"Chapter {number}",
        start_page=start,
        end_page=end,
        source_ids=[f"P{start}"],
        size=size,
    )


def inventory(chapters: list[Chapter]) -> ChapterInventory:
    return ChapterInventory(book_title="Test book", chapters=chapters)


def test_normal_chapters_become_one_week_each():
    plan = plan_semester(
        inventory([chapter(1, 1, 20), chapter(2, 21, 40), chapter(3, 41, 60)])
    )
    assert plan.week_count == 3
    assert [[part.chapter_id for part in week.chapters] for week in plan.weeks] == [
        ["C001"],
        ["C002"],
        ["C003"],
    ]


def test_small_book_keeps_one_chapter_per_week_regardless_of_page_size():
    plan = plan_semester(
        inventory(
            [
                chapter(1, 1, 4, ChapterSize.SMALL),
                chapter(2, 5, 9, ChapterSize.SMALL),
                chapter(3, 10, 29),
            ]
        )
    )
    assert [[part.chapter_id for part in week.chapters] for week in plan.weeks] == [
        ["C001"],
        ["C002"],
        ["C003"],
    ]


def test_tiny_chapters_are_still_distinct_theory_lectures_in_a_small_book():
    plan = plan_semester(
        inventory(
            [chapter(index, index * 2 - 1, index * 2, ChapterSize.TINY) for index in range(1, 5)]
        )
    )
    assert [len(week.chapters) for week in plan.weeks] == [1, 1, 1, 1]


def test_one_large_chapter_remains_one_lecture():
    source = inventory([chapter(1, 1, 100, ChapterSize.LARGE)])
    plan = plan_semester(source)
    assert plan.week_count == 1
    only = plan.weeks[0].chapters[0]
    assert (only.start_page, only.end_page) == (1, 100)
    assert plan.validate_against(source) is plan


def test_a_week_rejects_more_chapters_than_a_lecture_can_carry():
    # A week may now carry up to MAX_CHAPTERS_PER_WEEK: compressing chapters
    # into a week is how a large book fits inside the capped course length, so
    # the ceiling has to sit above what compression produces, not below it.
    parts = [
        ChapterPart(
            chapter_id=f"C{index:03d}",
            title=f"Chapter {index}",
            start_page=index,
            end_page=index,
            source_ids=[f"P{index}"],
        )
        for index in range(1, MAX_CHAPTERS_PER_WEEK + 2)
    ]
    with pytest.raises(ValidationError, match=f"at most {MAX_CHAPTERS_PER_WEEK} items"):
        SemesterWeek(
            week=1,
            chapters=parts,
            rationale="too many",
            learning_objectives=["learn"],
            source_ids=["P1"],
        )


def test_a_week_accepts_compressed_chapters():
    parts = [
        ChapterPart(
            chapter_id=f"C{index:03d}",
            title=f"Chapter {index}",
            start_page=index,
            end_page=index,
            source_ids=[f"P{index}"],
        )
        for index in range(1, MAX_CHAPTERS_PER_WEEK + 1)
    ]
    week = SemesterWeek(
        week=1,
        chapters=parts,
        rationale="compressed",
        learning_objectives=["learn"],
        source_ids=["P1"],
    )
    assert len(week.chapters) == MAX_CHAPTERS_PER_WEEK


def test_a_large_course_splits_across_semesters_instead_of_breaking_the_ceiling():
    source = inventory([chapter(index, index, index) for index in range(1, 41)])
    plan = plan_semester(source)
    assert plan.semester_count == 2
    assert [semester.week_count for semester in plan.semesters] == [12, 12]
    assert plan.week_count == 24
    assert all(semester.week_count <= MAX_SEMESTER_WEEKS for semester in plan.semesters)
    covered = sum(len(week.chapters) for week in plan.weeks)
    assert covered == 40, "compression must not drop chapters"
    assert plan.validate_against(source) is plan


def test_a_small_book_keeps_its_natural_length():
    source = inventory([chapter(index, index, index) for index in range(1, 6)])
    plan = plan_semester(source)
    assert plan.week_count == 5
    assert all(len(week.chapters) == 1 for week in plan.weeks)


def test_validation_rejects_missing_chapters():
    source = inventory([chapter(1, 1, 10), chapter(2, 11, 20)])
    plan = SemesterWeekPlan(
        book_title="Test book",
        week_count=1,
        weeks=[
            SemesterWeek(
                week=1,
                chapters=[
                    ChapterPart(
                        chapter_id="C001",
                        title="Chapter 1",
                        start_page=1,
                        end_page=10,
                        source_ids=["P1"],
                    )
                ],
                rationale="missing one",
                learning_objectives=["learn"],
                source_ids=["P1"],
            )
        ],
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        plan.validate_against(source)


def test_chapter_discovery_uses_page_top_headings():
    pages = [
        (1, "Front matter\nCopyright"),
        (3, "Chapter 1 Introduction\nThe first chapter."),
        (15, "Chapter 2 Search\nThe second chapter."),
    ]
    found = discover_chapters(pages, "AI")
    assert [item.title for item in found.chapters] == ["Introduction", "Search"]
    assert found.chapters[0].start_page == 1
    assert found.chapters[0].end_page == 14


def test_book_without_a_toc_has_an_explicit_low_confidence_fallback():
    pages = [(page, "ordinary paragraph text without headings") for page in range(1, 101)]
    found = discover_chapters(pages, "Mystery book")
    assert found.confidence < 0.5
    assert found.warnings
    assert plan_semester(found).week_count == 1


@pytest.mark.parametrize(
    ("chapter_count", "semester_weeks", "quiz_counts", "midterm_counts"),
    [
        (8, [8], [8], [1]),
        (12, [8], [8], [1]),
        (20, [12], [12], [1]),
        (30, [12, 12], [12, 12], [1, 1]),
    ],
)
def test_canonical_course_shapes(
    chapter_count: int,
    semester_weeks: list[int],
    quiz_counts: list[int],
    midterm_counts: list[int],
):
    source = inventory([chapter(index, index, index) for index in range(1, chapter_count + 1)])
    plan = plan_semester(source)

    assert plan.chapter_count == chapter_count
    assert [semester.week_count for semester in plan.semesters] == semester_weeks
    assert [semester.quiz_count for semester in plan.semesters] == quiz_counts
    assert [len(semester.midterms) for semester in plan.semesters] == midterm_counts
    assert all(
        semester.midterms[0].after_week == (semester.week_count + 1) // 2
        for semester in plan.semesters
    )
    assert all(semester.final_after_week == semester.week_count for semester in plan.semesters)
    assert sum(len(week.chapters) for week in plan.weeks) == chapter_count


def test_slide_deck_agenda_stays_inside_one_lecture_and_body_bullets_do_not_split_it():
    pages = [
        (1, "Course Learning Outcomes\n1. Understand the basic principles of the field of"),
        (
            2,
            "Agenda\n1. General introduction.\n2. When Simulation is the Appropriate Tool.\n"
            "3. When Simulation is Not Appropriate.\n"
            "4. Advantages and Disadvantages of Simulation.\n5. Areas of Application.",
        ),
        (3, "General Intoduction\nA simulation is the imitation of a real system."),
        (4, "When Simulation is the Appropriate Tool?"),
        (5, "When it Appropriate?\nTraining\nAnimation"),
        (6, "When Simulation is NOT the Appropriate Tool?"),
        (7, "Advantages of Simulation\nNew systems can be tested."),
        (8, "Disadvantages of Simulation\nModel building needs training."),
        (9, "Areas of Application\nHealthcare\nNetworks"),
    ]

    found = discover_chapters(pages, "Modeling and Simulation")

    assert len(found.chapters) == 1
    assert found.chapters[0].title == "Modeling and Simulation"
    assert (found.chapters[0].start_page, found.chapters[0].end_page) == (1, 9)
    assert found.confidence < 0.5


def test_a_qualified_bulleted_contents_page_defines_chapters():
    # A lecture deck's real shape: the contents page is titled for the session
    # ("Day 3 Contents") and its entries are bulleted, not numbered. Rejecting
    # either used to collapse the whole deck into one chapter, which turned a
    # multi-week course into a single hours-long lecture.
    pages = [
        (1, "DAY 3"),
        (2, "Day 3 Contents\n• Transactions\n• Built in functions\n• Triggers"),
        (3, "Transactions\nA transaction is a set of SQL statements."),
        (15, "Built in functions\nMySQL ships with many functions."),
        (30, "Triggers\nA trigger fires on a table event."),
    ]

    found = discover_chapters(pages, "MySQL")

    assert [chapter.title for chapter in found.chapters] == [
        "Transactions",
        "Built in functions",
        "Triggers",
    ]
    assert [chapter.start_page for chapter in found.chapters] == [1, 15, 30]
    assert found.confidence > 0.5
    assert plan_semester(found).week_count == 3


def test_explicit_contents_entries_can_define_unlabelled_chapters():
    pages = [
        (1, "Table of Contents\n1. Foundations\n2. Advanced Systems"),
        (2, "Foundations\nCore concepts"),
        (10, "Advanced Systems\nMore concepts"),
    ]

    found = discover_chapters(pages, "Systems")

    assert [chapter.title for chapter in found.chapters] == ["Foundations", "Advanced Systems"]
    assert [chapter.start_page for chapter in found.chapters] == [1, 10]
