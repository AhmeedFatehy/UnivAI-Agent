from __future__ import annotations

import pytest
from pydantic import ValidationError

from planning.semester_planner import (
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


def test_small_adjacent_chapters_can_share_a_week():
    plan = plan_semester(
        inventory(
            [
                chapter(1, 1, 4, ChapterSize.SMALL),
                chapter(2, 5, 9, ChapterSize.SMALL),
                chapter(3, 10, 29),
            ]
        )
    )
    assert [part.chapter_id for part in plan.weeks[0].chapters] == ["C001", "C002"]


def test_three_tiny_chapters_fit_but_a_fourth_starts_another_week():
    plan = plan_semester(
        inventory(
            [chapter(index, index * 2 - 1, index * 2, ChapterSize.TINY) for index in range(1, 5)]
        )
    )
    assert [len(week.chapters) for week in plan.weeks] == [3, 1]


def test_a_large_chapter_is_split_across_two_complete_ranges():
    source = inventory([chapter(1, 1, 100, ChapterSize.LARGE)])
    plan = plan_semester(source)
    assert plan.week_count == 2
    first, second = [week.chapters[0] for week in plan.weeks]
    assert (first.part_index, second.part_index) == (1, 2)
    assert first.end_page + 1 == second.start_page
    assert first.start_page == 1
    assert second.end_page == 100
    assert plan.validate_against(source) is plan


def test_a_week_rejects_more_than_three_chapters():
    parts = [
        ChapterPart(
            chapter_id=f"C{index:03d}",
            title=f"Chapter {index}",
            start_page=index,
            end_page=index,
            source_ids=[f"P{index}"],
        )
        for index in range(1, 5)
    ]
    with pytest.raises(ValidationError, match="at most 3 items"):
        SemesterWeek(
            week=1,
            chapters=parts,
            rationale="too many",
            learning_objectives=["learn"],
            source_ids=["P1"],
        )


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
    assert plan_semester(found).week_count == 2
