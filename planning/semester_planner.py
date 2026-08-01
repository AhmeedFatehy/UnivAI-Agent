"""Chapter-aware planning for one book's weekly semester layout.

The model may propose structure, but this module owns the rules: coverage,
ordering, adjacency, at most three chapters per week, and at most two parts for
a split chapter.
"""

from __future__ import annotations

import re
import statistics
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

SEMESTER_PLAN_SCHEMA = "univai.semester.week-plan"
SEMESTER_PLAN_VERSION = "1.0.0"


class ChapterSize(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    STANDARD = "standard"
    LARGE = "large"


class Chapter(BaseModel):
    chapter_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    source_ids: list[str] = Field(min_length=1)
    size: ChapterSize = ChapterSize.STANDARD

    @model_validator(mode="after")
    def _page_range_is_ordered(self) -> "Chapter":
        if self.end_page < self.start_page:
            raise ValueError("chapter end_page cannot precede start_page")
        return self

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


class ChapterInventory(BaseModel):
    book_title: str = Field(min_length=1)
    chapters: list[Chapter] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _chapters_are_unique_and_ordered(self) -> "ChapterInventory":
        ids = [chapter.chapter_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("chapter IDs must be unique")
        starts = [chapter.start_page for chapter in self.chapters]
        if starts != sorted(starts):
            raise ValueError("chapters must follow book order")
        for previous, current in zip(self.chapters, self.chapters[1:]):
            if previous.end_page >= current.start_page:
                raise ValueError("chapter page ranges cannot overlap")
        return self


class ChapterPart(BaseModel):
    chapter_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    part_index: int = Field(default=1, ge=1, le=2)
    part_count: int = Field(default=1, ge=1, le=2)
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _part_is_valid(self) -> "ChapterPart":
        if self.end_page < self.start_page:
            raise ValueError("chapter part end_page cannot precede start_page")
        if self.part_index > self.part_count:
            raise ValueError("part_index cannot exceed part_count")
        if self.part_count == 1 and self.part_index != 1:
            raise ValueError("an unsplit chapter must be part 1 of 1")
        return self


class SemesterWeek(BaseModel):
    week: int = Field(ge=1)
    chapters: list[ChapterPart] = Field(min_length=1, max_length=3)
    rationale: str = Field(min_length=1)
    learning_objectives: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _week_has_at_most_three_chapters(self) -> "SemesterWeek":
        ids = {chapter.chapter_id for chapter in self.chapters}
        if len(ids) > 3:
            raise ValueError("a week cannot contain more than three chapters")
        if len(ids) != len(self.chapters):
            raise ValueError("two parts of one chapter cannot share a week")
        return self


class SemesterWeekPlan(BaseModel):
    schema_name: str = SEMESTER_PLAN_SCHEMA
    schema_version: str = SEMESTER_PLAN_VERSION
    book_title: str = Field(min_length=1)
    week_count: int = Field(ge=1)
    weeks: list[SemesterWeek] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _weeks_are_contiguous(self) -> "SemesterWeekPlan":
        if self.week_count != len(self.weeks):
            raise ValueError("week_count must equal the number of weeks")
        expected = list(range(1, len(self.weeks) + 1))
        if [week.week for week in self.weeks] != expected:
            raise ValueError("week numbers must be contiguous and start at 1")
        return self

    def validate_against(self, inventory: ChapterInventory) -> "SemesterWeekPlan":
        """Enforce coverage, ordering, adjacency, and split boundaries."""
        expected = {chapter.chapter_id: chapter for chapter in inventory.chapters}
        flattened = [part for week in self.weeks for part in week.chapters]
        actual_ids = {part.chapter_id for part in flattened}
        if actual_ids != set(expected):
            missing = sorted(set(expected) - actual_ids)
            unknown = sorted(actual_ids - set(expected))
            raise ValueError(f"chapter coverage mismatch; missing={missing}, unknown={unknown}")

        positions = {chapter.chapter_id: index for index, chapter in enumerate(inventory.chapters)}
        seen_order = [positions[part.chapter_id] for part in flattened]
        if seen_order != sorted(seen_order):
            raise ValueError("semester plan changes chapter order")

        for week in self.weeks:
            week_positions = [positions[part.chapter_id] for part in week.chapters]
            if week_positions and week_positions != list(
                range(week_positions[0], week_positions[0] + len(week_positions))
            ):
                raise ValueError("chapters grouped in one week must be adjacent")

        by_chapter: dict[str, list[ChapterPart]] = {}
        for part in flattened:
            by_chapter.setdefault(part.chapter_id, []).append(part)
        for chapter_id, parts in by_chapter.items():
            chapter = expected[chapter_id]
            if len(parts) == 1:
                part = parts[0]
                if part.part_count != 1:
                    raise ValueError(f"chapter {chapter_id} is missing a split part")
                if (part.start_page, part.end_page) != (
                    chapter.start_page,
                    chapter.end_page,
                ):
                    raise ValueError(f"chapter {chapter_id} does not cover its full page range")
                continue
            if len(parts) != 2:
                raise ValueError(f"chapter {chapter_id} may use at most two weeks")
            parts.sort(key=lambda part: part.part_index)
            if [part.part_index for part in parts] != [1, 2] or any(
                part.part_count != 2 for part in parts
            ):
                raise ValueError(f"chapter {chapter_id} has invalid split numbering")
            if parts[0].start_page != chapter.start_page or parts[1].end_page != chapter.end_page:
                raise ValueError(f"chapter {chapter_id} split does not cover its boundaries")
            if parts[0].end_page + 1 != parts[1].start_page:
                raise ValueError(f"chapter {chapter_id} split overlaps or leaves a gap")
        return self


class ChapterDraft(BaseModel):
    chapter_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    source_ids: list[str] = Field(min_length=1)


class BookStructureDraftLLM(BaseModel):
    book_title: str = Field(min_length=1)
    chapters: list[ChapterDraft] = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class WeekDraft(BaseModel):
    chapter_ids: list[str] = Field(min_length=1, max_length=3)
    split_chapter_id: str | None = None
    split_part: int | None = Field(default=None, ge=1, le=2)
    rationale: str = Field(min_length=1)
    learning_objectives: list[str] = Field(min_length=1)


class SemesterPlanDraftLLM(BaseModel):
    weeks: list[WeekDraft] = Field(min_length=1)


_CHAPTER_RE = re.compile(
    r"^\s*chapter\s+(?P<number>\d+|[ivxlcdm]+)\s*[:.\-–—]?\s*(?P<title>[^.]{0,100})\s*$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,3})[.:\-]\s+(?P<title>[A-Z][^.!?]{2,90})\s*$"
)


def discover_chapters(
    pages: list[tuple[int, str]], book_title: str = "Uploaded book"
) -> ChapterInventory:
    """Find chapter starts from page-top headings, with an explicit fallback."""
    if not pages:
        raise ValueError("cannot discover chapters in an empty book")

    candidates: list[tuple[int, str, str]] = []
    seen_labels: set[str] = set()
    for page_number, text in pages:
        lines = [line.strip() for line in text.splitlines()[:8] if line.strip()]
        for line in lines:
            match = _CHAPTER_RE.match(line) or _NUMBERED_HEADING_RE.match(line)
            if not match:
                continue
            number = match.group("number").upper()
            title = match.group("title").strip(" :-–—") or f"Chapter {number}"
            label = f"{number}:{title.lower()}"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            candidates.append((page_number, number, title))
            break

    first_page, last_page = pages[0][0], pages[-1][0]
    warnings: list[str] = []
    confidence = 0.95
    if not candidates:
        warnings.append(
            "No reliable chapter headings were found; the whole readable book is one chapter."
        )
        candidates = [(first_page, "1", book_title)]
        confidence = 0.35

    candidates.sort(key=lambda item: item[0])
    deduped: list[tuple[int, str, str]] = []
    for candidate in candidates:
        if deduped and candidate[0] == deduped[-1][0]:
            continue
        deduped.append(candidate)

    chapters: list[Chapter] = []
    for index, (start, number, title) in enumerate(deduped):
        # The first chapter owns front matter so the weekly plan covers every
        # readable page rather than silently dropping introductions.
        start_page = first_page if index == 0 else start
        end_page = deduped[index + 1][0] - 1 if index + 1 < len(deduped) else last_page
        chapters.append(
            Chapter(
                chapter_id=f"C{index + 1:03d}",
                title=title or f"Chapter {number}",
                start_page=start_page,
                end_page=end_page,
                source_ids=[f"P{start}"],
            )
        )

    return ChapterInventory(
        book_title=book_title,
        chapters=classify_chapter_sizes(chapters),
        confidence=confidence,
        warnings=warnings,
    )


def classify_chapter_sizes(chapters: list[Chapter]) -> list[Chapter]:
    counts = [chapter.page_count for chapter in chapters]
    median = statistics.median(counts)
    classified: list[Chapter] = []
    for chapter in chapters:
        if chapter.page_count <= max(3, median * 0.35):
            size = ChapterSize.TINY
        elif chapter.page_count <= max(6, median * 0.65):
            size = ChapterSize.SMALL
        elif chapter.page_count >= 80 and (
            len(chapters) == 1 or chapter.page_count >= median * 1.65
        ):
            size = ChapterSize.LARGE
        else:
            size = ChapterSize.STANDARD
        classified.append(chapter.model_copy(update={"size": size}))
    return classified


def _whole_part(chapter: Chapter) -> ChapterPart:
    return ChapterPart(
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        start_page=chapter.start_page,
        end_page=chapter.end_page,
        source_ids=chapter.source_ids,
    )


def _split_parts(chapter: Chapter) -> tuple[ChapterPart, ChapterPart]:
    midpoint = chapter.start_page + (chapter.page_count // 2) - 1
    midpoint = max(chapter.start_page, min(midpoint, chapter.end_page - 1))
    first = ChapterPart(
        chapter_id=chapter.chapter_id,
        title=f"{chapter.title} — Part 1",
        start_page=chapter.start_page,
        end_page=midpoint,
        part_index=1,
        part_count=2,
        source_ids=chapter.source_ids,
    )
    second = ChapterPart(
        chapter_id=chapter.chapter_id,
        title=f"{chapter.title} — Part 2",
        start_page=midpoint + 1,
        end_page=chapter.end_page,
        part_index=2,
        part_count=2,
        source_ids=chapter.source_ids,
    )
    return first, second


def _week(week: int, parts: list[ChapterPart], rationale: str) -> SemesterWeek:
    titles = [part.title for part in parts]
    source_ids = list(dict.fromkeys(source for part in parts for source in part.source_ids))
    return SemesterWeek(
        week=week,
        chapters=parts,
        rationale=rationale,
        learning_objectives=[f"Explain the core ideas in {title}." for title in titles],
        source_ids=source_ids,
    )


def plan_semester(inventory: ChapterInventory) -> SemesterWeekPlan:
    """Build a deterministic valid plan from the chapter inventory."""
    chapters = inventory.chapters
    median = statistics.median(chapter.page_count for chapter in chapters)
    weeks: list[SemesterWeek] = []
    index = 0
    while index < len(chapters):
        chapter = chapters[index]
        if chapter.size is ChapterSize.LARGE and chapter.page_count >= 2:
            first, second = _split_parts(chapter)
            weeks.append(_week(len(weeks) + 1, [first], "Large chapter split across two weeks."))
            weeks.append(_week(len(weeks) + 1, [second], "Second half of the large chapter."))
            index += 1
            continue

        group = [chapter]
        total_pages = chapter.page_count
        if chapter.size in {ChapterSize.TINY, ChapterSize.SMALL}:
            cursor = index + 1
            while cursor < len(chapters) and len(group) < 3:
                candidate = chapters[cursor]
                if candidate.size is ChapterSize.LARGE:
                    break
                if total_pages + candidate.page_count > max(12, median * 1.35):
                    break
                group.append(candidate)
                total_pages += candidate.page_count
                cursor += 1
        rationale = (
            "Small adjacent chapters combined into one week."
            if len(group) > 1
            else "One chapter assigned to one week."
        )
        weeks.append(_week(len(weeks) + 1, [_whole_part(item) for item in group], rationale))
        index += len(group)

    plan = SemesterWeekPlan(
        book_title=inventory.book_title,
        week_count=len(weeks),
        weeks=weeks,
        confidence=inventory.confidence,
        warnings=inventory.warnings,
    )
    return plan.validate_against(inventory)


def pages_for_week(
    week: SemesterWeek, pages: list[tuple[int, str]]
) -> list[tuple[int, str]]:
    ranges = [(part.start_page, part.end_page) for part in week.chapters]
    selected = [
        page
        for page in pages
        if any(start <= page[0] <= end for start, end in ranges)
    ]
    if not selected:
        raise ValueError(f"week {week.week} has no readable source pages")
    return selected


def chapter_inventory_block(inventory: ChapterInventory) -> str:
    return "\n".join(
        f"[{chapter.chapter_id}] {chapter.title} | pages {chapter.start_page}-{chapter.end_page} "
        f"| {chapter.page_count} pages | size={chapter.size.value}"
        for chapter in inventory.chapters
    )


__all__ = [
    "SEMESTER_PLAN_SCHEMA",
    "SEMESTER_PLAN_VERSION",
    "BookStructureDraftLLM",
    "Chapter",
    "ChapterDraft",
    "ChapterInventory",
    "ChapterPart",
    "ChapterSize",
    "SemesterPlanDraftLLM",
    "SemesterWeek",
    "SemesterWeekPlan",
    "WeekDraft",
    "chapter_inventory_block",
    "classify_chapter_sizes",
    "discover_chapters",
    "pages_for_week",
    "plan_semester",
]
