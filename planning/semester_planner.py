"""Chapter-aware planning for one book's weekly semester layout.

The model may propose structure, but this module owns the rules: one book per
course, ordered chapter coverage, bounded weekly compression, semester splits,
and monthly assessment cadence.
"""

from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher
from enum import Enum
from math import ceil
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

SEMESTER_PLAN_SCHEMA = "univai.semester.week-plan"
SEMESTER_PLAN_VERSION = "2.0.0"

# ── The shape a course is allowed to take ─────────────────────────────
#
# One book is one course. A course normally fits one two-month semester; a
# genuinely large book may use one exceptional three-month semester, and a
# book above that ceiling is split into the minimum number of semesters.
WEEKS_PER_MONTH = 4
TARGET_SEMESTER_WEEKS = 2 * WEEKS_PER_MONTH  # 8 — two months, the normal course
MAX_SEMESTER_WEEKS = 3 * WEEKS_PER_MONTH  # 12 — exceptional three-month semester
NORMAL_SEMESTER_CHAPTERS = 12
MAX_CHAPTERS_PER_SEMESTER = 20


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


# 12 chapters / 8 weeks and 20 chapters / 12 weeks both require at most two
# adjacent chapters in one longer theoretical lecture.
MAX_CHAPTERS_PER_WEEK = 2


class SemesterWeek(BaseModel):
    week: int = Field(ge=1)
    semester: int = Field(default=1, ge=1)
    semester_week: int = Field(default=0, ge=0)
    chapters: list[ChapterPart] = Field(min_length=1, max_length=MAX_CHAPTERS_PER_WEEK)
    rationale: str = Field(min_length=1)
    learning_objectives: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _week_chapters_are_distinct(self) -> "SemesterWeek":
        if self.semester_week == 0:
            self.semester_week = self.week
        ids = {chapter.chapter_id for chapter in self.chapters}
        if len(ids) > MAX_CHAPTERS_PER_WEEK:
            raise ValueError(
                f"a week cannot contain more than {MAX_CHAPTERS_PER_WEEK} chapters"
            )
        if len(ids) != len(self.chapters):
            raise ValueError("two parts of one chapter cannot share a week")
        return self


class MidtermSchedule(BaseModel):
    number: int = Field(ge=1)
    after_week: int = Field(ge=1)
    covers_weeks: list[int] = Field(min_length=1)


class CourseSemester(BaseModel):
    semester: int = Field(ge=1)
    week_count: int = Field(ge=1, le=MAX_SEMESTER_WEEKS)
    starts_at_week: int = Field(ge=1)
    ends_at_week: int = Field(ge=1)
    chapter_ids: list[str] = Field(min_length=1)
    quiz_count: int = Field(ge=1)
    midterms: list[MidtermSchedule] = Field(default_factory=list)
    final_after_week: int = Field(ge=1)

    @model_validator(mode="after")
    def _assessment_cadence_is_valid(self) -> "CourseSemester":
        if self.ends_at_week - self.starts_at_week + 1 != self.week_count:
            raise ValueError("semester global week bounds must match week_count")
        if self.quiz_count != self.week_count:
            raise ValueError("every theoretical lecture must have one post-lecture quiz")
        expected_midterms = list(
            range(WEEKS_PER_MONTH, self.week_count + 1, WEEKS_PER_MONTH)
        )
        if [midterm.after_week for midterm in self.midterms] != expected_midterms:
            raise ValueError("a semester must schedule one midterm after every four weeks")
        if self.final_after_week != self.week_count:
            raise ValueError("the final must follow the semester's last week")
        return self


class SemesterWeekPlan(BaseModel):
    schema_name: str = SEMESTER_PLAN_SCHEMA
    schema_version: str = SEMESTER_PLAN_VERSION
    book_title: str = Field(min_length=1)
    chapter_count: int | None = Field(default=None, ge=1)
    semester_count: int = Field(default=1, ge=1)
    week_count: int = Field(ge=1)
    weeks: list[SemesterWeek] = Field(min_length=1)
    semesters: list[CourseSemester] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _weeks_are_contiguous(self) -> "SemesterWeekPlan":
        if self.week_count != len(self.weeks):
            raise ValueError("week_count must equal the number of weeks")
        expected = list(range(1, len(self.weeks) + 1))
        if [week.week for week in self.weeks] != expected:
            raise ValueError("week numbers must be contiguous and start at 1")
        if self.chapter_count is None:
            self.chapter_count = len(
                {part.chapter_id for week in self.weeks for part in week.chapters}
            )
        if not self.semesters:
            chapter_ids = list(
                dict.fromkeys(
                    part.chapter_id for week in self.weeks for part in week.chapters
                )
            )
            self.semesters = [
                CourseSemester(
                    semester=1,
                    week_count=self.week_count,
                    starts_at_week=1,
                    ends_at_week=self.week_count,
                    chapter_ids=chapter_ids,
                    quiz_count=self.week_count,
                    midterms=_midterms_for(self.week_count),
                    final_after_week=self.week_count,
                )
            ]
        if self.semester_count != len(self.semesters):
            raise ValueError("semester_count must equal the number of semesters")
        if [semester.semester for semester in self.semesters] != list(
            range(1, self.semester_count + 1)
        ):
            raise ValueError("semester numbers must be contiguous and start at 1")
        for semester in self.semesters:
            semester_weeks = [
                week for week in self.weeks if week.semester == semester.semester
            ]
            if len(semester_weeks) != semester.week_count:
                raise ValueError("semester week_count does not match its weeks")
            if [week.semester_week for week in semester_weeks] != list(
                range(1, semester.week_count + 1)
            ):
                raise ValueError("semester-local weeks must be contiguous and start at 1")
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
_CONTENTS_HEADING_RE = re.compile(r"^(contents|table of contents)$", re.I)
_CONTENTS_ITEM_RE = re.compile(
    r"^\s*(?P<number>\d{1,3})[.)]\s+(?P<title>.+?)\s*[.]?\s*$"
)
_HEADING_STOP_WORDS = {"a", "an", "and", "for", "is", "of", "the", "to", "when"}


def _normalise_heading(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _heading_similarity(expected: str, candidate: str) -> float:
    expected_normalised = _normalise_heading(expected)
    candidate_normalised = _normalise_heading(candidate)
    if not expected_normalised or not candidate_normalised:
        return 0.0
    expected_words = set(expected_normalised.split()) - _HEADING_STOP_WORDS
    candidate_words = set(candidate_normalised.split()) - _HEADING_STOP_WORDS
    if not expected_words or not candidate_words:
        return SequenceMatcher(None, expected_normalised, candidate_normalised).ratio()
    dice = 2 * len(expected_words & candidate_words) / (
        len(expected_words) + len(candidate_words)
    )
    sequence = SequenceMatcher(None, expected_normalised, candidate_normalised).ratio()
    # Sequence matching rescues small spelling mistakes ("Intoduction"), but
    # a merely moderate phrase resemblance must not erase a negation and map
    # "Not Appropriate" onto the earlier "Appropriate" slide.
    return max(dice, sequence if sequence >= 0.8 else 0.0)


def _contents_chapter_candidates(
    pages: list[tuple[int, str]],
) -> list[tuple[int, str, str]]:
    """Map an explicit contents page to the corresponding chapter starts."""
    for contents_page, text in pages:
        lines = [line.strip() for line in text.splitlines()[:30] if line.strip()]
        agenda_index = next(
            (index for index, line in enumerate(lines) if _CONTENTS_HEADING_RE.match(line)),
            None,
        )
        if agenda_index is None:
            continue
        agenda_items = []
        for line in lines[agenda_index + 1 :]:
            match = _CONTENTS_ITEM_RE.match(line)
            if match:
                agenda_items.append(
                    (match.group("number"), match.group("title").strip(" .:-–—"))
                )
        if len(agenda_items) < 2:
            continue

        located: list[tuple[int, str, str]] = []
        next_page = contents_page + 1
        for number, title in agenda_items:
            best: tuple[float, int] | None = None
            for page_number, page_text in pages:
                if page_number < next_page:
                    continue
                heading_lines = [
                    line.strip() for line in page_text.splitlines()[:3] if line.strip()
                ]
                score = max(
                    (_heading_similarity(title, line) for line in heading_lines),
                    default=0.0,
                )
                # Contents items sometimes combine two consecutive headings
                # (for example "Advantages and Disadvantages"). The
                # first credible match is the chapter boundary; choosing a
                # later, slightly closer wording would swallow the first half.
                if score >= 0.65:
                    best = (score, page_number)
                    break
                if best is None or score > best[0]:
                    best = (score, page_number)
            if best is not None and best[0] >= 0.65:
                located.append((best[1], number, title))
                next_page = best[1] + 1

        if len(located) >= 2:
            return located
    return []


def _numbered_chapter_candidates(
    pages: list[tuple[int, str]],
) -> list[tuple[int, str, str]]:
    """Accept numbered headings only when they form a clean 1..N page sequence."""
    candidates: list[tuple[int, str, str]] = []
    for page_number, text in pages:
        matches = [
            match
            for line in (line.strip() for line in text.splitlines()[:8] if line.strip())
            if (match := _NUMBERED_HEADING_RE.match(line))
        ]
        # Learning outcomes and ordinary numbered bullets place several items
        # on one page; a real numbered chapter start contributes one heading.
        if len(matches) != 1:
            continue
        match = matches[0]
        candidates.append(
            (page_number, match.group("number"), match.group("title").strip(" .:-–—"))
        )
    numbers = [int(number) for _, number, _ in candidates]
    return candidates if len(candidates) >= 2 and numbers == list(range(1, len(numbers) + 1)) else []


def discover_chapters(
    pages: list[tuple[int, str]], book_title: str = "Uploaded book"
) -> ChapterInventory:
    """Find chapter starts from page-top headings, with an explicit fallback."""
    if not pages:
        raise ValueError("cannot discover chapters in an empty book")

    explicit_candidates: list[tuple[int, str, str]] = []
    seen_labels: set[str] = set()
    for page_number, text in pages:
        lines = [line.strip() for line in text.splitlines()[:8] if line.strip()]
        for line in lines:
            match = _CHAPTER_RE.match(line)
            if not match:
                continue
            number = match.group("number").upper()
            title = match.group("title").strip(" :-–—") or f"Chapter {number}"
            label = f"{number}:{title.lower()}"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            explicit_candidates.append((page_number, number, title))
            break

    candidates = (
        explicit_candidates
        or _contents_chapter_candidates(pages)
        or _numbered_chapter_candidates(pages)
    )

    first_page, last_page = pages[0][0], pages[-1][0]
    warnings: list[str] = []
    confidence = 0.95 if explicit_candidates else 0.8
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


def _week(
    week: int,
    parts: list[ChapterPart],
    rationale: str,
    *,
    semester: int,
    semester_week: int,
) -> SemesterWeek:
    titles = [part.title for part in parts]
    source_ids = list(dict.fromkeys(source for part in parts for source in part.source_ids))
    return SemesterWeek(
        week=week,
        semester=semester,
        semester_week=semester_week,
        chapters=parts,
        rationale=rationale,
        learning_objectives=[f"Explain the core ideas in {title}." for title in titles],
        source_ids=source_ids,
    )


def _midterms_for(week_count: int) -> list[MidtermSchedule]:
    return [
        MidtermSchedule(
            number=number,
            after_week=after_week,
            covers_weeks=list(range(after_week - WEEKS_PER_MONTH + 1, after_week + 1)),
        )
        for number, after_week in enumerate(
            range(WEEKS_PER_MONTH, week_count + 1, WEEKS_PER_MONTH),
            start=1,
        )
    ]


def _semester_week_count(chapter_count: int) -> int:
    if chapter_count <= TARGET_SEMESTER_WEEKS:
        return chapter_count
    if chapter_count <= NORMAL_SEMESTER_CHAPTERS:
        return TARGET_SEMESTER_WEEKS
    return MAX_SEMESTER_WEEKS


def _semester_chapter_groups(chapters: list[Chapter]) -> list[list[Chapter]]:
    semester_count = ceil(len(chapters) / MAX_CHAPTERS_PER_SEMESTER)
    return [
        chapters[
            position * len(chapters) // semester_count :
            (position + 1) * len(chapters) // semester_count
        ]
        for position in range(semester_count)
    ]


def plan_semester(inventory: ChapterInventory) -> SemesterWeekPlan:
    """Build the book's complete course plan from its ordered chapters.

    The name is retained for API compatibility, but the returned contract may
    contain multiple semesters when a book has more than twenty chapters.
    """
    weeks: list[SemesterWeek] = []
    semesters: list[CourseSemester] = []
    warnings = list(inventory.warnings)
    chapter_groups = _semester_chapter_groups(inventory.chapters)
    if len(chapter_groups) > 1:
        warnings.append(
            f"The book has {len(inventory.chapters)} chapters, so its one course is split "
            f"across {len(chapter_groups)} semesters."
        )

    for semester_number, semester_chapters in enumerate(chapter_groups, start=1):
        semester_week_count = _semester_week_count(len(semester_chapters))
        semester_start = len(weeks) + 1
        for position in range(semester_week_count):
            group = semester_chapters[
                position * len(semester_chapters) // semester_week_count :
                (position + 1) * len(semester_chapters) // semester_week_count
            ]
            combined = len(group) > 1
            weeks.append(
                _week(
                    len(weeks) + 1,
                    [_whole_part(chapter) for chapter in group],
                    (
                        "Adjacent chapters combined into one longer theoretical lecture."
                        if combined
                        else "One chapter assigned to one theoretical lecture."
                    ),
                    semester=semester_number,
                    semester_week=position + 1,
                )
            )
        semesters.append(
            CourseSemester(
                semester=semester_number,
                week_count=semester_week_count,
                starts_at_week=semester_start,
                ends_at_week=len(weeks),
                chapter_ids=[chapter.chapter_id for chapter in semester_chapters],
                quiz_count=semester_week_count,
                midterms=_midterms_for(semester_week_count),
                final_after_week=semester_week_count,
            )
        )
        if len(semester_chapters) > NORMAL_SEMESTER_CHAPTERS:
            warnings.append(
                f"Semester {semester_number} is an exceptional {MAX_SEMESTER_WEEKS}-week "
                f"semester for {len(semester_chapters)} chapters."
            )
        elif len(semester_chapters) > TARGET_SEMESTER_WEEKS:
            warnings.append(
                f"Semester {semester_number} compresses {len(semester_chapters)} chapters "
                f"into {TARGET_SEMESTER_WEEKS} weekly lectures."
            )

    plan = SemesterWeekPlan(
        book_title=inventory.book_title,
        chapter_count=len(inventory.chapters),
        semester_count=len(semesters),
        week_count=len(weeks),
        weeks=weeks,
        semesters=semesters,
        confidence=inventory.confidence,
        warnings=warnings,
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
    "MAX_CHAPTERS_PER_SEMESTER",
    "MAX_CHAPTERS_PER_WEEK",
    "MAX_SEMESTER_WEEKS",
    "NORMAL_SEMESTER_CHAPTERS",
    "TARGET_SEMESTER_WEEKS",
    "BookStructureDraftLLM",
    "Chapter",
    "ChapterDraft",
    "ChapterInventory",
    "ChapterPart",
    "ChapterSize",
    "CourseSemester",
    "MidtermSchedule",
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
