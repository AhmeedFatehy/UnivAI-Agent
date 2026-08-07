"""Course identity: what makes one learner's generated course reusable for another.

Two learners who upload the same bytes should not each pay for the same course
to be written. ``cache.content_identity`` already solves this one layer down —
identical bytes are parsed, chunked and embedded once, and each tenant gets its
own grant over the one artifact. This module is the same idea for the layer
above: the lectures, quizzes and slides built *from* those bytes.

Byte identity alone is not enough to reuse a course, and the reason is on disk
right now. ``lectures/S-2026-000001`` and ``lectures/S-2026-000004`` were built
from a byte-identical book and disagree completely::

    S-2026-000001   1 week    ['MySQL_Lec3.pdf']          <- chapter discovery failed
    S-2026-000004   4 weeks   ['Transactions', ...]       <- discovery fixed

The first was produced before ``discover_chapters`` learned to read a qualified,
bulleted contents page. Reusing on content hash alone would have served that
stale one-week course to every later learner and quietly undone the fix.

So reuse is keyed by content hash AND :func:`course_fingerprint` — a hash over
every input that decides what a book becomes. Change the planner, the lecture
shape, a prompt or the generating model, and the fingerprint changes, so the
old course is no longer a candidate and the next learner gets a fresh build.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

COURSE_IDENTITY_SCHEMA = "univai.agent.course_identity"
COURSE_FINGERPRINT_VERSION = "1.0.0"

# Bump when discover_chapters or plan_semester change how many weeks a book
# becomes, or which chapters land in which week.
PLANNER_VERSION = "1.1.0"
# Bump when the lecture writer changes the slides, narration or questions a
# week becomes — prompts, batching, or the validity rules a batch must pass.
LECTURE_WRITER_VERSION = "1.1.0"


class CourseComponents(BaseModel):
    """Every input that shapes a generated course, in one fingerprint."""

    schema_version: str = COURSE_FINGERPRINT_VERSION
    planner_version: str = PLANNER_VERSION
    lecture_writer_version: str = LECTURE_WRITER_VERSION

    # The planner's caps: they decide the week count and the chapters per week.
    target_semester_weeks: int = Field(gt=0)
    max_semester_weeks: int = Field(gt=0)
    normal_semester_chapters: int = Field(gt=0)
    max_chapters_per_semester: int = Field(gt=0)
    max_chapters_per_week: int = Field(gt=0)

    # The lecture's shape: how long a week runs and how much it contains.
    lecture_minutes_min: int = Field(gt=0)
    lecture_minutes_max: int = Field(gt=0)
    minutes_per_page: float = Field(gt=0)
    spoken_words_per_minute: int = Field(gt=0)
    narration_sentences_per_slide: int = Field(gt=0)
    slides_per_batch: int = Field(gt=0)
    min_lecture_questions: int = Field(gt=0)

    # The words the model was given, and the model that answered them. A course
    # written by a different model is a different course.
    prompt_versions: str = Field(min_length=1)
    generation_model: str = Field(min_length=1)

    def fingerprint(self) -> str:
        """Stable SHA-256 over every component above."""
        components = [
            ("schema", self.schema_version),
            ("planner", self.planner_version),
            ("lecture_writer", self.lecture_writer_version),
            ("target_semester_weeks", str(self.target_semester_weeks)),
            ("max_semester_weeks", str(self.max_semester_weeks)),
            ("normal_semester_chapters", str(self.normal_semester_chapters)),
            ("max_chapters_per_semester", str(self.max_chapters_per_semester)),
            ("max_chapters_per_week", str(self.max_chapters_per_week)),
            ("lecture_minutes_min", str(self.lecture_minutes_min)),
            ("lecture_minutes_max", str(self.lecture_minutes_max)),
            ("minutes_per_page", f"{self.minutes_per_page:.4f}"),
            ("spoken_words_per_minute", str(self.spoken_words_per_minute)),
            ("narration_sentences_per_slide", str(self.narration_sentences_per_slide)),
            ("slides_per_batch", str(self.slides_per_batch)),
            ("min_lecture_questions", str(self.min_lecture_questions)),
            ("prompts", self.prompt_versions),
            ("generation_model", self.generation_model),
        ]
        seed = "\n".join(f"{name}:{value}" for name, value in components)
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def course_fingerprint(components: CourseComponents) -> str:
    """The reuse half of the key; the content hash is the other half."""
    return components.fingerprint()


__all__ = [
    "COURSE_IDENTITY_SCHEMA",
    "COURSE_FINGERPRINT_VERSION",
    "PLANNER_VERSION",
    "LECTURE_WRITER_VERSION",
    "CourseComponents",
    "course_fingerprint",
]
