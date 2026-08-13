"""Turn an uploaded book into a chapter-aware course with slides and quizzes.

    python UnivAI-Agent/generation/lecture_gen.py <absolute_pdf_path> <book_id>   (from the campus root)

For each week this stores a structured lecture, narration script, slide deck,
quiz, and grounded section pack in Postgres. Public identifiers are generated
by Postgres as UUIDs. Integrated generation never creates learner folders.
Progress is reported through books.progress so the upload page can show it.

The page numbers on slides and citations are OURS, taken from how the book was
split — the model is never trusted to invent one.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

# Model output lands in log prints; on Windows a redirected stdout defaults to
# cp1252 and one "≤" in a reply kills the whole course build.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agents.prompts import PromptOperation, load_prompt_for
from agents.schemas import strict_json_document
from guardrails.prompt_boundary import (
    PROMPT_BOUNDARY_POLICY_VERSION,
    quote_untrusted_data,
    split_prompt_roles,
)
from planning.semester_planner import (
    MAX_CHAPTERS_PER_SEMESTER,
    MAX_CHAPTERS_PER_WEEK,
    MAX_SEMESTER_WEEKS,
    NORMAL_SEMESTER_CHAPTERS,
    TARGET_SEMESTER_WEEKS,
    SemesterWeek,
    SemesterWeekPlan,
    discover_chapters,
    pages_for_week,
    plan_semester,
)
from generation.course_identity import CourseComponents
from generation.section_gen import generate_section_pack
from generation.slide_design import (
    SLIDE_DESIGN_INSTRUCTIONS,
    batch_design_problem,
    normalise_slide,
    render_slidev_markdown,
)
from planning.section_planner import SectionIdentity
from tools.registry import ToolContext

# The Brain cave is checked out inside the UnivAI campus repo; the shared
# plumbing (db, LLM adapter) lives there in services/.
ROOT = Path(__file__).resolve().parents[2]  # the UnivAI campus root
execute = None
fetch_one = None
fetch_all = None
complete = None
LLMError = RuntimeError


def load_integrated_dependencies() -> None:
    """Load parent-owned services only for the explicit integrated command."""
    global execute, fetch_one, fetch_all, complete, LLMError, ROOT
    configured = os.getenv("UNIVAI_INTEGRATION_ROOT")
    ROOT = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[2]
    )
    services = ROOT / "services"
    if not (services / "common").is_dir():
        raise RuntimeError(
            f"Integrated services were not found under {ROOT}. "
            "Set UNIVAI_INTEGRATION_ROOT to the main UnivAI checkout."
        )
    sys.path.insert(0, str(services))
    sys.path.insert(0, str(ROOT))
    from common.db import (
        execute as db_execute,
        fetch_all as db_fetch_all,
        fetch_one as db_fetch_one,
    )
    from common.llm import LLMError as SharedLLMError, complete as llm_complete

    execute = db_execute
    fetch_one = db_fetch_one
    fetch_all = db_fetch_all
    complete = llm_complete
    LLMError = SharedLLMError

# ── Lecture length ────────────────────────────────────────────────────
#
# A theoretical lecture runs 45 to 120 minutes and has to carry its week's chapters. There
# is no admin dial any more: the length follows the material. A week that
# compresses several chapters — the semester planner's answer to a book too big
# for three months — is simply a longer lecture, which is the trade the planner
# is making on purpose.
LECTURE_MINUTES_MIN = 45
LECTURE_MINUTES_MAX = 120
# Roughly what a page of textbook is worth once it is spoken rather than read.
MINUTES_PER_PAGE = 2.0
# Measured against the configured narration voice.
SPOKEN_WORDS_PER_MINUTE = 150
NARRATION_SENTENCES_PER_SLIDE = 8
WORDS_PER_NARRATION_SENTENCE = 18
MINUTES_PER_SLIDE = (
    NARRATION_SENTENCES_PER_SLIDE * WORDS_PER_NARRATION_SENTENCE
) / SPOKEN_WORDS_PER_MINUTE
# Slides asked for in ONE model call. A 120-minute lecture is ~125 slides and
# tens of thousands of tokens — far past what any single JSON reply survives,
# which is why the lecture is generated in batches and concatenated. Small
# enough that a reply still parses; large enough that a batch is worth a call.
SLIDES_PER_BATCH = 6
# The quiz bank per week: >=90% of any served paper must be answerable from
# what the lecturer SAID (easy if you attended); self-study questions from the
# wider pages exist but can never exceed 10% of a paper. Both scale with the
# lecture instead of a size dial.
QUESTIONS_PER_10_MINUTES = 3
SELF_STUDY_QUESTION_RATIO = 0.2
# The App's largest assessment profile serves 15 quiz questions. Even the
# shortest lecture must generate at least that many lecturer-grounded options;
# self-study questions are an additional tail, never a substitute for the
# served paper's core bank.
MIN_LECTURE_QUESTIONS = 15


def lecture_minutes(page_count: int) -> int:
    """How long this week's material is worth, inside the 45-120 bound."""
    return int(
        max(
            LECTURE_MINUTES_MIN,
            min(LECTURE_MINUTES_MAX, round(page_count * MINUTES_PER_PAGE)),
        )
    )


def lecture_shape(page_count: int) -> dict:
    """Slides and question counts for a week, derived from its own material."""
    minutes = lecture_minutes(page_count)
    slides = max(3, round(minutes / MINUTES_PER_SLIDE))
    lecture_qs = max(MIN_LECTURE_QUESTIONS, round(minutes / 10 * QUESTIONS_PER_10_MINUTES))
    return {
        "minutes": minutes,
        "slides": slides,
        "narration": f"{NARRATION_SENTENCES_PER_SLIDE - 2}-{NARRATION_SENTENCES_PER_SLIDE + 1}",
        "lecture_qs": lecture_qs,
        "self_qs": max(2, round(lecture_qs * SELF_STUDY_QUESTION_RATIO)),
    }


# A 3B model with an 8k window: keep the source well under it.
MAX_SOURCE_CHARS = 12000
MAX_CHARS_PER_PAGE = 1500
ATTEMPTS = 4
# How much of a rejected reply the repair prompt shows back to the model. A
# slide batch is allowed 800 + 340 * slides tokens — roughly 11k characters at
# SLIDES_PER_BATCH — so the old 2000-char excerpt cut off before the later
# slides. Asked to fix slide 5, the model could not see slide 5.
REPAIR_REPLY_CHARS = 16000

_LECTURE_PROMPT = load_prompt_for(PromptOperation.CONTENT_GENERATE_LECTURE)
_QUIZ_PROMPT = load_prompt_for(PromptOperation.ASSESSMENT_QUIZ)
_SLIDE_BATCH_PROMPT = load_prompt_for(PromptOperation.CONTENT_GENERATE_SLIDE_BATCH)
_QUIZ_BANK_PROMPT = load_prompt_for(PromptOperation.ASSESSMENT_QUIZ_BANK)
_SECTION_PROMPT = load_prompt_for(PromptOperation.CONTENT_GENERATE_SECTION)
LECTURE_SYSTEM = _SLIDE_BATCH_PROMPT.render_system()
QUIZ_SYSTEM = _QUIZ_BANK_PROMPT.render_system()


def course_components() -> CourseComponents:
    """The live inputs that decide what a book becomes, read at call time."""
    return CourseComponents(
        target_semester_weeks=TARGET_SEMESTER_WEEKS,
        max_semester_weeks=MAX_SEMESTER_WEEKS,
        normal_semester_chapters=NORMAL_SEMESTER_CHAPTERS,
        max_chapters_per_semester=MAX_CHAPTERS_PER_SEMESTER,
        max_chapters_per_week=MAX_CHAPTERS_PER_WEEK,
        lecture_minutes_min=LECTURE_MINUTES_MIN,
        lecture_minutes_max=LECTURE_MINUTES_MAX,
        minutes_per_page=MINUTES_PER_PAGE,
        spoken_words_per_minute=SPOKEN_WORDS_PER_MINUTE,
        narration_sentences_per_slide=NARRATION_SENTENCES_PER_SLIDE,
        slides_per_batch=SLIDES_PER_BATCH,
        min_lecture_questions=MIN_LECTURE_QUESTIONS,
        prompt_versions=(
            f"lecture={_LECTURE_PROMPT.version},quiz={_QUIZ_PROMPT.version},"
            f"slide_batch={_SLIDE_BATCH_PROMPT.version},"
            f"quiz_bank={_QUIZ_BANK_PROMPT.version},"
            f"section={_SECTION_PROMPT.version},"
            f"boundary={PROMPT_BOUNDARY_POLICY_VERSION}"
        ),
        # Whichever model actually writes the course. A course written by a
        # weaker model must never be reused for a run configured with a
        # stronger one, so the spec is part of the identity.
        generation_model=(
            os.getenv("LLM_GENERATION", "").strip()
            or os.getenv("LLM_PRIMARY", "").strip()
            or "unset"
        ),
    )


def progress(book_id: int, message: str) -> None:
    print(f"[lecture-gen] {message}", flush=True)
    execute(
        "UPDATE books SET progress = %s, heartbeat_at = CURRENT_TIMESTAMP WHERE id = %s",
        (message, book_id),
    )


def start_heartbeat(book_id: int) -> None:
    """Keep long LLM and TTS stages distinguishable from an abandoned run."""
    def beat() -> None:
        waiter = threading.Event()
        while True:
            try:
                execute(
                    "UPDATE books SET heartbeat_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (book_id,),
                )
            except Exception as exc:  # noqa: BLE001 - status reporting must not kill work
                print(f"[lecture-gen] heartbeat warning: {exc}", flush=True)
            waiter.wait(15)

    threading.Thread(target=beat, name=f"book-{book_id}-heartbeat", daemon=True).start()


def mark_milestone(
    book_id: int,
    sid: str,
    week: int,
    stage: str,
    status: str,
    *,
    message: str | None = None,
    error: str | None = None,
    artifact_ref: str | None = None,
) -> None:
    """Upsert one durable checkpoint without disturbing completed siblings."""
    execute(
        """
        INSERT INTO course_generation_milestones
          (book_id, student_id, week, stage, status, attempt_count, progress,
           error, artifact_ref, started_at, completed_at, updated_at)
        VALUES
          (%s, %s, %s, %s, %s,
           CASE WHEN %s = 'running' THEN 1 ELSE 0 END,
           %s, %s, %s,
           CASE WHEN %s = 'running' THEN CURRENT_TIMESTAMP ELSE NULL END,
           CASE WHEN %s = 'ready' THEN CURRENT_TIMESTAMP ELSE NULL END,
           CURRENT_TIMESTAMP)
        ON CONFLICT (book_id, week, stage) DO UPDATE SET
          status = EXCLUDED.status,
          attempt_count = course_generation_milestones.attempt_count
            + CASE WHEN EXCLUDED.status = 'running' THEN 1 ELSE 0 END,
          progress = EXCLUDED.progress,
          error = EXCLUDED.error,
          artifact_ref = COALESCE(EXCLUDED.artifact_ref, course_generation_milestones.artifact_ref),
          started_at = CASE WHEN EXCLUDED.status = 'running'
            THEN CURRENT_TIMESTAMP ELSE course_generation_milestones.started_at END,
          completed_at = CASE WHEN EXCLUDED.status = 'ready'
            THEN CURRENT_TIMESTAMP
            WHEN EXCLUDED.status IN ('running', 'failed') THEN NULL
            ELSE course_generation_milestones.completed_at END,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            book_id,
            sid,
            week,
            stage,
            status,
            status,
            message,
            error,
            artifact_ref,
            status,
            status,
        ),
    )


def milestone_is_ready(book_id: int, week: int, stage: str) -> bool:
    row = fetch_one(
        """SELECT status FROM course_generation_milestones
           WHERE book_id = %s AND week = %s AND stage = %s""",
        (book_id, week, stage),
    )
    return bool(row and row.get("status") == "ready")


def refresh_book_counts(book_id: int) -> tuple[int, int]:
    row = fetch_one(
        """
        WITH core AS (
          SELECT week
          FROM course_generation_milestones
          WHERE book_id = %s AND week > 0
            AND stage IN ('lecture', 'quiz', 'slides') AND status = 'ready'
          GROUP BY week
          HAVING COUNT(DISTINCT stage) = 3
        ), audio AS (
          SELECT COUNT(*)::int AS count
          FROM course_generation_milestones
          WHERE book_id = %s AND week > 0 AND stage = 'audio' AND status = 'ready'
        )
        SELECT (SELECT COUNT(*)::int FROM core) AS core_ready,
               (SELECT count FROM audio) AS audio_ready
        """,
        (book_id, book_id),
    ) or {"core_ready": 0, "audio_ready": 0}
    core_ready = int(row.get("core_ready") or 0)
    audio_ready = int(row.get("audio_ready") or 0)
    execute(
        """UPDATE books SET generation_ready_weeks = %s,
               generation_audio_ready_weeks = %s WHERE id = %s""",
        (core_ready, audio_ready, book_id),
    )
    return core_ready, audio_ready


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- book text


def read_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """(1-based page number, text) for every page that actually has text."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = re.sub(r"[ \t]+", " ", (page.extract_text() or "")).strip()
        except Exception:
            text = ""
        # Keep short title-only pages: slide decks often introduce a new
        # chapter with nothing except its heading. Dropping those pages erased
        # real chapter boundaries and made later body bullets look authoritative.
        if len(re.sub(r"\W+", "", text)) >= 3:  # blanks and pure-image pages stay out
            pages.append((index, text))
    return pages


def build_semester_plan(
    pages: list[tuple[int, str]], book_title: str
) -> tuple[SemesterWeekPlan, list[tuple[SemesterWeek, list[tuple[int, str]]]]]:
    """Discover chapters, enforce the semester rules, and bind real pages."""
    if not pages:
        raise RuntimeError("no readable text in the book - is it scanned images?")
    inventory = discover_chapters(pages, book_title)
    plan = plan_semester(inventory)
    return plan, [(week, pages_for_week(week, pages)) for week in plan.weeks]


def write_semester_plan(sid: str, plan: SemesterWeekPlan, book_id: int | None = None) -> None:
    """Persist the generated plan in Postgres; integrated runs write no course files."""
    if execute is None:
        raise RuntimeError("database is not loaded")
    payload = plan.model_dump_json()
    if book_id is None:
        execute(
            """UPDATE books SET semester_plan = %s::jsonb
                 WHERE id = (SELECT id FROM books WHERE student_id = %s ORDER BY id DESC LIMIT 1)""",
            (payload, sid),
        )
    else:
        execute(
            "UPDATE books SET semester_plan = %s::jsonb WHERE id = %s AND student_id = %s",
            (payload, book_id, sid),
        )


def remove_obsolete_weeks(sid: str, week_count: int, book_id: int | None = None) -> None:
    """Remove invalid tail artifacts from the database after a shorter re-plan."""
    if execute is None:
        return
    if book_id is None:
        execute(
            "DELETE FROM lecture_artifacts WHERE student_id = %s AND week > %s",
            (sid, week_count),
        )
    else:
        execute(
            "DELETE FROM lecture_artifacts WHERE book_id = %s AND student_id = %s AND week > %s",
            (book_id, sid, week_count),
        )


def source_block(pages: list[tuple[int, str]]) -> str:
    """The week's pages as '[page N] ...' lines, capped for a small context.

    A real textbook's week spans 100+ pages and cannot all fit: sample pages
    evenly across the stretch, so the lecture reflects the whole week rather
    than only its first pages."""
    max_pages = max(3, MAX_SOURCE_CHARS // MAX_CHARS_PER_PAGE)
    if len(pages) > max_pages:
        step = (len(pages) - 1) / (max_pages - 1)
        pages = [pages[round(i * step)] for i in range(max_pages)]

    budget = MAX_SOURCE_CHARS
    parts: list[str] = []
    for number, text in pages:
        chunk = text[: min(MAX_CHARS_PER_PAGE, budget)]
        if not chunk:
            break
        parts.append(
            f"[page {number}]\n"
            + quote_untrusted_data(chunk, label=f"textbook-page-{number}")
        )
        budget -= len(chunk)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- LLM helpers


def parse_json(raw: str) -> dict | None:
    """Parse exactly one JSON object; malformed output goes through repair."""
    try:
        payload = json.loads(strict_json_document(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def ask_json(prompt: str, system: str, max_tokens: int, check) -> dict:
    """complete() then validate; retry with the rejection explained, and hand the
    LAST attempt to the fallback model - a repeated JSON failure is an output-
    quality problem, and availability-failover alone would never switch models."""
    generation_model = os.getenv("LLM_GENERATION", "").strip() or None
    fallback = os.getenv("LLM_FALLBACK", "").strip() or None
    repair_template = load_prompt_for(PromptOperation.SHARED_REPAIR_JSON)
    last = "no attempts made"
    current_prompt = prompt
    current_system = system
    for attempt in range(1, ATTEMPTS + 1):
        force = fallback if (attempt == ATTEMPTS and fallback) else generation_model
        try:
            result = complete(
                current_prompt,
                current_system,
                max_tokens=max_tokens,
                force_spec=force,
            )
        except LLMError as exc:
            last = f"provider failure ({type(exc).__name__})"
            continue
        data = parse_json(result.text)
        problem = check(data) if data is not None else "reply was not JSON"
        if problem is None:
            return data
        last = f"attempt {attempt}: {problem}"
        print(f"[lecture-gen] retrying - {last}", flush=True)
        # Never copy provider output into logs: rejected text can contain source
        # content, prompt fragments, or provider diagnostics. The next attempt
        # gets it only inside the shared untrusted-data boundary.
        current_prompt = repair_template.render_user(
            original_prompt=prompt,
            previous_reply=result.text[:REPAIR_REPLY_CHARS],
            validation_errors=problem,
            json_schema="Use the exact JSON shape and rules in the original prompt.",
        )
        current_system = repair_template.render_system()
    raise RuntimeError(f"model never produced valid JSON ({last})")


# ---------------------------------------------------------------- lecture generation


def check_lecture(
    data: dict, expected_slides: int | None = None, require_intro: bool = True
) -> str | None:
    expected = lecture_shape(0)["slides"] if expected_slides is None else expected_slides
    if not isinstance(data, dict):
        return "reply must be one JSON object"
    unknown_top = set(data) - {"title", "intro", "slides"}
    if unknown_top:
        return f"unknown top-level fields: {sorted(unknown_top)}"
    if (
        not isinstance(data.get("title"), str)
        or not data["title"].strip()
        or len(data["title"]) > 200
    ):
        return "missing title"
    if "intro" in data and (
        not isinstance(data["intro"], str) or len(data["intro"]) > 2_000
    ):
        return "intro must be a string under 2000 characters"
    slides = data.get("slides")
    # A couple of slides short is a trim problem, not a rejection: demanding
    # exactly N well-formed slides from a small model kills whole builds over
    # cosmetics. Structural failures still reject.
    minimum = max(1, expected - 2)
    if not isinstance(slides, list) or len(slides) < minimum:
        return f"need at least {minimum} slides"
    if len(slides) > expected:
        return f"need no more than {expected} slides"
    # Name the offending slide. This string is handed straight back to the model
    # as the repair instruction (ask_json -> validation_errors), and a batch is
    # rejected whole even when a single slide is short, so "each slide needs..."
    # told it nothing about WHICH slide to fix — every retry reproduced the same
    # fault and the week died after ATTEMPTS identical rejections.
    for position, slide in enumerate(slides[:expected], start=1):
        if not isinstance(slide, dict):
            return f"slide {position} must be one JSON object"
        unknown_slide = set(slide) - {
            "heading", "layout", "bullets", "callout", "emphasis", "visual",
            "narration", "page",
        }
        if unknown_slide:
            return f"slide {position} has unknown fields: {sorted(unknown_slide)}"
        heading = slide.get("heading")
        if not isinstance(heading, str) or not heading.strip() or len(heading) > 200:
            return f"slide {position} is missing its heading"
        where = f"slide {position} ({heading.strip()!r})"
        bullets = slide.get("bullets")
        if not isinstance(bullets, list) or not any(
            isinstance(b, str) and b.strip() for b in bullets
        ):
            return f"{where} needs at least one bullet"
        if len(bullets) > 8 or any(
            not isinstance(bullet, str) or len(bullet) > 500 for bullet in bullets
        ):
            return f"{where} has invalid or oversized bullets"
        normalised_bullets = [bullet.strip().casefold() for bullet in bullets]
        if len(normalised_bullets) != len(set(normalised_bullets)):
            return f"{where} has duplicate bullets"
        narration = slide.get("narration")
        spoken = len(narration.split()) if isinstance(narration, str) else 0
        if spoken < 15:
            return (
                f"{where} has {spoken}-word narration; rewrite only that slide's "
                "narration so it is at least 15 spoken words. Leave the others as they are."
            )
        if len(narration) > 5_000 or spoken > 750:
            return f"{where} narration exceeds the safe output limit"
        if not isinstance(slide.get("page"), int) or isinstance(slide.get("page"), bool):
            return f"{where} needs the page number it came from"
    # Only the opening batch introduces the lecture; the ones that continue it
    # are told to leave intro empty so the lecturer does not greet the room
    # again halfway through.
    if require_intro and (
        not isinstance(data.get("intro"), str) or not data["intro"].strip()
    ):
        return "missing intro"
    return None


def _slide_check(slides: int, first: bool = True):
    """check_lecture bound to one batch's slide count."""

    def check(data: dict) -> str | None:
        return check_lecture(data, expected_slides=slides, require_intro=first)

    return check


def generate_week(
    week: int,
    total_weeks: int,
    assigned_chapters: str,
    pages: list[tuple[int, str]],
) -> dict:
    shape = lecture_shape(len(pages))
    print(
        f"[lecture-gen]   lecture {week}: ~{shape['minutes']} min, {shape['slides']} slides "
        f"from {len(pages)} pages",
        flush=True,
    )

    # A 45-120 minute lecture is roughly 47-125 slides. One call cannot return that as
    # valid JSON — it runs past the context window and comes back truncated
    # mid-string — so the lecture is built a batch at a time, each batch shown
    # only its own slice of the week's pages, and the slides concatenated.
    # Every call needs at least one source page. If there are fewer pages than
    # the ideal number of calls, use fewer slightly larger batches instead of
    # dropping the final slides (or constructing an empty source slice).
    batches = max(1, min(len(pages), math.ceil(shape["slides"] / SLIDES_PER_BATCH)))
    base_slides, extra_slides = divmod(shape["slides"], batches)
    lecture: dict = {}
    for index in range(batches):
        page_start = index * len(pages) // batches
        page_end = (index + 1) * len(pages) // batches
        slice_pages = pages[page_start:page_end]
        batch_slides = base_slides + (1 if index < extra_slides else 0)
        part = _generate_batch(
            week,
            total_weeks,
            assigned_chapters,
            slice_pages,
            slides=batch_slides,
            narration=shape["narration"],
            first=index == 0,
            batch=index + 1,
            batches=batches,
        )
        if not lecture:
            lecture = part
        else:
            lecture["slides"].extend(part["slides"])
    # ask_json deliberately tolerates a model returning up to two slides short
    # per call. Never claim the original target if that happened: the assembled
    # lecture is the source of truth for downstream slides, narration and quiz.
    if not lecture.get("slides"):
        raise RuntimeError("lecture generation produced no slides")
    lecture["durationMinutes"] = shape["minutes"]
    return lecture


def _generate_batch(
    week: int,
    total_weeks: int,
    assigned_chapters: str,
    pages: list[tuple[int, str]],
    *,
    slides: int,
    narration: str,
    first: bool,
    batch: int,
    batches: int,
) -> dict:
    """One model call: `slides` slides covering only `pages`."""
    valid_pages = [number for number, _ in pages]
    intro_line = (
        '  "intro": "2 spoken sentences welcoming students and saying what this lecture covers",\n'
        if first
        else '  "intro": "",\n'
    )
    continues = (
        ""
        if first
        else f"This is part {batch} of {batches} of the same lecture: continue from where "
        "part " + str(batch - 1) + " stopped, do not re-introduce the lecture or repeat "
        "material already covered.\n"
    )
    prompt = _SLIDE_BATCH_PROMPT.render_user(
        week=week,
        total_weeks=total_weeks,
        page_start=valid_pages[0],
        page_end=valid_pages[-1],
        assigned_chapters=assigned_chapters,
        continuation=continues or "Start the lecture; do not assume prior narration.",
        slide_count=slides,
        narration_range=narration,
        intro_field=intro_line.rstrip(),
        valid_pages=valid_pages,
        slide_design=SLIDE_DESIGN_INSTRUCTIONS,
        evidence=source_block(pages),
    )
    # Give the reply room to finish: a verbose narrator ran an M-size reply out
    # of tokens at 260/slide. Only ever one batch's worth, never the lecture's.
    data = ask_json(prompt, LECTURE_SYSTEM, 800 + 340 * slides, _slide_check(slides, first))
    # "Lecture 2: Consistency Models" — the deck already says Week N, and the
    # colon broke the deck's YAML headmatter once. Strip the redundant prefix.
    data["title"] = re.sub(r"^Lecture\s*\d+\s*[:\-–—]\s*", "", data["title"].strip())
    design_problem = batch_design_problem(data["slides"][:slides])
    if design_problem:
        # Visual polish must never make grounded teaching content unavailable.
        # The normalizer below reduces malformed optional visuals to a safe,
        # readable concept/card layout and keeps narration and citations.
        print(f"[lecture-gen]   visual fallback: {design_problem}", flush=True)
    data["slides"] = [
        normalise_slide(slide, position)
        for position, slide in enumerate(data["slides"][:slides], start=1)
    ]
    for slide in data["slides"]:
        # never trust a model with page numbers: clamp to the pages it was shown
        if slide["page"] not in valid_pages:
            slide["page"] = min(valid_pages, key=lambda p: abs(p - slide["page"]))
        # coerce cosmetic bullet violations instead of failing the build
        slide["bullets"] = [b.strip() for b in slide["bullets"] if isinstance(b, str) and b.strip()][:5]
    return data


def check_quiz(minimum: int, maximum: int | None = None):
    def check(data: dict) -> str | None:
        if not isinstance(data, dict):
            return "reply must be one JSON object"
        unknown_top = set(data) - {"questions"}
        if unknown_top:
            return f"unknown top-level fields: {sorted(unknown_top)}"
        questions = data.get("questions")
        if not isinstance(questions, list) or len(questions) < minimum:
            return f"need at least {minimum} questions"
        if maximum is not None and len(questions) > maximum:
            return f"need no more than {maximum} questions"
        for position, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                return f"question {position} must be one JSON object"
            unknown_question = set(question) - {"prompt", "options", "correct"}
            if unknown_question:
                return f"question {position} has unknown fields: {sorted(unknown_question)}"
            if (
                not isinstance(question.get("prompt"), str)
                or not question["prompt"].strip()
                or len(question["prompt"]) > 1_000
            ):
                return "a question is missing its prompt"
            options = question.get("options")
            if not isinstance(options, list) or len(options) != 4:
                return "each question needs exactly 4 options"
            if not all(isinstance(o, str) and o.strip() for o in options):
                return "empty option"
            if any(len(option) > 500 for option in options):
                return "an answer option exceeds the safe output limit"
            normalised = [
                re.sub(r"^[A-Da-d][).: ]+\s*", "", option).strip().casefold()
                for option in options
            ]
            if len(normalised) != len(set(normalised)):
                return "answer options must be unique"
            if question.get("correct") not in ("A", "B", "C", "D"):
                return 'correct must be "A", "B", "C" or "D"'
        return None

    return check


def lecture_text(title: str, segments: list[dict]) -> str:
    """Everything the lecturer says, isolated as untrusted downstream data."""
    return quote_untrusted_data(
        {
            "lecture_title": title,
            "narration": [
                str(segment.get("text") or "")
                for segment in segments
                if isinstance(segment, dict)
            ],
        },
        label="lecture-evidence",
    )


def ask_questions(prompt: str, count: int, source: str, minimum: int | None = None) -> list[dict]:
    # Accept a short reply rather than failing a whole course build — a 3-minute
    # lecture honestly supports about 5 distinct easy questions, not always 8.
    data = ask_json(
        prompt,
        QUIZ_SYSTEM,
        max(1800, 300 + 160 * count),
        check_quiz(minimum or max(1, count - 2), count),
    )

    # The exam system's shape: options carry the letter label, correct_option is
    # the letter. `source` says whether the lecturer taught it or it is homework.
    questions = []
    for question in data["questions"][:count]:
        options = [
            f"{letter}) {re.sub(r'^[A-Da-d][).: ]+\\s*', '', option.strip())}"
            for letter, option in zip("ABCD", question["options"])
        ]
        questions.append(
            {
                "prompt": question["prompt"].strip(),
                "type": "mcq",
                "options": options,
                "correct_option": question["correct"],
                "source": source,
            }
        )
    return questions


def generate_quiz(
    title: str, segments: list[dict], pages: list[tuple[int, str]]
) -> list[dict]:
    shape = lecture_shape(len(pages))
    # 1) The bulk of the bank: questions a student who WATCHED the lecture finds
    #    easy — every answer must have been said out loud by the lecturer.
    taught = ask_questions(
        _QUIZ_BANK_PROMPT.render_user(
            question_count=shape["lecture_qs"],
            source_kind="lecture-grounded",
            task_guidance=(
                "Test concepts actually covered by the lecturer. A learner who understood "
                "the lecture can answer every item. Test subject matter rather than wording; "
                "do not quote narration, ask what was 'said', or use fill-in-the-blank tricks."
            ),
            evidence=lecture_text(title, segments),
        ),
        shape["lecture_qs"],
        "lecture",
        # a full quiz paper must be coverable by lecturer-taught questions
        minimum=5,
    )

    # 2) The small self-study tail: from the week's wider pages, beyond the slides.
    homework = ask_questions(
        _QUIZ_BANK_PROMPT.render_user(
            question_count=shape["self_qs"],
            source_kind="self-study",
            task_guidance=(
                "Use only the textbook pages. Prefer grounded details beyond the short "
                "lecture that a learner is expected to read independently."
            ),
            evidence=(
                quote_untrusted_data(title, label="lecture-title")
                + "\n\n"
                + source_block(pages)
            ),
        ),
        shape["self_qs"],
        "self_study",
    )
    return taught + homework


# ---------------------------------------------------------------- database artifacts


def _book_for_student(sid: str, book_id: int | None = None) -> int:
    if book_id is not None:
        return book_id
    row = fetch_one(
        "SELECT id FROM books WHERE student_id = %s ORDER BY id DESC LIMIT 1",
        (sid,),
    )
    if not row:
        raise RuntimeError(f"no book found for {sid}")
    return int(row["id"])


def lecture_artifact(sid: str, week: int, book_id: int | None = None) -> dict | None:
    if fetch_one is None:
        return None
    if book_id is None:
        return fetch_one(
            """SELECT la.* FROM lecture_artifacts la
                 JOIN books b ON b.id = la.book_id
                WHERE la.student_id = %s AND la.week = %s
                ORDER BY b.id DESC LIMIT 1""",
            (sid, week),
        )
    return fetch_one(
        "SELECT * FROM lecture_artifacts WHERE book_id = %s AND student_id = %s AND week = %s",
        (book_id, sid, week),
    )


def write_lecture(
    sid: str, week: int, lecture: dict, book_id: int | None = None
) -> None:
    """Store a complete checkpoint as JSONB under a database-generated UUID."""
    if execute is None:
        raise RuntimeError("database is not loaded")
    resolved_book_id = _book_for_student(sid, book_id)
    title = lecture["title"].strip()
    normalised_slides = [
        normalise_slide(slide, index)
        for index, slide in enumerate(lecture["slides"], start=1)
    ]
    safe_lecture = deepcopy(lecture)
    safe_lecture["slides"] = normalised_slides
    slides = {
        "week": week,
        "title": title,
        "slides": [
            {
                "slide": index + 2,
                "heading": slide["heading"].strip(),
                "bullets": [bullet.strip() for bullet in slide["bullets"]],
                "layout": slide["layout"],
                "callout": slide["callout"],
                "emphasis": slide["emphasis"],
                "visual": slide["visual"],
                "page": slide["page"],
            }
            for index, slide in enumerate(normalised_slides)
        ],
    }
    segments = [
        {"slide": 1, "text": lecture["intro"].strip(), "citations": [{"page": normalised_slides[0]["page"]}]}
    ]
    for index, slide in enumerate(normalised_slides):
        segments.append(
            {
                "slide": index + 2,
                "text": slide["narration"].strip(),
                "citations": [{"page": slide["page"]}],
            }
        )
    script = {
        "title": title,
        "durationMinutes": lecture.get("durationMinutes", LECTURE_MINUTES_MIN),
        "segments": segments,
    }
    execute(
        """
        INSERT INTO lecture_artifacts
          (artifact_id, book_id, student_id, week, title, lecture_payload,
           script_payload, slides_payload, created_at, updated_at)
        SELECT generated_id, %s, %s, %s, %s, %s::jsonb,
               %s::jsonb || jsonb_build_object('lectureId', generated_id::text),
               %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
          FROM (SELECT gen_random_uuid() AS generated_id) generated
        ON CONFLICT (book_id, week) DO UPDATE SET
          student_id = EXCLUDED.student_id,
          title = EXCLUDED.title,
          lecture_payload = EXCLUDED.lecture_payload,
          script_payload = (EXCLUDED.script_payload - 'lectureId')
            || jsonb_build_object('lectureId', lecture_artifacts.artifact_id::text),
          slides_payload = EXCLUDED.slides_payload,
          quiz_payload = NULL,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            resolved_book_id,
            sid,
            week,
            title,
            json.dumps(safe_lecture, ensure_ascii=False),
            json.dumps(script, ensure_ascii=False),
            json.dumps(slides, ensure_ascii=False),
        ),
    )


def write_quiz(
    sid: str,
    week: int,
    title: str,
    quiz: list[dict],
    book_id: int | None = None,
) -> None:
    resolved_book_id = _book_for_student(sid, book_id)
    # Assessment banks are learner-owned even when the lecture was adopted
    # from a content-addressed donor. The generation id proves this payload was
    # materialized for this learner instead of copied verbatim.
    payload = {
        "schema_version": "learner-assessment-bank-v1",
        "owner_student_id": sid,
        "owner_book_id": resolved_book_id,
        "generation_id": str(uuid4()),
        "week": week,
        "title": title.strip(),
        "questions": quiz,
    }
    execute(
        """UPDATE lecture_artifacts
              SET quiz_payload = %s::jsonb, updated_at = CURRENT_TIMESTAMP
            WHERE book_id = %s AND student_id = %s AND week = %s""",
        (
            json.dumps(payload, ensure_ascii=False),
            resolved_book_id,
            sid,
            week,
        ),
    )


def generate_and_store_section(
    sid: str,
    book_id: int,
    week: int,
    lecture_title: str,
    *,
    focus: str,
) -> tuple[str, str | None]:
    """Generate one grounded section and persist it for the approved plan version.

    A refusal is durable generation state but produces no timetable entry.
    """
    programme = fetch_one(
        """SELECT id, name, plan_version, collection_id FROM programmes
             WHERE student_id = %s AND status = 'approved'
             ORDER BY id DESC LIMIT 1""",
        (sid,),
    )
    artifact = lecture_artifact(sid, week, book_id)
    if not programme:
        return "refused", "No approved programme version exists for this learner."
    if not artifact:
        raise RuntimeError("a lecture artifact is required before section generation")

    programme_id = str(programme["id"])
    plan_version = int(programme["plan_version"])
    topic_id = str(artifact["artifact_id"])
    identity = SectionIdentity(
        programme_title=str(programme["name"]),
        plan_schema="programme-plan-v1",
        plan_version=str(plan_version),
        user_id=sid,
        collection_id=str(programme["collection_id"]),
        course_id=f"book-{book_id}",
        week_number=week,
        topic_id=topic_id,
        lecture_title=lecture_title,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    # The section practises what the lecture taught, so it is grounded on that
    # lecture's own slide headings rather than on the week's title alone.
    slides = artifact.get("slides_payload") or {}
    topics = [
        str(slide.get("heading") or "").strip()
        for slide in (slides.get("slides") or [])
        if isinstance(slide, dict) and str(slide.get("heading") or "").strip()
    ]
    def complete_section(prompt: str) -> str:
        roles = split_prompt_roles(prompt)
        if roles is None:
            raise ValueError("section prompt is missing its trusted role boundary")
        system, user_task = roles
        return complete(user_task, system=system, max_tokens=4000).text

    run = generate_section_pack(
        llm=complete_section,
        identity=identity,
        tool_context=ToolContext(),
        focus=focus,
        topics=topics,
        on_call=lambda: progress(book_id, f"Writing grounded section {week}…"),
    )
    if run.section is None:
        execute(
            """DELETE FROM section_packs
                WHERE tenant_id = %s AND programme_id = %s
                  AND approved_plan_version = %s AND week = %s""",
            (sid, programme_id, plan_version, week),
        )
        reason = run.refusal.reason if run.refusal else "Section grounding refused"
        return "refused", reason

    payload = run.section.model_dump_json()
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    execute(
        """
        INSERT INTO section_packs
          (schema_version, tenant_id, programme_id, course_id, week, lecture_id,
           approved_plan_id, approved_plan_version, prompt_id, prompt_version,
           payload_hash, pack_payload, created_at)
        VALUES
          ('section-pack-v1', %s, %s, %s, %s, %s, %s, %s, %s, %s,
           %s, %s::jsonb, CURRENT_TIMESTAMP)
        ON CONFLICT (tenant_id, approved_plan_id, approved_plan_version, week)
        DO UPDATE SET
          programme_id = EXCLUDED.programme_id,
          course_id = EXCLUDED.course_id,
          lecture_id = EXCLUDED.lecture_id,
          prompt_id = EXCLUDED.prompt_id,
          prompt_version = EXCLUDED.prompt_version,
          payload_hash = EXCLUDED.payload_hash,
          pack_payload = EXCLUDED.pack_payload,
          created_at = CURRENT_TIMESTAMP
        """,
        (
            sid,
            programme_id,
            f"book-{book_id}",
            week,
            topic_id,
            programme_id,
            plan_version,
            run.prompt_id,
            run.prompt_version,
            payload_hash,
            payload,
        ),
    )
    return "ready", None


def section_checkpoint(
    sid: str, book_id: int, week: int
) -> dict | None:
    return fetch_one(
        """SELECT sp.section_pack_id FROM section_packs sp
             JOIN programmes p ON p.id::text = sp.programme_id
            WHERE sp.tenant_id = %s AND sp.course_id = %s AND sp.week = %s
              AND p.status = 'approved'
              AND sp.approved_plan_version = p.plan_version
            ORDER BY sp.created_at DESC LIMIT 1""",
        (sid, f"book-{book_id}", week),
    )


def register_week_artifacts(sid: str, week: int, book_id: int | None = None) -> None:
    if execute is None:
        return
    resolved_book_id = _book_for_student(sid, book_id)
    execute(
        """
        UPDATE lectures
        SET lecture_artifact_id = (
          SELECT artifact_id FROM lecture_artifacts
           WHERE book_id = %s AND student_id = %s AND week = %s
        ), book_id = %s
        WHERE student_id = %s AND week = %s
        """,
        (resolved_book_id, sid, week, resolved_book_id, sid, week),
    )


def write_week(
    sid: str, week: int, lecture: dict, quiz: list[dict], book_id: int | None = None
) -> None:
    """Compatibility helper used by focused callers; checkpoints write separately."""
    write_lecture(sid, week, lecture, book_id)
    write_quiz(sid, week, lecture["title"], quiz, book_id)
    register_week_artifacts(sid, week, book_id)



def _slidev_cache_dir(artifact_id: str) -> Path:
    return ROOT / ".cache" / "slidev" / artifact_id


def _slidev_markdown(deck: dict) -> str:
    """Compile the database deck into Slidev input without making it canonical."""
    return render_slidev_markdown(deck)


def build_slides(sid: str, week: int | None = None, book_id: int | None = None) -> None:
    """Build a disposable Slidev cache from the database-owned slide payload."""
    if week is None:
        return
    row = lecture_artifact(sid, week, book_id)
    if not row or not row.get("slides_payload"):
        raise RuntimeError(f"week {week} slide artifact is missing")
    artifact_id = str(row["artifact_id"])
    target = _slidev_cache_dir(artifact_id).resolve()
    cache_root = (ROOT / ".cache" / "slidev").resolve()
    if cache_root not in target.parents:
        raise RuntimeError("invalid Slidev cache target")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    executable = ROOT / "node_modules" / ".bin" / (
        "slidev.cmd" if os.name == "nt" else "slidev"
    )
    if not executable.is_file():
        raise RuntimeError("Slidev is not installed; run npm install from the UnivAI root")
    temp_root = ROOT / ".cache"
    temp_root.mkdir(parents=True, exist_ok=True)
    # Keep the transient deck under the repository cache so Slidev can resolve
    # the root-installed theme by walking up to node_modules.
    with tempfile.TemporaryDirectory(prefix="univai-slidev-", dir=temp_root) as temp_dir:
        deck_path = Path(temp_dir) / "slides.md"
        deck_path.write_text(_slidev_markdown(row["slides_payload"]), encoding="utf-8")
        subprocess.run(
            [
                str(executable),
                "build",
                str(deck_path),
                "--out",
                str(target),
                "--base",
                f"/api/presentation/{artifact_id}/",
            ],
            cwd=ROOT,
            check=True,
        )


def valid_lecture_checkpoint(
    sid: str, week: int, book_id: int | None = None
) -> bool:
    row = lecture_artifact(sid, week, book_id)
    if not row:
        return False
    script = row.get("script_payload")
    lecture = row.get("lecture_payload")
    slides = row.get("slides_payload")
    return bool(
        isinstance(script, dict)
        and isinstance(script.get("segments"), list)
        and script["segments"]
        and isinstance(lecture, dict)
        and isinstance(lecture.get("slides"), list)
        and isinstance(slides, dict)
    )


def valid_quiz_checkpoint(sid: str, week: int, book_id: int | None = None) -> bool:
    row = lecture_artifact(sid, week, book_id)
    quiz = row.get("quiz_payload") if row else None
    resolved_book_id = (
        int(row["book_id"])
        if row and row.get("book_id") is not None
        else book_id
    )
    return bool(
        isinstance(quiz, dict)
        and quiz.get("schema_version") == "learner-assessment-bank-v1"
        and quiz.get("owner_student_id") == sid
        and quiz.get("owner_book_id") == resolved_book_id
        and isinstance(quiz.get("generation_id"), str)
        and quiz["generation_id"].strip()
        and isinstance(quiz.get("questions"), list)
        and quiz["questions"]
    )


def valid_slides_checkpoint(sid: str, week: int, book_id: int | None = None) -> bool:
    row = lecture_artifact(sid, week, book_id)
    return bool(
        row
        and isinstance(row.get("slides_payload"), dict)
        and (_slidev_cache_dir(str(row["artifact_id"])) / "index.html").is_file()
    )


def prepare_generation_manifest(
    sid: str,
    book_id: int,
    source_sha256: str,
    total_weeks: int,
    course_fingerprint: str | None = None,
) -> bool:
    """Persist and compare the resumable generation manifest in Postgres."""
    row = fetch_one(
        "SELECT generation_manifest FROM books WHERE id = %s AND student_id = %s",
        (book_id, sid),
    )
    existing = row.get("generation_manifest") if row else None
    matching = bool(existing and existing.get("source_sha256") == source_sha256)
    manifest = {
        "schema_version": 3,
        "book_id": book_id,
        "source_sha256": source_sha256,
        "course_fingerprint": course_fingerprint,
        "total_weeks": total_weeks,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    execute(
        "UPDATE books SET generation_manifest = %s::jsonb WHERE id = %s AND student_id = %s",
        (json.dumps(manifest), book_id, sid),
    )
    return matching


# ── Course adoption ───────────────────────────────────────────────────
#
# The same textbook uploaded by a second learner must not be written twice. A
# course is a pure function of the bytes it was built from and the pipeline
# that built it, so when both match a finished course the content is already
# correct for the new learner and is copied instead of regenerated.
#
# Copied: lectures, narration, slides, section packs — shared teaching assets.
# Never copied: quiz banks, cumulative papers, attendance, attempts, or scores.
# Assessments are regenerated and stamped for the adopting learner after the
# teaching assets land, while the expensive lecture/slide work stays reused.


def normalize_reused_plan(plan: dict) -> dict:
    """Upgrade reusable plan metadata without changing its teaching weeks."""
    normalized = deepcopy(plan)
    for semester in normalized.get("semesters") or []:
        week_count = int(semester.get("week_count") or 0)
        if week_count < 1:
            continue
        midpoint = (week_count + 1) // 2
        semester["midterms"] = [{
            "number": 1,
            "after_week": midpoint,
            "covers_weeks": list(range(1, midpoint + 1)),
        }]
    return normalized


def find_reusable_course(
    sid: str,
    book_id: int,
    source_sha256: str,
    course_fingerprint: str,
    semester_plan: dict | None,
) -> dict | None:
    """Find a finished course built from these exact bytes by this pipeline.

    Every condition here is a reason the copy would otherwise be wrong:
    different bytes mean a different book, a different fingerprint means the
    prompts or lecture shape moved on, and a different semester plan means the
    weeks would not line up with the curriculum this learner approved.
    """
    # 'partial' is a statement about narration bookkeeping, not about teaching:
    # a course whose weeks are all written sits at 'partial' until its audio
    # counters agree. Completeness is therefore decided below, per week, from
    # the artifacts themselves — the only evidence that cannot be stale.
    #
    # ORDER BY b.id ascending is the tie-break, and it is deliberate: when the
    # same book has been taught more than once, the earliest course wins. That
    # is the one the most learners already hold, so a class converges on a
    # single shared course instead of splintering every time an admin
    # regenerates one learner's copy. Ordering by recency would hand each new
    # learner whichever version happened to be rebuilt last.
    donors = fetch_all(
        """SELECT b.id, b.student_id, b.pages,
                  b.generation_total_weeks AS total_weeks,
                  b.generation_manifest, b.semester_plan
             FROM books b
            WHERE b.source_sha256 = %s
              AND b.id <> %s
              AND b.status IN ('ready', 'partial')
              AND b.generation_total_weeks > 0
              AND b.semester_plan IS NOT NULL
            ORDER BY b.id ASC""",
        (source_sha256, book_id),
    )
    for donor in donors or []:
        manifest = donor.get("generation_manifest") or {}
        if manifest.get("course_fingerprint") != course_fingerprint:
            continue
        if (
            semester_plan is not None
            and normalize_reused_plan(donor.get("semester_plan") or {})
            != normalize_reused_plan(semester_plan)
        ):
            continue
        total_weeks = int(donor.get("total_weeks") or 0)
        complete_weeks = fetch_one(
            """SELECT count(*) AS ready FROM lecture_artifacts
                WHERE book_id = %s
                  AND jsonb_array_length(COALESCE(lecture_payload->'slides','[]'::jsonb)) > 0
                  AND jsonb_array_length(COALESCE(script_payload->'segments','[]'::jsonb)) > 0""",
            (donor["id"],),
        )
        if int((complete_weeks or {}).get("ready") or 0) < total_weeks:
            continue
        return donor
    return None


def course_reuse_allowed(sid: str, *, quizzes_only: bool, no_reuse: bool) -> bool:
    """Whether this run may take an existing course instead of writing one.

    Reuse is right for a learner meeting a book someone else has already been
    taught. It is wrong when an admin has explicitly asked to regenerate
    (``--no-reuse``), because returning a copy would make that request a silent
    no-op, and wrong when the learner reshaped their curriculum, because the
    existing course no longer describes what they approved.
    """
    if quizzes_only or no_reuse:
        return False
    return not learner_has_edited_curriculum(sid)


def learner_has_edited_curriculum(sid: str) -> bool:
    """A reshaped curriculum earns a real build, not somebody else's course."""
    programme = fetch_one(
        """SELECT plan_version FROM programmes
            WHERE student_id = %s AND status = 'approved'
            ORDER BY id DESC LIMIT 1""",
        (sid,),
    )
    return bool(programme and int(programme.get("plan_version") or 1) > 1)


def _reuse_slidev_cache(donor_artifact_id: str, artifact_id: str) -> bool:
    """Copy a built deck and re-key its embedded base URL for the adopter.

    Slidev writes the artifact id into HTML and JavaScript assets. A byte-for-
    byte copy therefore keeps requesting the donor's protected presentation,
    which the adopting learner cannot access. Use independent files (not hard
    links) so replacing that id never mutates the donor build.
    """
    donor_dir = _slidev_cache_dir(donor_artifact_id)
    if not (donor_dir / "index.html").is_file():
        return False
    target = _slidev_cache_dir(artifact_id)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(donor_dir, target)
        donor_id = donor_artifact_id.encode("utf-8")
        adopted_id = artifact_id.encode("utf-8")
        for copied in target.rglob("*"):
            if not copied.is_file():
                continue
            body = copied.read_bytes()
            if donor_id in body:
                copied.write_bytes(body.replace(donor_id, adopted_id))
    except OSError:
        shutil.rmtree(target, ignore_errors=True)
        return False
    return (target / "index.html").is_file()


def adopt_course(sid: str, book_id: int, donor: dict) -> int:
    """Copy a finished course onto this learner and return the weeks adopted."""
    donor_id = int(donor["id"])
    weeks = fetch_all(
        """SELECT artifact_id::text AS artifact_id, week, title, lecture_payload,
                  script_payload, slides_payload
             FROM lecture_artifacts WHERE book_id = %s ORDER BY week""",
        (donor_id,),
    )
    adopted = 0
    for row in weeks or []:
        week = int(row["week"])
        # A fresh artifact id per learner: the row is theirs, and script
        # payloads carry their own lectureId so nothing points back at the
        # donor. Everything else about the teaching is byte-identical.
        execute(
            """
            INSERT INTO lecture_artifacts
              (artifact_id, book_id, student_id, week, title, lecture_payload,
               script_payload, slides_payload, quiz_payload, created_at, updated_at)
            SELECT generated_id, %s, %s, %s, %s, %s::jsonb,
                   (%s::jsonb - 'lectureId')
                     || jsonb_build_object('lectureId', generated_id::text),
                   %s::jsonb, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
              FROM (SELECT gen_random_uuid() AS generated_id) generated
            ON CONFLICT (book_id, week) DO UPDATE SET
              student_id = EXCLUDED.student_id,
              title = EXCLUDED.title,
              lecture_payload = EXCLUDED.lecture_payload,
              script_payload = (EXCLUDED.script_payload - 'lectureId')
                || jsonb_build_object('lectureId', lecture_artifacts.artifact_id::text),
              slides_payload = EXCLUDED.slides_payload,
              quiz_payload = CASE
                WHEN lecture_artifacts.quiz_payload->>'schema_version'
                       = 'learner-assessment-bank-v1'
                 AND lecture_artifacts.quiz_payload->>'owner_student_id'
                       = EXCLUDED.student_id
                 AND lecture_artifacts.quiz_payload->>'owner_book_id'
                       = EXCLUDED.book_id::text
                 AND COALESCE(lecture_artifacts.quiz_payload->>'generation_id', '') <> ''
                THEN lecture_artifacts.quiz_payload
                ELSE NULL
              END,
              updated_at = CURRENT_TIMESTAMP
            """,
            (
                book_id,
                sid,
                week,
                row["title"],
                json.dumps(row["lecture_payload"], ensure_ascii=False),
                json.dumps(row["script_payload"], ensure_ascii=False),
                json.dumps(row["slides_payload"], ensure_ascii=False),
            ),
        )
        register_week_artifacts(sid, week, book_id)
        mark_milestone(
            book_id,
            sid,
            week,
            "lecture",
            "ready",
            message="Adopted from an identical book",
            artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}",
        )
        saved = lecture_artifact(sid, week, book_id)
        if saved and _reuse_slidev_cache(row["artifact_id"], str(saved["artifact_id"])):
            mark_milestone(
                book_id,
                sid,
                week,
                "slides",
                "ready",
                message="Adopted a built deck",
                artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:slides",
            )
        else:
            # The deck is derived from slides_payload, so a missing donor build
            # costs a local Slidev run, never a regenerated lecture.
            try:
                build_slides(sid, week, book_id)
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "slides",
                    "ready",
                    message="Slide deck built",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:slides",
                )
            except Exception as error:  # noqa: BLE001 - a deck must not sink the course
                mark_milestone(
                    book_id, sid, week, "slides", "failed", message=str(error)[:400]
                )
        mark_milestone(
            book_id,
            sid,
            week,
            "audio",
            "ready",
            message="On-demand lecture voice ready",
            artifact_ref="runtime:live-tts",
        )
        adopted += 1
    adopt_section_packs(sid, book_id, donor_id)
    return adopted


def rekey_section_payload(
    payload: dict,
    *,
    sid: str,
    book_id: int,
    topic_id: str,
    programme_title: str,
    plan_version: int,
) -> dict:
    """Make a copied pack describe the learner who is about to be taught it.

    Live re-reads the pack's own identity and refuses when it disagrees with
    the row it arrived in (protocols/section_session.py). Copying the payload
    unchanged therefore produced a section that failed its own contract at join
    time with ``section_artifact_unavailable`` — the columns said one learner
    and the payload still said another.
    """
    rekeyed = dict(payload)
    rekeyed["user_id"] = sid
    rekeyed["course_id"] = f"book-{book_id}"
    rekeyed["topic_id"] = topic_id
    rekeyed["programme_title"] = programme_title
    rekeyed["plan_version"] = str(plan_version)
    return rekeyed


def adopt_section_packs(sid: str, book_id: int, donor_id: int) -> None:
    """Re-key the donor's grounded sections onto this learner's approved plan.

    Section packs are addressed by tenant and approved plan version, not by
    book, so a copy has to be rewritten into the adopting learner's programme
    or the app would never find it.
    """
    programme = fetch_one(
        """SELECT id, name, plan_version FROM programmes
            WHERE student_id = %s AND status = 'approved'
            ORDER BY id DESC LIMIT 1""",
        (sid,),
    )
    if not programme:
        return
    programme_id = str(programme["id"])
    programme_title = str(programme["name"])
    plan_version = int(programme["plan_version"])
    packs = fetch_all(
        """SELECT week, prompt_id, prompt_version, payload_hash, pack_payload
             FROM section_packs WHERE course_id = %s ORDER BY week""",
        (f"book-{donor_id}",),
    )
    for pack in packs or []:
        week = int(pack["week"])
        artifact = lecture_artifact(sid, week, book_id)
        if not artifact:
            continue
        payload = rekey_section_payload(
            pack["pack_payload"],
            sid=sid,
            book_id=book_id,
            topic_id=str(artifact["artifact_id"]),
            programme_title=programme_title,
            plan_version=plan_version,
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        # The hash covers the payload as stored, so it has to follow the re-key.
        payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        execute(
            """
            INSERT INTO section_packs
              (schema_version, tenant_id, programme_id, course_id, week, lecture_id,
               approved_plan_id, approved_plan_version, prompt_id, prompt_version,
               payload_hash, pack_payload, created_at)
            VALUES
              ('section-pack-v1', %s, %s, %s, %s, %s, %s, %s, %s, %s,
               %s, %s::jsonb, CURRENT_TIMESTAMP)
            ON CONFLICT (tenant_id, approved_plan_id, approved_plan_version, week)
            DO UPDATE SET
              programme_id = EXCLUDED.programme_id,
              course_id = EXCLUDED.course_id,
              lecture_id = EXCLUDED.lecture_id,
              prompt_id = EXCLUDED.prompt_id,
              prompt_version = EXCLUDED.prompt_version,
              payload_hash = EXCLUDED.payload_hash,
              pack_payload = EXCLUDED.pack_payload,
              created_at = CURRENT_TIMESTAMP
            """,
            (
                sid,
                programme_id,
                f"book-{book_id}",
                week,
                str(artifact["artifact_id"]),
                programme_id,
                plan_version,
                pack["prompt_id"],
                pack["prompt_version"],
                payload_hash,
                serialized,
            ),
        )
        mark_milestone(
            book_id,
            sid,
            week,
            "section",
            "ready",
            message="Adopted a grounded section",
            artifact_ref=f"db:section_packs:{book_id}:week:{week}",
        )


def warm_narration_cache(book_id: int) -> bool:
    """Render this course's narration into the shared cache, in the background.

    Live synthesizes on demand, so a cold cache costs the first lecture its
    opening sentences rather than its voice. Warming it here moves that cost
    off the learner. Detached and best-effort by design: the course is already
    finished and usable, and a missing voice model must not fail it.
    """
    script = ROOT / "UnivAI-live" / "prerender_audio.py"
    python = ROOT / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not script.is_file() or not python.is_file():
        return False
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with open(logs / "prerender-audio.log", "a", encoding="utf-8") as log:
            subprocess.Popen(
                [str(python), str(script), "--book", str(book_id)],
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=os.name != "nt",
            )
    except OSError as error:
        print(f"[lecture-gen] narration warm-up did not start: {error}", flush=True)
        return False
    print(f"[lecture-gen] warming narration cache for book {book_id}", flush=True)
    return True


def initialize_milestones(book_id: int, sid: str, total_weeks: int) -> None:
    mark_milestone(book_id, sid, 0, "plan", "ready", message="Course plan saved")
    execute(
        """
        INSERT INTO course_generation_milestones
          (book_id, student_id, week, stage, status, updated_at)
        SELECT %s, %s, generated_week, stage, 'pending', CURRENT_TIMESTAMP
        FROM generate_series(1, %s) AS generated_week
        CROSS JOIN unnest(ARRAY['lecture', 'quiz', 'slides', 'section', 'audio']) AS stage
        ON CONFLICT (book_id, week, stage) DO NOTHING
        """,
        (book_id, sid, total_weeks),
    )


def update_book_state(
    book_id: int,
    status: str,
    stage: str,
    message: str,
    *,
    error: str | None = None,
) -> None:
    execute(
        """UPDATE books SET status = %s, generation_stage = %s,
               progress = %s, error = %s WHERE id = %s""",
        (status, stage, message, error, book_id),
    )


# ---------------------------------------------------------------- main


def regenerate_quizzes(
    sid: str,
    book_id: int,
    weeks: list[tuple[SemesterWeek, list[tuple[int, str]]]],
) -> None:
    """Rewrite only quiz JSONB per week from the saved lecture scripts."""
    total_weeks = len(weeks)
    for planned_week, week_pages in weeks:
        week = planned_week.week
        row = lecture_artifact(sid, week, book_id)
        script = row.get("script_payload") if row else None
        if not isinstance(script, dict):
            raise RuntimeError(f"week {week} lecture checkpoint is missing")
        message = (
            f"Writing learner assessment bank {week} of {total_weeks} "
            f"— “{script['title']}”…"
        )
        progress(book_id, message)
        mark_milestone(book_id, sid, week, "quiz", "running", message=message)
        quiz = generate_quiz(script["title"], script["segments"], week_pages)
        write_quiz(sid, week, script["title"], quiz, book_id)
        mark_milestone(
            book_id,
            sid,
            week,
            "quiz",
            "ready",
            message="Learner-owned assessment bank saved",
            artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:quiz",
        )
        register_week_artifacts(sid, week, book_id)


def assessment_weeks_for_plan(
    plan: SemesterWeekPlan,
    pdf_path: Path,
) -> list[tuple[SemesterWeek, list[tuple[int, str]]]]:
    """Bind an adopted plan to this upload's pages for fresh assessments."""
    pages = read_pages(pdf_path)
    if not pages:
        raise RuntimeError("no readable text in the book - is it scanned images?")
    return [(week, pages_for_week(week, pages)) for week in plan.weeks]


def adopt_course_with_learner_assessments(
    sid: str,
    book_id: int,
    donor: dict,
    plan: SemesterWeekPlan,
    pdf_path: Path,
) -> int:
    """Reuse teaching, then generate only this learner's missing quiz banks."""
    adopted = adopt_course(sid, book_id, donor)
    if not adopted:
        return 0
    missing = [
        item
        for item in assessment_weeks_for_plan(plan, pdf_path)
        if not valid_quiz_checkpoint(sid, item[0].week, book_id)
    ]
    if missing:
        regenerate_quizzes(sid, book_id, missing)
    return adopted


def main() -> int:
    if "--standalone" in sys.argv[1:]:
        import argparse

        os.environ["UNIVAI_MODE"] = "standalone"
        from runtime import runtime_mode, standalone_root
        from standalone_generation import generate_course

        runtime_mode()
        parser = argparse.ArgumentParser()
        parser.add_argument("--standalone", action="store_true")
        parser.add_argument(
            "--source",
            type=Path,
            default=AGENT_ROOT / "fixtures" / "sample_course.md",
        )
        parser.add_argument("--output-root", type=Path)
        args = parser.parse_args()
        source = args.source.resolve()
        output = args.output_root.resolve() if args.output_root else standalone_root() / "output"
        generate_course(source, output)
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "standalone",
                    "weeks": run["weeks"],
                    "output": str(output),
                    "side_effects": "database and integrated voice services skipped",
                }
            )
        )
        return 0

    load_integrated_dependencies()
    if len(sys.argv) < 3:
        print(json.dumps({
            "ok": False,
            "error": (
                "usage: lecture_gen.py <pdf_path> <book_id> "
                "[--quizzes-only|--plan-only] [--no-reuse]"
            ),
        }))
        return 2
    pdf_path = Path(sys.argv[1]).resolve()
    book_id = int(sys.argv[2])
    quizzes_only = "--quizzes-only" in sys.argv[3:]
    # Write this book's course even if an identical one already exists. The
    # admin's "Regenerate course" asks for new content, not a copy of it.
    no_reuse = "--no-reuse" in sys.argv[3:]
    # Discover the chapters and stop. The curriculum a learner approves is built
    # from semester-plan.json, so the plan has to exist before they can approve
    # anything — but writing lectures, quizzes, slides and voice for a course
    # they have not agreed to is hours of work thrown away the moment they
    # reshape it. Planning is pure Python (no model calls), so this pass is
    # cheap; the expensive stages wait for approval to spawn the full run.
    plan_only = "--plan-only" in sys.argv[3:]

    book = fetch_one(
        """SELECT id, student_id, title, filename, status, pages, source_sha256,
                  generation_total_weeks, generation_ready_weeks,
                  generation_manifest, semester_plan
           FROM books WHERE id = %s""",
        (book_id,),
    )
    if not book:
        print(json.dumps({"ok": False, "error": f"no book with id {book_id}"}))
        return 2
    start_heartbeat(book_id)
    # The owner. Every database write below is namespaced to this student.
    sid = book.get("student_id")
    if not sid:
        print(json.dumps({"ok": False, "error": f"book {book_id} has no owner (student_id)"}))
        return 2

    # No size dial: each week's lecture is sized from its own material in
    # generate_week, so a week carrying more chapters is simply a longer lecture.
    print(
        f"[lecture-gen] lectures run {LECTURE_MINUTES_MIN}-{LECTURE_MINUTES_MAX} min, "
        "sized per week from its pages",
        flush=True,
    )

    active_week = 0
    active_stage = "plan"
    total_weeks = 0
    try:
        source_sha256 = file_sha256(pdf_path)
        fingerprint = course_components().fingerprint()
        saved_manifest = book.get("generation_manifest")
        saved_total = int(book.get("generation_total_weeks") or 0)
        # --no-reuse has to switch off resuming this book's OWN checkpoints too,
        # not just adopting another learner's course. A finished book resumes
        # every week from the artifacts it already has, so without this an
        # admin's regeneration would skip straight past generation and rewrite
        # nothing — the same silent no-op by a different route.
        published_course_resume = bool(
            not quizzes_only
            and not no_reuse
            and saved_total > 0
            and int(book.get("generation_ready_weeks") or 0) >= saved_total
            and book.get("source_sha256") == source_sha256
            and saved_manifest
            and saved_manifest.get("source_sha256") == source_sha256
            and book.get("semester_plan")
        )
        reusable_donor = None
        if (
            not published_course_resume
            and course_reuse_allowed(sid, quizzes_only=quizzes_only, no_reuse=no_reuse)
        ):
            reusable_donor = find_reusable_course(
                sid,
                book_id,
                source_sha256,
                fingerprint,
                book.get("semester_plan"),
            )

        if published_course_resume:
            total_weeks = saved_total
            page_count = int(book.get("pages") or 0)
            plan = SemesterWeekPlan.model_validate(
                normalize_reused_plan(book["semester_plan"])
            )
            # Payloads written before learner ownership was enforced are not
            # trusted checkpoints. A resume keeps the lectures but rebuilds
            # every assessment bank from this learner's own upload.
            weeks = (
                []
                if all(
                    valid_quiz_checkpoint(sid, week.week, book_id)
                    for week in plan.weeks
                )
                else assessment_weeks_for_plan(plan, pdf_path)
            )
            resume_artifacts = True
            message = f"Reusing {total_weeks} published lecture checkpoints…"
            update_book_state(book_id, "generating", "resuming", message)
            progress(book_id, message)
            initialize_milestones(book_id, sid, total_weeks)
        elif reusable_donor:
            # The bytes and complete pipeline fingerprint match a finished
            # course. Reuse its verified plan directly: no PDF parsing, chapter
            # discovery, or planning work is repeated for this learner.
            plan = SemesterWeekPlan.model_validate(
                normalize_reused_plan(reusable_donor["semester_plan"])
            )
            weeks = []
            total_weeks = plan.week_count
            page_count = int(reusable_donor.get("pages") or 0)
            message = f"Found this book already taught — reusing its {total_weeks}-week plan…"
            update_book_state(book_id, "generating", "planning", message)
            progress(book_id, message)
            execute(
                "UPDATE books SET pages = %s, source_sha256 = %s WHERE id = %s",
                (page_count, source_sha256, book_id),
            )
            write_semester_plan(sid, plan, book_id)
            remove_obsolete_weeks(sid, total_weeks, book_id)
            resume_artifacts = prepare_generation_manifest(
                sid, book_id, source_sha256, total_weeks, fingerprint
            ) and not no_reuse
            execute(
                """UPDATE books SET generation_total_weeks = %s,
                       generation_stage = 'content', error = NULL WHERE id = %s""",
                (total_weeks, book_id),
            )
            initialize_milestones(book_id, sid, total_weeks)
        else:
            update_book_state(book_id, "generating", "reading", "Reading the book…")
            progress(book_id, "Reading the book…")
            pages = read_pages(pdf_path)
            page_count = len(pages)
            execute(
                "UPDATE books SET pages = %s, source_sha256 = %s WHERE id = %s",
                (page_count, source_sha256, book_id),
            )
            book_title = book.get("title") or book.get("filename") or pdf_path.stem
            update_book_state(
                book_id,
                "generating",
                "planning",
                "Finding chapters and planning the course…",
            )
            progress(book_id, "Finding chapters and planning the course…")
            plan, weeks = build_semester_plan(pages, book_title)
            total_weeks = plan.week_count
            write_semester_plan(sid, plan, book_id)
            remove_obsolete_weeks(sid, total_weeks, book_id)
            resume_artifacts = prepare_generation_manifest(
                sid, book_id, source_sha256, total_weeks, fingerprint
            ) and not no_reuse
            execute(
                """UPDATE books SET generation_total_weeks = %s,
                       generation_stage = 'content', error = NULL WHERE id = %s""",
                (total_weeks, book_id),
            )
            initialize_milestones(book_id, sid, total_weeks)

        if plan_only:
            update_book_state(
                book_id,
                "awaiting_approval",
                "plan",
                f"Course plan ready — {total_weeks} weeks. "
                "Approve your curriculum to start building the lectures.",
            )
            print(
                f"[lecture-gen] plan only: {total_weeks} weeks discovered, "
                "waiting for curriculum approval",
                flush=True,
            )
            print(json.dumps({"ok": True, "weeks": total_weeks, "plan_only": True}))
            return 0

        # The learner has approved a curriculum built from this plan. If another
        # learner already has a finished course from the same bytes and the same
        # pipeline, that course IS this course — writing it again would cost a
        # full run of model calls to arrive somewhere no better.
        #
        if course_reuse_allowed(sid, quizzes_only=quizzes_only, no_reuse=no_reuse):
            plan_row = fetch_one("SELECT semester_plan FROM books WHERE id = %s", (book_id,))
            donor = reusable_donor or find_reusable_course(
                sid, book_id, source_sha256, fingerprint,
                (plan_row or {}).get("semester_plan"),
            )
            if donor:
                message = f"Found this book already taught — reusing {donor['total_weeks']} weeks…"
                update_book_state(book_id, "generating", "adopting", message)
                progress(book_id, message)
                adopted_plan = SemesterWeekPlan.model_validate(
                    normalize_reused_plan(
                        (plan_row or {}).get("semester_plan")
                        or donor["semester_plan"]
                    )
                )
                active_stage = "quiz"
                adopted = adopt_course_with_learner_assessments(
                    sid,
                    book_id,
                    donor,
                    adopted_plan,
                    pdf_path,
                )
                if adopted:
                    total_weeks = adopted
                    execute(
                        """UPDATE books SET generation_total_weeks = %s,
                               generation_stage = 'content', error = NULL WHERE id = %s""",
                        (total_weeks, book_id),
                    )
                    core_ready, audio_ready = refresh_book_counts(book_id)
                    update_book_state(
                        book_id,
                        "ready",
                        "complete",
                        f"Course ready — {core_ready}/{total_weeks} lectures reused; "
                        "learner assessments generated.",
                    )
                    warm_narration_cache(book_id)
                    print(
                        f"[lecture-gen] adopted {adopted} weeks from book {donor['id']}",
                        flush=True,
                    )
                    print(json.dumps({
                        "ok": True,
                        "weeks": total_weeks,
                        "adopted_from": donor["id"],
                    }))
                    return 0

        if quizzes_only:
            regenerate_quizzes(sid, book_id, weeks)
            refresh_book_counts(book_id)
            update_book_state(
                book_id,
                "ready",
                "complete",
                f"Quizzes rewritten — {total_weeks} weeks.",
            )
            print(json.dumps({"ok": True, "weeks": total_weeks, "quizzes_only": True}))
            return 0

        for planned_week, week_pages in weeks:
            week = planned_week.week
            first, last = week_pages[0][0], week_pages[-1][0]
            chapter_titles = "; ".join(part.title for part in planned_week.chapters)
            active_week = week
            active_stage = "lecture"

            if resume_artifacts and valid_lecture_checkpoint(sid, week, book_id):
                saved = lecture_artifact(sid, week, book_id)
                script = saved.get("script_payload") if saved else None
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "lecture",
                    "ready",
                    message="Reused completed lecture checkpoint",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:script",
                )
            else:
                message = f"Writing lecture {week} of {total_weeks} (pages {first}-{last})…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "lecture", "running", message=message)
                lecture = generate_week(week, total_weeks, chapter_titles, week_pages)
                write_lecture(sid, week, lecture, book_id)
                saved = lecture_artifact(sid, week, book_id)
                script = saved.get("script_payload") if saved else None
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "lecture",
                    "ready",
                    message="Lecture script saved",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:script",
                )

            if not script:
                raise RuntimeError(f"week {week} lecture checkpoint is unreadable")

            active_stage = "quiz"
            if resume_artifacts and valid_quiz_checkpoint(sid, week, book_id):
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "quiz",
                    "ready",
                    message="Reused completed quiz checkpoint",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:quiz",
                )
            else:
                message = f"Writing quiz {week} of {total_weeks} — “{script['title']}”…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "quiz", "running", message=message)
                spoken = [{"text": segment["text"]} for segment in script["segments"]]
                quiz = generate_quiz(script["title"], spoken, week_pages)
                write_quiz(sid, week, script["title"], quiz, book_id)
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "quiz",
                    "ready",
                    message="Quiz saved",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:quiz",
                )

            active_stage = "section"
            if section_checkpoint(sid, book_id, week):
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "section",
                    "ready",
                    message="Reused grounded section checkpoint",
                    artifact_ref=f"db:section_packs:{book_id}:week:{week}",
                )
            else:
                message = f"Writing section {week} of {total_weeks}…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "section", "running", message=message)
                section_status, refusal = generate_and_store_section(
                    sid,
                    book_id,
                    week,
                    script["title"],
                    focus=f"guided practice and application of {script['title']}",
                )
                if section_status == "ready":
                    mark_milestone(
                        book_id,
                        sid,
                        week,
                        "section",
                        "ready",
                        message="Grounded section saved",
                        artifact_ref=f"db:section_packs:{book_id}:week:{week}",
                    )
                else:
                    mark_milestone(
                        book_id,
                        sid,
                        week,
                        "section",
                        "deferred",
                        message="No grounded section published",
                        error=refusal,
                    )

            active_stage = "slides"
            if resume_artifacts and valid_slides_checkpoint(sid, week, book_id):
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "slides",
                    "ready",
                    message="Reused published slide deck",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:slides",
                )
            else:
                message = f"Publishing lecture {week} of {total_weeks}…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "slides", "running", message=message)
                build_slides(sid, week, book_id)
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "slides",
                    "ready",
                    message="Slide deck published",
                    artifact_ref=f"db:lecture_artifacts:{book_id}:week:{week}:slides",
                )

            register_week_artifacts(sid, week, book_id)
            execute(
                "UPDATE lectures SET title = %s WHERE week = %s AND student_id = %s",
                (script["title"].strip(), week, sid),
            )
            core_ready, audio_ready = refresh_book_counts(book_id)
            update_book_state(
                book_id,
                "generating",
                "content",
                f"Published week {week} of {total_weeks} — {core_ready} ready to use.",
            )

        # Live synthesizes narration from the database script. Readiness means
        # the on-demand path is available; no per-learner audio folder is built.
        for week in range(1, total_weeks + 1):
            active_week = week
            active_stage = "audio"
            mark_milestone(
                book_id,
                sid,
                week,
                "audio",
                "ready",
                message="On-demand lecture voice ready",
                artifact_ref="runtime:live-tts",
            )
            refresh_book_counts(book_id)

        core_ready, audio_ready = refresh_book_counts(book_id)
        if audio_ready == total_weeks:
            final_status = "ready"
            final_stage = "complete"
            final_message = (
                f"Course complete — {core_ready}/{total_weeks} lectures and "
                f"{audio_ready}/{total_weeks} narration runtimes ready."
            )
        else:
            final_status = "partial"
            final_stage = "paused"
            final_message = (
                f"Course usable — {core_ready}/{total_weeks} lectures ready; "
                f"{audio_ready}/{total_weeks} narration runtimes ready. Continue when convenient."
            )
        update_book_state(book_id, final_status, final_stage, final_message)
        if core_ready:
            warm_narration_cache(book_id)
        print(
            json.dumps(
                {
                    "ok": True,
                    "weeks": total_weeks,
                    "pages": page_count,
                    "core_ready": core_ready,
                    "audio_ready": audio_ready,
                    "complete": final_status == "ready",
                }
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - a failed run must land in books.error
        detail = f"{type(exc).__name__}: {exc}"
        if total_weeks:
            mark_milestone(
                book_id,
                sid,
                active_week,
                active_stage,
                "failed",
                message=f"{active_stage.title()} failed",
                error=detail,
            )
        core_ready, audio_ready = refresh_book_counts(book_id)
        failed_status = "partial_failed" if core_ready > 0 else "failed"
        failed_message = (
            f"Paused after {core_ready}/{total_weeks or '?'} usable lectures; "
            f"week {active_week} {active_stage} failed. Resume continues from this checkpoint."
        )
        update_book_state(
            book_id,
            failed_status,
            active_stage,
            failed_message,
            error=detail,
        )
        print(json.dumps({"ok": False, "error": detail}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
