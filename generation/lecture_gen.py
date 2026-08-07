"""Turn an uploaded book into a chapter-aware course with slides and quizzes.

    python UnivAI-Agent/generation/lecture_gen.py <absolute_pdf_path> <book_id>   (from the campus root)

For each week this writes, under lectures/week-N/:
    slides.md    Slidev deck — title slide + 3 content slides, built from the book
    script.json  what the Lecturer speaks, aligned slide-by-slide, citing real pages
    quiz.json    8 MCQs in the exam system's question shape (its question bank)

and then rebuilds the static decks (scripts/build-slides.mjs). Progress is
reported through books.progress so the upload page can show where it is.

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
import threading
from datetime import datetime, timezone
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

# Model output lands in log prints; on Windows a redirected stdout defaults to
# cp1252 and one "≤" in a reply kills the whole course build.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agents.prompts import PromptOperation, load_prompt_for
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

# The Brain cave is checked out inside the UnivAI campus repo; the shared
# plumbing (db, LLM adapter) lives there in services/.
ROOT = Path(__file__).resolve().parents[2]  # the UnivAI campus root
LECTURES_DIR = ROOT / "lectures"
execute = None
fetch_one = None
fetch_all = None
complete = None
LLMError = RuntimeError


def load_integrated_dependencies() -> None:
    """Load parent-owned services only for the explicit integrated command."""
    global execute, fetch_one, fetch_all, complete, LLMError, ROOT, LECTURES_DIR
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
    LECTURES_DIR = Path(os.getenv("LECTURES_DIR", str(ROOT / "lectures"))).resolve()

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
# Measured against the pre-rendered Kokoro voice.
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
LECTURE_SYSTEM = _LECTURE_PROMPT.system
QUIZ_SYSTEM = _QUIZ_PROMPT.system


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
        prompt_versions=f"lecture={_LECTURE_PROMPT.version},quiz={_QUIZ_PROMPT.version}",
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


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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


def write_semester_plan(sid: str, plan: SemesterWeekPlan) -> None:
    folder = LECTURES_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "semester-plan.json").write_text(
        plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def remove_obsolete_weeks(sid: str, week_count: int) -> None:
    """Remove only a regenerated course's now-invalid tail weeks.

    A corrected detector may shorten a plan. Leaving the old folders behind
    makes slide builds and manual inspection disagree with semester-plan.json.
    Both roots are resolved and every deletion is containment-checked before
    it happens; no path outside this student's generated namespace is touched.
    """

    roots = [
        (LECTURES_DIR / sid).resolve(),
        (ROOT / "UnivAI-app" / "public" / "slides" / sid).resolve(),
    ]
    for course_root in roots:
        if not course_root.is_dir():
            continue
        for folder in course_root.glob("week-*"):
            match = re.fullmatch(r"week-(\d+)", folder.name)
            if not match or int(match.group(1)) <= week_count:
                continue
            target = folder.resolve()
            if course_root not in target.parents:
                raise RuntimeError(f"refusing to remove week outside {course_root}")
            shutil.rmtree(target)


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
        parts.append(f"[page {number}]\n{chunk}")
        budget -= len(chunk)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- LLM helpers


def parse_json(raw: str) -> dict | None:
    """Small models wrap JSON in fences or chatter; dig the object out."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    text = text[start : end + 1]
    # The classic small-model sins, repaired before giving up: smart quotes
    # around/inside strings and trailing commas before a closing bracket.
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text.replace("“", '"').replace("”", '"'))):
        try:
            # strict=False: literal newlines/tabs inside strings are the other
            # classic small-model sin, and they carry no ambiguity for us.
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def ask_json(prompt: str, system: str, max_tokens: int, check) -> dict:
    """complete() then validate; retry with the rejection explained, and hand the
    LAST attempt to the fallback model - a repeated JSON failure is an output-
    quality problem, and availability-failover alone would never switch models."""
    generation_model = os.getenv("LLM_GENERATION", "").strip() or None
    fallback = os.getenv("LLM_FALLBACK", "").strip() or None
    repair_template = load_prompt_for(PromptOperation.SHARED_REPAIR_JSON)
    last = "no attempts made"
    current_prompt = prompt
    for attempt in range(1, ATTEMPTS + 1):
        force = fallback if (attempt == ATTEMPTS and fallback) else generation_model
        try:
            result = complete(
                current_prompt,
                system,
                max_tokens=max_tokens,
                force_spec=force,
            )
        except LLMError as exc:
            last = str(exc)
            continue
        data = parse_json(result.text)
        problem = check(data) if data is not None else "reply was not JSON"
        if problem is None:
            return data
        last = f"attempt {attempt}: {problem}"
        print(f"[lecture-gen] retrying - {last}", flush=True)
        print(f"[lecture-gen]   reply began: {result.text[:200]!r}", flush=True)
        print(f"[lecture-gen]   reply ended: {result.text[-200:]!r}", flush=True)
        current_prompt = repair_template.render(
            original_prompt=prompt,
            previous_reply=result.text[:REPAIR_REPLY_CHARS],
            validation_errors=problem,
            json_schema="Use the exact JSON shape and rules in the original prompt.",
        )
    raise RuntimeError(f"model never produced valid JSON ({last})")


# ---------------------------------------------------------------- lecture generation


def check_lecture(
    data: dict, expected_slides: int | None = None, require_intro: bool = True
) -> str | None:
    expected = lecture_shape(0)["slides"] if expected_slides is None else expected_slides
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        return "missing title"
    slides = data.get("slides")
    # A couple of slides short is a trim problem, not a rejection: demanding
    # exactly N well-formed slides from a small model kills whole builds over
    # cosmetics. Structural failures still reject.
    minimum = max(1, expected - 2)
    if not isinstance(slides, list) or len(slides) < minimum:
        return f"need at least {minimum} slides"
    # Name the offending slide. This string is handed straight back to the model
    # as the repair instruction (ask_json -> validation_errors), and a batch is
    # rejected whole even when a single slide is short, so "each slide needs..."
    # told it nothing about WHICH slide to fix — every retry reproduced the same
    # fault and the week died after ATTEMPTS identical rejections.
    for position, slide in enumerate(slides[:expected], start=1):
        heading = slide.get("heading")
        if not isinstance(heading, str) or not heading.strip():
            return f"slide {position} is missing its heading"
        where = f"slide {position} ({heading.strip()!r})"
        bullets = slide.get("bullets")
        if not isinstance(bullets, list) or not any(
            isinstance(b, str) and b.strip() for b in bullets
        ):
            return f"{where} needs at least one bullet"
        narration = slide.get("narration")
        spoken = len(narration.split()) if isinstance(narration, str) else 0
        if spoken < 15:
            return (
                f"{where} has {spoken}-word narration; rewrite only that slide's "
                "narration so it is at least 15 spoken words. Leave the others as they are."
            )
        if not isinstance(slide.get("page"), int):
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
    prompt = (
        f"These are pages {valid_pages[0]}-{valid_pages[-1]} of a textbook. "
        f"Create lecture {week} of a {total_weeks}-week course from them. "
        f"This week covers: {assigned_chapters}.\n"
        + continues
        + "\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "title": "short lecture title",\n'
        + intro_line
        + '  "slides": [\n'
        '    {"heading": "...", "bullets": ["...", "...", "..."], '
        f'"narration": "{narration} spoken sentences explaining this slide", "page": <page number the content came from>}}\n'
        "  ]\n"
        "}\n\n"
        f"Rules: exactly {slides} slides. Bullets are short phrases (under 12 words). "
        "Narration is natural speech - no bullet symbols, no 'as you can see'. "
        f'"page" must be one of {valid_pages}.\n\n'
        "Textbook pages:\n" + source_block(pages)
    )
    # Give the reply room to finish: a verbose narrator ran an M-size reply out
    # of tokens at 260/slide. Only ever one batch's worth, never the lecture's.
    data = ask_json(prompt, LECTURE_SYSTEM, 800 + 340 * slides, _slide_check(slides, first))
    # "Lecture 2: Consistency Models" — the deck already says Week N, and the
    # colon broke the deck's YAML headmatter once. Strip the redundant prefix.
    data["title"] = re.sub(r"^Lecture\s*\d+\s*[:\-–—]\s*", "", data["title"].strip())
    data["slides"] = data["slides"][:slides]
    for slide in data["slides"]:
        # never trust a model with page numbers: clamp to the pages it was shown
        if slide["page"] not in valid_pages:
            slide["page"] = min(valid_pages, key=lambda p: abs(p - slide["page"]))
        # coerce cosmetic bullet violations instead of failing the build
        slide["bullets"] = [b.strip() for b in slide["bullets"] if isinstance(b, str) and b.strip()][:5]
    return data


def check_quiz(minimum: int):
    def check(data: dict) -> str | None:
        questions = data.get("questions")
        if not isinstance(questions, list) or len(questions) < minimum:
            return f"need at least {minimum} questions"
        for question in questions:
            if not isinstance(question.get("prompt"), str) or not question["prompt"].strip():
                return "a question is missing its prompt"
            options = question.get("options")
            if not isinstance(options, list) or len(options) != 4:
                return "each question needs exactly 4 options"
            if not all(isinstance(o, str) and o.strip() for o in options):
                return "empty option"
            if question.get("correct") not in ("A", "B", "C", "D"):
                return 'correct must be "A", "B", "C" or "D"'
        return None

    return check


QUESTION_SHAPE = (
    "Return exactly this JSON shape:\n"
    "{\n"
    '  "questions": [\n'
    '    {"prompt": "the question?", "options": ["first", "second", "third", "fourth"], "correct": "A"}\n'
    "  ]\n"
    "}\n\n"
    'Rules: 4 options each, exactly one correct, "correct" is the letter of the correct '
    "option (A = first, B = second, C = third, D = fourth). Options must NOT start with "
    "letter labels. Spread the correct letters around - not all the same. "
    "No trick questions about page numbers or formatting.\n\n"
)


def lecture_text(title: str, segments: list[dict]) -> str:
    """Everything the lecturer actually says, as the quiz's source of truth."""
    return f"Lecture: {title}\n\n" + "\n\n".join(seg["text"] for seg in segments)


def ask_questions(prompt: str, count: int, source: str, minimum: int | None = None) -> list[dict]:
    # Accept a short reply rather than failing a whole course build — a 3-minute
    # lecture honestly supports about 5 distinct easy questions, not always 8.
    data = ask_json(
        prompt, QUIZ_SYSTEM, max(1800, 300 + 160 * count), check_quiz(minimum or max(1, count - 2))
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
        f'Write {shape["lecture_qs"]} multiple-choice questions testing the TOPICS this lecturer '
        "covered. A student who understood the lecture must be able to answer every one; do not "
        "ask about anything the lecture does not cover. Test the concept, not the wording: never "
        "quote the lecturer's sentences verbatim, never ask what the lecturer 'said' or "
        "'mentioned', and never turn a sentence into a fill-in-the-blank. Plain questions about "
        "the subject matter itself.\n\n" + QUESTION_SHAPE +
        "The lecture:\n" + lecture_text(title, segments),
        shape["lecture_qs"],
        "lecture",
        # a full quiz paper must be coverable by lecturer-taught questions
        minimum=5,
    )

    # 2) The small self-study tail: from the week's wider pages, beyond the slides.
    homework = ask_questions(
        f'Write {shape["self_qs"]} multiple-choice SELF-STUDY questions for the week on '
        f'"{title}", using ONLY these textbook pages. Pick details a short lecture would not '
        "have covered - the student is expected to have read the pages themselves.\n\n"
        + QUESTION_SHAPE + "Textbook pages:\n" + source_block(pages),
        shape["self_qs"],
        "self_study",
    )
    return taught + homework


# ---------------------------------------------------------------- writing files


def write_lecture(sid: str, week: int, lecture: dict) -> None:
    # Per-student course layout: lectures/<studentId>/week-N/ (matches the app's
    # lib/lectures.ts and the UnivAI-live worker).
    folder = LECTURES_DIR / sid / f"week-{week}"
    folder.mkdir(parents=True, exist_ok=True)
    title = lecture["title"].strip()

    yaml_title = f"Week {week} — {title}".replace('"', "'")
    deck = [
        "---",
        "theme: default",
        "routerMode: hash",
        # quoted: a colon inside an unquoted YAML value kills the whole build
        f'title: "{yaml_title}"',
        "---",
        "",
        f"# Week {week}",
        f"## {title}",
    ]
    for slide in lecture["slides"]:
        deck += ["", "---", "", f"# {slide['heading'].strip()}", ""]
        deck += [f"- {bullet.strip()}" for bullet in slide["bullets"]]
        deck += ["", f"<small>Source: p.{slide['page']}</small>"]
    atomic_write(folder / "slides.md", "\n".join(deck) + "\n")

    # Slidev's hash router is 1-based and the title slide is 1: the intro plays
    # there, and slide N of content lives at N+1. This alignment was exactly the
    # bug in the premade decks - keep it in one place.
    segments = [
        {"slide": 1, "text": lecture["intro"].strip(), "citations": [{"page": lecture["slides"][0]["page"]}]}
    ]
    for index, slide in enumerate(lecture["slides"]):
        segments.append(
            {
                "slide": index + 2,
                "text": slide["narration"].strip(),
                "citations": [{"page": slide["page"]}],
            }
        )
    script = {
        "lectureId": f"week-{week}",
        "title": title,
        "durationMinutes": lecture.get("durationMinutes", LECTURE_MINUTES_MIN),
        "segments": segments,
    }
    atomic_write(
        folder / "script.json",
        json.dumps(script, indent=2, ensure_ascii=False) + "\n",
    )
    # The full lecture checkpoint lets a retry rebuild the deck without asking
    # the model again. New narration invalidates any older pre-rendered audio.
    atomic_write(
        folder / "lecture.json",
        json.dumps(lecture, indent=2, ensure_ascii=False) + "\n",
    )
    audio_dir = folder / "audio"
    if audio_dir.is_dir():
        for old_audio in audio_dir.glob("*.npy"):
            old_audio.unlink(missing_ok=True)
        (audio_dir / "meta.json").unlink(missing_ok=True)


def write_quiz(sid: str, week: int, title: str, quiz: list[dict]) -> None:
    folder = LECTURES_DIR / sid / f"week-{week}"
    atomic_write(
        folder / "quiz.json",
        json.dumps(
            {"week": week, "title": title.strip(), "questions": quiz},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )


def create_artifact(filepath: Path, state: str = "ready") -> str:
    content_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
    pipeline_hash = hashlib.sha256(b"lecture_gen").hexdigest()
    content_key = f"sha256:{content_hash}.pipeline:{pipeline_hash}"

    try:
        execute(
            """
            INSERT INTO content_artifacts
            (content_key, schema_version, original_sha256, pipeline_fingerprint, state, byte_length, page_count, artifact_checksum, storage_ref, created_at, updated_at)
            VALUES (%s, 'content-artifact-v1', %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_key) DO NOTHING
            """,
            (
                content_key,
                content_hash,
                json.dumps({"source": "lecture_gen"}),
                state,
                len(filepath.read_bytes()),
                1,
                content_hash,
                str(filepath.relative_to(ROOT)),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ),
        )
    except Exception as exc:
        print(f"Error inserting artifact {filepath}: {exc}", flush=True)
    return content_key


def register_week_artifacts(sid: str, week: int) -> None:
    if execute is None:
        return
    folder = LECTURES_DIR / sid / f"week-{week}"
    slides_key = create_artifact(folder / "slides.md")
    script_key = create_artifact(folder / "script.json")
    quiz_key = create_artifact(folder / "quiz.json")
    execute(
        """
        UPDATE lectures
        SET script_artifact_key = %s, slides_artifact_key = %s, quiz_artifact_key = %s
        WHERE student_id = %s AND week = %s
        """,
        (script_key, slides_key, quiz_key, sid, week),
    )


def write_week(sid: str, week: int, lecture: dict, quiz: list[dict]) -> None:
    """Compatibility helper used by focused callers; checkpoints write separately."""
    write_lecture(sid, week, lecture)
    write_quiz(sid, week, lecture["title"], quiz)
    register_week_artifacts(sid, week)



def build_slides(sid: str, week: int | None = None) -> None:
    # sid tells the builder to read lectures/<sid>/week-N/slides.md and emit the
    # decks under public/slides/<sid>/week-N/.
    command = ["node", str(ROOT / "scripts" / "build-slides.mjs"), sid]
    if week is not None:
        command.append(f"week-{week}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"slidev build failed: {result.stderr[-800:]}")


def prerender_voice(sid: str, week: int, book_id: int, total_weeks: int) -> None:
    """Record one resumable week in a bounded subprocess."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "UnivAI-live" / "prerender_audio.py"),
            sid,
            str(week),
            str(book_id),
            str(total_weeks),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30 * 60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"voice pre-render failed: {(result.stdout + result.stderr)[-500:]}")


GENERATION_MANIFEST = "generation-manifest.json"
AUDIO_WEEKS_PER_RUN = max(1, int(os.getenv("AUDIO_WEEKS_PER_RUN", "1")))


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def valid_lecture_checkpoint(sid: str, week: int) -> bool:
    folder = LECTURES_DIR / sid / f"week-{week}"
    script = _read_json(folder / "script.json")
    lecture = _read_json(folder / "lecture.json")
    if not script or not isinstance(script.get("segments"), list) or not script["segments"]:
        return False
    if not (folder / "slides.md").is_file():
        return False
    # Legacy runs did not retain lecture.json. Their script + slides are still
    # complete checkpoints; new runs keep the richer file for deck rebuilding.
    return lecture is None or isinstance(lecture.get("slides"), list)


def valid_quiz_checkpoint(sid: str, week: int) -> bool:
    quiz = _read_json(LECTURES_DIR / sid / f"week-{week}" / "quiz.json")
    return bool(quiz and isinstance(quiz.get("questions"), list) and quiz["questions"])


def valid_slides_checkpoint(sid: str, week: int) -> bool:
    return (
        ROOT
        / "UnivAI-app"
        / "public"
        / "slides"
        / sid
        / f"week-{week}"
        / "index.html"
    ).is_file()


def valid_audio_checkpoint(sid: str, week: int) -> bool:
    audio = LECTURES_DIR / sid / f"week-{week}" / "audio"
    meta = _read_json(audio / "meta.json")
    return bool(meta and meta.get("sample_rate") and next(audio.glob("*.npy"), None))


def prepare_generation_manifest(
    sid: str,
    book_id: int,
    source_sha256: str,
    total_weeks: int,
    course_fingerprint: str | None = None,
) -> bool:
    """Return whether existing files belong to a resumable generation.

    The one-time legacy adoption preserves the files produced before manifests
    existed. Every subsequent source is protected by its SHA-256 identity.

    The fingerprint is recorded so another learner's run can decide whether this
    course is safe to adopt (:func:`find_reusable_course`), but it is
    deliberately NOT part of the resume test: a learner resuming their own
    half-built course should keep their completed weeks even after the
    generator changes. Cross-learner reuse is the strict case, and it demands an
    exact match.
    """
    path = LECTURES_DIR / sid / GENERATION_MANIFEST
    existing = _read_json(path)
    matching = bool(existing and existing.get("source_sha256") == source_sha256)
    legacy = existing is None and any((LECTURES_DIR / sid).glob("week-*/script.json"))
    atomic_write(
        path,
        json.dumps(
            {
                "schema_version": 2,
                "book_id": book_id,
                "source_sha256": source_sha256,
                "course_fingerprint": course_fingerprint,
                "total_weeks": total_weeks,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
    )
    return matching or legacy


# The text of one reusable week. Small (68-112 KB), and genuinely copied:
# every learner owns their files, because a shared directory would let one
# learner's regenerate rewrite another's course.
REUSABLE_WEEK_FILES = ("script.json", "slides.md", "quiz.json", "lecture.json")


def link_or_copy(source: Path, target: Path) -> None:
    """Hard-link a file, falling back to a copy when the link cannot be made.

    A link fails across filesystems (EXDEV) and on filesystems without hard
    links; a copy is always correct, just larger. Either way the caller ends up
    with a readable file at ``target``.
    """
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def adopt_week_audio(donor: str, sid: str, week: int) -> bool:
    """Give this learner the donor's rendered voice without re-rendering it.

    Audio dwarfs everything else — 86 MB a week against ~100 KB of text — so
    copying it per learner would cost gigabytes across a cohort, and
    re-rendering it costs ~20 minutes of TTS to produce byte-identical output
    from the same script and the same engine. Hard links cost neither: one copy
    on disk, many names for it.

    This is safe because prerender_audio never writes a clip in place. It
    renders to ``.<name>.tmp.npy`` and calls ``replace()``, which swaps in a NEW
    inode — so a learner who later regenerates gets their own file and the
    donor's bytes are untouched. Stale cleanup unlinks, which only drops a name.
    """
    source_dir = LECTURES_DIR / donor / f"week-{week}" / "audio"
    if not valid_audio_checkpoint(donor, week):
        return False
    target_dir = LECTURES_DIR / sid / f"week-{week}" / "audio"
    target_dir.mkdir(parents=True, exist_ok=True)
    for clip in sorted(source_dir.glob("*.npy")):
        link_or_copy(clip, target_dir / clip.name)
    meta = source_dir / "meta.json"
    if meta.is_file():
        link_or_copy(meta, target_dir / meta.name)
    return valid_audio_checkpoint(sid, week)


def find_reusable_course(
    sid: str, source_sha256: str, course_fingerprint: str
) -> tuple[str, int] | None:
    """Another learner's finished course for these exact bytes and generator.

    Returns ``(donor_sid, total_weeks)``, or None when nothing is adoptable.
    Both halves of the key must match: the bytes, so it is the same book, and
    the fingerprint, so it was built by the generator running now.
    """
    if fetch_all is None:
        return None
    owners = fetch_all(
        """SELECT DISTINCT student_id FROM books
            WHERE source_sha256 = %s AND student_id <> %s
              AND status IN ('ready', 'partial')""",
        (source_sha256, sid),
    )
    for owner in owners:
        donor = owner.get("student_id")
        if not donor:
            continue
        manifest = _read_json(LECTURES_DIR / donor / GENERATION_MANIFEST)
        if not manifest:
            continue
        if manifest.get("source_sha256") != source_sha256:
            continue
        if manifest.get("course_fingerprint") != course_fingerprint:
            continue
        total = int(manifest.get("total_weeks") or 0)
        if total < 1:
            continue
        # Only adopt a course whose weeks are all actually on disk. A donor
        # halfway through its own build would hand over gaps.
        if all(valid_lecture_checkpoint(donor, week) for week in range(1, total + 1)):
            return donor, total
    return None


def adopt_course(donor: str, sid: str, total_weeks: int) -> None:
    """Copy a donor's generated weeks into this learner's namespace.

    Every learner keeps their OWN copy rather than sharing a directory: the
    live lecture, the slide build and the exam sync all read
    ``lectures/<sid>/``, and a shared path would make one learner's regenerate
    silently rewrite another's course.
    """
    source_root = (LECTURES_DIR / donor).resolve()
    target_root = (LECTURES_DIR / sid).resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"donor course {donor} is not on disk")
    target_root.mkdir(parents=True, exist_ok=True)

    plan = source_root / "semester-plan.json"
    if not plan.is_file():
        raise RuntimeError(f"donor course {donor} has no semester plan")
    shutil.copy2(plan, target_root / "semester-plan.json")

    for week in range(1, total_weeks + 1):
        source_week = source_root / f"week-{week}"
        target_week = target_root / f"week-{week}"
        target_week.mkdir(parents=True, exist_ok=True)
        for name in REUSABLE_WEEK_FILES:
            candidate = source_week / name
            if candidate.is_file():
                shutil.copy2(candidate, target_week / name)
        # Any week the donor has already voiced arrives voiced.
        adopt_week_audio(donor, sid, week)


def finish_adopted_course(sid: str, book_id: int, total_weeks: int) -> None:
    """Publish an adopted course: decks, artifacts, milestones, final state.

    Everything the model would have produced is already on disk, so only the
    per-learner work is left. Slides are rebuilt rather than copied because a
    built deck hard-codes its own ``/slides/<sid>/`` base path, and audio is
    left for the normal deferred pass — it is regenerable from script.json and
    a learner is not blocked from attending without it.
    """
    for week in range(1, total_weeks + 1):
        script = _read_json(LECTURES_DIR / sid / f"week-{week}" / "script.json") or {}
        for stage, note in (("lecture", "Reused lecture"), ("quiz", "Reused quiz")):
            mark_milestone(
                book_id,
                sid,
                week,
                stage,
                "ready",
                message=f"{note} from an identical book",
                artifact_ref=(
                    f"lectures/{sid}/week-{week}/"
                    f"{'script.json' if stage == 'lecture' else 'quiz.json'}"
                ),
            )
        message = f"Publishing lecture {week} of {total_weeks}…"
        progress(book_id, message)
        mark_milestone(book_id, sid, week, "slides", "running", message=message)
        build_slides(sid, week)
        mark_milestone(
            book_id,
            sid,
            week,
            "slides",
            "ready",
            message="Slide deck published",
            artifact_ref=f"UnivAI-app/public/slides/{sid}/week-{week}/index.html",
        )
        # A week the donor had already voiced is ready now; the rest queue for
        # a later step exactly as a freshly generated course's would.
        if valid_audio_checkpoint(sid, week):
            mark_milestone(
                book_id,
                sid,
                week,
                "audio",
                "ready",
                message="Reused lecture audio from an identical book",
                artifact_ref=f"lectures/{sid}/week-{week}/audio/meta.json",
            )
        else:
            mark_milestone(
                book_id,
                sid,
                week,
                "audio",
                "deferred",
                message="Queued for a later generation step",
            )
        register_week_artifacts(sid, week)
        title = str(script.get("title") or "").strip()
        if title:
            execute(
                "UPDATE lectures SET title = %s WHERE week = %s AND student_id = %s",
                (title, week, sid),
            )
        refresh_book_counts(book_id)

    core_ready, audio_ready = refresh_book_counts(book_id)
    if audio_ready >= total_weeks:
        update_book_state(
            book_id,
            "ready",
            "complete",
            f"Course ready from an identical book — {core_ready}/{total_weeks} lectures "
            f"and {audio_ready}/{total_weeks} audio tracks.",
        )
        return
    update_book_state(
        book_id,
        "partial",
        "paused",
        f"Course ready from an identical book — {core_ready}/{total_weeks} lectures; "
        f"{audio_ready}/{total_weeks} audio tracks. Continue when convenient.",
    )


def initialize_milestones(book_id: int, sid: str, total_weeks: int) -> None:
    mark_milestone(book_id, sid, 0, "plan", "ready", message="Course plan saved")
    execute(
        """
        INSERT INTO course_generation_milestones
          (book_id, student_id, week, stage, status, updated_at)
        SELECT %s, %s, generated_week, stage, 'pending', CURRENT_TIMESTAMP
        FROM generate_series(1, %s) AS generated_week
        CROSS JOIN unnest(ARRAY['lecture', 'quiz', 'slides', 'audio']) AS stage
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
    """Rewrite only quiz.json per week, from the ALREADY generated lecture scripts."""
    total_weeks = len(weeks)
    for planned_week, week_pages in weeks:
        week = planned_week.week
        script = json.loads(
            (LECTURES_DIR / sid / f"week-{week}" / "script.json").read_text("utf-8")
        )
        progress(book_id, f"Rewriting quiz {week} of {total_weeks} — “{script['title']}”…")
        quiz = generate_quiz(script["title"], script["segments"], week_pages)
        (LECTURES_DIR / sid / f"week-{week}" / "quiz.json").write_text(
            json.dumps(
                {"week": week, "title": script["title"], "questions": quiz},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


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
                    "side_effects": "database, Slidev, and voice prerender skipped",
                }
            )
        )
        return 0

    load_integrated_dependencies()
    if len(sys.argv) < 3:
        print(json.dumps({
            "ok": False,
            "error": "usage: lecture_gen.py <pdf_path> <book_id> [--quizzes-only|--plan-only]",
        }))
        return 2
    pdf_path = Path(sys.argv[1]).resolve()
    book_id = int(sys.argv[2])
    quizzes_only = "--quizzes-only" in sys.argv[3:]
    # Discover the chapters and stop. The curriculum a learner approves is built
    # from semester-plan.json, so the plan has to exist before they can approve
    # anything — but writing lectures, quizzes, slides and voice for a course
    # they have not agreed to is hours of work thrown away the moment they
    # reshape it. Planning is pure Python (no model calls), so this pass is
    # cheap; the expensive stages wait for approval to spawn the full run.
    plan_only = "--plan-only" in sys.argv[3:]

    book = fetch_one(
        """SELECT id, student_id, title, filename, status, pages, source_sha256,
                  generation_total_weeks, generation_ready_weeks
           FROM books WHERE id = %s""",
        (book_id,),
    )
    if not book:
        print(json.dumps({"ok": False, "error": f"no book with id {book_id}"}))
        return 2
    start_heartbeat(book_id)
    # The owner. Every write below is namespaced to this student (disk + DB).
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
        saved_manifest = _read_json(LECTURES_DIR / sid / GENERATION_MANIFEST)
        saved_total = int(book.get("generation_total_weeks") or 0)
        fast_audio_resume = bool(
            not quizzes_only
            and saved_total > 0
            and int(book.get("generation_ready_weeks") or 0) >= saved_total
            and book.get("source_sha256") == source_sha256
            and saved_manifest
            and saved_manifest.get("source_sha256") == source_sha256
            and (LECTURES_DIR / sid / "semester-plan.json").is_file()
        )

        # Someone has already paid for this exact book to be turned into this
        # exact course. Adopt it instead of spending the model again — the
        # learner gets their course in seconds rather than half an hour.
        # Skipped for a resume (this learner already has work in progress) and
        # for the cheap plan-only pass, which is pure Python anyway.
        adopted = None
        if not quizzes_only and not plan_only and not fast_audio_resume:
            adopted = find_reusable_course(sid, source_sha256, fingerprint)

        if adopted:
            donor, total_weeks = adopted
            message = f"Reusing a course already built from this book — {total_weeks} weeks."
            update_book_state(book_id, "generating", "reusing", message)
            progress(book_id, message)
            adopt_course(donor, sid, total_weeks)
            execute(
                """UPDATE books SET pages = %s, source_sha256 = %s,
                       generation_total_weeks = %s, generation_stage = 'content',
                       error = NULL WHERE id = %s""",
                (int(book.get("pages") or 0), source_sha256, total_weeks, book_id),
            )
            prepare_generation_manifest(
                sid, book_id, source_sha256, total_weeks, fingerprint
            )
            initialize_milestones(book_id, sid, total_weeks)
            finish_adopted_course(sid, book_id, total_weeks)
            print(
                json.dumps(
                    {"ok": True, "weeks": total_weeks, "reused_from": donor}
                )
            )
            return 0

        if fast_audio_resume:
            total_weeks = saved_total
            page_count = int(book.get("pages") or 0)
            weeks = []
            resume_files = True
            message = f"Reusing {total_weeks} published lecture checkpoints…"
            update_book_state(book_id, "generating", "resuming", message)
            progress(book_id, message)
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
            write_semester_plan(sid, plan)
            remove_obsolete_weeks(sid, total_weeks)
            resume_files = prepare_generation_manifest(
                sid, book_id, source_sha256, total_weeks, fingerprint
            )
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

        if quizzes_only:
            regenerate_quizzes(sid, book_id, weeks)
            for planned_week, _week_pages in weeks:
                mark_milestone(
                    book_id,
                    sid,
                    planned_week.week,
                    "quiz",
                    "ready",
                    message="Quiz checkpoint rewritten",
                    artifact_ref=f"lectures/{sid}/week-{planned_week.week}/quiz.json",
                )
                register_week_artifacts(sid, planned_week.week)
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

            if resume_files and valid_lecture_checkpoint(sid, week):
                script = _read_json(LECTURES_DIR / sid / f"week-{week}" / "script.json")
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "lecture",
                    "ready",
                    message="Reused completed lecture checkpoint",
                    artifact_ref=f"lectures/{sid}/week-{week}/script.json",
                )
            else:
                message = f"Writing lecture {week} of {total_weeks} (pages {first}-{last})…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "lecture", "running", message=message)
                lecture = generate_week(week, total_weeks, chapter_titles, week_pages)
                write_lecture(sid, week, lecture)
                script = _read_json(LECTURES_DIR / sid / f"week-{week}" / "script.json")
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "lecture",
                    "ready",
                    message="Lecture script saved",
                    artifact_ref=f"lectures/{sid}/week-{week}/script.json",
                )

            if not script:
                raise RuntimeError(f"week {week} lecture checkpoint is unreadable")

            active_stage = "quiz"
            if resume_files and valid_quiz_checkpoint(sid, week):
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "quiz",
                    "ready",
                    message="Reused completed quiz checkpoint",
                    artifact_ref=f"lectures/{sid}/week-{week}/quiz.json",
                )
            else:
                message = f"Writing quiz {week} of {total_weeks} — “{script['title']}”…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "quiz", "running", message=message)
                spoken = [{"text": segment["text"]} for segment in script["segments"]]
                quiz = generate_quiz(script["title"], spoken, week_pages)
                write_quiz(sid, week, script["title"], quiz)
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "quiz",
                    "ready",
                    message="Quiz saved",
                    artifact_ref=f"lectures/{sid}/week-{week}/quiz.json",
                )

            active_stage = "slides"
            if resume_files and valid_slides_checkpoint(sid, week):
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "slides",
                    "ready",
                    message="Reused published slide deck",
                    artifact_ref=f"UnivAI-app/public/slides/{sid}/week-{week}/index.html",
                )
            else:
                message = f"Publishing lecture {week} of {total_weeks}…"
                progress(book_id, message)
                mark_milestone(book_id, sid, week, "slides", "running", message=message)
                build_slides(sid, week)
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "slides",
                    "ready",
                    message="Slide deck published",
                    artifact_ref=f"UnivAI-app/public/slides/{sid}/week-{week}/index.html",
                )

            register_week_artifacts(sid, week)
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

        # Audio is enrichment, not an all-or-nothing gate. Adopt complete legacy
        # weeks, render at most one new week per run, and defer the rest.
        rendered_this_run = 0
        for week in range(1, total_weeks + 1):
            active_week = week
            active_stage = "audio"
            if valid_audio_checkpoint(sid, week):
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "audio",
                    "ready",
                    message="Audio checkpoint ready",
                    artifact_ref=f"lectures/{sid}/week-{week}/audio/meta.json",
                )
                refresh_book_counts(book_id)
                continue
            if rendered_this_run >= AUDIO_WEEKS_PER_RUN:
                mark_milestone(
                    book_id,
                    sid,
                    week,
                    "audio",
                    "deferred",
                    message="Queued for a later generation step",
                )
                continue

            message = f"Recording lecture audio {week} of {total_weeks}…"
            progress(book_id, message)
            update_book_state(book_id, "generating", "audio", message)
            mark_milestone(book_id, sid, week, "audio", "running", message=message)
            prerender_voice(sid, week, book_id, total_weeks)
            mark_milestone(
                book_id,
                sid,
                week,
                "audio",
                "ready",
                message="Lecture audio ready",
                artifact_ref=f"lectures/{sid}/week-{week}/audio/meta.json",
            )
            refresh_book_counts(book_id)
            rendered_this_run += 1

        core_ready, audio_ready = refresh_book_counts(book_id)
        if audio_ready == total_weeks:
            final_status = "ready"
            final_stage = "complete"
            final_message = (
                f"Course complete — {core_ready}/{total_weeks} lectures and "
                f"{audio_ready}/{total_weeks} audio tracks ready."
            )
        else:
            final_status = "partial"
            final_stage = "paused"
            final_message = (
                f"Course usable — {core_ready}/{total_weeks} lectures ready; "
                f"{audio_ready}/{total_weeks} audio tracks ready. Continue when convenient."
            )
        update_book_state(book_id, final_status, final_stage, final_message)
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
