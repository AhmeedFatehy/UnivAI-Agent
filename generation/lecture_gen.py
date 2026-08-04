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
import os
import re
import subprocess
import sys
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
    SemesterWeek,
    SemesterWeekPlan,
    discover_chapters,
    pages_for_week,
    plan_semester,
)

# The Brain cave is checked out inside the UnivAI campus repo; the shared
# plumbing (db, LLM adapter) lives there in services/.
ROOT = Path(__file__).resolve().parents[2]  # the UnivAI campus root
LECTURES_DIR = ROOT / "lectures"
execute = None
fetch_one = None
complete = None
LLMError = RuntimeError


def load_integrated_dependencies() -> None:
    """Load parent-owned services only for the explicit integrated command."""
    global execute, fetch_one, complete, LLMError, ROOT, LECTURES_DIR
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
    from common.db import execute as db_execute, fetch_one as db_fetch_one
    from common.llm import LLMError as SharedLLMError, complete as llm_complete

    execute = db_execute
    fetch_one = db_fetch_one
    complete = llm_complete
    LLMError = SharedLLMError
    LECTURES_DIR = Path(os.getenv("LECTURES_DIR", str(ROOT / "lectures"))).resolve()

# The course size dial (settings.course_size, set from the admin page). One
# knob scales the lecture and the quiz bank together. The app holds the SAME
# table in app/lib/course-size.ts — keep them in sync.
# The quiz bank per week: >=90% of any served paper must be answerable from
# what the lecturer SAID (easy if you attended); self-study questions from the
# wider pages exist but can never exceed 10% of a paper.
SIZES = {
    "XS": {"slides": 3, "narration": "4-6", "lecture_qs": 8, "self_qs": 2},
    "S": {"slides": 5, "narration": "4-6", "lecture_qs": 10, "self_qs": 2},
    "M": {"slides": 8, "narration": "5-7", "lecture_qs": 14, "self_qs": 3},
    "L": {"slides": 12, "narration": "6-8", "lecture_qs": 18, "self_qs": 4},
    "XL": {"slides": 16, "narration": "6-9", "lecture_qs": 22, "self_qs": 5},
}
# Filled in main() from the settings table; XS keeps the original behaviour.
CFG = SIZES["XS"]
# A 3B model with an 8k window: keep the source well under it.
MAX_SOURCE_CHARS = 12000
MAX_CHARS_PER_PAGE = 1500
ATTEMPTS = 4

LECTURE_SYSTEM = load_prompt_for(PromptOperation.CONTENT_GENERATE_LECTURE).system
QUIZ_SYSTEM = load_prompt_for(PromptOperation.ASSESSMENT_QUIZ).system


def progress(book_id: int, message: str) -> None:
    print(f"[lecture-gen] {message}", flush=True)
    execute("UPDATE books SET progress = %s WHERE id = %s", (message, book_id))


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
        if len(text) >= 40:  # covers, blank pages, pure-image pages
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
            previous_reply=result.text[:2000],
            validation_errors=problem,
            json_schema="Use the exact JSON shape and rules in the original prompt.",
        )
    raise RuntimeError(f"model never produced valid JSON ({last})")


# ---------------------------------------------------------------- lecture generation


def check_lecture(data: dict) -> str | None:
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        return "missing title"
    slides = data.get("slides")
    # A couple of slides short is a trim problem, not a rejection: demanding
    # exactly N well-formed slides from a small model kills whole builds over
    # cosmetics. Structural failures still reject.
    if not isinstance(slides, list) or len(slides) < max(3, CFG["slides"] - 2):
        return f"need at least {max(3, CFG['slides'] - 2)} slides"
    for slide in slides[: CFG["slides"]]:
        if not isinstance(slide.get("heading"), str) or not slide["heading"].strip():
            return "a slide is missing its heading"
        bullets = slide.get("bullets")
        if not isinstance(bullets, list) or not any(
            isinstance(b, str) and b.strip() for b in bullets
        ):
            return "each slide needs at least one bullet"
        if not isinstance(slide.get("narration"), str) or len(slide["narration"].split()) < 15:
            return "each slide needs spoken narration of at least 15 words"
        if not isinstance(slide.get("page"), int):
            return "each slide needs the page number it came from"
    if not isinstance(data.get("intro"), str) or not data["intro"].strip():
        return "missing intro"
    return None


def generate_week(
    week: int,
    total_weeks: int,
    assigned_chapters: str,
    pages: list[tuple[int, str]],
) -> dict:
    valid_pages = [number for number, _ in pages]
    prompt = (
        f"These are pages {valid_pages[0]}-{valid_pages[-1]} of a textbook. "
        f"Create lecture {week} of a {total_weeks}-week course from them. "
        f"This week covers: {assigned_chapters}.\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "title": "short lecture title",\n'
        '  "intro": "2 spoken sentences welcoming students and saying what this lecture covers",\n'
        '  "slides": [\n'
        '    {"heading": "...", "bullets": ["...", "...", "..."], '
        f'"narration": "{CFG["narration"]} spoken sentences explaining this slide", "page": <page number the content came from>}}\n'
        "  ]\n"
        "}\n\n"
        f"Rules: exactly {CFG['slides']} slides. Bullets are short phrases (under 12 words). "
        "Narration is natural speech - no bullet symbols, no 'as you can see'. "
        f'"page" must be one of {valid_pages}.\n\n'
        "Textbook pages:\n" + source_block(pages)
    )
    # Bigger sizes produce longer JSON: give the reply room to finish. A small
    # model narrates verbosely — an M-size reply got cut at 260 tokens/slide.
    data = ask_json(prompt, LECTURE_SYSTEM, 800 + 340 * CFG["slides"], check_lecture)
    # "Lecture 2: Consistency Models" — the deck already says Week N, and the
    # colon broke the deck's YAML headmatter once. Strip the redundant prefix.
    data["title"] = re.sub(r"^Lecture\s*\d+\s*[:\-–—]\s*", "", data["title"].strip())
    data["slides"] = data["slides"][: CFG["slides"]]
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
    # 1) The bulk of the bank: questions a student who WATCHED the lecture finds
    #    easy — every answer must have been said out loud by the lecturer.
    taught = ask_questions(
        f'Write {CFG["lecture_qs"]} multiple-choice questions testing the TOPICS this lecturer '
        "covered. A student who understood the lecture must be able to answer every one; do not "
        "ask about anything the lecture does not cover. Test the concept, not the wording: never "
        "quote the lecturer's sentences verbatim, never ask what the lecturer 'said' or "
        "'mentioned', and never turn a sentence into a fill-in-the-blank. Plain questions about "
        "the subject matter itself.\n\n" + QUESTION_SHAPE +
        "The lecture:\n" + lecture_text(title, segments),
        CFG["lecture_qs"],
        "lecture",
        # a full quiz paper must be coverable by lecturer-taught questions
        minimum=5,
    )

    # 2) The small self-study tail: from the week's wider pages, beyond the slides.
    homework = ask_questions(
        f'Write {CFG["self_qs"]} multiple-choice SELF-STUDY questions for the week on '
        f'"{title}", using ONLY these textbook pages. Pick details a short lecture would not '
        "have covered - the student is expected to have read the pages themselves.\n\n"
        + QUESTION_SHAPE + "Textbook pages:\n" + source_block(pages),
        CFG["self_qs"],
        "self_study",
    )
    return taught + homework


# ---------------------------------------------------------------- writing files


def write_week(sid: str, week: int, lecture: dict, quiz: list[dict]) -> None:
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
    (folder / "slides.md").write_text("\n".join(deck) + "\n", encoding="utf-8")

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
    script = {"lectureId": f"week-{week}", "title": title, "segments": segments}
    (folder / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    (folder / "quiz.json").write_text(
        json.dumps({"week": week, "title": title, "questions": quiz}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    import hashlib
    from datetime import datetime, timezone

    def create_artifact(filepath, state):
        content_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        pipeline_hash = hashlib.sha256(b"lecture_gen").hexdigest()
        content_key = f"sha256:{content_hash}.pipeline:{pipeline_hash}"
        
        try:
            execute(
                """
                INSERT INTO content_artifacts 
                (content_key, schema_version, original_sha256, pipeline_fingerprint, state, byte_length, page_count, artifact_checksum, storage_ref, created_at, updated_at)
                VALUES (%s, 'content-artifact-v1', %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_key) DO UPDATE SET storage_ref = EXCLUDED.storage_ref, updated_at = EXCLUDED.updated_at
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
                    datetime.now(timezone.utc)
                )
            )
        except Exception as e:
            print(f"Error inserting artifact {filepath}: {e}")
        return content_key

    if execute is not None:
        slides_key = create_artifact(folder / "slides.md", "ready")
        script_key = create_artifact(folder / "script.json", "ready")
        quiz_key = create_artifact(folder / "quiz.json", "ready")
        execute(
            """
            UPDATE lectures
            SET script_artifact_key = %s, slides_artifact_key = %s, quiz_artifact_key = %s
            WHERE student_id = %s AND week = %s
            """,
            (script_key, slides_key, quiz_key, sid, week)
        )



def build_slides(sid: str) -> None:
    # sid tells the builder to read lectures/<sid>/week-N/slides.md and emit the
    # decks under public/slides/<sid>/week-N/.
    result = subprocess.run(
        ["node", str(ROOT / "scripts" / "build-slides.mjs"), sid],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"slidev build failed: {result.stderr[-800:]}")


def prerender_voice(sid: str) -> None:
    """Record the whole lecture to disk (UnivAI-live/prerender_audio.py — the
    Mouth cave's job) in a subprocess, so the TTS memory returns when done.
    sid scopes it to this student's lectures/<sid>/week-N/audio/."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "UnivAI-live" / "prerender_audio.py"), sid],
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
        print(json.dumps({"ok": False, "error": "usage: lecture_gen.py <pdf_path> <book_id> [--quizzes-only]"}))
        return 2
    pdf_path = Path(sys.argv[1]).resolve()
    book_id = int(sys.argv[2])
    quizzes_only = "--quizzes-only" in sys.argv[3:]

    book = fetch_one("SELECT id, student_id, title, filename FROM books WHERE id = %s", (book_id,))
    if not book:
        print(json.dumps({"ok": False, "error": f"no book with id {book_id}"}))
        return 2
    # The owner. Every write below is namespaced to this student (disk + DB).
    sid = book.get("student_id")
    if not sid:
        print(json.dumps({"ok": False, "error": f"book {book_id} has no owner (student_id)"}))
        return 2

    # The admin's size dial. Set on the admin page, honoured here.
    global CFG
    size_row = fetch_one("SELECT value FROM settings WHERE key = 'course_size'")
    size = (size_row or {}).get("value", "XS")
    CFG = SIZES.get(size, SIZES["XS"])
    print(f"[lecture-gen] course size: {size} ({CFG['slides']} slides, "
          f"{CFG['lecture_qs']}+{CFG['self_qs']} questions per week)", flush=True)

    try:
        progress(book_id, "Reading the book…")
        pages = read_pages(pdf_path)
        execute("UPDATE books SET pages = %s WHERE id = %s", (len(pages), book_id))
        book_title = book.get("title") or book.get("filename") or pdf_path.stem
        progress(book_id, "Finding chapters and planning the semester…")
        plan, weeks = build_semester_plan(pages, book_title)
        total_weeks = plan.week_count
        write_semester_plan(sid, plan)

        if quizzes_only:
            regenerate_quizzes(sid, book_id, weeks)
            execute(
                "UPDATE books SET status = 'ready', progress = %s WHERE id = %s",
                (f"Quizzes rewritten — {total_weeks} weeks.", book_id),
            )
            print(json.dumps({"ok": True, "weeks": total_weeks, "quizzes_only": True}))
            return 0

        for planned_week, week_pages in weeks:
            week = planned_week.week
            first, last = week_pages[0][0], week_pages[-1][0]
            chapter_titles = "; ".join(part.title for part in planned_week.chapters)
            progress(book_id, f"Writing lecture {week} of {total_weeks} (pages {first}-{last})…")
            lecture = generate_week(week, total_weeks, chapter_titles, week_pages)
            progress(book_id, f"Writing quiz {week} of {total_weeks} — “{lecture['title']}”…")
            spoken = [{"text": lecture["intro"]}] + [
                {"text": slide["narration"]} for slide in lecture["slides"]
            ]
            quiz = generate_quiz(lecture["title"], spoken, week_pages)
            write_week(sid, week, lecture, quiz)
            execute(
                "UPDATE lectures SET title = %s WHERE week = %s AND student_id = %s",
                (lecture["title"].strip(), week, sid),
            )

        progress(book_id, "Building the slide decks…")
        build_slides(sid)

        progress(book_id, "Recording the lecturer's voice…")
        prerender_voice(sid)

        execute(
            "UPDATE books SET status = 'ready', progress = %s WHERE id = %s",
            (f"Course ready — {total_weeks} lectures generated from {len(pages)} pages.", book_id),
        )
        print(json.dumps({"ok": True, "weeks": total_weeks, "pages": len(pages)}))
        return 0
    except Exception as exc:  # noqa: BLE001 - a failed run must land in books.error
        detail = f"{type(exc).__name__}: {exc}"
        execute(
            "UPDATE books SET status = 'failed', error = %s, progress = NULL WHERE id = %s",
            (detail, book_id),
        )
        print(json.dumps({"ok": False, "error": detail}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
