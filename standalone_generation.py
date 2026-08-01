"""Deterministic section-aware course generation for explicit standalone mode."""

from __future__ import annotations

import json
import re
from pathlib import Path

from contracts import validate_course

def _source_sections(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8")
    parts = [part.strip() for part in re.split(r"\n(?=## )", text) if part.strip()]
    body = [part for part in parts if part.startswith("## ")]
    if not body:
        raise ValueError("standalone source needs Markdown level-two sections")
    return body


def _question(week: int, index: int, source: str) -> dict:
    correct = "ABCD"[index % 4]
    return {
        "prompt": f"Which statement matches the Week {week} learning material (item {index + 1})?",
        "type": "mcq",
        "options": [
            f"A) Week {week} uses explicit evidence",
            "B) Results may ignore ownership",
            "C) Configuration should be guessed",
            "D) External services are always required",
        ],
        "correct_option": correct,
        "source": source,
    }


def generate_course(source: Path, output_root: Path) -> Path:
    sections = _source_sections(source)
    week_count = len(sections)
    output_root.mkdir(parents=True, exist_ok=True)
    for week, section in enumerate(sections, start=1):
        lines = [line.strip("# ").strip() for line in section.splitlines() if line.strip()]
        title = lines[0] if lines else f"Standalone Week {week}"
        content = " ".join(lines[1:]) or "Use deterministic, tenant-safe development data."
        pages = [week]
        lecture = {
            "lectureId": f"week-{week}",
            "title": title,
            "segments": [
                {
                    "slide": 1,
                    "text": f"Welcome to week {week}. We will study {title}.",
                    "citations": [{"page": pages[0]}],
                },
                {
                    "slide": 2,
                    "text": content,
                    "citations": [{"page": pages[0]}],
                },
                {
                    "slide": 3,
                    "text": "Apply the idea with explicit configuration, stable contracts, and tenant isolation.",
                    "citations": [{"page": pages[0]}],
                },
            ],
        }
        questions = [_question(week, index, "lecture") for index in range(8)]
        questions += [_question(week, index + 8, "self_study") for index in range(2)]
        folder = output_root / f"week-{week}"
        folder.mkdir(parents=True, exist_ok=True)
        deck = (
            "---\n"
            "theme: default\n"
            "routerMode: hash\n"
            f'title: "Week {week} - {title}"\n'
            "---\n\n"
            f"# Week {week}\n## {title}\n\n---\n\n# Core idea\n\n"
            f"- {content[:120]}\n\n<small>Source: p.{week}</small>\n"
        )
        (folder / "slides.md").write_text(deck, encoding="utf-8")
        (folder / "script.json").write_text(
            json.dumps(lecture, indent=2) + "\n", encoding="utf-8"
        )
        (folder / "quiz.json").write_text(
            json.dumps(
                {"week": week, "title": title, "questions": questions}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    validate_course(output_root)
    (output_root / "run.json").write_text(
        json.dumps(
            {
                "mode": "standalone",
                "weeks": week_count,
                "integration_side_effects": {
                    "database": "skipped",
                    "slide_build": "skipped",
                    "voice_prerender": "skipped",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_root
