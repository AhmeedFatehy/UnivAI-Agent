from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from generation.slide_design import (
    batch_design_problem,
    normalise_slide,
    render_slidev_markdown,
    slide_design_problem,
)


def slide(layout: str, visual: dict) -> dict:
    return {
        "heading": "A concrete idea",
        "layout": layout,
        "bullets": ["A short grounded point", "A second grounded point"],
        "callout": "Keep the decision explicit.",
        "emphasis": [{"text": "grounded", "style": "underline"}],
        "visual": visual,
        "narration": "This is a sufficiently detailed spoken explanation for the slide content.",
        "page": 12,
    }


def test_visual_contract_accepts_each_bounded_layout():
    visuals = {
        "concept": {},
        "cards": {},
        "process": {"steps": ["Receive", "Validate", "Commit"]},
        "comparison": {
            "leftTitle": "Before",
            "leftItems": ["Uncommitted"],
            "rightTitle": "After",
            "rightItems": ["Durable"],
        },
        "diagram": {"center": "Coordinator", "nodes": ["Worker A", "Worker B"]},
        "code": {"language": "python", "code": "value = read()\ncommit(value)", "highlightLines": [2]},
        "formula": {"latex": "T = R + W", "explanation": "Latency combines read and write work."},
        "data": {
            "metrics": [
                {"value": "3", "label": "replicas"},
                {"value": "2", "label": "acknowledgements"},
            ]
        },
        "quote": {"quote": "Failures are part of the design.", "attribution": "Textbook, p.12"},
    }

    for layout, visual in visuals.items():
        assert slide_design_problem(slide(layout, visual), 1) is None


def test_batch_requires_visual_variety_without_arbitrary_markup():
    batch = [
        slide("concept", {}),
        slide("process", {"steps": ["Read", "Check", "Write"]}),
        slide(
            "comparison",
            {
                "leftTitle": "Weak",
                "leftItems": ["One copy"],
                "rightTitle": "Strong",
                "rightItems": ["Replicated"],
            },
        ),
        slide("cards", {}),
    ]

    assert batch_design_problem(batch) is None
    assert "at least 3" in (batch_design_problem([slide("concept", {})] * 4) or "")


def test_invalid_or_unsafe_visual_degrades_to_readable_cards():
    unsafe = slide(
        "formula",
        {"latex": r"\href{https://example.com}{x}", "explanation": "Remote link"},
    )

    normalised = normalise_slide(unsafe, 2)

    assert normalised["layout"] == "cards"
    assert normalised["visual"] == {}
    assert "unsafe LaTeX" in (slide_design_problem(unsafe, 2) or "")


def test_renderer_escapes_model_text_and_applies_only_known_emphasis():
    unsafe = slide("cards", {})
    unsafe["heading"] = "Why <script>alert(1)</script> fails"
    unsafe["bullets"] = ["Keep this grounded", '<img src="remote">']
    unsafe["emphasis"] = [
        {"text": "grounded", "style": "underline"},
        {"text": "Why", "style": "bold"},
    ]
    deck = render_slidev_markdown(
        {"week": 3, "title": "Safe rendering", "slides": [unsafe]}
    )

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in deck
    assert '<span class="ua-underline">grounded</span>' in deck
    assert "<strong>Why</strong>" in deck
    assert '<img src="remote">' not in deck
    assert "&lt;img src=&quot;remote&quot;&gt;" in deck


def test_renderer_uses_slidev_code_highlights_and_safe_diagram_shapes():
    code = slide(
        "code",
        {
            "language": "python",
            "code": "value = read()\ncommit(value)",
            "highlightLines": [2],
        },
    )
    diagram = slide("diagram", {"center": "Leader", "nodes": ["Replica A", "Replica B"]})

    deck = render_slidev_markdown(
        {"week": 4, "title": "Replication", "slides": [code, diagram]}
    )

    assert "```python {2}" in deck
    assert '<div class="ua-hub">Leader</div>' in deck
    assert deck.count('class="ua-node"') == 2
    assert "transition: fade" in deck


def test_generated_markdown_builds_with_installed_slidev():
    campus_root = Path(__file__).resolve().parents[3]
    executable = campus_root / "node_modules" / ".bin" / (
        "slidev.cmd" if os.name == "nt" else "slidev"
    )
    if not executable.is_file():
        pytest.skip("the parent checkout has not installed Slidev")

    slides = [
        slide("cards", {}),
        slide("process", {"steps": ["Receive", "Validate", "Commit"]}),
        slide(
            "formula",
            {"latex": r"T = R + W", "explanation": "Total work combines reads and writes."},
        ),
        slide(
            "code",
            {
                "language": "python",
                "code": "value = read()\ncommit(value)",
                "highlightLines": [2],
            },
        ),
    ]
    cache_root = campus_root / ".cache"
    cache_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="univai-slidev-test-", dir=cache_root) as temp:
        temp_path = Path(temp)
        entry = temp_path / "slides.md"
        target = temp_path / "dist"
        entry.write_text(
            render_slidev_markdown({"week": 2, "title": "Build check", "slides": slides}),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                str(executable),
                "build",
                str(entry),
                "--out",
                str(target),
                "--base",
                "/test-deck/",
            ],
            cwd=campus_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert (target / "index.html").is_file()
