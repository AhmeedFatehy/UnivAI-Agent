"""A small, safe visual language for AI-authored Slidev decks.

The model chooses from semantic layouts and supplies plain data.  It never
writes Vue, HTML, CSS, URLs, or Slidev frontmatter.  This module validates that
data, repairs cosmetic failures deterministically, and compiles it into the
only markup the presentation runtime is allowed to execute.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


LAYOUTS = (
    "concept",
    "cards",
    "process",
    "comparison",
    "diagram",
    "code",
    "formula",
    "data",
    "quote",
)
RICH_LAYOUTS = frozenset(LAYOUTS) - {"concept", "cards"}
EMPHASIS_STYLES = frozenset({"bold", "underline", "accent"})
CODE_LANGUAGES = frozenset(
    {
        "bash",
        "c",
        "cpp",
        "csharp",
        "css",
        "go",
        "html",
        "java",
        "javascript",
        "json",
        "kotlin",
        "markdown",
        "php",
        "plaintext",
        "python",
        "rust",
        "sql",
        "typescript",
        "yaml",
    }
)

SLIDE_DESIGN_INSTRUCTIONS = """\
Visual design contract (plain JSON data only; never output Markdown, HTML, CSS, URLs, or image paths):
- Every slide has one concrete idea, a heading of at most 9 words, 1-5 short bullets,
  layout, callout, emphasis, visual, narration, and page.
- layout is one of: concept, cards, process, comparison, diagram, code, formula, data, quote.
- callout is an optional one-sentence takeaway, otherwise "".
- emphasis is 0-3 items like {"text":"exact phrase already on the slide","style":"bold"};
  style is bold, underline, or accent. Do not add new facts through emphasis.
- visual is {} for concept/cards. Otherwise use exactly the matching shape:
  process: {"steps":["ordered step", "ordered step", "ordered step"]}
  comparison: {"leftTitle":"...","leftItems":["..."],"rightTitle":"...","rightItems":["..."]}
  diagram: {"center":"central idea","nodes":["related node", "related node"]}
  code: {"language":"python","code":"max 12 lines","highlightLines":[2,3]}
  formula: {"latex":"safe KaTeX expression","explanation":"what the symbols mean"}
  data: {"metrics":[{"value":"grounded value","label":"what it measures"}]}
  quote: {"quote":"short exact excerpt from a supplied page","attribution":"source name or page"}
- In a batch of 4+ slides, use at least 3 layout types and at least 2 rich layouts
  (process/comparison/diagram/code/formula/data/quote). Never repeat one layout 3 times.
- Choose a rich layout only when the textbook supports its structure. Never invent a
  number, quotation, formula, relationship, or code behavior for decoration.
- Prefer comparison for genuine contrasts, process for ordered actions, diagram for
  relationships, data only for supplied values, and code/formula only when they teach the idea.

Example process slide:
{"heading":"A Write Becomes Durable","layout":"process","bullets":["Acknowledge only after durable storage"],"callout":"Durability changes the safe acknowledgement point.","emphasis":[{"text":"durable storage","style":"underline"}],"visual":{"steps":["Receive write","Append to log","Flush safely","Acknowledge"]},"narration":"...","page":12}
"""

_STYLE_PATH = Path(__file__).with_name("slidev_theme.css")
_WHITESPACE = re.compile(r"\s+")
_UNSAFE_LATEX = re.compile(
    r"(?:<|>|\$|\\(?:href|url|includegraphics|html(?:Class|Id|Style|Data)?|class|style|cssId|data|def|gdef|newcommand|renewcommand|input|include)\b)",
    re.IGNORECASE,
)


def _plain(value: Any, *, limit: int = 220) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()[:limit]


def _items(value: Any, *, count: int = 5, limit: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _plain(item, limit=limit)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) == count:
            break
    return cleaned


def _safe_latex(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    latex = value.strip()[:320]
    if (
        not latex
        or "---" in latex
        or _UNSAFE_LATEX.search(latex)
        or latex.count("{") != latex.count("}")
        or latex.count("[") != latex.count("]")
    ):
        return ""
    return latex


def _safe_code(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lines = value.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line[:140] for line in lines[:12]).strip()


def _normalise_visual(layout: str, raw: Any) -> dict[str, Any]:
    visual = raw if isinstance(raw, dict) else {}
    if layout == "process":
        return {"steps": _items(visual.get("steps"), count=5, limit=80)}
    if layout == "comparison":
        return {
            "leftTitle": _plain(visual.get("leftTitle"), limit=50),
            "leftItems": _items(visual.get("leftItems"), count=4, limit=90),
            "rightTitle": _plain(visual.get("rightTitle"), limit=50),
            "rightItems": _items(visual.get("rightItems"), count=4, limit=90),
        }
    if layout == "diagram":
        return {
            "center": _plain(visual.get("center"), limit=70),
            "nodes": _items(visual.get("nodes"), count=6, limit=70),
        }
    if layout == "code":
        language = _plain(visual.get("language"), limit=20).lower()
        if language not in CODE_LANGUAGES:
            language = "plaintext"
        code = _safe_code(visual.get("code"))
        line_count = max(1, len(code.splitlines()))
        raw_highlights = visual.get("highlightLines")
        highlights = []
        if isinstance(raw_highlights, list):
            highlights = sorted(
                {
                    line
                    for line in raw_highlights
                    if isinstance(line, int) and not isinstance(line, bool) and 1 <= line <= line_count
                }
            )[:6]
        return {"language": language, "code": code, "highlightLines": highlights}
    if layout == "formula":
        return {
            "latex": _safe_latex(visual.get("latex")),
            "explanation": _plain(visual.get("explanation"), limit=150),
        }
    if layout == "data":
        metrics = []
        raw_metrics = visual.get("metrics")
        if isinstance(raw_metrics, list):
            for metric in raw_metrics[:4]:
                if not isinstance(metric, dict):
                    continue
                value = _plain(metric.get("value"), limit=35)
                label = _plain(metric.get("label"), limit=75)
                if value and label:
                    metrics.append({"value": value, "label": label})
        return {"metrics": metrics}
    if layout == "quote":
        return {
            "quote": _plain(visual.get("quote"), limit=240),
            "attribution": _plain(visual.get("attribution"), limit=90),
        }
    return {}


def _visual_ready(layout: str, visual: dict[str, Any]) -> bool:
    if layout in {"concept", "cards"}:
        return True
    if layout == "process":
        return len(visual.get("steps", [])) >= 3
    if layout == "comparison":
        return bool(
            visual.get("leftTitle")
            and visual.get("rightTitle")
            and visual.get("leftItems")
            and visual.get("rightItems")
        )
    if layout == "diagram":
        return bool(visual.get("center") and len(visual.get("nodes", [])) >= 2)
    if layout == "code":
        return bool(visual.get("code"))
    if layout == "formula":
        return bool(visual.get("latex") and visual.get("explanation"))
    if layout == "data":
        return len(visual.get("metrics", [])) >= 2
    if layout == "quote":
        return bool(visual.get("quote"))
    return False


def _fallback_layout(position: int, bullets: list[str]) -> str:
    return "cards" if len(bullets) >= 2 and position % 2 == 0 else "concept"


def normalise_slide(raw: Any, position: int = 1) -> dict[str, Any]:
    """Return a bounded slide; malformed visuals degrade to readable text."""
    slide = raw if isinstance(raw, dict) else {}
    bullets = _items(slide.get("bullets"), count=5, limit=120)
    layout = _plain(slide.get("layout"), limit=20).lower()
    if layout not in LAYOUTS:
        layout = _fallback_layout(position, bullets)
    visual = _normalise_visual(layout, slide.get("visual"))
    if not _visual_ready(layout, visual):
        layout = _fallback_layout(position, bullets)
        visual = {}

    emphasis = []
    raw_emphasis = slide.get("emphasis")
    if isinstance(raw_emphasis, list):
        for item in raw_emphasis[:3]:
            if not isinstance(item, dict):
                continue
            text = _plain(item.get("text"), limit=80)
            style = _plain(item.get("style"), limit=16).lower()
            if text and style in EMPHASIS_STYLES:
                emphasis.append({"text": text, "style": style})

    page = slide.get("page")
    if not isinstance(page, int) or isinstance(page, bool):
        page = 1
    return {
        "heading": _plain(slide.get("heading"), limit=100) or "Key idea",
        "layout": layout,
        "bullets": bullets or ["Review the lecturer's explanation"],
        "callout": _plain(slide.get("callout"), limit=180),
        "emphasis": emphasis,
        "visual": visual,
        "narration": _plain(slide.get("narration"), limit=2600),
        "page": max(1, page),
    }


def slide_design_problem(raw: Any, position: int) -> str | None:
    """Explain the first contract error precisely enough for an LLM repair."""
    if not isinstance(raw, dict):
        return f"slide {position} must be an object"
    heading = _plain(raw.get("heading"), limit=500)
    if len(heading.split()) > 9:
        return f"slide {position} heading must be at most 9 words"
    bullets = raw.get("bullets")
    if not isinstance(bullets, list) or not 1 <= len(bullets) <= 5:
        return f"slide {position} needs 1-5 bullets"
    for bullet in bullets:
        if not isinstance(bullet, str) or not bullet.strip():
            return f"slide {position} has an empty bullet"
        if len(bullet.split()) > 14:
            return f"slide {position} bullets must be at most 14 words each"

    layout = raw.get("layout")
    if layout not in LAYOUTS:
        return f"slide {position} layout must be one of {', '.join(LAYOUTS)}"
    raw_visual = raw.get("visual")
    if not isinstance(raw_visual, dict):
        return f"slide {position} visual must be an object"
    if layout == "formula" and not _safe_latex(raw_visual.get("latex")):
        return f"slide {position} formula contains unsupported or unsafe LaTeX"
    if layout == "code":
        language = _plain(raw_visual.get("language"), limit=20).lower()
        if language not in CODE_LANGUAGES:
            return f"slide {position} code language is not in the supported allowlist"
        raw_code = raw_visual.get("code")
        if isinstance(raw_code, str) and len(raw_code.splitlines()) > 12:
            return f"slide {position} code must be at most 12 lines"
    visual = _normalise_visual(layout, raw.get("visual"))
    if not _visual_ready(layout, visual):
        return f"slide {position} has incomplete visual data for its {layout} layout"

    emphasis = raw.get("emphasis")
    if not isinstance(emphasis, list) or len(emphasis) > 3:
        return f"slide {position} emphasis must be a list of at most 3 items"
    for item in emphasis:
        if not isinstance(item, dict) or item.get("style") not in EMPHASIS_STYLES:
            return f"slide {position} emphasis style must be bold, underline, or accent"
        if not _plain(item.get("text"), limit=80):
            return f"slide {position} emphasis text is empty"
        searchable = " ".join(
            [
                heading,
                *[str(bullet) for bullet in bullets],
                str(raw.get("callout") or ""),
                json.dumps(raw_visual, ensure_ascii=False),
            ]
        )
        if str(item["text"]).lower() not in searchable.lower():
            return f"slide {position} emphasis text must already appear on the slide"
    if not isinstance(raw.get("callout"), str):
        return f"slide {position} callout must be a string (use an empty string when unused)"
    return None


def batch_design_problem(slides: Any) -> str | None:
    if not isinstance(slides, list):
        return "slides must be a list"
    for position, slide in enumerate(slides, start=1):
        problem = slide_design_problem(slide, position)
        if problem:
            return problem
    if len(slides) >= 4:
        layouts = [slide["layout"] for slide in slides]
        if len(set(layouts)) < 3:
            return "the slide batch must use at least 3 different layout types"
        if sum(layout in RICH_LAYOUTS for layout in layouts) < 2:
            return "the slide batch needs at least 2 rich visual layouts"
        for index in range(len(layouts) - 2):
            if len(set(layouts[index : index + 3])) == 1:
                return f"slides {index + 1}-{index + 3} repeat the same layout"
    return None


def _emphasised(value: str, emphasis: list[dict[str, str]]) -> str:
    phrases: dict[str, str] = {}
    for item in emphasis:
        phrase = item["text"]
        if phrase.lower() in value.lower():
            phrases.setdefault(phrase.lower(), item["style"])
    if not phrases:
        return html.escape(value)
    pattern = re.compile(
        "|".join(re.escape(phrase) for phrase in sorted(phrases, key=len, reverse=True)),
        re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        escaped = html.escape(match.group(0))
        style = phrases[match.group(0).lower()]
        if style == "bold":
            return f"<strong>{escaped}</strong>"
        if style == "underline":
            return f'<span class="ua-underline">{escaped}</span>'
        return f'<span class="ua-accent">{escaped}</span>'

    output: list[str] = []
    cursor = 0
    for match in pattern.finditer(value):
        output.append(html.escape(value[cursor : match.start()]))
        output.append(replace(match))
        cursor = match.end()
    output.append(html.escape(value[cursor:]))
    return "".join(output)


def _bullet_list(items: list[str], emphasis: list[dict[str, str]], class_name: str = "ua-list") -> str:
    body = "".join(f"<li>{_emphasised(item, emphasis)}</li>" for item in items)
    return f'<ul class="{class_name}">{body}</ul>'


def _callout(slide: dict[str, Any]) -> str:
    if not slide["callout"]:
        return ""
    return (
        '<div class="ua-callout"><span>Remember</span>'
        f'<p>{_emphasised(slide["callout"], slide["emphasis"])}</p></div>'
    )


def _render_html_body(slide: dict[str, Any]) -> str:
    layout = slide["layout"]
    visual = slide["visual"]
    emphasis = slide["emphasis"]
    if layout == "cards":
        cards = "".join(
            '<div class="ua-card">'
            f'<span>{index:02}</span><p>{_emphasised(item, emphasis)}</p></div>'
            for index, item in enumerate(slide["bullets"], start=1)
        )
        return f'<div class="ua-card-grid">{cards}</div>{_callout(slide)}'
    if layout == "process":
        steps = "".join(
            '<div class="ua-step">'
            f'<span>{index}</span><p>{_emphasised(item, emphasis)}</p></div>'
            + ('<div class="ua-arrow">&#8594;</div>' if index < len(visual["steps"]) else "")
            for index, item in enumerate(visual["steps"], start=1)
        )
        return f'<div class="ua-process">{steps}</div>{_callout(slide)}'
    if layout == "comparison":
        left = _bullet_list(visual["leftItems"], emphasis, "ua-compare-list")
        right = _bullet_list(visual["rightItems"], emphasis, "ua-compare-list")
        return (
            '<div class="ua-compare">'
            f'<section><span>A</span><h2>{_emphasised(visual["leftTitle"], emphasis)}</h2>{left}</section>'
            f'<section><span>B</span><h2>{_emphasised(visual["rightTitle"], emphasis)}</h2>{right}</section>'
            f'</div>{_callout(slide)}'
        )
    if layout == "diagram":
        nodes = "".join(
            f'<div class="ua-node">{_emphasised(node, emphasis)}</div>'
            for node in visual["nodes"]
        )
        return (
            '<div class="ua-diagram">'
            f'<div class="ua-hub">{_emphasised(visual["center"], emphasis)}</div>'
            f'<div class="ua-nodes">{nodes}</div></div>{_callout(slide)}'
        )
    if layout == "data":
        metrics = "".join(
            '<div class="ua-metric">'
            f'<strong>{_emphasised(metric["value"], emphasis)}</strong>'
            f'<span>{_emphasised(metric["label"], emphasis)}</span></div>'
            for metric in visual["metrics"]
        )
        return f'<div class="ua-metrics">{metrics}</div>{_callout(slide)}'
    if layout == "quote":
        attribution = (
            f'<footer>{_emphasised(visual["attribution"], emphasis)}</footer>'
            if visual["attribution"]
            else ""
        )
        return (
            '<blockquote class="ua-quote"><span>&ldquo;</span>'
            f'<p>{_emphasised(visual["quote"], emphasis)}</p>{attribution}</blockquote>'
            f'{_callout(slide)}'
        )
    return (
        '<div class="ua-concept">'
        f'{_bullet_list(slide["bullets"], emphasis)}{_callout(slide)}</div>'
    )


def _code_fence(code: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


def _render_special_body(slide: dict[str, Any]) -> str:
    visual = slide["visual"]
    if slide["layout"] == "code":
        highlights = visual["highlightLines"]
        marker = "{" + ",".join(str(line) for line in highlights) + "}" if highlights else ""
        fence = _code_fence(visual["code"])
        return (
            '<div class="ua-code-label">Worked example</div>\n\n'
            f'{fence}{visual["language"]} {marker}\n{visual["code"]}\n{fence}\n\n'
            f'{_callout(slide)}'
        )
    if slide["layout"] == "formula":
        return (
            '<div class="ua-formula">\n\n'
            f'$$\n{visual["latex"]}\n$$\n\n'
            f'<p>{_emphasised(visual["explanation"], slide["emphasis"])}</p>\n'
            f'</div>{_callout(slide)}'
        )
    return _render_html_body(slide)


def _slide_markup(slide: dict[str, Any], *, week: int, position: int) -> str:
    heading = _emphasised(slide["heading"], slide["emphasis"])
    header = (
        '<div class="ua-meta">'
        f'<span>WEEK {week:02}</span><span>{position:02}</span></div>\n'
        f'<h1 class="ua-heading">{heading}</h1>\n'
        '<div class="ua-rule"></div>\n'
    )
    body = _render_special_body(slide)
    source = f'<div class="ua-source">SOURCE&nbsp;&middot;&nbsp;P.{slide["page"]}</div>'
    return header + body + "\n" + source


def render_slidev_markdown(deck: dict[str, Any]) -> str:
    """Compile a database-owned deck into deterministic, self-contained Slidev."""
    title = _plain(deck.get("title"), limit=120) or "Course lecture"
    raw_week = deck.get("week")
    week = raw_week if isinstance(raw_week, int) and not isinstance(raw_week, bool) else 1
    styles = _STYLE_PATH.read_text(encoding="utf-8")
    pages = [
        "---\n"
        "theme: default\n"
        "layout: cover\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        "transition: fade\n"
        "clickAnimation: fade\n"
        "colorSchema: light\n"
        "mdc: true\n"
        "---\n\n"
        f"<style>\n{styles}\n</style>\n\n"
        '<div class="ua-cover">\n'
        '  <div class="ua-cover-shapes" aria-hidden="true"><i></i><i></i><i></i></div>\n'
        f'  <div class="ua-cover-kicker">UNIVAI &nbsp;/&nbsp; WEEK {week:02}</div>\n'
        f'  <h1>{html.escape(title)}</h1>\n'
        '  <p>Ideas, evidence, and the relationships that connect them.</p>\n'
        '  <div class="ua-cover-rule"></div>\n'
        "</div>\n"
    ]
    raw_slides = deck.get("slides") if isinstance(deck.get("slides"), list) else []
    for position, raw_slide in enumerate(raw_slides, start=1):
        slide = normalise_slide(raw_slide, position)
        pages.append(
            "---\nlayout: default\nclass: ua-slide\ntransition: fade\n---\n\n"
            + _slide_markup(slide, week=week, position=position)
            + "\n"
        )
    return "\n".join(pages)


__all__ = [
    "CODE_LANGUAGES",
    "EMPHASIS_STYLES",
    "LAYOUTS",
    "RICH_LAYOUTS",
    "SLIDE_DESIGN_INSTRUCTIONS",
    "batch_design_problem",
    "normalise_slide",
    "render_slidev_markdown",
    "slide_design_problem",
]
