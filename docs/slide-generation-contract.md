# Agentic slide generation contract

## TL;DR

Keep Slidev. The weak decks were not a Slidev limitation: UnivAI only asked the
model for a heading and bullets, then compiled every answer into the same
default layout. The new contract lets the model choose from nine semantic
layouts, while Python owns all markup, styling, validation, escaping, and
fallbacks.

## Why Slidev stays

Slidev already fits the live lecture pipeline and is installed by the parent
repository. Its official documentation confirms that it supports reusable
layouts and Vue components, built-in slide transitions and motion, Shiki code
highlighting, KaTeX, and text-defined diagrams:

- Layouts: <https://sli.dev/guide/layout>
- Components: <https://sli.dev/guide/component>
- Animations and transitions: <https://sli.dev/guide/animations>
- Code syntax and highlighting: <https://sli.dev/guide/syntax>
- LaTeX via KaTeX: <https://sli.dev/features/latex>
- Mermaid diagrams: <https://sli.dev/features/mermaid>

Marp is a good lightweight Markdown-to-slide converter with themes and exports
(<https://marp.app/>). Reveal.js is a strong web presentation framework with
Markdown, code, math, and Auto-Animate (<https://revealjs.com/> and
<https://revealjs.com/auto-animate/>). Neither solves UnivAI's real problem:
the missing visual-generation contract. Switching would also replace a working
build, cache, route, and live slide-sync integration. This is why UnivAI keeps
Slidev and fixes the layer above it.

## The model's allowed vocabulary

The model returns plain JSON. It never returns Slidev frontmatter, Markdown,
HTML, CSS, Vue, URLs, image paths, or animation code.

| Layout | Use it for | Required visual data |
| --- | --- | --- |
| `concept` | One idea and its takeaway | `{}` |
| `cards` | Two to five peer ideas | `{}` |
| `process` | A real ordered sequence | `steps` |
| `comparison` | A genuine A/B contrast | Two titles and two item lists |
| `diagram` | One center and related nodes | `center`, `nodes` |
| `code` | A grounded worked snippet | Allowed language, up to 12 lines, highlighted lines |
| `formula` | A supported mathematical relationship | Safe KaTeX and a plain explanation |
| `data` | Values present in the source | Two to four value/label metrics |
| `quote` | A short exact excerpt | Quote and optional attribution |

Every slide also has:

- one specific heading, at most nine words;
- one to five short supporting points;
- an optional callout;
- zero to three structured emphasis instructions: `bold`, `underline`, or
  `accent`;
- grounded narration and one page citation.

For a normal batch, the prompt asks for at least three layout types and at least
two rich layouts. A layout cannot repeat three times in a row. Visual structure
must come from the supplied textbook pages; decoration is never permission to
invent a number, quote, formula, code behavior, or relationship.

## Example

```json
{
  "heading": "A Write Becomes Durable",
  "layout": "process",
  "bullets": ["Acknowledge only after durable storage"],
  "callout": "Durability changes the safe acknowledgement point.",
  "emphasis": [
    {"text": "durable storage", "style": "underline"}
  ],
  "visual": {
    "steps": ["Receive write", "Append to log", "Flush safely", "Acknowledge"]
  },
  "narration": "The lecturer's grounded spoken explanation goes here.",
  "page": 12
}
```

## Validation and fallback behavior

1. The existing lecture checks still reject missing titles, empty bullets,
   missing citations, and unusably short narration.
2. The visual validator checks layout names, required fields, batch variety,
   code limits, the language allowlist, emphasis references, and safe LaTeX.
3. Visual errors do not destroy an otherwise grounded lecture. They are logged,
   bounded, and deterministically reduced to a readable concept or card slide.
4. Legacy heading-and-bullet decks use the same fallback and receive the new
   visual theme when rebuilt.
5. The compiler escapes every model-authored text field. Only static,
   repository-owned markup and CSS can execute.

Code fences are sized around any backticks in model output. LaTeX rejects HTML,
remote-link, macro-definition, include, and slide-separator constructs. Code
languages and highlighted line numbers are allowlisted and bounded. Remote
assets are not part of the schema.

## Live lecture effects

The compiled deck uses a restrained built-in `fade` transition, because live
voice controls slides directly and click-to-reveal content could hide material
when no presenter click occurs. Bold, underline, and accent emphasis are
compiled from structured instructions and are therefore visible immediately
when the synced slide arrives. Shapes, comparison panels, process steps,
relationship hubs, metric cards, code, and formula treatments are static and
do not disturb slide-number synchronization.

## Tests

`tests/generation/test_slide_design.py` verifies all layout contracts, batch
variety, unsafe-LaTeX fallback, HTML escaping, structured emphasis, code-line
highlighting, diagram shapes, and a real production Slidev build when Slidev is
installed in the parent checkout.
