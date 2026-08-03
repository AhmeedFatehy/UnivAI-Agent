# System prompt catalog

All UnivAI Agent system prompts live in this folder. Python code selects a
prompt by `PromptOperation`; it must not choose a YAML filename directly.
`registry.yaml` is the single operation-to-prompt map.

## Folder map

| Folder | Used for |
|---|---|
| `curriculum/` | book structure, topic extraction, semester planning and repair |
| `teaching/` | lecture writing, summaries and learning objectives |
| `assessment/` | one prompt per assessment type, repair and explanations |
| `retrieval/` | query rewriting, grounded answers and refusals |
| `evaluation/` | retrieval, answer and assessment quality checks |
| `shared/` | structured-output repair shared by every agent |

## Semester rules

The approved chapter inventory is the source of truth. The semester prompt may
propose a layout, but `planning/semester_planner.py` enforces these rules:

- one normal chapter is one week;
- adjacent small chapters may be combined;
- a tiny chapter may share with two adjacent chapters;
- no week contains more than three chapters;
- one large chapter may be split into exactly two complete, non-overlapping weeks;
- chapters cannot be dropped, invented, duplicated, reordered, or grouped when non-adjacent.

If headings cannot be found, the planner records a low-confidence warning and
treats the readable book as one chapter. A very large fallback chapter can
still become two weeks. The upload generator saves the validated result as
`semester-plan.json` so other endpoints can use the same week boundaries.

## Assessment routing

These operations intentionally use different prompts:

| Assessment | Operation | Prompt |
|---|---|---|
| Diagnostic | `assessment.generate:diagnostic` | `assessment/diagnostic.yaml` |
| Practice | `assessment.generate:practice` | `assessment/practice.yaml` |
| Quiz | `assessment.generate:quiz` | `assessment/quiz.yaml` |
| Assignment | `assessment.generate:assignment` | `assessment/assignment.yaml` |
| Midterm | `assessment.generate:midterm` | `assessment/midterm.yaml` |
| Final | `assessment.generate:final` | `assessment/final.yaml` |
| Oral exam | `assessment.generate:oral_exam` | `assessment/oral_exam.yaml` |

Every assessment receives a covered scope, question count, difficulty mix,
allowed formats, and grounded evidence. Every returned assessment records its
type and rejects a reply that claims to be a different type.

## Post-lecture section packs

`teaching/section_generation` (`content.generate_section`) drafts a section for a
live follow-up: ordered activities, worked examples and explicit learner TODOs,
each forced to cite supplied evidence. `teaching/section_repair`
(`content.repair_section`) is the bounded repair prompt used when that draft is
malformed. `planning/section_planner.py` enforces the deterministic invariants:
every source id resolves against the evidence, the total activity duration stays
inside the configured budget, and objectives are non-empty and distinct. A pack
that the evidence cannot support is refused, never padded with generic teaching
content.

## Adding or changing a prompt

1. Add one stable ID to `PromptId` and one caller-facing operation to
   `PromptOperation` in `agents/prompts.py`.
2. Add the YAML file with its version, owner, variables, output schema,
   grounding policy, safety rules, system message, and user template.
3. Map the operation to the prompt ID in `registry.yaml`.
4. Load it with `load_prompt_for(PromptOperation.…)` in the caller.
5. Add or update a test and run `uv run pytest -q`.

`validate_prompt_catalog()` fails if an enum operation has no route, a prompt
ID has no route, a YAML file is missing or extra, two prompts claim the same
operation, or required template variables are not supplied. Prompt changes
use semantic versions so traces can show exactly which prompt produced output.

## Cross-book learning paths

`curriculum/book_prerequisite_analysis` may only propose edges supported on both
sides by supplied excerpts. `planning/learning_path.py` validates document and
collection provenance, topologically orders whole books, and appends each
chapter-complete per-book plan serially. Cycles, weak/missing evidence, duplicate
editions, overlap, ambiguous choices, and disconnected books remain visible as
versioned warnings. The artifact stays pending (or blocked) until the App records
approval of the exact schema version and any warning overrides.
