# UnivAI Standalone Course

This project-authored sample exists only for deterministic local development.

## Evidence and Sources

Reliable answers cite the supplied learning material. A system should say when
the material does not cover a question instead of inventing an answer.

## Tenant Isolation

Each learner owns a separate document collection. Listing, retrieval, and
deletion must filter by the learner identifier so one learner cannot see
another learner's material.

## Explicit Runtime Modes

Standalone mode uses deterministic local fixtures. Integrated mode uses the
real vector database, embedding models, language model, database, slide build,
and voice prerender services. The system never chooses fixtures silently.

## Stable Contracts

Lecture scripts align narration to slide numbers and include page citations.
Quiz questions contain four labelled options, one correct option, and a source
value of lecture or self_study.
