"""Topic identity and overlap detection across books.

Three books on adjacent subjects will teach the same thing more than once. The
planner has to notice that before it schedules the same material twice, and it
has to be able to *show* why it thought two topics were the same.

Overlap here is lexical and deterministic — Jaccard similarity over the content
terms of a topic's title, keywords and cited evidence. No LLM call, so the
decision is reproducible and a reviewer can recompute it by hand.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from document_processing.metadata import SourceLocation

_WORD_RE = re.compile(r"[a-z][a-z0-9+#-]{2,}")

#: Words carrying no subject signal. Small and explicit rather than a corpus.
STOPWORDS = frozenset(
    """
    the and for with that this from into their there here what when where which
    who whom whose how why are was were been being have has had does did done
    not but you your they them then than also such some any all can could should
    would may might must shall will about above after again against because
    before below between both during each few further more most other over same
    under until while these those very own too its introduction chapter
    section overview summary example examples use used using topic topics
    """.split()
)

MERGE_THRESHOLD = 0.55
RELATED_THRESHOLD = 0.30


class TopicEvidence(BaseModel):
    """A quoted span plus exactly where it came from."""

    model_config = {"frozen": True}

    quote: str = Field(min_length=1)
    location: SourceLocation


class Topic(BaseModel):
    """A candidate unit of teaching, always backed by at least one source."""

    topic_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    difficulty: int = Field(default=3, ge=1, le=5)
    evidence: list[TopicEvidence] = Field(min_length=1)

    @field_validator("keywords", "prerequisites")
    @classmethod
    def _clean(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen

    @property
    def source_documents(self) -> list[str]:
        seen: list[str] = []
        for item in self.evidence:
            if item.location.document_id not in seen:
                seen.append(item.location.document_id)
        return seen

    @property
    def citations(self) -> list[SourceLocation]:
        return [item.location for item in self.evidence]

    def page_span(self) -> int:
        """How much of a book this topic covers, in distinct pages."""
        pages = {item.location.page for item in self.evidence if item.location.page is not None}
        return max(1, len(pages))


class TopicOverlap(BaseModel):
    """Two topics that teach overlapping material, and what to do about it."""

    topic_a: str
    topic_b: str
    similarity: float = Field(ge=0.0, le=1.0)
    shared_terms: list[str]
    decision: Literal["merge", "sequence", "keep_separate"]
    rationale: str
    evidence: list[SourceLocation] = Field(min_length=1)


def content_terms(text: str) -> set[str]:
    """Lowercased content words, stopwords and short tokens dropped."""
    return {word for word in _WORD_RE.findall(text.lower()) if word not in STOPWORDS}


def topic_terms(topic: Topic) -> set[str]:
    """The term signature of a topic: title, keywords and cited evidence."""
    parts = [topic.title, topic.summary, " ".join(topic.keywords)]
    parts.extend(item.quote for item in topic.evidence)
    parts.extend(item.location.section or "" for item in topic.evidence)
    return content_terms(" ".join(parts))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def find_overlaps(
    topics: list[Topic],
    *,
    merge_threshold: float = MERGE_THRESHOLD,
    related_threshold: float = RELATED_THRESHOLD,
) -> list[TopicOverlap]:
    """Compare every pair of topics and classify the ones that overlap.

    ``merge`` — near-duplicates, teach once.
    ``sequence`` — related enough that teaching them apart wastes the overlap,
    so they are kept but scheduled together.
    Pairs below ``related_threshold`` are not reported at all.
    """
    signatures = {topic.topic_id: topic_terms(topic) for topic in topics}
    by_id = {topic.topic_id: topic for topic in topics}
    overlaps: list[TopicOverlap] = []

    for index, left in enumerate(topics):
        for right in topics[index + 1 :]:
            left_terms = signatures[left.topic_id]
            right_terms = signatures[right.topic_id]
            similarity = jaccard(left_terms, right_terms)
            if similarity < related_threshold:
                continue

            shared = sorted(left_terms & right_terms)[:12]
            same_book = set(left.source_documents) == set(right.source_documents)
            if similarity >= merge_threshold:
                decision = "merge"
                rationale = (
                    f"'{left.title}' and '{right.title}' share {similarity:.0%} of their "
                    f"content terms ({', '.join(shared[:5])}); teaching both would repeat "
                    "the same material"
                    + ("" if same_book else " across different books")
                )
            else:
                decision = "sequence"
                rationale = (
                    f"'{left.title}' and '{right.title}' overlap at {similarity:.0%} "
                    f"({', '.join(shared[:5])}); they stay separate but are scheduled "
                    "in the same semester so the shared ground is taught once"
                )

            overlaps.append(
                TopicOverlap(
                    topic_a=left.topic_id,
                    topic_b=right.topic_id,
                    similarity=round(similarity, 4),
                    shared_terms=shared,
                    decision=decision,
                    rationale=rationale,
                    evidence=_pair_evidence(by_id[left.topic_id], by_id[right.topic_id]),
                )
            )

    overlaps.sort(key=lambda item: (-item.similarity, item.topic_a, item.topic_b))
    return overlaps


def _evidence_key(item: TopicEvidence) -> tuple:
    return (
        item.location.document_id,
        item.location.chunk_index,
        item.location.page,
        item.quote,
    )


def _pair_evidence(left: Topic, right: Topic) -> list[SourceLocation]:
    """One citation from each side, so the claim of overlap is checkable."""
    return [left.evidence[0].location, right.evidence[0].location]


def merge_overlapping(
    topics: list[Topic], overlaps: list[TopicOverlap]
) -> tuple[list[Topic], dict[str, str]]:
    """Fold ``merge`` pairs into a single topic that keeps both books' evidence.

    Returns the surviving topics and a map from every absorbed topic id to the
    id that replaced it, so prerequisites pointing at the absorbed topic can be
    rewritten instead of silently dangling.
    """
    by_id = {topic.topic_id: topic for topic in topics}
    replaced_by: dict[str, str] = {}

    def resolve(topic_id: str) -> str:
        seen = set()
        while topic_id in replaced_by and topic_id not in seen:
            seen.add(topic_id)
            topic_id = replaced_by[topic_id]
        return topic_id

    for overlap in overlaps:
        if overlap.decision != "merge":
            continue
        keeper_id = resolve(overlap.topic_a)
        absorbed_id = resolve(overlap.topic_b)
        if keeper_id == absorbed_id:
            continue

        keeper, absorbed = by_id[keeper_id], by_id[absorbed_id]
        # Deduplicate by *where* the evidence is, not by what it says: two books
        # can word the same idea identically and both citations still matter.
        seen = {_evidence_key(item) for item in keeper.evidence}
        evidence = list(keeper.evidence) + [
            item for item in absorbed.evidence if _evidence_key(item) not in seen
        ]
        by_id[keeper_id] = keeper.model_copy(
            update={
                "summary": keeper.summary,
                "keywords": sorted(set(keeper.keywords) | set(absorbed.keywords)),
                "prerequisites": [
                    value
                    for value in dict.fromkeys(keeper.prerequisites + absorbed.prerequisites)
                    if value not in (keeper_id, absorbed_id)
                ],
                "difficulty": max(keeper.difficulty, absorbed.difficulty),
                "evidence": evidence,
            }
        )
        del by_id[absorbed_id]
        replaced_by[absorbed_id] = keeper_id

    surviving = [by_id[topic.topic_id] for topic in topics if topic.topic_id in by_id]
    rewritten = [
        topic.model_copy(
            update={
                "prerequisites": [
                    resolved
                    for resolved in dict.fromkeys(
                        resolve(value) for value in topic.prerequisites
                    )
                    if resolved != topic.topic_id
                ]
            }
        )
        for topic in surviving
    ]
    return rewritten, replaced_by


__all__ = [
    "MERGE_THRESHOLD",
    "RELATED_THRESHOLD",
    "STOPWORDS",
    "Topic",
    "TopicEvidence",
    "TopicOverlap",
    "content_terms",
    "find_overlaps",
    "jaccard",
    "merge_overlapping",
    "topic_terms",
]
