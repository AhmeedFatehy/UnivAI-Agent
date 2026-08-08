"""What a section is grounded on: the lecture's own slides.

Retrieval scores a passage by ``term_coverage`` — the share of the QUERY's
terms it contains. The old single query appended a fixed phrase ("the material
covered in the lecture") to the title, so a textbook passage had to contain the
words "material", "covered" and "lecture" to count. For a one-word title the
best attainable coverage was 0.25 against a 0.34 threshold, which made those
sections impossible to ground rather than merely hard.
"""

from __future__ import annotations

from agents.schemas import SectionPackV1  # noqa: F401  - breaks a known import cycle
from generation.section_gen import MAX_SECTION_SEEDS, section_seed_queries
from planning.section_planner import SectionIdentity
from tools.registry import DEFAULT_MIN_TERM_COVERAGE, content_terms, term_coverage


def identity(title: str = "Triggers") -> SectionIdentity:
    return SectionIdentity(
        programme_title="Computer Science Foundations",
        plan_schema="univai.programme.plan",
        plan_version="1.3.0",
        user_id="student-1",
        collection_id="collection-1",
        course_id="CS101",
        week_number=4,
        topic_id="T04",
        lecture_title=title,
        created_at="2026-08-03T00:00:00+00:00",
    )


def test_a_one_word_lecture_can_be_grounded_at_all():
    """The regression that left three of four weeks without a section."""
    passage = (
        "Triggers are named database objects that activate when an INSERT, "
        "UPDATE or DELETE event occurs on their table."
    )
    old_query = "Triggers. the material covered in the lecture"
    assert term_coverage(old_query, passage) < DEFAULT_MIN_TERM_COVERAGE

    seeds = section_seed_queries(identity(), ["Creating Triggers"])
    assert any(term_coverage(seed, passage) >= DEFAULT_MIN_TERM_COVERAGE for seed in seeds)


def test_term_matching_is_still_literal():
    """A known limit of shared retrieval, recorded rather than worked around.

    ``content_terms`` does not stem, so a slide headed "Triggers" scores zero
    against a passage that only ever writes "trigger". Seeding from several
    headings makes a section resilient to that; it does not cure it, and any
    fix belongs in retrieval where every caller would get it.
    """
    singular = "A trigger is a named database object that fires on INSERT."
    assert term_coverage("Creating Triggers", singular) == 0.0


def test_seeds_are_the_slide_headings():
    seeds = section_seed_queries(
        identity("Triggers"), ["Creating Triggers", "OLD and NEW", "Dropping a Trigger"]
    )
    assert seeds[:3] == ["Creating Triggers", "OLD and NEW", "Dropping a Trigger"]
    # The title still earns a seed of its own, for decks whose headings drift.
    assert "Triggers" in seeds


def test_no_seed_carries_boilerplate_terms():
    seeds = section_seed_queries(identity(), ["Creating Triggers"])
    for seed in seeds:
        assert not ({"material", "covered", "lecture"} & content_terms(seed))


def test_repeated_headings_are_asked_once():
    seeds = section_seed_queries(
        identity("Triggers"), ["Creating Triggers", "creating triggers", "Creating Triggers"]
    )
    assert seeds.count("Creating Triggers") == 1


def test_a_long_deck_is_sampled_rather_than_asked_slide_by_slide():
    """~50 slides must not become ~50 retrievals for one section."""
    headings = [f"Heading {index}" for index in range(75)]
    seeds = section_seed_queries(identity("Stored Functions"), headings)

    assert len(seeds) == MAX_SECTION_SEEDS + 1  # sampled headings, plus the title
    # Sampled across the whole deck, not just its introduction.
    assert seeds[0] == "Heading 0"
    assert any(heading.endswith(("60", "61", "62", "63", "64", "65", "66", "67")) for heading in seeds)


def test_a_lecture_without_headings_still_asks_for_its_title():
    assert section_seed_queries(identity("Triggers"), []) == ["Triggers"]
    assert section_seed_queries(identity("Triggers"), None) == ["Triggers"]
    assert section_seed_queries(identity("Triggers"), ["  ", ""]) == ["Triggers"]
