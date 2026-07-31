"""Prerequisite ordering for a set of topics.

A programme is only valid if nothing is taught before the thing it depends on.
This module turns the per-topic ``prerequisites`` lists into a dependency graph,
reports what is wrong with it (dangling references, cycles) and produces a
deterministic teaching order.

Cycles are reported rather than raised: a cycle in three books' worth of
extracted topics is a data problem, not a crash, and the planner still has to
produce a plan — it breaks the cycle at a named, reported edge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from planning.overlap import Topic


class PrerequisiteIssue(BaseModel):
    """Something wrong with the dependency graph, stated plainly."""

    kind: Literal["unknown_prerequisite", "self_reference", "cycle"]
    topics: list[str] = Field(min_length=1)
    detail: str


def build_graph(topics: list[Topic]) -> dict[str, list[str]]:
    """Map dependencies, dropping one deterministic back-edge per cycle."""
    return _break_cycles(_raw_graph(topics))


def _raw_graph(topics: list[Topic]) -> dict[str, list[str]]:
    """Map dependencies before cycle repair, keeping only known topics."""
    known = {topic.topic_id for topic in topics}
    return {
        topic.topic_id: [
            value
            for value in topic.prerequisites
            if value in known and value != topic.topic_id
        ]
        for topic in topics
    }


def validate_prerequisites(topics: list[Topic]) -> list[PrerequisiteIssue]:
    """Report prerequisite references that cannot be satisfied."""
    known = {topic.topic_id for topic in topics}
    issues: list[PrerequisiteIssue] = []

    for topic in topics:
        for value in topic.prerequisites:
            if value == topic.topic_id:
                issues.append(
                    PrerequisiteIssue(
                        kind="self_reference",
                        topics=[topic.topic_id],
                        detail=f"'{topic.title}' lists itself as a prerequisite; ignored",
                    )
                )
            elif value not in known:
                issues.append(
                    PrerequisiteIssue(
                        kind="unknown_prerequisite",
                        topics=[topic.topic_id],
                        detail=(
                            f"'{topic.title}' requires '{value}', which is not covered by "
                            "the indexed books; the dependency is dropped"
                        ),
                    )
                )

    issues.extend(_cycles(_raw_graph(topics), topics))
    return issues


def _break_cycles(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return an acyclic copy by removing deterministic DFS back-edges."""
    repaired = {node: list(parents) for node, parents in graph.items()}
    while True:
        edge = _cycle_back_edge(repaired)
        if edge is None:
            return repaired
        node, parent = edge
        repaired[node] = [value for value in repaired[node] if value != parent]


def _cycle_back_edge(graph: dict[str, list[str]]) -> tuple[str, str] | None:
    colour: dict[str, int] = dict.fromkeys(graph, 0)

    def visit(node: str) -> tuple[str, str] | None:
        colour[node] = 1
        for parent in sorted(graph.get(node, [])):
            if colour.get(parent, 0) == 1:
                return node, parent
            if colour.get(parent, 0) == 0:
                found = visit(parent)
                if found is not None:
                    return found
        colour[node] = 2
        return None

    for node in sorted(graph):
        if colour[node] == 0:
            found = visit(node)
            if found is not None:
                return found
    return None


def _cycles(graph: dict[str, list[str]], topics: list[Topic]) -> list[PrerequisiteIssue]:
    """Find dependency cycles with an iterative depth-first search."""
    titles = {topic.topic_id: topic.title for topic in topics}
    colour: dict[str, int] = dict.fromkeys(graph, 0)  # 0 unseen, 1 on stack, 2 done
    found: list[PrerequisiteIssue] = []
    reported: set[frozenset[str]] = set()

    for root in graph:
        if colour[root] != 0:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        path: list[str] = []
        while stack:
            node, child_index = stack.pop()
            if child_index == 0:
                if colour[node] == 2:
                    continue
                colour[node] = 1
                path.append(node)

            children = graph.get(node, [])
            if child_index < len(children):
                stack.append((node, child_index + 1))
                child = children[child_index]
                if colour.get(child, 0) == 1:
                    cycle = path[path.index(child) :]
                    key = frozenset(cycle)
                    if key not in reported:
                        reported.add(key)
                        names = " → ".join(titles.get(item, item) for item in cycle + [child])
                        found.append(
                            PrerequisiteIssue(
                                kind="cycle",
                                topics=list(cycle),
                                detail=(
                                    f"circular prerequisites: {names}; the edge back to "
                                    f"'{titles.get(child, child)}' is dropped so the "
                                    "programme can still be ordered"
                                ),
                            )
                        )
                elif colour.get(child, 0) == 0:
                    stack.append((child, 0))
            else:
                colour[node] = 2
                path.pop()

    return found


def teaching_order(topics: list[Topic]) -> tuple[list[str], list[PrerequisiteIssue]]:
    """Deterministic topological order: prerequisites first.

    Ties are broken by topic title so the same input always yields the same
    programme. Topics left over after Kahn's algorithm are inside a cycle; they
    are appended in title order and the cycle is reported.
    """
    graph = build_graph(topics)
    issues = validate_prerequisites(topics)
    titles = {topic.topic_id: topic.title for topic in topics}

    remaining = {node: set(parents) for node, parents in graph.items()}
    ordered: list[str] = []

    while remaining:
        ready = sorted(
            (node for node, parents in remaining.items() if not parents),
            key=lambda node: (titles.get(node, node), node),
        )
        if not ready:
            # Everything left is in a cycle. Break it deterministically.
            stuck = sorted(remaining, key=lambda node: (titles.get(node, node), node))
            ordered.extend(stuck)
            break
        for node in ready:
            ordered.append(node)
            del remaining[node]
        for parents in remaining.values():
            parents.difference_update(ready)

    return ordered, issues


def dependency_depth(topics: list[Topic]) -> dict[str, int]:
    """Longest prerequisite chain ending at each topic (roots are 0)."""
    graph = build_graph(topics)
    order, _ = teaching_order(topics)
    depth: dict[str, int] = {}
    for node in order:
        parents = [depth[parent] for parent in graph.get(node, []) if parent in depth]
        depth[node] = max(parents, default=-1) + 1
    return depth


__all__ = [
    "PrerequisiteIssue",
    "build_graph",
    "dependency_depth",
    "teaching_order",
    "validate_prerequisites",
]
