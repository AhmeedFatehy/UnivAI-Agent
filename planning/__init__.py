"""Programme planning — overlap, prerequisites, workload and semester layout."""

from planning.overlap import Topic, TopicEvidence, TopicOverlap, find_overlaps
from planning.prerequisites import PrerequisiteIssue, teaching_order
from planning.programme_planner import (
    PlannedTopic,
    PlanningDecision,
    ProgrammePlan,
    SemesterPlan,
    create_programme_plan,
)
from planning.workload import WorkloadEstimate, estimate_all, pack_semesters

__all__ = [
    "PlannedTopic",
    "PlanningDecision",
    "PrerequisiteIssue",
    "ProgrammePlan",
    "SemesterPlan",
    "Topic",
    "TopicEvidence",
    "TopicOverlap",
    "WorkloadEstimate",
    "create_programme_plan",
    "estimate_all",
    "find_overlaps",
    "pack_semesters",
    "teaching_order",
]
