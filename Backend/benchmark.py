"""
FOUNDER 1 — TASK 4: EVALUATION SYSTEM

Most teams never define success.
This file defines it.

The evaluation system contains:
  1. Benchmark question dataset (100 questions across all query types)
  2. Answer quality criteria per question
  3. Automated scoring logic
  4. Regression testing framework

This becomes the permanent testing baseline.
Before any model change, retrieval change, or prompt change —
run the benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Schema
# ─────────────────────────────────────────────────────────────────────────────

class QuestionCategory(str, Enum):
    PERSON = "person"
    COMPANY = "company"
    PROJECT = "project"
    COMMITMENT = "commitment"
    RISK = "risk"
    TIMELINE = "timeline"
    SUMMARY = "summary"
    CROSS_ENTITY = "cross_entity"


class AnswerCriteria(str, Enum):
    MUST_INCLUDE_DATES = "must_include_dates"
    MUST_INCLUDE_SOURCES = "must_include_sources"
    MUST_INCLUDE_COMMITMENTS = "must_include_commitments"
    MUST_INCLUDE_RISK_LEVELS = "must_include_risk_levels"
    MUST_INCLUDE_PERSON_NAMES = "must_include_person_names"
    MUST_INCLUDE_STATUS = "must_include_status"
    MUST_NOT_HALLUCINATE = "must_not_hallucinate"
    MUST_BE_ACTIONABLE = "must_be_actionable"
    MUST_CITE_ENTITIES = "must_cite_entities"
    MUST_INCLUDE_DEADLINES = "must_include_deadlines"


@dataclass
class BenchmarkQuestion:
    """
    A single benchmark question with evaluation criteria.

    Note: We define characteristics, not exact answers.
    The system must meet criteria regardless of phrasing.
    """
    id: str
    question: str
    category: QuestionCategory
    required_criteria: list[AnswerCriteria]
    optional_criteria: list[AnswerCriteria] = field(default_factory=list)
    difficulty: int = 2                               # 1–3 (easy/medium/hard)
    expected_entity_types: list[str] = field(default_factory=list)
    notes: str = ""

    # For regression testing — populated after first passing run
    baseline_score: Optional[float] = None
    baseline_answer: Optional[str] = None
    baseline_recorded_at: Optional[datetime] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a single answer against a benchmark."""
    question_id: str
    question: str
    answer: str
    score: float                                      # 0.0–1.0
    criteria_passed: list[AnswerCriteria]
    criteria_failed: list[AnswerCriteria]
    retrieval_sources_count: int
    latency_ms: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""


@dataclass
class BenchmarkRun:
    """A complete benchmark run across all questions."""
    id: str = field(default_factory=lambda: str(uuid4()))
    run_at: datetime = field(default_factory=datetime.utcnow)
    model_version: str = ""
    retrieval_version: str = ""
    results: list[EvaluationResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def score_by_category(self) -> dict[str, float]:
        category_scores: dict[str, list[float]] = {}
        for result in self.results:
            q = BENCHMARK_DATASET.get(result.question_id)
            if q:
                cat = q.category.value
                category_scores.setdefault(cat, []).append(result.score)
        return {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

    def to_report(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "run_at": self.run_at.isoformat(),
            "model_version": self.model_version,
            "overall_score": round(self.overall_score, 4),
            "total_questions": len(self.results),
            "by_category": {k: round(v, 4) for k, v in self.score_by_category.items()},
            "results": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "score": round(r.score, 4),
                    "passed": r.criteria_passed,
                    "failed": r.criteria_failed,
                    "sources": r.retrieval_sources_count,
                    "latency_ms": r.latency_ms,
                }
                for r in sorted(self.results, key=lambda r: r.score)
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Dataset — 100 Questions
# ─────────────────────────────────────────────────────────────────────────────

def _q(
    qid: str,
    question: str,
    category: QuestionCategory,
    required: list[AnswerCriteria],
    optional: Optional[list[AnswerCriteria]] = None,
    difficulty: int = 2,
    entities: Optional[list[str]] = None,
    notes: str = "",
) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=qid,
        question=question,
        category=category,
        required_criteria=required,
        optional_criteria=optional or [],
        difficulty=difficulty,
        expected_entity_types=entities or [],
        notes=notes,
    )


_AC = AnswerCriteria
_QC = QuestionCategory

BENCHMARK_QUESTIONS: list[BenchmarkQuestion] = [

    # ── PERSON QUERIES (20 questions) ─────────────────────────────────────────

    _q("P01", "What is pending with John Smith?",
       _QC.PERSON, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DEADLINES, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P02", "What commitments does John Smith owe us?",
       _QC.PERSON, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P03", "When did we last speak with Sarah Johnson?",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       difficulty=1, entities=["person"]),

    _q("P04", "What projects is Rahul Sharma involved in?",
       _QC.PERSON, [_AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_STATUS],
       entities=["person", "project"]),

    _q("P05", "Has the regulator responded to our submission?",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3, entities=["person"]),

    _q("P06", "What is the relationship status with our key investor contacts?",
       _QC.PERSON, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P07", "Who have we not contacted in over two weeks?",
       _QC.PERSON, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_INCLUDE_DATES],
       difficulty=3),

    _q("P08", "What did we promise to the vendor last week?",
       _QC.PERSON, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P09", "What risks is Michael Chen associated with?",
       _QC.PERSON, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P10", "Summarize our relationship with the procurement team.",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_CITE_ENTITIES],
       difficulty=3, entities=["person"]),

    _q("P11", "What emails came from the CFO this month?",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P12", "Which contacts have open commitments due this week?",
       _QC.PERSON, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_INCLUDE_DEADLINES],
       difficulty=3),

    _q("P13", "What is the status of Arun Mehta's deliverables?",
       _QC.PERSON, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P14", "Who are our most important external contacts?",
       _QC.PERSON, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_CITE_ENTITIES],
       difficulty=2),

    _q("P15", "Has anyone followed up on the MoU signing?",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),

    _q("P16", "What meetings did we have with regulators last month?",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_PERSON_NAMES],
       difficulty=2, entities=["person"]),

    _q("P17", "List contacts we have not replied to.",
       _QC.PERSON, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),

    _q("P18", "What did the board members discuss last week?",
       _QC.PERSON, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["person"]),

    _q("P19", "What is the last known status of Priya's deliverable?",
       _QC.PERSON, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_INCLUDE_DATES],
       difficulty=2, entities=["person"]),

    _q("P20", "Who is responsible for the approval pending from KEI?",
       _QC.PERSON, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_CITE_ENTITIES],
       difficulty=3, entities=["person", "company"]),


    # ── COMPANY QUERIES (20 questions) ───────────────────────────────────────

    _q("C01", "What is happening with Schneider Electric?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_STATUS, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),

    _q("C02", "What is the current status of our engagement with KEI?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),

    _q("C03", "What open commitments do we have with our key vendor?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DEADLINES],
       entities=["company"]),

    _q("C04", "Has the investor group responded to our proposal?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3, entities=["company"]),

    _q("C05", "What risks are associated with our government contracts?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),

    _q("C06", "Summarize all interactions with Schneider this quarter.",
       _QC.COMPANY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES, _AC.MUST_CITE_ENTITIES],
       difficulty=3, entities=["company"]),

    _q("C07", "What projects are active with our top three partners?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_CITE_ENTITIES],
       entities=["company", "project"]),

    _q("C08", "Which company has the most overdue commitments?",
       _QC.COMPANY, [_AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_COMMITMENTS],
       difficulty=3),

    _q("C09", "What documents have we exchanged with KEI this year?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES],
       entities=["company"]),

    _q("C10", "What decisions were made in recent meetings with the regulator?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       difficulty=3, entities=["company"]),

    _q("C11", "What is the relationship health score with our main client?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),

    _q("C12", "Which companies have not been contacted in 30 days?",
       _QC.COMPANY, [_AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_DATES],
       difficulty=3),

    _q("C13", "What is the latest update from our logistics vendor?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_NOT_HALLUCINATE],
       entities=["company"]),

    _q("C14", "Which companies are involved in the Smart Meter project?",
       _QC.COMPANY, [_AC.MUST_CITE_ENTITIES],
       entities=["company", "project"]),

    _q("C15", "What outstanding items exist with our largest customer?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_RISK_LEVELS],
       entities=["company"]),

    _q("C16", "Has the contract with our new partner been finalized?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3, entities=["company"]),

    _q("C17", "What commitments have been made to the regulator?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DEADLINES],
       entities=["company"]),

    _q("C18", "Give a brief overview of our investor relationships.",
       _QC.COMPANY, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),

    _q("C19", "Which companies have the highest risk exposure?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("C20", "What meetings are scheduled with external partners this week?",
       _QC.COMPANY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),


    # ── PROJECT QUERIES (20 questions) ───────────────────────────────────────

    _q("PR01", "What is the status of the Smart Meter Rollout?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_CITE_ENTITIES],
       entities=["project"]),

    _q("PR02", "What risks exist in the Infrastructure Upgrade project?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       entities=["project"]),

    _q("PR03", "Which projects are currently behind schedule?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("PR04", "Who are the stakeholders on the Regulatory Initiative?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_CITE_ENTITIES],
       entities=["project"]),

    _q("PR05", "What open items remain in Phase 2 of the deployment?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_STATUS],
       entities=["project"]),

    _q("PR06", "Summarize the history of the Infrastructure Upgrade project.",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES, _AC.MUST_CITE_ENTITIES],
       difficulty=3, entities=["project"]),

    _q("PR07", "What decisions have been made on the Smart Grid project?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       entities=["project"]),

    _q("PR08", "Which projects have critical risks?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=2),

    _q("PR09", "What documents have been submitted for the regulatory approval project?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES],
       entities=["project"]),

    _q("PR10", "Is the Smart Meter Rollout on track for its deadline?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_STATUS, _AC.MUST_NOT_HALLUCINATE],
       entities=["project"]),

    _q("PR11", "What companies are involved in the Infrastructure Upgrade?",
       _QC.PROJECT, [_AC.MUST_CITE_ENTITIES],
       entities=["project", "company"]),

    _q("PR12", "List all projects with overdue commitments.",
       _QC.PROJECT, [_AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_COMMITMENTS],
       difficulty=3),

    _q("PR13", "What is the overall health of our active project portfolio?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("PR14", "What were the last three events in the Smart Meter project?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES],
       difficulty=2, entities=["project"]),

    _q("PR15", "Who is responsible for the grid testing milestone?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_CITE_ENTITIES],
       entities=["project"]),

    _q("PR16", "How many commitments are open across all projects?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_COMMITMENTS],
       difficulty=1),

    _q("PR17", "What projects are scheduled to complete this quarter?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_STATUS],
       entities=["project"]),

    _q("PR18", "What unresolved issues remain in the compliance project?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_STATUS],
       entities=["project"]),

    _q("PR19", "What was the last major update on the infrastructure project?",
       _QC.PROJECT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_NOT_HALLUCINATE],
       difficulty=2, entities=["project"]),

    _q("PR20", "Summarize all project risks in one view.",
       _QC.PROJECT, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),


    # ── COMMITMENT QUERIES (15 questions) ────────────────────────────────────

    _q("CM01", "What commitments remain unresolved?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DEADLINES],
       difficulty=1),

    _q("CM02", "Which commitments are overdue?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DATES, _AC.MUST_BE_ACTIONABLE],
       difficulty=1),

    _q("CM03", "What did we promise to deliver this week?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DEADLINES],
       difficulty=2),

    _q("CM04", "List all commitments due in the next 7 days.",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_DEADLINES, _AC.MUST_BE_ACTIONABLE],
       difficulty=2),

    _q("CM05", "Which commitments were made by the partner companies?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_CITE_ENTITIES],
       difficulty=2),

    _q("CM06", "What has the regulator committed to?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_SOURCES],
       difficulty=3),

    _q("CM07", "Are there any commitments without clear owners?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),

    _q("CM08", "What was the most recent commitment made in writing?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES],
       difficulty=2),

    _q("CM09", "Summarize all inbound commitments from external parties.",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("CM10", "List commitments by urgency.",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_BE_ACTIONABLE],
       difficulty=2),

    _q("CM11", "What follow-ups do we owe to KEI?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_CITE_ENTITIES],
       entities=["company"]),

    _q("CM12", "How many total open commitments exist in the system?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS],
       difficulty=1),

    _q("CM13", "Which project has the most open commitments?",
       _QC.COMMITMENT, [_AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_COMMITMENTS],
       difficulty=3),

    _q("CM14", "What commitments were created from meeting notes?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_SOURCES],
       difficulty=2),

    _q("CM15", "Have any commitments been missed without escalation?",
       _QC.COMMITMENT, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),


    # ── RISK QUERIES (15 questions) ───────────────────────────────────────────

    _q("R01", "What risks currently exist in the system?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=1),

    _q("R02", "What are the critical risks across all projects?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=2),

    _q("R03", "What is the risk status around the vendor delay?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_INCLUDE_STATUS],
       entities=["company"]),

    _q("R04", "Which risks are unmitigated?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=2),

    _q("R05", "What regulatory risks exist?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_INCLUDE_STATUS],
       difficulty=2),

    _q("R06", "What is the highest-scoring risk in the system?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=1),

    _q("R07", "Summarize all risks associated with the Smart Meter project.",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_STATUS],
       difficulty=3, entities=["project"]),

    _q("R08", "Have any risks escalated recently?",
       _QC.RISK, [_AC.MUST_INCLUDE_DATES, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),

    _q("R09", "What is the risk exposure around contract approvals?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_INCLUDE_STATUS],
       difficulty=3),

    _q("R10", "Which companies carry the most risk?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("R11", "What risks were identified in the last 30 days?",
       _QC.RISK, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_RISK_LEVELS],
       difficulty=2),

    _q("R12", "Are there any risks without a mitigation plan?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),

    _q("R13", "What financial risks exist?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=2),

    _q("R14", "Summarize the overall risk posture of the organization.",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_BE_ACTIONABLE],
       difficulty=3),

    _q("R15", "Which risks involve third-party dependencies?",
       _QC.RISK, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),


    # ── CROSS-ENTITY / SUMMARY QUERIES (10 questions) ─────────────────────────

    _q("X01", "Summarize investor discussions from this quarter.",
       _QC.SUMMARY, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_INCLUDE_SOURCES],
       difficulty=3),

    _q("X02", "Give me an executive briefing for Monday.",
       _QC.SUMMARY, [_AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_BE_ACTIONABLE],
       difficulty=3),

    _q("X03", "What happened across the organization last week?",
       _QC.TIMELINE, [_AC.MUST_INCLUDE_DATES, _AC.MUST_INCLUDE_SOURCES],
       difficulty=3),

    _q("X04", "What are the top three things requiring my attention today?",
       _QC.SUMMARY, [_AC.MUST_BE_ACTIONABLE, _AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_RISK_LEVELS],
       difficulty=3),

    _q("X05", "What contracts are pending signature?",
       _QC.CROSS_ENTITY, [_AC.MUST_INCLUDE_STATUS, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("X06", "What major decisions were made this month?",
       _QC.TIMELINE, [_AC.MUST_INCLUDE_DATES, _AC.MUST_CITE_ENTITIES],
       difficulty=3),

    _q("X07", "Which relationships need immediate attention?",
       _QC.CROSS_ENTITY, [_AC.MUST_INCLUDE_PERSON_NAMES, _AC.MUST_BE_ACTIONABLE],
       difficulty=3),

    _q("X08", "Are there any alignment issues between our internal teams and external partners?",
       _QC.CROSS_ENTITY, [_AC.MUST_INCLUDE_RISK_LEVELS, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),

    _q("X09", "List all entities with both overdue commitments and open risks.",
       _QC.CROSS_ENTITY, [_AC.MUST_CITE_ENTITIES, _AC.MUST_INCLUDE_COMMITMENTS, _AC.MUST_INCLUDE_RISK_LEVELS],
       difficulty=3),

    _q("X10", "What is the single most important thing to act on right now?",
       _QC.SUMMARY, [_AC.MUST_BE_ACTIONABLE, _AC.MUST_NOT_HALLUCINATE],
       difficulty=3),
]

# Dictionary for fast lookup
BENCHMARK_DATASET: dict[str, BenchmarkQuestion] = {
    q.id: q for q in BENCHMARK_QUESTIONS
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────

class AnswerScorer:
    """
    Scores an answer against its benchmark criteria.
    Criteria are checked heuristically in Month 1.
    Future: LLM-as-judge scoring.
    """

    DATE_PATTERNS = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(yesterday|today|tomorrow|last week|this week|this month)",
    ]

    def score(
        self,
        question: BenchmarkQuestion,
        answer: str,
        sources_count: int,
    ) -> EvaluationResult:
        import re
        answer_lower = answer.lower()

        passed = []
        failed = []

        for criterion in question.required_criteria:

            if criterion == AnswerCriteria.MUST_INCLUDE_DATES:
                has_date = any(re.search(p, answer_lower) for p in self.DATE_PATTERNS)
                (passed if has_date else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_INCLUDE_SOURCES:
                has_source = sources_count > 0 and any(
                    kw in answer_lower for kw in ["email", "meeting", "document", "slack", "teams"]
                )
                (passed if has_source else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_INCLUDE_COMMITMENTS:
                has_commitment = any(kw in answer_lower for kw in [
                    "commit", "pending", "due", "promise", "deliver", "send", "provide"
                ])
                (passed if has_commitment else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_INCLUDE_RISK_LEVELS:
                has_risk = any(kw in answer_lower for kw in [
                    "critical", "high", "medium", "low", "risk"
                ])
                (passed if has_risk else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_INCLUDE_PERSON_NAMES:
                has_names = bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", answer))
                (passed if has_names else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_INCLUDE_STATUS:
                has_status = any(kw in answer_lower for kw in [
                    "active", "pending", "complete", "open", "closed", "on hold", "in progress"
                ])
                (passed if has_status else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_NOT_HALLUCINATE:
                # Basic check: no "I don't have" + definitive claim pattern
                has_hedge = any(kw in answer_lower for kw in [
                    "based on available", "according to", "no record", "not found", "no information"
                ])
                (passed if has_hedge or sources_count > 0 else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_BE_ACTIONABLE:
                has_action = any(kw in answer_lower for kw in [
                    "follow up", "action", "required", "review", "contact", "send",
                    "schedule", "approve", "respond", "check"
                ])
                (passed if has_action else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_CITE_ENTITIES:
                has_entities = any(kw in answer_lower for kw in [
                    "project", "company", "person", "team"
                ]) or bool(re.search(r"[A-Z][a-z]+", answer))
                (passed if has_entities else failed).append(criterion)

            elif criterion == AnswerCriteria.MUST_INCLUDE_DEADLINES:
                has_deadline = any(kw in answer_lower for kw in [
                    "due", "deadline", "by", "before", "expires", "overdue"
                ]) and any(re.search(p, answer_lower) for p in self.DATE_PATTERNS)
                (passed if has_deadline else failed).append(criterion)

        required_count = len(question.required_criteria)
        score = len(passed) / required_count if required_count > 0 else 1.0

        # Bonus for short, complete answers (penalize padding)
        if len(answer) > 3000:
            score *= 0.95

        return EvaluationResult(
            question_id=question.id,
            question=question.question,
            answer=answer,
            score=round(score, 4),
            criteria_passed=passed,
            criteria_failed=failed,
            retrieval_sources_count=sources_count,
            latency_ms=0,
        )
