from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from task1_query_understanding.query_understanding import QueryClassifier
from task2_retrieval_strategy.retrieval_strategy import RetrievalStrategyEngine
from task3_hybrid_search.hybrid_search import HybridSearchEngine, MemoryChunk
from task4_context_assembly.context_assembly import ContextAssemblyEngine
from task5_memory_fusion.memory_fusion import MemoryFusionEngine
from task6_source_attribution.source_attribution import SourceAttributionEngine
from task7_ranking.ranking_system import RankingEngine
from task8_context_window.context_window import ContextWindowManager
from task9_answer_generation.answer_generation import AnswerGenerationEngine, StubLLMAdapter
from task10_failure_handling.failure_handling import FailureHandler


# ---------------------------------------------------------------------------
# Scoring enums
# ---------------------------------------------------------------------------

class ScoreLevel(str, Enum):
    PASS    = "PASS"
    PARTIAL = "PARTIAL"
    FAIL    = "FAIL"


# ---------------------------------------------------------------------------
# Benchmark question bank (Phase 0 questions)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuestion:
    question_id: str
    query: str
    expected_category: str
    expected_intent:   str
    expected_entities: list[str]
    expected_sources:  list[str]       # which source types should appear
    required_topics:   list[str]       # key topics the answer must cover
    must_have_citations: bool = True
    must_not_hallucinate: bool = True
    max_latency_ms: float = 800.0


BENCHMARK_QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(
        question_id="BQ-001",
        query="What is pending with KEI?",
        expected_category="company",
        expected_intent="pending",
        expected_entities=["KEI"],
        expected_sources=["commitment", "meeting", "email"],
        required_topics=["pending", "commitment", "KEI"],
    ),
    BenchmarkQuestion(
        question_id="BQ-002",
        query="What commitments remain unresolved?",
        expected_category="commitment",
        expected_intent="pending",
        expected_entities=[],
        expected_sources=["commitment"],
        required_topics=["commitment", "unresolved", "open"],
    ),
    BenchmarkQuestion(
        question_id="BQ-003",
        query="What risks exist around deployment?",
        expected_category="risk",
        expected_intent="risk_scan",
        expected_entities=[],
        expected_sources=["risk", "project"],
        required_topics=["risk", "deployment", "delay"],
    ),
    BenchmarkQuestion(
        question_id="BQ-004",
        query="Summarize investor discussions.",
        expected_category="company",
        expected_intent="summary",
        expected_entities=[],
        expected_sources=["meeting", "email"],
        required_topics=["investor"],
    ),
    BenchmarkQuestion(
        question_id="BQ-005",
        query="What is happening with Schneider Electric?",
        expected_category="company",
        expected_intent="status",
        expected_entities=["Schneider Electric"],
        expected_sources=["email", "meeting", "risk", "commitment"],
        required_topics=["Schneider", "approval", "deployment"],
    ),
    BenchmarkQuestion(
        question_id="BQ-006",
        query="Who is John Smith?",
        expected_category="person",
        expected_intent="identification",
        expected_entities=["John Smith"],
        expected_sources=["email", "commitment"],
        required_topics=["John Smith"],
    ),
    BenchmarkQuestion(
        question_id="BQ-007",
        query="Status of Smart Meter Rollout?",
        expected_category="project",
        expected_intent="status",
        expected_entities=["Smart Meter"],
        expected_sources=["project", "meeting", "risk"],
        required_topics=["Smart Meter", "timeline", "delay"],
    ),
    BenchmarkQuestion(
        question_id="BQ-008",
        query="What are the latest regulator discussions?",
        expected_category="company",
        expected_intent="status",
        expected_entities=[],
        expected_sources=["email", "meeting"],
        required_topics=["regulator"],
    ),
]


# ---------------------------------------------------------------------------
# Score card
# ---------------------------------------------------------------------------

@dataclass
class CriterionScore:
    name: str
    score: ScoreLevel
    value: Any
    notes: str = ""


@dataclass
class EvaluationScorecard:
    """Full evaluation result for a single benchmark question."""
    question_id: str
    query: str
    overall: ScoreLevel
    scores: list[CriterionScore]
    answer_text: str
    latency_ms: float
    sources_found: list[str]
    entities_found: list[str]
    category_detected: str
    intent_detected: str
    citation_coverage: float
    evidence_count: int
    warnings: list[str]
    raw_pipeline_output: dict = field(default_factory=dict)

    def passed(self) -> bool:
        return self.overall == ScoreLevel.PASS

    def to_text(self) -> str:
        lines = [
            f"┌─ {self.question_id}: {self.query}",
            f"│  Overall: {self.overall.value}  |  Latency: {self.latency_ms:.0f}ms  |  "
            f"Citations: {self.citation_coverage:.0%}  |  Evidence: {self.evidence_count}",
        ]
        for c in self.scores:
            icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}[c.score.value]
            lines.append(f"│  {icon} {c.name:<30} {c.score.value:<8} {c.notes}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"│  ⚠  {w}")
        lines.append(f"└─ Answer (truncated): {self.answer_text[:200]}…")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

class AccuracyScorer:
    """
    Checks that the answer addresses the query topic.
    Heuristic: required topics appear in the answer.
    """

    def score(
        self, answer: str, required_topics: list[str]
    ) -> CriterionScore:
        answer_lower = answer.lower()
        found = [t for t in required_topics if t.lower() in answer_lower]
        coverage = len(found) / max(len(required_topics), 1)

        if coverage >= 0.80:
            level = ScoreLevel.PASS
        elif coverage >= 0.50:
            level = ScoreLevel.PARTIAL
        else:
            level = ScoreLevel.FAIL

        return CriterionScore(
            name="Accuracy",
            score=level,
            value=coverage,
            notes=f"{len(found)}/{len(required_topics)} required topics covered: "
                  f"{found if found else 'none'}",
        )


class CompletenessScorer:
    """
    Checks that required source types were actually retrieved.
    """

    def score(
        self, sources_found: list[str], expected_sources: list[str]
    ) -> CriterionScore:
        found_set = set(s.lower() for s in sources_found)
        expected_set = set(s.lower() for s in expected_sources)
        matched = found_set & expected_set
        coverage = len(matched) / max(len(expected_set), 1)

        if coverage >= 0.80:
            level = ScoreLevel.PASS
        elif coverage >= 0.50:
            level = ScoreLevel.PARTIAL
        else:
            level = ScoreLevel.FAIL

        missing = expected_set - found_set
        return CriterionScore(
            name="Completeness",
            score=level,
            value=coverage,
            notes=f"Found: {sorted(matched)}  Missing: {sorted(missing)}",
        )


class CitationQualityScorer:
    """
    Evaluates how well the answer cites its sources.
    """

    def score(self, citation_coverage: float, answer: str) -> CriterionScore:
        import re
        citation_count = len(re.findall(r"\[.+?\]", answer))

        if citation_coverage >= 0.80 and citation_count >= 2:
            level = ScoreLevel.PASS
        elif citation_coverage >= 0.40:
            level = ScoreLevel.PARTIAL
        else:
            level = ScoreLevel.FAIL

        return CriterionScore(
            name="Citation Quality",
            score=level,
            value=citation_coverage,
            notes=f"Coverage: {citation_coverage:.0%}  |  Inline citations: {citation_count}",
        )


class MissingContextScorer:
    """
    Flags if the answer is very short (likely missing context).
    """

    MIN_ANSWER_WORDS = 30

    def score(self, answer: str, evidence_count: int) -> CriterionScore:
        word_count = len(answer.split())
        if evidence_count == 0:
            return CriterionScore(
                name="Missing Context",
                score=ScoreLevel.FAIL,
                value=0,
                notes="Zero evidence sources. Answer cannot be grounded.",
            )
        if word_count < self.MIN_ANSWER_WORDS:
            return CriterionScore(
                name="Missing Context",
                score=ScoreLevel.PARTIAL,
                value=word_count,
                notes=f"Answer only {word_count} words. May be missing context.",
            )
        return CriterionScore(
            name="Missing Context",
            score=ScoreLevel.PASS,
            value=word_count,
            notes=f"Answer has {word_count} words and {evidence_count} sources.",
        )


class HallucinationRiskScorer:
    """
    Detects phrases that signal speculation or invention.
    """

    SPECULATION_PHRASES = [
        "i believe", "i think", "probably", "i assume",
        "in my opinion", "it seems likely", "it's possible that",
        "generally speaking", "typically", "usually",
    ]

    def score(self, answer: str) -> CriterionScore:
        answer_lower = answer.lower()
        found = [p for p in self.SPECULATION_PHRASES if p in answer_lower]
        if found:
            return CriterionScore(
                name="Hallucination Risk",
                score=ScoreLevel.FAIL,
                value=found,
                notes=f"Speculation phrases detected: {found}",
            )
        insufficient = "insufficient evidence" in answer_lower
        if insufficient:
            return CriterionScore(
                name="Hallucination Risk",
                score=ScoreLevel.PASS,
                value=[],
                notes="Correctly reported insufficient evidence (no speculation).",
            )
        return CriterionScore(
            name="Hallucination Risk",
            score=ScoreLevel.PASS,
            value=[],
            notes="No speculation phrases detected.",
        )


class LatencyScorer:
    """Checks whether query was answered within acceptable time."""

    def score(self, latency_ms: float, max_ms: float = 800.0) -> CriterionScore:
        if latency_ms <= max_ms * 0.75:
            level = ScoreLevel.PASS
        elif latency_ms <= max_ms:
            level = ScoreLevel.PARTIAL
        else:
            level = ScoreLevel.FAIL
        return CriterionScore(
            name="Latency",
            score=level,
            value=latency_ms,
            notes=f"{latency_ms:.0f}ms vs threshold {max_ms:.0f}ms",
        )


class ClassificationScorer:
    """Checks query was classified into the right category and intent."""

    def score(
        self,
        detected_category: str,
        detected_intent: str,
        expected_category: str,
        expected_intent: str,
    ) -> CriterionScore:
        cat_ok = detected_category == expected_category
        intent_ok = detected_intent == expected_intent
        if cat_ok and intent_ok:
            level = ScoreLevel.PASS
        elif cat_ok or intent_ok:
            level = ScoreLevel.PARTIAL
        else:
            level = ScoreLevel.FAIL
        return CriterionScore(
            name="Query Classification",
            score=level,
            value={"category": detected_category, "intent": detected_intent},
            notes=(
                f"Category: {detected_category} (expected {expected_category}) "
                f"Intent: {detected_intent} (expected {expected_intent})"
            ),
        )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Wires all Phase 4 tasks together to answer a single query.
    Returns a structured dict with every pipeline output.
    """

    def __init__(self, corpus: list[MemoryChunk]) -> None:
        self._corpus = corpus
        self._classifier = QueryClassifier()
        self._strategy   = RetrievalStrategyEngine()
        self._search     = HybridSearchEngine()
        self._assembly   = ContextAssemblyEngine()
        self._fusion     = MemoryFusionEngine()
        self._attribution = SourceAttributionEngine()
        self._ranking    = RankingEngine()
        self._context_mgr = ContextWindowManager(budget_tokens=5000)
        self._answer     = AnswerGenerationEngine(llm=StubLLMAdapter())
        self._failures   = FailureHandler()

    def run(self, query: str) -> dict:
        t0 = time.perf_counter()

        # 1. Classify
        classified = self._classifier.classify(query)

        # 2. Build plan
        plan = self._strategy.build_plan(classified)

        # 3. Search
        entity_ids = [e.text for e in classified.entities]
        first_source = plan.sources[0] if plan.sources else None
        filters = first_source.filters if first_source else {}
        results = self._search.search(
            query, self._corpus,
            filters=filters,
            entity_ids=entity_ids,
            modes=plan.search_modes,
            top_k=15,
        )

        # 4. Failure check
        required_sources = [s.source_type.value for s in plan.sources]
        failure_response = self._failures.handle(
            query, results, required_sources=required_sources
        )

        # 5. Rank
        ranked = self._ranking.rank(results, focus_entity_ids=entity_ids)

        # 6. Fuse
        fusion_report = self._fusion.fuse_from_results(results)

        # 7. Assemble + compress context
        hybrid_results = [sr.result for sr in ranked]
        assembled = self._assembly.assemble(query, hybrid_results, strategy=plan.assembly_strategy)
        managed   = self._context_mgr.compress(assembled)

        # 8. Attribution
        attributed = self._attribution.attribute(query, "",
                                                  results=results,
                                                  fused_memories=fusion_report.fused_memories)

        # 9. Generate answer
        answer = self._answer.generate(query, managed, attributed)

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "query":             query,
            "category":          classified.category.value,
            "intent":            classified.intent.value,
            "entities":          [e.text for e in classified.entities],
            "sources_found":     list({r.chunk.source_type for r in results}),
            "results_count":     len(results),
            "answer_text":       answer.answer_text,
            "quality":           answer.quality.value,
            "citation_coverage": attributed.coverage_score if attributed else 0.0,
            "evidence_count":    answer.evidence_count,
            "sections_used":     managed.sections_included,
            "warnings":          answer.uncertainty_flags,
            "failure_events":    [e.failure_type.value for e in failure_response.events],
            "latency_ms":        total_ms,
        }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """
    Runs all benchmark questions through the full pipeline
    and produces scorecards.

    Usage
    -----
    evaluator = RetrievalEvaluator(corpus=your_chunks)
    report = evaluator.run_all()
    print(report.to_text())
    """

    def __init__(self, corpus: list[MemoryChunk]) -> None:
        self._runner   = PipelineRunner(corpus)
        self._accuracy = AccuracyScorer()
        self._complete = CompletenessScorer()
        self._citation = CitationQualityScorer()
        self._missing  = MissingContextScorer()
        self._halluc   = HallucinationRiskScorer()
        self._latency  = LatencyScorer()
        self._classify = ClassificationScorer()

    def evaluate_one(self, bq: BenchmarkQuestion) -> EvaluationScorecard:
        output = self._runner.run(bq.query)

        scores: list[CriterionScore] = [
            self._classify.score(
                output["category"], output["intent"],
                bq.expected_category, bq.expected_intent,
            ),
            self._accuracy.score(output["answer_text"], bq.required_topics),
            self._complete.score(output["sources_found"], bq.expected_sources),
            self._citation.score(output["citation_coverage"], output["answer_text"]),
            self._missing.score(output["answer_text"], output["evidence_count"]),
            self._halluc.score(output["answer_text"]),
            self._latency.score(output["latency_ms"], bq.max_latency_ms),
        ]

        # Overall: PASS if no FAILs; PARTIAL if any PARTIAL; FAIL if any FAIL
        levels = {s.score for s in scores}
        if ScoreLevel.FAIL in levels:
            overall = ScoreLevel.FAIL
        elif ScoreLevel.PARTIAL in levels:
            overall = ScoreLevel.PARTIAL
        else:
            overall = ScoreLevel.PASS

        return EvaluationScorecard(
            question_id=bq.question_id,
            query=bq.query,
            overall=overall,
            scores=scores,
            answer_text=output["answer_text"],
            latency_ms=output["latency_ms"],
            sources_found=output["sources_found"],
            entities_found=output["entities"],
            category_detected=output["category"],
            intent_detected=output["intent"],
            citation_coverage=output["citation_coverage"],
            evidence_count=output["evidence_count"],
            warnings=output["warnings"],
            raw_pipeline_output=output,
        )

    def run_all(
        self, questions: Optional[list[BenchmarkQuestion]] = None
    ) -> "EvaluationReport":
        questions = questions or BENCHMARK_QUESTIONS
        scorecards: list[EvaluationScorecard] = []
        for bq in questions:
            sc = self.evaluate_one(bq)
            scorecards.append(sc)
        return EvaluationReport(scorecards=scorecards)


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    scorecards: list[EvaluationScorecard]

    def pass_rate(self) -> float:
        passed = sum(1 for sc in self.scorecards if sc.passed())
        return passed / max(len(self.scorecards), 1)

    def phase5_ready(self) -> bool:
        """Returns True only if ALL benchmark questions passed."""
        return all(sc.passed() for sc in self.scorecards)

    def to_text(self) -> str:
        lines = [
            "=" * 70,
            "PHASE 4 EVALUATION REPORT — RETRIEVAL BENCHMARK",
            f"  Run at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Questions evaluated: {len(self.scorecards)}",
            f"  Pass rate: {self.pass_rate():.0%}",
            "=" * 70,
        ]

        for sc in self.scorecards:
            lines.append("")
            lines.append(sc.to_text())

        # Summary table
        lines += ["", "=" * 70, "CRITERION SUMMARY", ""]
        criteria_agg: dict[str, dict[str, int]] = {}
        for sc in self.scorecards:
            for c in sc.scores:
                if c.name not in criteria_agg:
                    criteria_agg[c.name] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
                criteria_agg[c.name][c.score.value] += 1

        n = len(self.scorecards)
        lines.append(f"  {'Criterion':<30} {'PASS':>6} {'PARTIAL':>8} {'FAIL':>6}  Rate")
        lines.append("  " + "─" * 60)
        for name, agg in criteria_agg.items():
            rate = agg["PASS"] / n
            icon = "✅" if rate == 1.0 else ("⚠️" if rate >= 0.75 else "❌")
            lines.append(
                f"  {name:<30} {agg['PASS']:>6} {agg['PARTIAL']:>8} {agg['FAIL']:>6}  "
                f"{rate:.0%} {icon}"
            )

        # Phase gate
        lines += ["", "=" * 70]
        if self.phase5_ready():
            lines.append("✅ PHASE 4 COMPLETE — System is ready to proceed to PHASE 5.")
        else:
            failed = [sc.question_id for sc in self.scorecards if not sc.passed()]
            lines.append(f"❌ PHASE 4 INCOMPLETE — Failed questions: {failed}")
            lines.append("   Fix all failing criteria before proceeding to Phase 5.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "run_at":        datetime.utcnow().isoformat(),
            "total":         len(self.scorecards),
            "pass_rate":     self.pass_rate(),
            "phase5_ready":  self.phase5_ready(),
            "scorecards": [
                {
                    "question_id":  sc.question_id,
                    "query":        sc.query,
                    "overall":      sc.overall.value,
                    "latency_ms":   sc.latency_ms,
                    "scores": [
                        {"criterion": c.name, "score": c.score.value, "notes": c.notes}
                        for c in sc.scores
                    ],
                }
                for sc in self.scorecards
            ],
        }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time

    now = _time.time()

    # Build a realistic demo corpus
    corpus = [
        MemoryChunk("email_001", "email",
                    "Approval delayed pending regulator sign-off on Smart Meter deployment.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter",
                              "person": "John Smith", "tags": ["delay", "approval"]},
                    relationships=["meeting_002"],
                    timestamp=now - 86400 * 2),
        MemoryChunk("meeting_002", "meeting",
                    "KEI confirmed Smart Meter timeline shifted 2 weeks. Regulator response pending.",
                    metadata={"company": "KEI", "project": "Smart Meter", "tags": ["delay"]},
                    relationships=["email_001", "risk_003"],
                    timestamp=now - 86400 * 4),
        MemoryChunk("risk_003", "risk",
                    "Risk: Q2 delivery at risk. Smart Meter deployment pending approval.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter",
                              "tags": ["risk", "delivery"]},
                    timestamp=now - 86400 * 1),
        MemoryChunk("commitment_004", "commitment",
                    "John Smith committed to following up with regulator by March 25.",
                    metadata={"person": "John Smith", "status": "open",
                              "project": "Smart Meter", "tags": ["commitment"]},
                    timestamp=now - 86400 * 3),
        MemoryChunk("investor_005", "meeting",
                    "KEI investor call: strong Q1 pipeline. Discussed Smart Meter prospects.",
                    metadata={"company": "KEI", "tags": ["investor", "q1"]},
                    timestamp=now - 86400 * 7),
        MemoryChunk("person_006", "email",
                    "John Smith is the project lead for Smart Meter. Managing regulator liaison.",
                    metadata={"person": "John Smith", "project": "Smart Meter"},
                    timestamp=now - 86400 * 10),
        MemoryChunk("project_007", "project",
                    "Smart Meter Rollout project: Q1 install phase complete. "
                    "Q2 regulator sign-off required. Timeline shifted 2 weeks.",
                    metadata={"project": "Smart Meter", "company": "Schneider Electric"},
                    timestamp=now - 86400 * 2),
        MemoryChunk("regulator_008", "email",
                    "Latest regulator discussion: approval committee reviewing Smart Meter specs. "
                    "Decision expected within 2 weeks.",
                    metadata={"tags": ["regulator"], "project": "Smart Meter"},
                    timestamp=now - 86400 * 1),
    ]

    evaluator = RetrievalEvaluator(corpus=corpus)
    report = evaluator.run_all()

    print(report.to_text())
    print()
    print("JSON report saved to: phase4_evaluation_report.json")
    with open("/home/claude/phase4/joint_evaluation/phase4_evaluation_report.json", "w") as f:
        f.write(report.to_json())
