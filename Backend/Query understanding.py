
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueryCategory(str, Enum):
    PERSON = "person"
    COMPANY = "company"
    PROJECT = "project"
    RISK = "risk"
    COMMITMENT = "commitment"
    GENERAL = "general"


class QueryIntent(str, Enum):
    STATUS = "status"            # "What is happening with X?"
    SUMMARY = "summary"          # "Summarize X"
    IDENTIFICATION = "identification"   # "Who is X?"
    HISTORY = "history"          # "What has X worked on?"
    PENDING = "pending"          # "What is pending / unresolved?"
    RISK_SCAN = "risk_scan"      # "What risks exist?"
    TIMELINE = "timeline"        # "Walk me through the timeline of X"
    PEOPLE = "people"            # "Who is involved in X?"


class TemporalScope(str, Enum):
    RECENT = "recent"            # last 7 days
    MONTH = "month"              # last 30 days
    QUARTER = "quarter"          # last 90 days
    ALL_TIME = "all_time"        # no restriction
    SPECIFIC = "specific"        # a specific date or range was mentioned


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """A named entity pulled from the query."""
    text: str
    entity_type: str   # person | company | project | generic
    confidence: float = 1.0


@dataclass
class ClassifiedQuery:
    """
    Full understanding of a single executive query.
    Drives the retrieval strategy in Task 2.
    """
    raw_query: str
    normalized_query: str

    category: QueryCategory
    intent: QueryIntent
    temporal_scope: TemporalScope

    entities: list[ExtractedEntity] = field(default_factory=list)

    # Which memory sources are needed (populated by classifier)
    required_memory_sources: list[str] = field(default_factory=list)

    # Free-form enrichment notes
    notes: str = ""

    # Confidence in the classification (0-1)
    confidence: float = 1.0

    def primary_entity(self) -> Optional[ExtractedEntity]:
        return self.entities[0] if self.entities else None

    def __repr__(self) -> str:
        ents = [e.text for e in self.entities]
        return (
            f"ClassifiedQuery("
            f"category={self.category.value}, "
            f"intent={self.intent.value}, "
            f"temporal={self.temporal_scope.value}, "
            f"entities={ents})"
        )


# ---------------------------------------------------------------------------
# Signal banks  (keyword / pattern → classification signal)
# ---------------------------------------------------------------------------

PERSON_SIGNALS = [
    r"\bwho is\b",
    r"\bwho's\b",
    r"\btell me about\b.{0,20}\bperson\b",
    r"\bwhat has\b.{0,40}\bworked on\b",
    r"\bcommitments.{0,20}involve\b",
    r"\binvolving\b.{0,20}\b[A-Z][a-z]+\b",
]

COMPANY_SIGNALS = [
    r"\bwhat is happening with\b",
    r"\bstatus of\b",
    r"\bsummariz\w+ interaction",
    r"\bsummariz\w+.{0,20}\bwith\b",
    r"\blatest.{0,20}(discussion|update|news)",
    r"\bclient\b",
    r"\baccount\b",
]

PROJECT_SIGNALS = [
    r"\bproject\b",
    r"\brollout\b",
    r"\bdeployment\b",
    r"\binitiative\b",
    r"\bcampaign\b",
    r"\bprogram\b",
    r"\bphase\b",
    r"\bimplementation\b",
    r"\bstatus of\b.{0,40}\b(project|initiative|rollout)",
]

RISK_SIGNALS = [
    r"\brisk\b",
    r"\bissu\w+\b",
    r"\bblocke\w+\b",
    r"\bthrea\w+\b",
    r"\bdelay\w*\b",
    r"\bconcern\w*\b",
    r"\bproblem\w*\b",
    r"\bescalat\w+\b",
    r"\bwhat could go wrong\b",
]

COMMITMENT_SIGNALS = [
    r"\bcommitment\w*\b",
    r"\bunresolv\w+\b",
    r"\bpending\b",
    r"\bfollow.up\b",
    r"\bopen item\w*\b",
    r"\bremaining\b",
    r"\bdue\b",
    r"\bdeadline\b",
    r"\bpromise\w*\b",
    r"\baction item\w*\b",
]

# Intent patterns
INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
    QueryIntent.IDENTIFICATION:  [r"\bwho is\b", r"\bwho's\b", r"\btell me about\b"],
    QueryIntent.SUMMARY:         [r"\bsummariz\w+\b", r"\boverall\b", r"\bbrief\w*\b", r"\brecap\b"],
    QueryIntent.STATUS:          [r"\bwhat is happening\b", r"\bstatus\b", r"\blatest\b", r"\bupdate\b"],
    QueryIntent.HISTORY:         [r"\bhistory\b", r"\btimeline\b", r"\bprevious\b", r"\bever\b"],
    QueryIntent.PENDING:         [r"\bpending\b", r"\bremain\w+\b", r"\bunresolved\b", r"\bopen\b"],
    QueryIntent.RISK_SCAN:       [r"\brisk\b", r"\bissues?\b", r"\bproblems?\b", r"\bdanger\b"],
    QueryIntent.TIMELINE:        [r"\btimeline\b", r"\bchron\w+\b", r"\bwhen\b", r"\bhistory\b"],
    QueryIntent.PEOPLE:          [r"\bwho\b", r"\bteam\b", r"\bstakeholder\b", r"\binvolved\b"],
}

# Temporal signals
TEMPORAL_PATTERNS: dict[TemporalScope, list[str]] = {
    TemporalScope.RECENT:   [r"\brecent\w*\b", r"\blatest\b", r"\blast week\b", r"\bthis week\b", r"\bnew\b"],
    TemporalScope.MONTH:    [r"\blast month\b", r"\bthis month\b", r"\bpast month\b"],
    TemporalScope.QUARTER:  [r"\bquarter\b", r"\bq\d\b", r"\blast 90\b"],
    TemporalScope.SPECIFIC: [r"\b\d{4}\b", r"\bjanuary|february|march|april|may|june|"
                              r"july|august|september|october|november|december\b"],
}

# Memory source map per category
MEMORY_SOURCE_MAP: dict[QueryCategory, list[str]] = {
    QueryCategory.PERSON: [
        "person_entity",
        "conversations",
        "commitments",
        "projects",
        "emails",
    ],
    QueryCategory.COMPANY: [
        "company_memory",
        "projects",
        "conversations",
        "commitments",
        "risks",
        "emails",
        "meetings",
    ],
    QueryCategory.PROJECT: [
        "project_timeline",
        "stakeholders",
        "documents",
        "risks",
        "commitments",
        "meetings",
    ],
    QueryCategory.RISK: [
        "risk_objects",
        "delays",
        "follow_ups",
        "commitments",
        "projects",
    ],
    QueryCategory.COMMITMENT: [
        "commitment_memory",
        "conversations",
        "deadlines",
        "projects",
    ],
    QueryCategory.GENERAL: [
        "conversations",
        "documents",
        "emails",
        "meetings",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many regex patterns match in text."""
    count = 0
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            count += 1
    return count


def _detect_temporal_scope(text: str) -> TemporalScope:
    for scope, patterns in TEMPORAL_PATTERNS.items():
        if _count_pattern_matches(text, patterns) > 0:
            return scope
    return TemporalScope.ALL_TIME


def _detect_intent(text: str) -> QueryIntent:
    scores: dict[QueryIntent, int] = {}
    for intent, patterns in INTENT_PATTERNS.items():
        scores[intent] = _count_pattern_matches(text, patterns)
    if not any(scores.values()):
        return QueryIntent.STATUS
    return max(scores, key=lambda k: scores[k])


def _extract_entities(text: str) -> list[ExtractedEntity]:
    """
    Lightweight entity extraction.
    In production, replace with spaCy NER or an LLM-based extractor.
    """
    entities: list[ExtractedEntity] = []
    seen: set[str] = set()

    # Capitalised multi-word phrases that look like names/companies
    for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", text):
        phrase = match.group(1)
        # Skip common question words
        if phrase.lower() in {
            "What", "Who", "Where", "When", "Why", "How", "Is", "Are",
            "The", "This", "That", "These", "Those", "Has", "Have",
            "Can", "Could", "Should", "Would", "Will", "Do", "Does",
            "Summarize", "Tell", "Show", "Give", "Find", "List",
        }:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)

        # Heuristic type detection
        company_hints = re.compile(
            r"\b(Electric|Corp|Inc|Ltd|GmbH|Partners|Group|Energy|Capital|"
            r"Investments|Industries|Technologies|Solutions|Systems|Associates)\b",
            re.IGNORECASE,
        )
        person_hints = re.compile(r"\b(Mr|Mrs|Ms|Dr|Sr|Jr)\b", re.IGNORECASE)

        if company_hints.search(phrase):
            etype = "company"
        elif person_hints.search(phrase) or (len(phrase.split()) == 2):
            etype = "person"
        else:
            etype = "generic"

        entities.append(ExtractedEntity(text=phrase, entity_type=etype))

    return entities


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class QueryClassifier:
    """
    Classifies executive queries into structured ClassifiedQuery objects.

    Usage
    -----
    classifier = QueryClassifier()
    result = classifier.classify("What is happening with Schneider Electric?")
    print(result)
    """

    def classify(self, raw_query: str) -> ClassifiedQuery:
        """Main entry point."""
        normalized = self._normalize(raw_query)

        category, confidence = self._detect_category(normalized)
        intent = _detect_intent(normalized)
        temporal = _detect_temporal_scope(normalized)
        entities = _extract_entities(raw_query)

        # Refine entity types using category context
        entities = self._refine_entities(entities, category)

        memory_sources = MEMORY_SOURCE_MAP.get(category, MEMORY_SOURCE_MAP[QueryCategory.GENERAL])

        notes = self._generate_notes(category, intent, entities, temporal)

        return ClassifiedQuery(
            raw_query=raw_query,
            normalized_query=normalized,
            category=category,
            intent=intent,
            temporal_scope=temporal,
            entities=entities,
            required_memory_sources=list(memory_sources),
            notes=notes,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Lowercase, strip extra spaces, remove punctuation extremes."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    def _detect_category(self, text: str) -> tuple[QueryCategory, float]:
        """Score each category and return the winner with a confidence score."""
        scores: dict[QueryCategory, int] = {
            QueryCategory.PERSON:     _count_pattern_matches(text, PERSON_SIGNALS),
            QueryCategory.COMPANY:    _count_pattern_matches(text, COMPANY_SIGNALS),
            QueryCategory.PROJECT:    _count_pattern_matches(text, PROJECT_SIGNALS),
            QueryCategory.RISK:       _count_pattern_matches(text, RISK_SIGNALS),
            QueryCategory.COMMITMENT: _count_pattern_matches(text, COMMITMENT_SIGNALS),
        }

        total = sum(scores.values())
        if total == 0:
            return QueryCategory.GENERAL, 0.5

        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]
        confidence = round(best_score / total, 2) if total > 0 else 0.5

        # If scores are tied or very close, fall back to GENERAL
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
            return QueryCategory.GENERAL, 0.4

        return best_category, min(confidence, 1.0)

    def _refine_entities(
        self,
        entities: list[ExtractedEntity],
        category: QueryCategory,
    ) -> list[ExtractedEntity]:
        """Use category context to improve entity type labels."""
        refined = []
        for e in entities:
            if category == QueryCategory.PERSON and e.entity_type == "generic":
                e.entity_type = "person"
            elif category == QueryCategory.COMPANY and e.entity_type == "generic":
                e.entity_type = "company"
            elif category == QueryCategory.PROJECT and e.entity_type == "generic":
                e.entity_type = "project"
            refined.append(e)
        return refined

    def _generate_notes(
        self,
        category: QueryCategory,
        intent: QueryIntent,
        entities: list[ExtractedEntity],
        temporal: TemporalScope,
    ) -> str:
        entity_names = [e.text for e in entities]
        parts = [
            f"Category detected: {category.value}.",
            f"Intent: {intent.value}.",
            f"Temporal scope: {temporal.value}.",
        ]
        if entity_names:
            parts.append(f"Key entities: {', '.join(entity_names)}.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Batch classifier
# ---------------------------------------------------------------------------

class BatchQueryClassifier:
    """Classify multiple queries at once."""

    def __init__(self) -> None:
        self._classifier = QueryClassifier()

    def classify_all(self, queries: list[str]) -> list[ClassifiedQuery]:
        return [self._classifier.classify(q) for q in queries]

    def classify_and_group(
        self, queries: list[str]
    ) -> dict[QueryCategory, list[ClassifiedQuery]]:
        results = self.classify_all(queries)
        grouped: dict[QueryCategory, list[ClassifiedQuery]] = {c: [] for c in QueryCategory}
        for r in results:
            grouped[r.category].append(r)
        return grouped


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    classifier = QueryClassifier()

    test_queries = [
        "Who is John Smith?",
        "What has John been working on?",
        "What commitments involve John?",
        "What is happening with Schneider Electric?",
        "Summarize interactions with KEI.",
        "Status of Smart Meter Rollout?",
        "What risks exist around deployment?",
        "What remains unresolved?",
        "What commitments remain open?",
        "Summarize investor discussions.",
        "What are the latest regulator discussions?",
        "What is pending with KEI?",
    ]

    print("=" * 70)
    print("PHASE 4 - TASK 1: QUERY UNDERSTANDING LAYER — DEMO")
    print("=" * 70)

    for q in test_queries:
        result = classifier.classify(q)
        print(f"\nQuery : {q}")
        print(f"  Category       : {result.category.value}")
        print(f"  Intent         : {result.intent.value}")
        print(f"  Temporal       : {result.temporal_scope.value}")
        print(f"  Entities       : {[e.text for e in result.entities]}")
        print(f"  Memory Sources : {result.required_memory_sources}")
        print(f"  Confidence     : {result.confidence}")
        print(f"  Notes          : {result.notes}")

    print("\n" + "=" * 70)
    print("Batch classify + group by category:")
    batch = BatchQueryClassifier()
    grouped = batch.classify_and_group(test_queries)
    for cat, items in grouped.items():
        if items:
            print(f"\n  [{cat.value.upper()}]")
            for item in items:
                print(f"    - {item.raw_query}")
