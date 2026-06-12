
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from app.models.canonical import CanonicalDocument, ExtractedMetadata

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for fast extraction
# ---------------------------------------------------------------------------

# Email addresses
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# URLs
URL_PATTERN = re.compile(
    r"https?://[^\s\)\]\>\"\']+",
    re.IGNORECASE,
)

# Dates (ISO, US, EU formats)
DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"                                    # 2024-03-15
    r"|\d{1,2}/\d{1,2}/\d{2,4}"                             # 3/15/2024
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}"  # 15 March 2024
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}"  # March 15, 2024
    r"|Q[1-4]\s+\d{4}"                                       # Q2 2024
    r")\b",
    re.IGNORECASE,
)

# Relative time expressions
RELATIVE_TIME_PATTERN = re.compile(
    r"\b(?:today|tomorrow|yesterday|next\s+(?:week|month|quarter|year|monday|tuesday|wednesday|thursday|friday)|this\s+(?:week|month|quarter|year|friday)|by\s+(?:end\s+of\s+)?(?:week|month|quarter|eow|eom)|asap|urgent)\b",
    re.IGNORECASE,
)

# Deadline indicators
DEADLINE_PATTERN = re.compile(
    r"(?:deadline|due|by|no later than|must be|deliver by|submit by|complete by)\s+([^.!?\n]{3,60})",
    re.IGNORECASE,
)

# Action item indicators
ACTION_ITEM_PATTERN = re.compile(
    r"(?:^|\n)\s*[-•*]\s+(?:TODO|ACTION|AP|FOLLOW.?UP|TASK):?\s*(.+)",
    re.IGNORECASE | re.MULTILINE,
)

# Commitment indicators
COMMITMENT_PATTERN = re.compile(
    r"(?:I will|we will|I'll|we'll|I am going to|we are going to|committed to|will deliver|will send|will complete|will share|will review|will get back)\s+([^.!?\n]{5,100})",
    re.IGNORECASE,
)

# Meeting references
MEETING_PATTERN = re.compile(
    r"\b(?:meeting|call|sync|standup|standup|kickoff|review|retrospective|demo|presentation|conference|webinar)\b(?:\s+(?:with|about|on|regarding|for)\s+[^\n.!?]{3,50})?",
    re.IGNORECASE,
)

# Project patterns (common naming conventions)
PROJECT_PATTERN = re.compile(
    r"\b(?:project|proj|initiative|program|workstream|sprint|milestone)\s+([A-Z][A-Za-z0-9\s\-_]{2,40})",
    re.IGNORECASE,
)

# Jira/linear/GitHub ticket patterns
TICKET_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,10}-\d{1,6}|#\d{3,6})\b"
)

# Phone numbers (basic)
PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
)


# ---------------------------------------------------------------------------
# Common organization name fragments (lightweight alternative to NER)
# ---------------------------------------------------------------------------

ORGANIZATION_SUFFIXES = re.compile(
    r"\b([A-Z][A-Za-z\s&\-\.]{2,40})\s+(?:Inc\.?|LLC\.?|Ltd\.?|Corp\.?|Co\.?|GmbH|AG|SA|Pvt\.?\s*Ltd\.?|Pte\.?\s*Ltd\.?|Foundation|Group|Holdings|Solutions|Technologies|Consulting|Services|Partners)\b"
)

# ---------------------------------------------------------------------------
# Main enrichment class
# ---------------------------------------------------------------------------


class MetadataEnrichmentPipeline:
    """
    Runs all extraction passes on a CanonicalDocument.
    Returns the document with populated ExtractedMetadata.
    """

    def __init__(self, use_spacy: bool = True, use_ollama: bool = False):
        self.use_spacy = use_spacy
        self.use_ollama = use_ollama
        self._nlp = None  # Lazy-loaded spaCy model

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                # Use small model for speed; upgrade to en_core_web_lg for better accuracy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning("spaCy not available, using regex-only extraction", error=str(e))
                self.use_spacy = False
        return self._nlp

    async def enrich(self, doc: CanonicalDocument) -> CanonicalDocument:
        """
        Enrich a canonical document with extracted metadata.
        This is the main entry point called from the processing queue.
        """
        text = doc.content
        if not text or len(text.strip()) < 10:
            doc.extracted_metadata = ExtractedMetadata()
            return doc

        metadata = ExtractedMetadata()

        # Regex-based extraction (fast, always runs)
        metadata.dates_mentioned = self._extract_dates(text)
        metadata.deadlines = self._extract_deadlines(text)
        metadata.references = self._extract_references(text)
        metadata.meetings_referenced = self._extract_meetings(text)
        metadata.has_action_items = bool(ACTION_ITEM_PATTERN.search(text))
        metadata.has_commitments = bool(COMMITMENT_PATTERN.search(text))
        metadata.organizations_mentioned = self._extract_organizations(text)
        metadata.word_count = len(text.split())

        # Add people from email fields
        all_people: Set[str] = set()
        for person in doc.participants + doc.recipients + doc.cc:
            if person.name:
                all_people.add(person.name)
            if person.email:
                all_people.add(person.email)

        # Extract emails mentioned in body text
        body_emails = EMAIL_PATTERN.findall(text)
        all_people.update(body_emails)
        metadata.people_mentioned = list(all_people)

        # spaCy NER (better person/org extraction)
        nlp = self._get_nlp()
        if nlp and len(text) < 100000:  # Skip NER for huge documents
            try:
                ner_result = self._run_spacy(nlp, text[:50000])  # Process first 50k chars
                metadata.people_mentioned = list(set(metadata.people_mentioned + ner_result["persons"]))
                metadata.organizations_mentioned = list(set(metadata.organizations_mentioned + ner_result["organizations"]))
                metadata.locations_mentioned = ner_result["locations"]
                metadata.key_topics = ner_result["key_phrases"][:10]
                metadata.language = ner_result.get("language", "en")
            except Exception as e:
                logger.warning("spaCy enrichment failed", error=str(e))

        # Projects
        metadata.projects_mentioned = self._extract_projects(text)

        doc.extracted_metadata = metadata
        return doc

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def _extract_dates(self, text: str) -> List[str]:
        dates = set()
        for match in DATE_PATTERN.finditer(text):
            dates.add(match.group(0).strip())
        for match in RELATIVE_TIME_PATTERN.finditer(text):
            dates.add(match.group(0).strip().lower())
        return list(dates)

    def _extract_deadlines(self, text: str) -> List[Dict[str, str]]:
        deadlines = []
        for match in DEADLINE_PATTERN.finditer(text):
            deadline_text = match.group(0).strip()
            extracted_date = match.group(1).strip()
            # Find surrounding context (30 chars before)
            start = max(0, match.start() - 30)
            context = text[start:match.end()]
            deadlines.append({
                "text": deadline_text,
                "date_phrase": extracted_date,
                "context": context.strip(),
            })
        return deadlines[:20]  # Cap at 20

    def _extract_references(self, text: str) -> List[str]:
        refs = set()
        for url in URL_PATTERN.findall(text):
            refs.add(url)
        for ticket in TICKET_PATTERN.findall(text):
            refs.add(ticket)
        return list(refs)[:50]

    def _extract_meetings(self, text: str) -> List[str]:
        meetings = set()
        for match in MEETING_PATTERN.finditer(text):
            meetings.add(match.group(0).strip())
        return list(meetings)[:20]

    def _extract_organizations(self, text: str) -> List[str]:
        orgs = set()
        for match in ORGANIZATION_SUFFIXES.finditer(text):
            org = match.group(0).strip()
            if len(org) > 3:
                orgs.add(org)
        return list(orgs)[:30]

    def _extract_projects(self, text: str) -> List[str]:
        projects = set()
        for match in PROJECT_PATTERN.finditer(text):
            name = match.group(1).strip()
            if len(name) > 2:
                projects.add(name)
        return list(projects)[:20]

    def _run_spacy(self, nlp, text: str) -> Dict[str, Any]:
        """Run spaCy NER for person, org, location extraction."""
        doc = nlp(text)
        persons = set()
        organizations = set()
        locations = set()

        for ent in doc.ents:
            if ent.label_ in ("PERSON",):
                if len(ent.text) > 2:
                    persons.add(ent.text.strip())
            elif ent.label_ in ("ORG",):
                if len(ent.text) > 2:
                    organizations.add(ent.text.strip())
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                locations.add(ent.text.strip())

        # Extract noun chunks as key phrases (simple approach)
        key_phrases = []
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if len(text.split()) > 1 and len(text) > 4:
                key_phrases.append(text)

        return {
            "persons": list(persons)[:30],
            "organizations": list(organizations)[:30],
            "locations": list(locations)[:20],
            "key_phrases": key_phrases[:20],
            "language": "en",
        }
