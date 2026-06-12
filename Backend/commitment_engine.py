
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy.orm import Session
from dateutil import parser as dateutil_parser

from ..models.commitment import Commitment, CommitmentHistory, CommitmentType, CommitmentStatus

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://ollama:11434"
LLM_MODEL = "qwen3"


# ─────────────────────────────────────────────────────────────
# TASK 1: COMMITMENT DETECTION ENGINE
# ─────────────────────────────────────────────────────────────

# Step 1: Commitment type patterns (deterministic first pass)
COMMITMENT_PATTERNS = {
    CommitmentType.DELIVERABLE: [
        r"i[''`]?ll\s+(send|deliver|share|provide|submit|upload|prepare|write|create|complete)",
        r"will\s+(send|deliver|share|provide|submit|upload|prepare|write|create|complete)",
        r"(sending|delivering|sharing|providing)\s+(by|before|on)",
        r"(report|proposal|document|contract|draft)\s+will\s+be\s+(ready|sent|delivered|shared)",
        r"(i|we)\s+will\s+get\s+(it|this|that)\s+(to|over)\s+to\s+you",
    ],
    CommitmentType.APPROVAL: [
        r"need(s)?\s+(approval|signoff|sign-off|sign\s+off|authorization|clearance|green\s+light)",
        r"(approval|signoff)\s+(required|needed|pending)",
        r"waiting\s+for\s+(your\s+)?(approval|signoff|authorization|go-ahead)",
        r"(please|kindly)\s+(approve|authorize|confirm|sign\s+off)",
        r"pending\s+(your\s+)?(review|approval|decision)",
    ],
    CommitmentType.RESPONSE: [
        r"will\s+(get\s+back|revert|respond|follow\s+up|follow-up)",
        r"(get\s+back|revert)\s+to\s+you",
        r"(i[''`]?ll|we[''`]?ll)\s+(check|look\s+into|investigate|look\s+at\s+this)",
        r"(will|shall)\s+(confirm|clarify|let\s+you\s+know|update\s+you)",
        r"(checking|looking\s+into)\s+(this|it|that)",
    ],
    CommitmentType.MEETING: [
        r"(let[''`]?s|we\s+should)\s+(discuss|connect|meet|sync|catch\s+up|talk)",
        r"(schedule|set\s+up|arrange)\s+a\s+(meeting|call|session|discussion)",
        r"(meeting|call|discussion)\s+(next|this)\s+(week|month|tuesday|wednesday|thursday|friday|monday)",
        r"(revisit|reconnect|follow\s+up)\s+(next|this|in)",
        r"(let[''`]?s)\s+(circle\s+back|touch\s+base|regroup)",
    ],
    CommitmentType.EXTERNAL_DEPENDENCY: [
        r"waiting\s+(for|on)\s+(regulator|vendor|client|partner|bank|investor|government|authority)",
        r"(pending|awaiting)\s+(regulator|vendor|client|external|third[-\s]?party)",
        r"(regulator|vendor|client|partner)\s+(has\s+not|hasn[''`]?t)\s+(responded|replied|confirmed)",
        r"(external|third[-\s]?party)\s+(approval|confirmation|response)\s+(pending|awaited|needed)",
        r"(no\s+response|silent)\s+from\s+(regulator|vendor|client|partner)",
    ],
}

# Deadline extraction patterns
DEADLINE_PATTERNS = [
    r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"by\s+(end\s+of\s+(day|week|month|quarter))",
    r"(before|by)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"(this|next)\s+(monday|tuesday|wednesday|thursday|friday|week|month)",
    r"(in|within)\s+(\d+)\s+(days?|hours?|weeks?)",
    r"(tomorrow|today)",
    r"(monday|tuesday|wednesday|thursday|friday)\b",
]

# Resolution signals — what proves a commitment was completed
RESOLUTION_PATTERNS = [
    r"(done|completed|finished|delivered|sent|submitted|approved|signed|confirmed|resolved)",
    r"(please\s+find\s+attached|attached\s+(herewith|please\s+find))",
    r"(i\s+have\s+sent|we\s+have\s+sent|sent\s+over)",
    r"(approval\s+granted|approved|sign(ed|ature)\s+(received|obtained))",
    r"(meeting\s+(held|completed|done)|we\s+(met|connected|synced|discussed))",
    r"(contract\s+(signed|executed|finalized))",
    r"(proposal\s+(sent|delivered|shared|submitted))",
    r"(response\s+received|responded|replied)",
]

# Negative / stall signals
STALL_PATTERNS = [
    r"(following\s+up|checking\s+in|any\s+update|any\s+news)",
    r"(still\s+waiting|no\s+response|hasn[''`]?t\s+(responded|replied))",
    r"(reminder|gentle\s+reminder|nudge)",
    r"(as\s+discussed|as\s+mentioned|as\s+per\s+my\s+(previous|last)\s+(email|message))",
]


class CommitmentDetectionEngine:
    """
    Task 1: Detects commitments from organizational text.
    Uses deterministic pattern matching first, then LLM for edge cases.
    """

    def __init__(self, db: Session):
        self.db = db

    # ── Step 1: Type detection via regex ──────────────────────────────────────

    def detect_commitment_type(self, text: str) -> Optional[CommitmentType]:
        """Classify commitment type using deterministic patterns."""
        text_lower = text.lower()
        scores = {}
        for ctype, patterns in COMMITMENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text_lower))
            if score > 0:
                scores[ctype] = score
        if not scores:
            return None
        return max(scores, key=scores.get)

    # ── Step 2: Extraction rules ───────────────────────────────────────────────

    def has_future_orientation(self, text: str) -> bool:
        future_words = r"\b(will|shall|going\s+to|would|should|must|need\s+to|plan\s+to|intend\s+to|by|before|next|tomorrow|upcoming)\b"
        return bool(re.search(future_words, text.lower()))

    def has_action_statement(self, text: str) -> bool:
        action_words = r"\b(send|deliver|review|approve|submit|complete|prepare|provide|share|schedule|meet|confirm|sign|respond)\b"
        return bool(re.search(action_words, text.lower()))

    def extract_deadline(self, text: str) -> Optional[datetime]:
        """Extract deadline from text if present."""
        text_lower = text.lower()
        for pattern in DEADLINE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    date_str = match.group(0)
                    # Resolve relative dates
                    now = datetime.utcnow()
                    relative_map = {
                        "tomorrow": now + timedelta(days=1),
                        "today": now,
                        "end of day": now.replace(hour=17, minute=0),
                        "end of week": now + timedelta(days=(4 - now.weekday())),
                        "end of month": (now.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1),
                        "next week": now + timedelta(weeks=1),
                        "this week": now + timedelta(days=(4 - now.weekday())),
                    }
                    for key, val in relative_map.items():
                        if key in date_str:
                            return val
                    return dateutil_parser.parse(date_str, default=now, fuzzy=True)
                except Exception:
                    continue
        return None

    def extract_owner(self, text: str) -> Optional[str]:
        """Extract who made the commitment (simple heuristic)."""
        patterns = [
            r"^i\s+will",
            r"^i[''`]ll",
            r"^we\s+will",
            r"^we[''`]ll",
        ]
        for p in patterns:
            if re.search(p, text.lower().strip()):
                return "sender"
        return None

    def is_candidate(self, text: str) -> bool:
        """
        Step 2: Decide if a sentence is a commitment candidate.
        Must have: future orientation OR action statement.
        """
        return self.has_future_orientation(text) or self.has_action_statement(text)

    def calculate_confidence(self, text: str, ctype: CommitmentType) -> float:
        """Score extraction confidence 0.0-1.0."""
        score = 0.0
        text_lower = text.lower()

        # Pattern strength
        patterns = COMMITMENT_PATTERNS.get(ctype, [])
        matches = sum(1 for p in patterns if re.search(p, text_lower))
        score += min(matches * 0.2, 0.5)

        # Modifiers
        if self.has_future_orientation(text):
            score += 0.15
        if self.has_action_statement(text):
            score += 0.15
        if self.extract_deadline(text):
            score += 0.1
        if self.extract_owner(text):
            score += 0.1

        return min(score, 1.0)

    # ── LLM-assisted extraction for low-confidence candidates ─────────────────

    async def llm_extract_commitment(self, text: str) -> Optional[dict]:
        """
        Use Qwen3 via Ollama to extract structured commitment data.
        Only used when deterministic confidence is below threshold.
        """
        prompt = f"""Analyze this text and determine if it contains an organizational commitment.

Text: "{text}"

A commitment is:
- A promise to deliver something
- A request for approval or signoff
- A statement about following up or responding
- A meeting or discussion scheduled
- Dependency on an external party

Respond ONLY with valid JSON in this exact format:
{{
  "is_commitment": true/false,
  "commitment_type": "deliverable|approval|response|meeting|external",
  "owner": "who made the commitment or null",
  "counterparty": "who it was made to or null",
  "due_date_text": "any deadline mentioned or null",
  "confidence": 0.0-1.0,
  "normalized_text": "clean restatement of the commitment"
}}"""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"},
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "{}")
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"LLM commitment extraction failed: {e}")
            return None

    # ── Main extraction pipeline ───────────────────────────────────────────────

    async def extract_commitments(
        self,
        text: str,
        source_document_id: str,
        source_document_type: str,
        sender_email: Optional[str] = None,
        sender_name: Optional[str] = None,
        recipient_email: Optional[str] = None,
        company_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> list[Commitment]:
        """
        Step 2 (full pipeline): Extract all commitments from a document/email/meeting.
        Returns list of persisted Commitment objects.
        """
        commitments = []

        # Split into sentences for granular analysis
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Candidate filter
            if not self.is_candidate(sentence):
                continue

            ctype = self.detect_commitment_type(sentence)
            confidence = 0.0
            normalized_text = sentence
            owner = sender_name or self.extract_owner(sentence)
            due_date = self.extract_deadline(sentence)

            if ctype:
                confidence = self.calculate_confidence(sentence, ctype)
            else:
                # Fall back to LLM
                llm_result = await self.llm_extract_commitment(sentence)
                if llm_result and llm_result.get("is_commitment"):
                    type_map = {
                        "deliverable": CommitmentType.DELIVERABLE,
                        "approval": CommitmentType.APPROVAL,
                        "response": CommitmentType.RESPONSE,
                        "meeting": CommitmentType.MEETING,
                        "external": CommitmentType.EXTERNAL_DEPENDENCY,
                    }
                    ctype = type_map.get(llm_result.get("commitment_type", ""))
                    confidence = llm_result.get("confidence", 0.3)
                    normalized_text = llm_result.get("normalized_text", sentence)
                    owner = llm_result.get("owner") or owner

            if not ctype or confidence < 0.2:
                continue

            commitment = Commitment(
                raw_text=sentence,
                normalized_text=normalized_text,
                commitment_type=ctype,
                status=CommitmentStatus.OPEN,
                owner=owner,
                owner_email=sender_email,
                counterparty=recipient_email,
                counterparty_email=recipient_email,
                due_date=due_date,
                source_document_id=source_document_id,
                source_document_type=source_document_type,
                source_excerpt=sentence,
                company_name=company_name,
                project_name=project_name,
                confidence_score=confidence,
            )
            self.db.add(commitment)
            self.db.flush()

            # Add initial history entry
            history = CommitmentHistory(
                commitment_id=commitment.id,
                previous_status=None,
                new_status=CommitmentStatus.OPEN,
                reason="Auto-detected by commitment extraction engine",
                evidence_excerpt=sentence,
                evidence_document_id=source_document_id,
            )
            self.db.add(history)
            commitments.append(commitment)

        self.db.commit()
        logger.info(f"Extracted {len(commitments)} commitments from document {source_document_id}")
        return commitments


# ─────────────────────────────────────────────────────────────
# TASK 2: COMMITMENT RESOLUTION ENGINE
# ─────────────────────────────────────────────────────────────

class CommitmentResolutionEngine:
    """
    Task 2: Moves commitments through their lifecycle by detecting resolution evidence.
    Matches follow-up text against open commitments.
    """

    def __init__(self, db: Session):
        self.db = db

    def detect_resolution_signals(self, text: str) -> list[str]:
        """Find resolution signals in a piece of text."""
        signals = []
        text_lower = text.lower()
        for pattern in RESOLUTION_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                signals.append(match.group(0))
        return signals

    def detect_stall_signals(self, text: str) -> list[str]:
        """Find stall/follow-up signals that indicate a commitment is in progress but blocked."""
        signals = []
        text_lower = text.lower()
        for pattern in STALL_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                signals.append(match.group(0))
        return signals

    def _compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """
        Simple token overlap similarity (no ML dependency).
        Replace with BGE-M3 embeddings in production for precision.
        """
        words_a = set(re.findall(r'\b\w{4,}\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w{4,}\b', text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def find_matching_commitment(
        self,
        text: str,
        company_name: Optional[str] = None,
        person_email: Optional[str] = None,
    ) -> Optional[Commitment]:
        """
        Match a piece of text to an open commitment.
        Uses token similarity + context filters.
        """
        query = self.db.query(Commitment).filter(
            Commitment.status.in_([CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS])
        )
        if company_name:
            query = query.filter(Commitment.company_name == company_name)
        if person_email:
            query = query.filter(
                (Commitment.owner_email == person_email) |
                (Commitment.counterparty_email == person_email)
            )

        candidates = query.all()
        best_match = None
        best_score = 0.3  # Minimum threshold

        for commitment in candidates:
            score = self._compute_text_similarity(text, commitment.normalized_text or commitment.raw_text)
            if score > best_score:
                best_score = score
                best_match = commitment

        return best_match

    def _transition_status(
        self,
        commitment: Commitment,
        new_status: CommitmentStatus,
        reason: str,
        evidence_excerpt: Optional[str] = None,
        evidence_document_id: Optional[str] = None,
        changed_by: Optional[str] = None,
    ):
        """Core lifecycle transition with full audit history."""
        old_status = commitment.status
        commitment.status = new_status
        commitment.updated_at = datetime.utcnow()

        if new_status == CommitmentStatus.RESOLVED:
            commitment.resolved_at = datetime.utcnow()
            commitment.resolution_evidence = evidence_excerpt
            commitment.resolution_document_id = evidence_document_id

        history = CommitmentHistory(
            commitment_id=commitment.id,
            previous_status=old_status,
            new_status=new_status,
            changed_by=changed_by,
            reason=reason,
            evidence_excerpt=evidence_excerpt,
            evidence_document_id=evidence_document_id,
        )
        self.db.add(history)
        logger.info(f"Commitment {commitment.id}: {old_status} → {new_status} ({reason})")

    # ── Step 3: Lifecycle transition logic ────────────────────────────────────

    def process_document_for_resolutions(
        self,
        text: str,
        source_document_id: str,
        company_name: Optional[str] = None,
        person_email: Optional[str] = None,
    ) -> list[dict]:
        """
        Full resolution pipeline:
        1. Detect resolution signals in text
        2. Match to open commitments
        3. Transition commitment lifecycle accordingly
        Returns list of transition events.
        """
        transitions = []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            resolution_signals = self.detect_resolution_signals(sentence)
            stall_signals = self.detect_stall_signals(sentence)

            if not resolution_signals and not stall_signals:
                continue

            matched = self.find_matching_commitment(sentence, company_name, person_email)
            if not matched:
                continue

            if resolution_signals:
                # Evidence of completion → Resolved
                self._transition_status(
                    matched,
                    CommitmentStatus.RESOLVED,
                    reason=f"Resolution evidence detected: {', '.join(resolution_signals)}",
                    evidence_excerpt=sentence,
                    evidence_document_id=source_document_id,
                )
                transitions.append({
                    "commitment_id": str(matched.id),
                    "transition": f"{CommitmentStatus.OPEN} → {CommitmentStatus.RESOLVED}",
                    "evidence": sentence,
                    "signals": resolution_signals,
                })

            elif stall_signals and matched.status == CommitmentStatus.OPEN:
                # Follow-up detected → In Progress (being chased)
                self._transition_status(
                    matched,
                    CommitmentStatus.IN_PROGRESS,
                    reason=f"Follow-up detected: {', '.join(stall_signals)}",
                    evidence_excerpt=sentence,
                    evidence_document_id=source_document_id,
                )
                transitions.append({
                    "commitment_id": str(matched.id),
                    "transition": f"{CommitmentStatus.OPEN} → {CommitmentStatus.IN_PROGRESS}",
                    "evidence": sentence,
                    "signals": stall_signals,
                })

        self.db.commit()
        return transitions

    def mark_overdue_commitments(self) -> int:
        """
        Scheduled job: Mark commitments past due date as Overdue.
        Returns count of newly overdue commitments.
        """
        now = datetime.utcnow()
        overdue_candidates = self.db.query(Commitment).filter(
            Commitment.status.in_([CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS]),
            Commitment.due_date < now,
            Commitment.due_date.isnot(None),
        ).all()

        count = 0
        for commitment in overdue_candidates:
            self._transition_status(
                commitment,
                CommitmentStatus.OVERDUE,
                reason=f"Due date {commitment.due_date.date()} passed",
            )
            commitment.overdue_notified = False
            count += 1

        self.db.commit()
        logger.info(f"Marked {count} commitments as overdue")
        return count

    def get_commitment_dashboard(self) -> dict:
        """
        Summary stats for executive dashboard.
        """
        from sqlalchemy import func
        counts = (
            self.db.query(Commitment.status, func.count(Commitment.id))
            .group_by(Commitment.status)
            .all()
        )
        stats = {status: count for status, count in counts}

        overdue = self.db.query(Commitment).filter(
            Commitment.status == CommitmentStatus.OVERDUE
        ).order_by(Commitment.due_date).limit(10).all()

        return {
            "summary": {s.value: stats.get(s, 0) for s in CommitmentStatus},
            "overdue_commitments": [
                {
                    "id": str(c.id),
                    "text": c.normalized_text or c.raw_text,
                    "owner": c.owner,
                    "company": c.company_name,
                    "due_date": c.due_date.isoformat() if c.due_date else None,
                    "type": c.commitment_type.value,
                }
                for c in overdue
            ],
        }
