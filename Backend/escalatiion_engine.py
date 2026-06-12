import re
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.escalation import Escalation, EscalationSignal
from ..models.commitment import Commitment, CommitmentStatus

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Signal thresholds
# ─────────────────────────────────────────────────────────────

# Signal 1: How many follow-ups before it becomes an escalation
FOLLOWUP_THRESHOLD = 2

# Signal 2: Silence thresholds by relationship type
SILENCE_THRESHOLDS = {
    "vendor": 14,
    "investor": 21,
    "regulator": 14,
    "partner": 21,
    "default": 21,
}

# Signal 4: Negative words indicating a problem
NEGATIVE_PATTERNS = [
    r"\b(concern|concerned|concerns)\b",
    r"\b(delay|delayed|delays|slipping|slip)\b",
    r"\b(issue|issues|problem|problems|trouble)\b",
    r"\b(escalat|escalation|escalating)\b",
    r"\b(block|blocked|blocking|bottleneck)\b",
    r"\b(urgent|urgently|critical|critical)\b",
    r"\b(disappoint|disappointed|disappointing)\b",
    r"\b(unacceptable|unresponsive)\b",
    r"\b(overdue|missed|miss|missed deadline)\b",
    r"\b(no\s+response|haven[''`]?t\s+heard|silence)\b",
    r"\b(pushback|push\s+back|resistance)\b",
    r"\b(risk|risky|jeopardize)\b",
]

# Follow-up patterns
FOLLOWUP_PATTERNS = [
    r"\b(following\s+up|follow[-\s]?up|following\s+up\s+on)\b",
    r"\b(checking\s+in|checking\s+back|touching\s+base)\b",
    r"\b(any\s+update|any\s+news|any\s+progress|any\s+word)\b",
    r"\b(gentle\s+reminder|just\s+a\s+reminder|reminder)\b",
    r"\b(as\s+per\s+my\s+(previous|last|earlier)\s+(email|message|note))\b",
    r"\b(still\s+waiting|still\s+haven[''`]?t)\b",
    r"\b(nudge|pinging\s+again|reaching\s+out\s+again)\b",
]


class EscalationDetectionEngine:
    """
    Task 4: Detects escalation conditions across the five signal types.
    Every escalation is grounded in evidence.
    """

    def __init__(self, db: Session):
        self.db = db

    # ─────────────────────────────────────────────────────────
    # SIGNAL 1: REPEATED FOLLOW-UPS
    # ─────────────────────────────────────────────────────────

    def detect_followup_signal(self, text: str) -> list[str]:
        """Identify follow-up language in text."""
        matches = []
        text_lower = text.lower()
        for pattern in FOLLOWUP_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                matches.append(m.group(0))
        return matches

    def scan_for_repeated_followups(
        self,
        company_name: str,
        window_days: int = 30,
    ) -> Optional[Escalation]:
        """
        Scan interaction history for repeated follow-ups to the same company.
        Fires when FOLLOWUP_THRESHOLD is exceeded.
        """
        from ..models.relationship import Interaction
        since = datetime.utcnow() - timedelta(days=window_days)

        interactions = (
            self.db.query(Interaction)
            .filter(
                Interaction.company_id.isnot(None),
            )
            .filter(
                func.lower(
                    self.db.query(
                        __import__('sqlalchemy').text("SELECT name FROM companies WHERE id = interactions.company_id")
                    )
                ) == company_name.lower()
            )
            .filter(Interaction.occurred_at >= since)
            .order_by(Interaction.occurred_at)
            .all()
        )

        # Simpler approach: count follow-up interactions by checking summaries
        followup_events = []
        for interaction in interactions:
            text_to_check = f"{interaction.subject or ''} {interaction.summary or ''}"
            signals = self.detect_followup_signal(text_to_check)
            if signals:
                followup_events.append({
                    "date": interaction.occurred_at.isoformat() if interaction.occurred_at else None,
                    "document_id": interaction.source_document_id,
                    "signals": signals,
                    "excerpt": text_to_check[:200],
                })

        if len(followup_events) >= FOLLOWUP_THRESHOLD:
            return self._create_or_update_escalation(
                signal_type=EscalationSignal.REPEATED_FOLLOWUP,
                company_name=company_name,
                title=f"Repeated follow-ups to {company_name} ({len(followup_events)} times in {window_days} days)",
                description=(
                    f"Follow-up communication has been sent {len(followup_events)} times "
                    f"in the past {window_days} days, suggesting blocked progress or unresponsiveness."
                ),
                evidence=followup_events,
                trigger_count=len(followup_events),
                signal_dates=[e["date"] for e in followup_events],
            )
        return None

    # ─────────────────────────────────────────────────────────
    # SIGNAL 2: LONG SILENCE
    # ─────────────────────────────────────────────────────────

    def scan_for_long_silence(self, silence_days_override: Optional[int] = None) -> list[Escalation]:
        """
        Scan all companies for communication silence beyond their threshold.
        """
        from ..models.relationship import Company
        companies = self.db.query(Company).filter(Company.last_interaction_at.isnot(None)).all()
        escalations = []
        now = datetime.utcnow()

        for company in companies:
            threshold = silence_days_override or SILENCE_THRESHOLDS.get(
                company.relationship_type or "default",
                SILENCE_THRESHOLDS["default"],
            )
            last_contact = company.last_interaction_at.replace(tzinfo=None)
            days_silent = (now - last_contact).days

            if days_silent >= threshold:
                esc = self._create_or_update_escalation(
                    signal_type=EscalationSignal.LONG_SILENCE,
                    company_name=company.name,
                    title=f"No contact with {company.name} for {days_silent} days",
                    description=(
                        f"Last interaction with {company.name} was {days_silent} days ago "
                        f"(threshold: {threshold} days for {company.relationship_type or 'default'} relationships)."
                    ),
                    evidence=[{
                        "type": "silence",
                        "last_contact": last_contact.isoformat(),
                        "days_silent": days_silent,
                        "threshold": threshold,
                    }],
                    last_contact_date=last_contact,
                    silence_days=days_silent,
                )
                escalations.append(esc)

        return escalations

    # ─────────────────────────────────────────────────────────
    # SIGNAL 3: DEADLINE PASSED
    # ─────────────────────────────────────────────────────────

    def scan_for_overdue_commitments(self) -> list[Escalation]:
        """
        Scan for commitments past their due date — each is an escalation signal.
        Groups by company for concise executive reporting.
        """
        now = datetime.utcnow()
        overdue = (
            self.db.query(Commitment)
            .filter(
                Commitment.status == CommitmentStatus.OVERDUE,
                Commitment.due_date < now,
            )
            .all()
        )

        # Group by company
        by_company: dict[str, list] = {}
        for c in overdue:
            key = c.company_name or "Unknown"
            by_company.setdefault(key, []).append(c)

        escalations = []
        for company_name, commitments in by_company.items():
            most_overdue = max(
                (c for c in commitments if c.due_date),
                key=lambda x: (now - x.due_date.replace(tzinfo=None)).days,
                default=None,
            )
            days_overdue = 0
            if most_overdue and most_overdue.due_date:
                days_overdue = (now - most_overdue.due_date.replace(tzinfo=None)).days

            esc = self._create_or_update_escalation(
                signal_type=EscalationSignal.DEADLINE_PASSED,
                company_name=company_name,
                title=f"{len(commitments)} overdue commitment(s) with {company_name}",
                description=f"{len(commitments)} commitment(s) past due date. Most overdue: {days_overdue} days.",
                evidence=[{
                    "commitment_id": str(c.id),
                    "text": c.normalized_text or c.raw_text,
                    "due_date": c.due_date.isoformat() if c.due_date else None,
                    "type": c.commitment_type.value,
                } for c in commitments],
                days_overdue=days_overdue,
                original_deadline=most_overdue.due_date if most_overdue else None,
                linked_commitment_ids=[str(c.id) for c in commitments],
            )
            escalations.append(esc)

        return escalations

    # ─────────────────────────────────────────────────────────
    # SIGNAL 4: NEGATIVE COMMUNICATION PATTERN
    # ─────────────────────────────────────────────────────────

    def detect_negative_signals(self, text: str) -> list[str]:
        """Find negative language in text."""
        matches = []
        text_lower = text.lower()
        for pattern in NEGATIVE_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                matches.append(m.group(0))
        return matches

    def scan_for_negative_patterns(
        self,
        company_name: str,
        window_days: int = 14,
        threshold_count: int = 3,
    ) -> Optional[Escalation]:
        """
        Detect repeated negative language in communications with a company.
        Threshold: 3+ negative signals in a 14-day window.
        """
        from ..models.relationship import Interaction, Company
        since = datetime.utcnow() - timedelta(days=window_days)

        company = self.db.query(Company).filter(
            func.lower(Company.name) == company_name.lower()
        ).first()
        if not company:
            return None

        interactions = (
            self.db.query(Interaction)
            .filter(
                Interaction.company_id == company.id,
                Interaction.occurred_at >= since,
            )
            .all()
        )

        negative_evidence = []
        for interaction in interactions:
            text_to_check = f"{interaction.subject or ''} {interaction.summary or ''}"
            signals = self.detect_negative_signals(text_to_check)
            if signals:
                negative_evidence.append({
                    "date": interaction.occurred_at.isoformat() if interaction.occurred_at else None,
                    "document_id": interaction.source_document_id,
                    "negative_signals": signals,
                    "excerpt": text_to_check[:200],
                })

        if len(negative_evidence) >= threshold_count:
            return self._create_or_update_escalation(
                signal_type=EscalationSignal.NEGATIVE_PATTERN,
                company_name=company_name,
                title=f"Negative communication pattern detected with {company_name}",
                description=(
                    f"Found {len(negative_evidence)} interactions with negative signals "
                    f"(concern, delay, issue, blocked, etc.) in the past {window_days} days."
                ),
                evidence=negative_evidence,
                trigger_count=len(negative_evidence),
                signal_dates=[e["date"] for e in negative_evidence],
                severity="high" if len(negative_evidence) >= 5 else "medium",
            )
        return None

    # ─────────────────────────────────────────────────────────
    # SIGNAL 5: PROJECT DRIFT
    # ─────────────────────────────────────────────────────────

    def register_project_drift(
        self,
        project_name: str,
        company_name: str,
        planned_date: datetime,
        current_estimated_date: datetime,
        evidence_document_id: Optional[str] = None,
        evidence_excerpt: Optional[str] = None,
    ) -> Optional[Escalation]:
        """
        Detect when a project's timeline has drifted beyond acceptable bounds.
        Drift threshold: 7+ days.
        """
        drift_days = (current_estimated_date - planned_date).days

        if drift_days < 7:
            return None  # Minor drift, not an escalation

        severity = "low"
        if drift_days >= 30:
            severity = "critical"
        elif drift_days >= 14:
            severity = "high"
        elif drift_days >= 7:
            severity = "medium"

        return self._create_or_update_escalation(
            signal_type=EscalationSignal.PROJECT_DRIFT,
            company_name=company_name,
            project_name=project_name,
            title=f"Project drift: {project_name} is {drift_days} days behind schedule",
            description=(
                f"Originally planned for {planned_date.date()}, "
                f"now estimated {current_estimated_date.date()} — "
                f"{drift_days} days late."
            ),
            evidence=[{
                "type": "project_drift",
                "project": project_name,
                "planned_date": planned_date.isoformat(),
                "current_date": current_estimated_date.isoformat(),
                "drift_days": drift_days,
                "document_id": evidence_document_id,
                "excerpt": evidence_excerpt,
            }],
            planned_date=planned_date,
            current_estimated_date=current_estimated_date,
            drift_days=drift_days,
            severity=severity,
        )

    # ─────────────────────────────────────────────────────────
    # SHARED: CREATE OR UPDATE ESCALATION
    # ─────────────────────────────────────────────────────────

    def _create_or_update_escalation(
        self,
        signal_type: EscalationSignal,
        company_name: str,
        title: str,
        description: str,
        evidence: list,
        trigger_count: int = 1,
        signal_dates: Optional[list] = None,
        last_contact_date: Optional[datetime] = None,
        silence_days: int = 0,
        original_deadline: Optional[datetime] = None,
        days_overdue: int = 0,
        planned_date: Optional[datetime] = None,
        current_estimated_date: Optional[datetime] = None,
        drift_days: int = 0,
        linked_commitment_ids: Optional[list] = None,
        linked_risk_ids: Optional[list] = None,
        severity: str = "medium",
        project_name: Optional[str] = None,
    ) -> Escalation:
        """Upsert an escalation — avoid duplicate escalations for the same signal+company."""
        existing = (
            self.db.query(Escalation)
            .filter(
                Escalation.signal_type == signal_type,
                Escalation.company_name == company_name,
                Escalation.is_active == True,
            )
            .first()
        )

        if existing:
            existing.title = title
            existing.description = description
            existing.evidence = evidence
            existing.trigger_count = trigger_count
            existing.signal_dates = signal_dates or []
            existing.silence_days = silence_days
            existing.days_overdue = days_overdue
            existing.drift_days = drift_days
            existing.severity = severity
            existing.updated_at = datetime.utcnow()
            self.db.flush()
            return existing

        esc = Escalation(
            signal_type=signal_type,
            company_name=company_name,
            project_name=project_name,
            title=title,
            description=description,
            evidence=evidence,
            trigger_count=trigger_count,
            signal_dates=signal_dates or [],
            last_contact_date=last_contact_date,
            silence_days=silence_days,
            original_deadline=original_deadline,
            days_overdue=days_overdue,
            planned_date=planned_date,
            current_estimated_date=current_estimated_date,
            drift_days=drift_days,
            linked_commitment_ids=linked_commitment_ids or [],
            linked_risk_ids=linked_risk_ids or [],
            severity=severity,
            first_detected_at=datetime.utcnow(),
        )
        self.db.add(esc)
        self.db.flush()
        logger.warning(f"New escalation [{signal_type.value}]: {title}")
        return esc

    def run_full_escalation_scan(self) -> dict:
        """
        Scheduled job: Run all escalation detectors.
        Returns summary of active escalations by signal type.
        """
        results = {
            "silence": [],
            "overdue": [],
            "total_active": 0,
        }

        # Signal 2: Long silence (scan all companies)
        silence_escalations = self.scan_for_long_silence()
        results["silence"] = [{"company": e.company_name, "days": e.silence_days} for e in silence_escalations]

        # Signal 3: Deadline passed
        overdue_escalations = self.scan_for_overdue_commitments()
        results["overdue"] = [{"company": e.company_name, "count": e.trigger_count} for e in overdue_escalations]

        self.db.commit()

        results["total_active"] = self.db.query(Escalation).filter(Escalation.is_active == True).count()
        return results

    def get_active_escalations_summary(self) -> list[dict]:
        """Return all active escalations for the executive briefing."""
        escalations = (
            self.db.query(Escalation)
            .filter(Escalation.is_active == True)
            .order_by(Escalation.severity.desc(), Escalation.first_detected_at.desc())
            .all()
        )
        return [
            {
                "id": str(e.id),
                "signal_type": e.signal_type.value,
                "title": e.title,
                "company": e.company_name,
                "project": e.project_name,
                "severity": e.severity,
                "description": e.description,
                "first_detected": e.first_detected_at.isoformat() if e.first_detected_at else None,
                "evidence_count": len(e.evidence or []),
            }
            for e in escalations
        ]
