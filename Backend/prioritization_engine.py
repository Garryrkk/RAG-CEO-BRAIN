
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.briefing import Priority, PriorityLevel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Scoring weights (sum to 1.0)
# ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "ceo_involvement": 0.25,
    "revenue_impact": 0.25,
    "deadline_urgency": 0.20,
    "escalation_level": 0.20,
    "relationship_importance": 0.10,
}

# Priority thresholds
HIGH_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35

# CEO involvement keywords
CEO_INVOLVEMENT_KEYWORDS = [
    "ceo", "chief executive", "founder", "board", "chairman",
    "executive", "c-suite", "managing director", "md", "president",
]

# High revenue impact keywords
REVENUE_KEYWORDS = [
    "revenue", "deal", "contract value", "payment", "invoice",
    "million", "crore", "lakh", "billion", "budget", "funding",
    "investment", "acquisition", "merger",
]

# High relationship importance keywords
KEY_RELATIONSHIP_KEYWORDS = [
    "investor", "regulator", "bank", "key client", "strategic partner",
    "government", "authority", "fca", "sec", "rbi",
]


class PrioritizationEngine:
    """
    Task 8: Scores and ranks executive action items.
    Every priority score is transparent and explainable.
    """

    def __init__(self, db: Session):
        self.db = db

    # ─────────────────────────────────────────────────────────
    # INDIVIDUAL FACTOR SCORERS
    # ─────────────────────────────────────────────────────────

    def score_ceo_involvement(self, text: str, metadata: Optional[dict] = None) -> float:
        """
        0.0-1.0: Is the CEO directly involved or mentioned?
        """
        text_lower = (text or "").lower()
        if any(kw in text_lower for kw in CEO_INVOLVEMENT_KEYWORDS):
            return 1.0
        if metadata and metadata.get("ceo_mentioned"):
            return 1.0
        if metadata and metadata.get("forwarded_by_ceo"):
            return 0.8
        return 0.0

    def score_revenue_impact(self, text: str, company_name: Optional[str] = None) -> float:
        """
        0.0-1.0: Does this item have revenue implications?
        """
        text_lower = (text or "").lower()

        # Direct revenue mention
        if any(kw in text_lower for kw in REVENUE_KEYWORDS):
            # Higher score for larger amounts
            import re
            amount_patterns = [
                (r"\$[\d,]+\s*[mb]illion", 1.0),
                (r"\$[\d,]+\s*million", 0.9),
                (r"[\d,]+\s*crore", 0.9),
                (r"\$[\d,]+\s*thousand", 0.5),
                (r"revenue", 0.7),
                (r"deal|contract value", 0.8),
            ]
            for pattern, score in amount_patterns:
                if re.search(pattern, text_lower):
                    return score
            return 0.6

        # Company-level revenue impact
        if company_name:
            from ..models.relationship import Company
            company = self.db.query(Company).filter(
                Company.name.ilike(f"%{company_name}%")
            ).first()
            if company:
                impact_map = {"high": 0.9, "medium": 0.5, "low": 0.2}
                return impact_map.get(company.revenue_impact or "low", 0.2)

        return 0.2

    def score_deadline_urgency(self, due_date: Optional[datetime]) -> float:
        """
        0.0-1.0: How urgent is the deadline?
        Overdue = 1.0, Due today = 0.95, 1 week = 0.7, 1 month = 0.3
        """
        if not due_date:
            return 0.1  # No deadline = low urgency

        now = datetime.utcnow()
        due = due_date.replace(tzinfo=None) if due_date.tzinfo else due_date
        days_until = (due - now).days

        if days_until < 0:
            return 1.0   # Overdue
        elif days_until == 0:
            return 0.95  # Due today
        elif days_until <= 1:
            return 0.90  # Due tomorrow
        elif days_until <= 3:
            return 0.80  # Due this week (near)
        elif days_until <= 7:
            return 0.70  # Due this week
        elif days_until <= 14:
            return 0.55  # Due in 2 weeks
        elif days_until <= 30:
            return 0.35  # Due this month
        else:
            return 0.15  # Far future

    def score_escalation_level(self, entity_name: Optional[str] = None, entity_type: Optional[str] = None) -> float:
        """
        0.0-1.0: Is there an active escalation for this entity?
        """
        if not entity_name:
            return 0.0

        from ..models.escalation import Escalation
        escalations = (
            self.db.query(Escalation)
            .filter(
                Escalation.company_name == entity_name,
                Escalation.is_active == True,
            )
            .all()
        )
        if not escalations:
            return 0.0

        severity_scores = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
        max_score = max(severity_scores.get(e.severity, 0.3) for e in escalations)
        # Boost for multiple escalations
        count_boost = min((len(escalations) - 1) * 0.1, 0.2)
        return min(max_score + count_boost, 1.0)

    def score_relationship_importance(self, company_name: Optional[str] = None, text: Optional[str] = None) -> float:
        """
        0.0-1.0: Is this a strategically important relationship?
        """
        text_lower = (text or "").lower()
        if any(kw in text_lower for kw in KEY_RELATIONSHIP_KEYWORDS):
            return 0.9

        if company_name:
            from ..models.relationship import Company
            company = self.db.query(Company).filter(
                Company.name.ilike(f"%{company_name}%")
            ).first()
            if company:
                importance_map = {"high": 1.0, "medium": 0.6, "low": 0.2}
                return importance_map.get(company.strategic_importance or "low", 0.2)

        return 0.2

    # ─────────────────────────────────────────────────────────
    # COMPOSITE SCORING
    # ─────────────────────────────────────────────────────────

    def compute_priority_score(
        self,
        text: str,
        due_date: Optional[datetime] = None,
        company_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[float, dict]:
        """
        Compute composite priority score with factor breakdown.
        Returns (score, {factor: score} breakdown).
        """
        factors = {
            "ceo_involvement": self.score_ceo_involvement(text, metadata),
            "revenue_impact": self.score_revenue_impact(text, company_name),
            "deadline_urgency": self.score_deadline_urgency(due_date),
            "escalation_level": self.score_escalation_level(company_name),
            "relationship_importance": self.score_relationship_importance(company_name, text),
        }

        composite = sum(factors[k] * WEIGHTS[k] for k in factors)
        return composite, factors

    def determine_priority_level(self, score: float) -> PriorityLevel:
        if score >= HIGH_THRESHOLD:
            return PriorityLevel.HIGH
        elif score >= MEDIUM_THRESHOLD:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    # ─────────────────────────────────────────────────────────
    # COMMITMENT PRIORITIZATION
    # ─────────────────────────────────────────────────────────

    def prioritize_commitments(self) -> list[Priority]:
        """Rank all open/overdue commitments by priority."""
        from ..models.commitment import Commitment, CommitmentStatus
        commitments = (
            self.db.query(Commitment)
            .filter(
                Commitment.status.in_([
                    CommitmentStatus.OPEN,
                    CommitmentStatus.IN_PROGRESS,
                    CommitmentStatus.OVERDUE,
                ])
            )
            .all()
        )

        priorities = []
        for commitment in commitments:
            text = commitment.normalized_text or commitment.raw_text
            score, factors = self.compute_priority_score(
                text=text,
                due_date=commitment.due_date,
                company_name=commitment.company_name,
            )
            level = self.determine_priority_level(score)

            # Update commitment priority score
            commitment.priority_score = score

            priority = self._upsert_priority(
                title=text[:200],
                description=f"[{commitment.commitment_type.value}] {text}",
                level=level,
                priority_score=score,
                entity_type="commitment",
                entity_id=str(commitment.id),
                entity_name=commitment.owner or "Unknown",
                due_date=commitment.due_date,
                company_name=commitment.company_name,
                factors=factors,
            )
            priorities.append(priority)

        self.db.commit()
        return priorities

    # ─────────────────────────────────────────────────────────
    # RISK PRIORITIZATION
    # ─────────────────────────────────────────────────────────

    def prioritize_risks(self) -> list[Priority]:
        """Rank all active risks by priority."""
        from ..models.risk import Risk, RiskSeverity
        risks = self.db.query(Risk).filter(Risk.is_active == True).all()

        severity_due_dates = {
            RiskSeverity.CRITICAL: datetime.utcnow(),
            RiskSeverity.HIGH: datetime.utcnow() + timedelta(days=3),
            RiskSeverity.MEDIUM: datetime.utcnow() + timedelta(days=14),
            RiskSeverity.LOW: datetime.utcnow() + timedelta(days=30),
        }

        priorities = []
        for risk in risks:
            score, factors = self.compute_priority_score(
                text=risk.title,
                due_date=severity_due_dates.get(risk.severity),
                company_name=risk.company_name,
            )
            # Boost by intrinsic risk score
            score = min(score * 0.7 + risk.risk_score * 0.3, 1.0)
            level = self.determine_priority_level(score)

            priority = self._upsert_priority(
                title=risk.title[:200],
                description=risk.description,
                level=level,
                priority_score=score,
                entity_type="risk",
                entity_id=str(risk.id),
                entity_name=risk.company_name or risk.title[:50],
                company_name=risk.company_name,
                factors=factors,
            )
            priorities.append(priority)

        self.db.commit()
        return priorities

    def _upsert_priority(
        self,
        title: str,
        description: Optional[str],
        level: PriorityLevel,
        priority_score: float,
        entity_type: str,
        entity_id: str,
        entity_name: str,
        due_date: Optional[datetime] = None,
        company_name: Optional[str] = None,
        factors: Optional[dict] = None,
    ) -> Priority:
        factors = factors or {}
        existing = (
            self.db.query(Priority)
            .filter(
                Priority.entity_type == entity_type,
                Priority.entity_id == entity_id,
            )
            .first()
        )
        if existing:
            existing.level = level
            existing.priority_score = priority_score
            existing.ceo_involvement_score = factors.get("ceo_involvement", 0.0)
            existing.revenue_impact_score = factors.get("revenue_impact", 0.0)
            existing.deadline_urgency_score = factors.get("deadline_urgency", 0.0)
            existing.escalation_level_score = factors.get("escalation_level", 0.0)
            existing.relationship_importance_score = factors.get("relationship_importance", 0.0)
            existing.evaluated_at = datetime.utcnow()
            self.db.flush()
            return existing

        priority = Priority(
            title=title,
            description=description,
            level=level,
            priority_score=priority_score,
            ceo_involvement_score=factors.get("ceo_involvement", 0.0),
            revenue_impact_score=factors.get("revenue_impact", 0.0),
            deadline_urgency_score=factors.get("deadline_urgency", 0.0),
            escalation_level_score=factors.get("escalation_level", 0.0),
            relationship_importance_score=factors.get("relationship_importance", 0.0),
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            due_date=due_date,
            company_name=company_name,
            evaluated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        self.db.add(priority)
        self.db.flush()
        return priority

    def run_full_prioritization(self) -> dict:
        """Scheduled job: Re-prioritize all items."""
        commitment_priorities = self.prioritize_commitments()
        risk_priorities = self.prioritize_risks()

        high = sum(1 for p in commitment_priorities + risk_priorities if p.level == PriorityLevel.HIGH)
        medium = sum(1 for p in commitment_priorities + risk_priorities if p.level == PriorityLevel.MEDIUM)
        low = sum(1 for p in commitment_priorities + risk_priorities if p.level == PriorityLevel.LOW)

        return {
            "commitment_priorities": len(commitment_priorities),
            "risk_priorities": len(risk_priorities),
            "high": high,
            "medium": medium,
            "low": low,
        }

    def get_top_priorities(self, limit: int = 10) -> list[dict]:
        """Return top-ranked priorities for executive briefing."""
        priorities = (
            self.db.query(Priority)
            .filter(Priority.level.in_([PriorityLevel.HIGH, PriorityLevel.MEDIUM]))
            .order_by(Priority.priority_score.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": str(p.id),
                "title": p.title,
                "level": p.level.value,
                "score": round(p.priority_score, 2),
                "entity_type": p.entity_type,
                "company": p.company_name,
                "due_date": p.due_date.isoformat() if p.due_date else None,
                "factors": {
                    "ceo_involvement": round(p.ceo_involvement_score, 2),
                    "revenue_impact": round(p.revenue_impact_score, 2),
                    "deadline_urgency": round(p.deadline_urgency_score, 2),
                    "escalation_level": round(p.escalation_level_score, 2),
                    "relationship_importance": round(p.relationship_importance_score, 2),
                },
            }
            for p in priorities
        ]
