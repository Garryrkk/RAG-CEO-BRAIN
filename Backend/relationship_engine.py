import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..models.relationship import (
    Company, Person, Interaction, Relationship, RelationshipHealth
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# HEALTH SCORING RULES
# Simple rule-based scoring — avoids ML complexity in Month 1.
# ─────────────────────────────────────────────────────────────

HEALTH_THRESHOLDS = {
    RelationshipHealth.HEALTHY: 0.70,
    RelationshipHealth.NEUTRAL: 0.45,
    RelationshipHealth.ATTENTION_REQUIRED: 0.25,
    RelationshipHealth.AT_RISK: 0.0,
}

# Days of silence → health penalty
SILENCE_PENALTIES = [
    (7, -0.05),
    (14, -0.10),
    (30, -0.20),
    (60, -0.35),
    (90, -0.50),
]

# Overdue commitment penalty per commitment
OVERDUE_PENALTY_PER_COMMITMENT = 0.08

# Open risk penalty per risk
RISK_PENALTY = 0.10

# Active escalation penalty
ESCALATION_PENALTY = 0.15


class RelationshipMemoryEngine:
    """
    Task 3: Builds and maintains relationship memory for companies and persons.

    Every email, meeting, document interaction gets recorded.
    Health scores are computed from multiple signals.
    Executives can ask: "What's happening with Schneider?" and get a complete picture.
    """

    def __init__(self, db: Session):
        self.db = db

    # ─────────────────────────────────────────────────────────
    # COMPANY MEMORY
    # ─────────────────────────────────────────────────────────

    def get_or_create_company(self, name: str, domain: Optional[str] = None, **kwargs) -> Company:
        """Upsert a company record."""
        company = self.db.query(Company).filter(
            func.lower(Company.name) == name.lower()
        ).first()

        if not company:
            company = Company(
                name=name,
                domain=domain,
                first_interaction_at=datetime.utcnow(),
                **kwargs,
            )
            self.db.add(company)
            self.db.flush()
            # Create relationship record
            rel = Relationship(company_id=company.id)
            self.db.add(rel)
            self.db.flush()
            logger.info(f"Created company memory: {name}")

        return company

    def get_company_memory(self, company_name: str) -> Optional[dict]:
        """
        Return complete company memory — the answer to "What's happening with X?"
        Includes: interaction history, projects, risks, commitments, recent activity.
        """
        company = self.db.query(Company).filter(
            func.lower(Company.name) == company_name.lower()
        ).first()

        if not company:
            return None

        # Recent interactions (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_interactions = (
            self.db.query(Interaction)
            .filter(
                Interaction.company_id == company.id,
                Interaction.occurred_at >= thirty_days_ago,
            )
            .order_by(Interaction.occurred_at.desc())
            .limit(20)
            .all()
        )

        # Persons at this company
        persons = self.db.query(Person).filter(Person.company_id == company.id).all()

        # Commitments (import here to avoid circular imports)
        from ..models.commitment import Commitment, CommitmentStatus
        open_commitments = (
            self.db.query(Commitment)
            .filter(
                Commitment.company_name == company_name,
                Commitment.status.in_([CommitmentStatus.OPEN, CommitmentStatus.IN_PROGRESS, CommitmentStatus.OVERDUE]),
            )
            .order_by(Commitment.due_date)
            .all()
        )

        # Risks
        from ..models.risk import Risk
        active_risks = (
            self.db.query(Risk)
            .filter(Risk.company_name == company_name, Risk.is_active == True)
            .order_by(Risk.risk_score.desc())
            .all()
        )

        # Escalations
        from ..models.escalation import Escalation
        active_escalations = (
            self.db.query(Escalation)
            .filter(Escalation.company_name == company_name, Escalation.is_active == True)
            .all()
        )

        return {
            "company": {
                "id": str(company.id),
                "name": company.name,
                "domain": company.domain,
                "industry": company.industry,
                "health": company.health.value,
                "health_score": company.health_score,
                "relationship_type": company.relationship_type,
                "last_interaction": company.last_interaction_at.isoformat() if company.last_interaction_at else None,
                "days_since_contact": company.days_since_last_contact,
                "total_interactions": company.total_interactions,
                "communication_30d": company.communication_frequency_30d,
                "communication_prev_30d": company.communication_frequency_prev_30d,
                "revenue_impact": company.revenue_impact,
                "strategic_importance": company.strategic_importance,
            },
            "persons": [
                {
                    "name": p.name,
                    "email": p.email,
                    "title": p.title,
                    "last_contact": p.last_contact_at.isoformat() if p.last_contact_at else None,
                    "open_commitments": p.open_commitments_count,
                }
                for p in persons
            ],
            "recent_interactions": [
                {
                    "type": i.interaction_type,
                    "subject": i.subject,
                    "summary": i.summary,
                    "sentiment": i.sentiment,
                    "date": i.occurred_at.isoformat() if i.occurred_at else None,
                }
                for i in recent_interactions
            ],
            "open_commitments": [
                {
                    "id": str(c.id),
                    "text": c.normalized_text or c.raw_text,
                    "type": c.commitment_type.value,
                    "status": c.status.value,
                    "owner": c.owner,
                    "due_date": c.due_date.isoformat() if c.due_date else None,
                }
                for c in open_commitments
            ],
            "active_risks": [
                {
                    "id": str(r.id),
                    "title": r.title,
                    "category": r.category.value,
                    "severity": r.severity.value,
                }
                for r in active_risks
            ],
            "active_escalations": [
                {
                    "id": str(e.id),
                    "title": e.title,
                    "signal_type": e.signal_type.value,
                    "severity": e.severity,
                }
                for e in active_escalations
            ],
        }

    # ─────────────────────────────────────────────────────────
    # PERSON MEMORY
    # ─────────────────────────────────────────────────────────

    def get_or_create_person(
        self,
        email: str,
        name: Optional[str] = None,
        company_name: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Person:
        """Upsert a person record."""
        person = self.db.query(Person).filter(Person.email == email).first()

        if not person:
            company = None
            if company_name:
                company = self.get_or_create_company(company_name)
            person = Person(
                email=email,
                name=name or email.split("@")[0],
                title=title,
                company_id=company.id if company else None,
                first_contact_at=datetime.utcnow(),
            )
            self.db.add(person)
            self.db.flush()
            logger.info(f"Created person memory: {name} <{email}>")
        else:
            if name and not person.name:
                person.name = name
            if title and not person.title:
                person.title = title

        return person

    def get_person_memory(self, email: str) -> Optional[dict]:
        """Full person memory — communications, commitments, follow-ups."""
        person = self.db.query(Person).filter(Person.email == email).first()
        if not person:
            return None

        recent_interactions = (
            self.db.query(Interaction)
            .filter(Interaction.person_id == person.id)
            .order_by(Interaction.occurred_at.desc())
            .limit(20)
            .all()
        )

        from ..models.commitment import Commitment, CommitmentStatus
        commitments = (
            self.db.query(Commitment)
            .filter(
                (Commitment.owner_email == email) | (Commitment.counterparty_email == email),
                Commitment.status != CommitmentStatus.RESOLVED,
            )
            .all()
        )

        return {
            "person": {
                "name": person.name,
                "email": person.email,
                "title": person.title,
                "health": person.health.value,
                "relationship_strength": person.relationship_strength,
                "last_contact": person.last_contact_at.isoformat() if person.last_contact_at else None,
                "total_emails": person.total_emails,
                "total_meetings": person.total_meetings,
                "open_commitments": person.open_commitments_count,
                "overdue_commitments": person.overdue_commitments_count,
            },
            "recent_interactions": [
                {
                    "type": i.interaction_type,
                    "subject": i.subject,
                    "summary": i.summary,
                    "date": i.occurred_at.isoformat() if i.occurred_at else None,
                }
                for i in recent_interactions
            ],
            "commitments": [
                {
                    "id": str(c.id),
                    "text": c.normalized_text or c.raw_text,
                    "type": c.commitment_type.value,
                    "status": c.status.value,
                    "due_date": c.due_date.isoformat() if c.due_date else None,
                }
                for c in commitments
            ],
        }

    # ─────────────────────────────────────────────────────────
    # INTERACTION RECORDING
    # ─────────────────────────────────────────────────────────

    def record_interaction(
        self,
        interaction_type: str,
        company_name: Optional[str] = None,
        person_email: Optional[str] = None,
        subject: Optional[str] = None,
        summary: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        source_document_id: Optional[str] = None,
        sentiment: Optional[str] = None,
        sentiment_score: float = 0.0,
        had_commitment: bool = False,
        had_escalation: bool = False,
        had_risk_signal: bool = False,
    ) -> Interaction:
        """Record a single interaction and update company/person stats."""
        occurred_at = occurred_at or datetime.utcnow()

        company = None
        if company_name:
            company = self.get_or_create_company(company_name)

        person = None
        if person_email:
            person = self.get_or_create_person(person_email, company_name=company_name)

        interaction = Interaction(
            company_id=company.id if company else None,
            person_id=person.id if person else None,
            interaction_type=interaction_type,
            subject=subject,
            summary=summary,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            occurred_at=occurred_at,
            source_document_id=source_document_id,
            had_commitment=had_commitment,
            had_escalation=had_escalation,
            had_risk_signal=had_risk_signal,
        )
        self.db.add(interaction)

        # Update company stats
        if company:
            company.total_interactions += 1
            company.last_interaction_at = max(
                occurred_at,
                company.last_interaction_at or datetime.min.replace(tzinfo=None),
            )

        # Update person stats
        if person:
            if interaction_type == "email":
                person.total_emails += 1
            elif interaction_type == "meeting":
                person.total_meetings += 1
            person.last_contact_at = max(
                occurred_at,
                person.last_contact_at or datetime.min.replace(tzinfo=None),
            )

        self.db.flush()
        return interaction

    # ─────────────────────────────────────────────────────────
    # RELATIONSHIP HEALTH SCORING
    # Simple rule-based — every score is explainable.
    # ─────────────────────────────────────────────────────────

    def compute_health_score(self, company: Company) -> tuple[float, list[str]]:
        """
        Compute relationship health score (0.0-1.0) with explanation.
        Returns (score, [explanation_strings]).
        """
        score = 0.7  # Start healthy
        reasons = []
        now = datetime.utcnow()

        # ── Silence penalty ───────────────────────────────────
        if company.last_interaction_at:
            days_silent = (now - company.last_interaction_at.replace(tzinfo=None)).days
            company.days_since_last_contact = days_silent
            for threshold, penalty in SILENCE_PENALTIES:
                if days_silent >= threshold:
                    score += penalty
                    reasons.append(f"No contact for {days_silent} days (penalty: {penalty:.0%})")
                    break

        # ── Overdue commitments ───────────────────────────────
        from ..models.commitment import Commitment, CommitmentStatus
        overdue_count = self.db.query(Commitment).filter(
            Commitment.company_name == company.name,
            Commitment.status == CommitmentStatus.OVERDUE,
        ).count()
        if overdue_count > 0:
            penalty = overdue_count * OVERDUE_PENALTY_PER_COMMITMENT
            score -= penalty
            reasons.append(f"{overdue_count} overdue commitment(s) (penalty: {penalty:.0%})")

        # ── Active risks ──────────────────────────────────────
        from ..models.risk import Risk
        risk_count = self.db.query(Risk).filter(
            Risk.company_name == company.name,
            Risk.is_active == True,
        ).count()
        if risk_count > 0:
            penalty = risk_count * RISK_PENALTY
            score -= penalty
            reasons.append(f"{risk_count} active risk(s) (penalty: {penalty:.0%})")

        # ── Active escalations ────────────────────────────────
        from ..models.escalation import Escalation
        esc_count = self.db.query(Escalation).filter(
            Escalation.company_name == company.name,
            Escalation.is_active == True,
        ).count()
        if esc_count > 0:
            penalty = esc_count * ESCALATION_PENALTY
            score -= penalty
            reasons.append(f"{esc_count} active escalation(s) (penalty: {penalty:.0%})")

        # ── Communication trend ───────────────────────────────
        if company.communication_frequency_30d == 0 and company.communication_frequency_prev_30d > 0:
            score -= 0.15
            reasons.append("Communication dropped to zero this month")
        elif company.communication_frequency_30d < company.communication_frequency_prev_30d * 0.5:
            score -= 0.10
            reasons.append("Communication frequency dropped >50%")

        # Clamp
        score = max(0.0, min(1.0, score))
        return score, reasons

    def determine_health_status(self, score: float) -> RelationshipHealth:
        if score >= HEALTH_THRESHOLDS[RelationshipHealth.HEALTHY]:
            return RelationshipHealth.HEALTHY
        elif score >= HEALTH_THRESHOLDS[RelationshipHealth.NEUTRAL]:
            return RelationshipHealth.NEUTRAL
        elif score >= HEALTH_THRESHOLDS[RelationshipHealth.ATTENTION_REQUIRED]:
            return RelationshipHealth.ATTENTION_REQUIRED
        else:
            return RelationshipHealth.AT_RISK

    def evaluate_all_relationships(self) -> list[dict]:
        """
        Scheduled job: Re-evaluate health for all companies.
        Returns summary of all relationship health changes.
        """
        companies = self.db.query(Company).all()
        results = []

        for company in companies:
            old_health = company.health
            score, reasons = self.compute_health_score(company)
            new_health = self.determine_health_status(score)

            company.health_score = score
            company.health = new_health

            # Update relationship record
            rel = self.db.query(Relationship).filter(
                Relationship.company_id == company.id
            ).first()
            if rel:
                rel.health = new_health
                rel.health_score = score
                rel.last_evaluated_at = datetime.utcnow()

                # Append to health history
                history = rel.health_history or []
                history.append({
                    "date": datetime.utcnow().isoformat(),
                    "score": score,
                    "health": new_health.value,
                    "reasons": reasons,
                })
                # Keep last 90 days
                rel.health_history = history[-90:]

            results.append({
                "company": company.name,
                "old_health": old_health.value,
                "new_health": new_health.value,
                "score": round(score, 2),
                "reasons": reasons,
                "changed": old_health != new_health,
            })

            if old_health != new_health:
                logger.warning(f"Relationship health change: {company.name} {old_health.value} → {new_health.value}")

        self.db.commit()
        return results

    def get_relationships_requiring_attention(self) -> list[dict]:
        """Return companies whose relationships need executive attention."""
        companies = self.db.query(Company).filter(
            Company.health.in_([RelationshipHealth.ATTENTION_REQUIRED, RelationshipHealth.AT_RISK])
        ).order_by(Company.health_score).all()

        return [
            {
                "company": c.name,
                "health": c.health.value,
                "score": c.health_score,
                "days_since_contact": c.days_since_last_contact,
                "relationship_type": c.relationship_type,
                "revenue_impact": c.revenue_impact,
            }
            for c in companies
        ]
