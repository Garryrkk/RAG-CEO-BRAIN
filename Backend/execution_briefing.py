
import json
import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional
import httpx
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.briefing import ExecutiveBriefing
from ..models.commitment import Commitment, CommitmentStatus
from ..models.risk import Risk, RiskSeverity
from ..models.escalation import Escalation
from ..models.relationship import Company, RelationshipHealth

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://ollama:11434"
LLM_MODEL = "qwen3"


class ExecutiveBriefingEngine:
    """
    Task 6: Generates the daily executive awareness report.

    Assembles intelligence from all Phase 5 engines into a structured,
    actionable briefing. Every section is evidence-backed.
    """

    def __init__(self, db: Session):
        self.db = db

    # ─────────────────────────────────────────────────────────
    # BRIEF SECTION 1: TOP PRIORITIES
    # ─────────────────────────────────────────────────────────

    def _compile_top_priorities(self) -> list[dict]:
        """Pull highest-priority items from the prioritization engine."""
        from .prioritization_engine import PrioritizationEngine
        engine = PrioritizationEngine(self.db)
        return engine.get_top_priorities(limit=5)

    # ─────────────────────────────────────────────────────────
    # BRIEF SECTION 2: COMMITMENTS REQUIRING ATTENTION
    # ─────────────────────────────────────────────────────────

    def _compile_commitment_attention(self) -> tuple[list, list, list]:
        """
        Returns (overdue, stalled, open_high_priority) commitment lists.
        """
        # Overdue
        overdue = (
            self.db.query(Commitment)
            .filter(Commitment.status == CommitmentStatus.OVERDUE)
            .order_by(Commitment.due_date)
            .limit(10)
            .all()
        )

        # Stalled (In Progress > 7 days without resolution)
        stall_threshold = datetime.utcnow() - timedelta(days=7)
        stalled = (
            self.db.query(Commitment)
            .filter(
                Commitment.status == CommitmentStatus.IN_PROGRESS,
                Commitment.updated_at < stall_threshold,
            )
            .limit(5)
            .all()
        )

        # Open high-priority
        open_high = (
            self.db.query(Commitment)
            .filter(
                Commitment.status == CommitmentStatus.OPEN,
                Commitment.priority_score >= 0.65,
            )
            .order_by(Commitment.priority_score.desc())
            .limit(5)
            .all()
        )

        def format_commitment(c: Commitment) -> dict:
            return {
                "id": str(c.id),
                "text": c.normalized_text or c.raw_text,
                "type": c.commitment_type.value,
                "status": c.status.value,
                "owner": c.owner,
                "company": c.company_name,
                "due_date": c.due_date.isoformat() if c.due_date else None,
                "priority_score": round(c.priority_score or 0, 2),
                "source_excerpt": c.source_excerpt,
            }

        return (
            [format_commitment(c) for c in overdue],
            [format_commitment(c) for c in stalled],
            [format_commitment(c) for c in open_high],
        )

    # ─────────────────────────────────────────────────────────
    # BRIEF SECTION 3: RELATIONSHIP UPDATES
    # ─────────────────────────────────────────────────────────

    def _compile_relationship_updates(self) -> list[dict]:
        """
        Returns companies with notable relationship changes or status.
        Focus on: AT_RISK, ATTENTION_REQUIRED, or recent health changes.
        """
        at_risk = (
            self.db.query(Company)
            .filter(
                Company.health.in_([RelationshipHealth.AT_RISK, RelationshipHealth.ATTENTION_REQUIRED])
            )
            .order_by(Company.health_score)
            .limit(8)
            .all()
        )

        result = []
        for company in at_risk:
            open_commitments = self.db.query(Commitment).filter(
                Commitment.company_name == company.name,
                Commitment.status.in_([CommitmentStatus.OPEN, CommitmentStatus.OVERDUE]),
            ).count()

            active_risks = self.db.query(Risk).filter(
                Risk.company_name == company.name,
                Risk.is_active == True,
            ).count()

            result.append({
                "company": company.name,
                "health": company.health.value,
                "health_score": round(company.health_score, 2),
                "days_since_contact": company.days_since_last_contact,
                "relationship_type": company.relationship_type,
                "open_commitments": open_commitments,
                "active_risks": active_risks,
                "revenue_impact": company.revenue_impact,
                "summary": self._summarize_relationship_status(company, open_commitments, active_risks),
            })

        return result

    def _summarize_relationship_status(
        self, company: Company, open_commitments: int, active_risks: int
    ) -> str:
        parts = []
        if company.days_since_last_contact > 14:
            parts.append(f"No contact for {company.days_since_last_contact} days")
        if open_commitments > 0:
            parts.append(f"{open_commitments} open commitment(s)")
        if active_risks > 0:
            parts.append(f"{active_risks} active risk(s)")
        if company.health == RelationshipHealth.AT_RISK:
            parts.append("Relationship at risk")
        return "; ".join(parts) if parts else "Requires monitoring"

    # ─────────────────────────────────────────────────────────
    # BRIEF SECTION 4: EMERGING RISKS
    # ─────────────────────────────────────────────────────────

    def _compile_emerging_risks(self) -> list[dict]:
        """New or escalating risks from the last 48 hours."""
        since = datetime.utcnow() - timedelta(hours=48)
        new_risks = (
            self.db.query(Risk)
            .filter(
                Risk.is_active == True,
                Risk.first_detected_at >= since,
            )
            .order_by(Risk.risk_score.desc())
            .limit(5)
            .all()
        )

        # Also include any critical risks regardless of age
        critical_risks = (
            self.db.query(Risk)
            .filter(
                Risk.is_active == True,
                Risk.severity == RiskSeverity.CRITICAL,
            )
            .order_by(Risk.last_evaluated_at.desc())
            .limit(3)
            .all()
        )

        all_risks = {str(r.id): r for r in new_risks + critical_risks}

        return [
            {
                "id": str(r.id),
                "title": r.title,
                "category": r.category.value,
                "severity": r.severity.value,
                "company": r.company_name,
                "description": r.description,
                "evidence_count": len(r.evidence or []),
                "first_detected": r.first_detected_at.isoformat() if r.first_detected_at else None,
                "is_new": r.first_detected_at >= since if r.first_detected_at else False,
            }
            for r in all_risks.values()
        ]

    # ─────────────────────────────────────────────────────────
    # BRIEF SECTION 5: MAJOR ACTIVITY SUMMARY
    # ─────────────────────────────────────────────────────────

    def _compile_activity_summary(self) -> list[dict]:
        """Key developments from the last 24 hours."""
        since = datetime.utcnow() - timedelta(hours=24)
        from ..models.relationship import Interaction

        recent = (
            self.db.query(Interaction)
            .filter(
                Interaction.occurred_at >= since,
                Interaction.summary.isnot(None),
            )
            .order_by(Interaction.occurred_at.desc())
            .limit(10)
            .all()
        )

        return [
            {
                "type": i.interaction_type,
                "subject": i.subject,
                "summary": i.summary,
                "company": None,  # Would join company via company_id
                "date": i.occurred_at.isoformat() if i.occurred_at else None,
                "had_commitment": i.had_commitment,
                "had_escalation": i.had_escalation,
            }
            for i in recent
        ]

    # ─────────────────────────────────────────────────────────
    # SUGGESTED FOCUS AREAS
    # ─────────────────────────────────────────────────────────

    def _compute_suggested_focus_areas(
        self,
        overdue_commitments: list,
        relationship_updates: list,
        emerging_risks: list,
        active_escalations: list,
    ) -> list[dict]:
        """
        Rule-based focus area suggestions.
        The system tells the CEO what to pay attention to.
        """
        focus_areas = []

        # Overdue commitments
        if overdue_commitments:
            focus_areas.append({
                "area": "Overdue Commitments",
                "reason": f"{len(overdue_commitments)} commitment(s) past due date require resolution",
                "urgency": "high",
                "action": "Review and resolve overdue commitments — many may be blocking downstream work",
            })

        # At-risk relationships
        at_risk = [r for r in relationship_updates if r["health"] == "at_risk"]
        if at_risk:
            names = ", ".join(r["company"] for r in at_risk[:3])
            focus_areas.append({
                "area": "Relationship Recovery",
                "reason": f"Relationship(s) at risk: {names}",
                "urgency": "high",
                "action": f"Re-engage with {names} before relationship deteriorates further",
            })

        # Critical risks
        critical = [r for r in emerging_risks if r["severity"] == "critical"]
        if critical:
            focus_areas.append({
                "area": "Critical Risk Mitigation",
                "reason": f"{len(critical)} critical risk(s) require immediate attention",
                "urgency": "critical",
                "action": "Address critical risks immediately — escalate or assign owners",
            })

        # Active escalations
        if active_escalations:
            focus_areas.append({
                "area": "Escalation Resolution",
                "reason": f"{len(active_escalations)} active escalation(s) require executive decision",
                "urgency": "high",
                "action": "Review escalations and make decisions to unblock progress",
            })

        return focus_areas[:5]  # Top 5 focus areas

    # ─────────────────────────────────────────────────────────
    # LLM NARRATIVE GENERATION
    # ─────────────────────────────────────────────────────────

    async def _generate_narrative(self, briefing_data: dict) -> str:
        """
        Use Qwen3 to generate a concise executive narrative from structured data.
        """
        stats = briefing_data.get("stats", {})
        priorities_count = len(briefing_data.get("top_priorities", []))
        overdue_count = len(briefing_data.get("overdue_commitments", []))
        risk_count = len(briefing_data.get("emerging_risks", []))
        escalation_count = len(briefing_data.get("active_escalations", []))
        at_risk_count = len([r for r in briefing_data.get("relationship_updates", []) if r.get("health") == "at_risk"])

        prompt = f"""You are an executive assistant generating a morning briefing for a CEO.

Based on these operational facts, write a concise 3-paragraph executive summary:

FACTS:
- {overdue_count} overdue commitments requiring attention
- {risk_count} emerging risks (including any critical ones)
- {escalation_count} active escalations
- {at_risk_count} relationships at risk
- {priorities_count} high-priority items demanding attention
- Total open commitments: {stats.get('open_commitments', 0)}

TOP RISKS: {json.dumps([r['title'] for r in briefing_data.get('emerging_risks', [])[:3]])}
TOP PRIORITIES: {json.dumps([p['title'][:80] for p in briefing_data.get('top_priorities', [])[:3]])}
AT-RISK RELATIONSHIPS: {json.dumps([r['company'] for r in briefing_data.get('relationship_updates', []) if r.get('health') == 'at_risk'][:3])}

Write in direct, executive tone. Be specific. Flag what is most urgent.
Do not use bullet points. Maximum 150 words. Focus on what needs action today."""

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Narrative generation failed: {e}")
            # Deterministic fallback
            parts = []
            if overdue_count:
                parts.append(f"{overdue_count} commitments are overdue")
            if risk_count:
                parts.append(f"{risk_count} emerging risks require attention")
            if escalation_count:
                parts.append(f"{escalation_count} active escalations")
            if at_risk_count:
                parts.append(f"{at_risk_count} relationships at risk")
            return (
                f"Today's briefing highlights {priorities_count} high-priority items. "
                + (", ".join(parts) + "." if parts else "All indicators are within normal range.")
            )

    # ─────────────────────────────────────────────────────────
    # MAIN BRIEFING GENERATION
    # ─────────────────────────────────────────────────────────

    async def generate_daily_briefing(self, briefing_date: Optional[date] = None) -> ExecutiveBriefing:
        """
        Generate the complete daily executive briefing.
        This is the flagship feature of Phase 5.
        """
        t_start = time.time()
        briefing_date = briefing_date or date.today()

        # Check if already generated today
        existing = (
            self.db.query(ExecutiveBriefing)
            .filter(ExecutiveBriefing.briefing_date == briefing_date)
            .first()
        )
        if existing:
            logger.info(f"Briefing for {briefing_date} already exists, regenerating")
            self.db.delete(existing)
            self.db.flush()

        # ── Compile all sections ───────────────────────────────────
        top_priorities = self._compile_top_priorities()
        overdue, stalled, open_high = self._compile_commitment_attention()
        relationship_updates = self._compile_relationship_updates()
        emerging_risks = self._compile_emerging_risks()
        activity_summary = self._compile_activity_summary()

        # Get active escalations
        from .escalation_engine import EscalationDetectionEngine
        esc_engine = EscalationDetectionEngine(self.db)
        active_escalations = esc_engine.get_active_escalations_summary()

        # Suggested focus areas
        suggested_focus = self._compute_suggested_focus_areas(
            overdue, relationship_updates, emerging_risks, active_escalations
        )

        # ── Stats snapshot ─────────────────────────────────────────
        stats = {
            "open_commitments": self.db.query(Commitment).filter(
                Commitment.status == CommitmentStatus.OPEN
            ).count(),
            "overdue_commitments": len(overdue),
            "active_risks": self.db.query(Risk).filter(Risk.is_active == True).count(),
            "active_escalations": self.db.query(Escalation).filter(Escalation.is_active == True).count(),
            "relationships_at_risk": self.db.query(Company).filter(
                Company.health == RelationshipHealth.AT_RISK
            ).count(),
        }

        # ── Generate narrative ─────────────────────────────────────
        briefing_data = {
            "top_priorities": top_priorities,
            "overdue_commitments": overdue,
            "relationship_updates": relationship_updates,
            "emerging_risks": emerging_risks,
            "active_escalations": active_escalations,
            "stats": stats,
        }
        narrative = await self._generate_narrative(briefing_data)

        # ── Persist briefing ───────────────────────────────────────
        duration_ms = int((time.time() - t_start) * 1000)
        briefing = ExecutiveBriefing(
            briefing_date=briefing_date,
            top_priorities=top_priorities,
            open_commitments=open_high,
            overdue_commitments=overdue,
            stalled_commitments=stalled,
            relationship_updates=relationship_updates,
            emerging_risks=emerging_risks,
            active_escalations=active_escalations,
            major_activity_summary=activity_summary,
            suggested_focus_areas=suggested_focus,
            total_open_commitments=stats["open_commitments"],
            total_overdue_commitments=stats["overdue_commitments"],
            total_active_risks=stats["active_risks"],
            total_active_escalations=stats["active_escalations"],
            relationships_at_risk=stats["relationships_at_risk"],
            narrative_summary=narrative,
            generated_at=datetime.utcnow(),
            generation_duration_ms=duration_ms,
            generation_model=LLM_MODEL,
        )
        self.db.add(briefing)
        self.db.commit()

        logger.info(f"Executive briefing generated for {briefing_date} in {duration_ms}ms")
        return briefing

    def get_latest_briefing(self) -> Optional[dict]:
        """Return most recent briefing as dict."""
        briefing = (
            self.db.query(ExecutiveBriefing)
            .order_by(ExecutiveBriefing.briefing_date.desc())
            .first()
        )
        if not briefing:
            return None

        return {
            "date": briefing.briefing_date.isoformat(),
            "narrative": briefing.narrative_summary,
            "top_priorities": briefing.top_priorities,
            "open_commitments": briefing.open_commitments,
            "overdue_commitments": briefing.overdue_commitments,
            "stalled_commitments": briefing.stalled_commitments,
            "relationship_updates": briefing.relationship_updates,
            "emerging_risks": briefing.emerging_risks,
            "active_escalations": briefing.active_escalations,
            "major_activity_summary": briefing.major_activity_summary,
            "suggested_focus_areas": briefing.suggested_focus_areas,
            "stats": {
                "open_commitments": briefing.total_open_commitments,
                "overdue_commitments": briefing.total_overdue_commitments,
                "active_risks": briefing.total_active_risks,
                "active_escalations": briefing.total_active_escalations,
                "relationships_at_risk": briefing.relationships_at_risk,
            },
            "generated_at": briefing.generated_at.isoformat() if briefing.generated_at else None,
            "generation_ms": briefing.generation_duration_ms,
        }
