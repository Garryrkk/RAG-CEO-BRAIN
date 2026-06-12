
import re
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.risk import Risk, RiskCategory, RiskSeverity
from ..models.commitment import Commitment, CommitmentStatus
from ..models.escalation import Escalation, EscalationSignal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Risk rule definitions
# Each rule: (name, pattern_list, category, base_severity)
# ─────────────────────────────────────────────────────────────

RISK_RULES = {
    RiskCategory.VENDOR: [
        (r"vendor\s+(delay|delayed|delaying|not\s+responding|unresponsive|miss)", "Vendor delivery delay"),
        (r"(supplier|vendor)\s+(issue|problem|failure|defaulted|terminated)", "Vendor relationship issue"),
        (r"(no\s+delivery|missed\s+delivery|delivery\s+postponed)", "Missed vendor delivery"),
        (r"(sla|service\s+level)\s+(breach|breached|violation|missed)", "SLA breach"),
    ],
    RiskCategory.APPROVAL: [
        (r"(approval|sign.off|authorization)\s+(pending|overdue|stalled|not\s+received|delayed)", "Approval pending"),
        (r"(waiting|wait)\s+for\s+(approval|sign.off|green\s+light)\s+(for\s+\d+|since|overdue)", "Approval overdue"),
        (r"(board|committee|management)\s+(approval|decision)\s+(pending|awaited|not\s+given)", "Board approval pending"),
        (r"(budget|spend)\s+(approval|authorization)\s+(not|pending|required|delayed)", "Budget approval blocked"),
    ],
    RiskCategory.REGULATORY: [
        (r"(regulator|regulatory|compliance|authority)\s+(concern|issue|query|question|review|audit)", "Regulatory attention"),
        (r"(regulator|fca|sec|rbi|sebi)\s+(has\s+not|hasn.t)\s+(responded|replied|confirmed)", "Regulator non-response"),
        (r"(compliance\s+(deadline|requirement)|regulatory\s+deadline)\s+(approaching|missed|overdue)", "Compliance deadline risk"),
        (r"(license|permit|certification)\s+(expired|expiring|renewal\s+pending|not\s+renewed)", "License/permit risk"),
        (r"(regulatory\s+approval|noc|clearance)\s+(pending|awaited|not\s+received)", "Regulatory clearance pending"),
    ],
    RiskCategory.CONTRACT: [
        (r"(contract|agreement)\s+(stalled|stuck|negotiation|dispute|breach|terminated|lapsed)", "Contract issue"),
        (r"(contract\s+signing|execution)\s+(delayed|pending|not\s+completed|overdue)", "Contract execution delay"),
        (r"(terms|clause|condition)\s+(not\s+agreed|disputed|rejected|unacceptable)", "Contract terms dispute"),
        (r"(renewal|extension)\s+(not\s+signed|overdue|expired|pending|at\s+risk)", "Contract renewal risk"),
        (r"(penalty|liquidated\s+damages|breach\s+of\s+contract)", "Contractual penalty risk"),
    ],
    RiskCategory.DEPLOYMENT: [
        (r"(deployment|release|launch|go.live)\s+(delayed|postponed|at\s+risk|blocked|slipping)", "Deployment delay"),
        (r"(milestone|deliverable)\s+(missed|slipping|delayed|overdue|not\s+met)", "Milestone slippage"),
        (r"(timeline|schedule|roadmap)\s+(slipping|at\s+risk|revised|pushed\s+back)", "Timeline risk"),
        (r"(testing|qa|uat)\s+(failed|blocked|incomplete|pending|not\s+started)", "Testing/QA risk"),
        (r"(production|live\s+environment)\s+(issue|incident|outage|failure)", "Production risk"),
    ],
}

# Severity multipliers based on context words
SEVERITY_BOOSTERS = {
    RiskSeverity.CRITICAL: [
        r"\b(critical|emergency|immediate|immediately|crisis)\b",
        r"\b(breach|default|terminated|collapsed|failed)\b",
    ],
    RiskSeverity.HIGH: [
        r"\b(urgent|serious|major|significant|important)\b",
        r"\b(overdue|missed|blocked|stalled)\b",
    ],
    RiskSeverity.MEDIUM: [
        r"\b(pending|awaiting|delayed|slipping|concern)\b",
    ],
    RiskSeverity.LOW: [
        r"\b(minor|small|watch|monitor|noted)\b",
    ],
}


class RiskDetectionEngine:
    """
    Task 5: Detects organizational risks using rule-based scanning.
    Every risk has: category, severity, evidence, and history.
    No risk is reported without traceable evidence.
    """

    def __init__(self, db: Session):
        self.db = db

    # ─────────────────────────────────────────────────────────
    # Text-based risk scanning
    # ─────────────────────────────────────────────────────────

    def scan_text_for_risks(
        self,
        text: str,
        source_document_id: str,
        company_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> list[Risk]:
        """
        Scan a document/email/meeting transcript for risk signals.
        Returns list of created/updated Risk objects.
        """
        text_lower = text.lower()
        detected = []

        for category, rules in RISK_RULES.items():
            for pattern, risk_name in rules:
                matches = list(re.finditer(pattern, text_lower))
                if not matches:
                    continue

                for match in matches:
                    # Extract surrounding context (±100 chars)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    excerpt = text[start:end].strip()

                    severity = self._compute_severity(excerpt)
                    confidence = self._compute_confidence(excerpt, category)

                    if confidence < 0.3:
                        continue

                    risk = self._upsert_risk(
                        title=f"{risk_name}" + (f" — {company_name}" if company_name else ""),
                        description=f"Risk detected in document {source_document_id}: {excerpt[:200]}",
                        category=category,
                        severity=severity,
                        evidence={
                            "document_id": source_document_id,
                            "excerpt": excerpt,
                            "matched_pattern": pattern,
                            "match_text": match.group(0),
                            "detected_at": datetime.utcnow().isoformat(),
                        },
                        company_name=company_name,
                        project_name=project_name,
                        confidence=confidence,
                    )
                    detected.append(risk)

        self.db.commit()
        return detected

    def _compute_severity(self, context: str) -> RiskSeverity:
        """Determine risk severity from surrounding context."""
        context_lower = context.lower()
        for severity, patterns in SEVERITY_BOOSTERS.items():
            for pattern in patterns:
                if re.search(pattern, context_lower):
                    return severity
        return RiskSeverity.MEDIUM

    def _compute_confidence(self, text: str, category: RiskCategory) -> float:
        """Confidence score for this risk detection."""
        score = 0.4  # Base score for pattern match
        text_lower = text.lower()

        rules = RISK_RULES.get(category, [])
        extra_matches = sum(
            1 for p, _ in rules if re.search(p, text_lower)
        )
        score += min(extra_matches * 0.1, 0.3)

        # Higher confidence if company is known
        if any(word in text_lower for word in ["vendor", "client", "regulator", "company"]):
            score += 0.1

        return min(score, 1.0)

    def _upsert_risk(
        self,
        title: str,
        description: str,
        category: RiskCategory,
        severity: RiskSeverity,
        evidence: dict,
        company_name: Optional[str] = None,
        project_name: Optional[str] = None,
        confidence: float = 0.5,
    ) -> Risk:
        """Create or update a risk, appending new evidence."""
        # Match by title + category + company (avoid duplicates)
        existing = self.db.query(Risk).filter(
            Risk.title == title,
            Risk.category == category,
            Risk.company_name == company_name,
            Risk.is_active == True,
        ).first()

        if existing:
            # Append evidence
            ev_list = existing.evidence or []
            ev_list.append(evidence)
            existing.evidence = ev_list
            existing.last_evaluated_at = datetime.utcnow()
            # Upgrade severity if needed
            severity_order = [RiskSeverity.LOW, RiskSeverity.MEDIUM, RiskSeverity.HIGH, RiskSeverity.CRITICAL]
            if severity_order.index(severity) > severity_order.index(existing.severity):
                history = existing.history or []
                history.append({
                    "date": datetime.utcnow().isoformat(),
                    "old_severity": existing.severity.value,
                    "new_severity": severity.value,
                    "reason": "Additional evidence detected",
                })
                existing.history = history
                existing.severity = severity
            existing.updated_at = datetime.utcnow()
            return existing

        risk = Risk(
            title=title,
            description=description,
            category=category,
            severity=severity,
            evidence=[evidence],
            evidence_summary=description,
            company_name=company_name,
            project_name=project_name,
            confidence=confidence,
            risk_score=self._severity_to_score(severity),
            first_detected_at=datetime.utcnow(),
            last_evaluated_at=datetime.utcnow(),
        )
        self.db.add(risk)
        self.db.flush()
        logger.warning(f"New risk detected [{category.value}/{severity.value}]: {title}")
        return risk

    def _severity_to_score(self, severity: RiskSeverity) -> float:
        return {
            RiskSeverity.LOW: 0.25,
            RiskSeverity.MEDIUM: 0.50,
            RiskSeverity.HIGH: 0.75,
            RiskSeverity.CRITICAL: 1.0,
        }.get(severity, 0.5)

    # ─────────────────────────────────────────────────────────
    # DERIVED RISK DETECTION: from commitments and escalations
    # ─────────────────────────────────────────────────────────

    def derive_risks_from_commitments(self) -> list[Risk]:
        """
        Derive risks from commitment state.
        Multiple overdue external-dependency commitments = regulatory/vendor risk.
        """
        from ..models.commitment import CommitmentType
        risks = []

        # Overdue external dependencies → regulatory/vendor risk
        overdue_external = (
            self.db.query(Commitment)
            .filter(
                Commitment.status == CommitmentStatus.OVERDUE,
                Commitment.commitment_type == CommitmentType.EXTERNAL_DEPENDENCY,
            )
            .all()
        )

        # Group by company
        by_company: dict[str, list] = {}
        for c in overdue_external:
            key = c.company_name or "Unknown"
            by_company.setdefault(key, []).append(c)

        for company_name, commitments in by_company.items():
            risk = self._upsert_risk(
                title=f"Unresolved external dependencies — {company_name}",
                description=f"{len(commitments)} external dependency commitment(s) overdue",
                category=RiskCategory.REGULATORY if "regulator" in company_name.lower() else RiskCategory.VENDOR,
                severity=RiskSeverity.HIGH if len(commitments) >= 3 else RiskSeverity.MEDIUM,
                evidence={
                    "type": "derived_from_commitments",
                    "commitment_ids": [str(c.id) for c in commitments],
                    "count": len(commitments),
                    "detected_at": datetime.utcnow().isoformat(),
                },
                company_name=company_name,
            )
            risks.append(risk)

        # Overdue approvals → approval risk
        overdue_approvals = (
            self.db.query(Commitment)
            .filter(
                Commitment.status == CommitmentStatus.OVERDUE,
                Commitment.commitment_type == CommitmentType.APPROVAL,
            )
            .all()
        )
        for c in overdue_approvals:
            risk = self._upsert_risk(
                title=f"Overdue approval: {(c.normalized_text or c.raw_text)[:100]}",
                description=f"Approval commitment has passed due date without resolution",
                category=RiskCategory.APPROVAL,
                severity=RiskSeverity.HIGH,
                evidence={
                    "type": "derived_from_commitment",
                    "commitment_id": str(c.id),
                    "due_date": c.due_date.isoformat() if c.due_date else None,
                    "detected_at": datetime.utcnow().isoformat(),
                },
                company_name=c.company_name,
            )
            risks.append(risk)

        self.db.commit()
        return risks

    def derive_risks_from_escalations(self) -> list[Risk]:
        """Promote active escalations to risks when severity warrants it."""
        critical_escalations = (
            self.db.query(Escalation)
            .filter(
                Escalation.is_active == True,
                Escalation.severity.in_(["high", "critical"]),
            )
            .all()
        )
        risks = []
        for esc in critical_escalations:
            risk = self._upsert_risk(
                title=f"Escalation risk: {esc.title[:100]}",
                description=esc.description,
                category=RiskCategory.VENDOR,  # Default; could be refined with LLM
                severity=RiskSeverity.HIGH if esc.severity == "high" else RiskSeverity.CRITICAL,
                evidence={
                    "type": "derived_from_escalation",
                    "escalation_id": str(esc.id),
                    "signal_type": esc.signal_type.value,
                    "detected_at": datetime.utcnow().isoformat(),
                },
                company_name=esc.company_name,
            )
            esc.linked_risk_ids = list(set((esc.linked_risk_ids or []) + [str(risk.id)]))
            risks.append(risk)

        self.db.commit()
        return risks

    def run_full_risk_evaluation(self) -> dict:
        """
        Scheduled job: Full risk evaluation pass.
        Returns summary.
        """
        commitment_risks = self.derive_risks_from_commitments()
        escalation_risks = self.derive_risks_from_escalations()

        total_active = self.db.query(Risk).filter(Risk.is_active == True).count()
        critical = self.db.query(Risk).filter(
            Risk.is_active == True, Risk.severity == RiskSeverity.CRITICAL
        ).count()
        high = self.db.query(Risk).filter(
            Risk.is_active == True, Risk.severity == RiskSeverity.HIGH
        ).count()

        return {
            "new_from_commitments": len(commitment_risks),
            "new_from_escalations": len(escalation_risks),
            "total_active": total_active,
            "critical": critical,
            "high": high,
        }

    def get_active_risks_summary(self) -> list[dict]:
        """Return active risks ordered by severity for executive reporting."""
        severity_order = {
            RiskSeverity.CRITICAL: 0,
            RiskSeverity.HIGH: 1,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.LOW: 3,
        }
        risks = (
            self.db.query(Risk)
            .filter(Risk.is_active == True)
            .all()
        )
        risks.sort(key=lambda r: (severity_order.get(r.severity, 4), -r.risk_score))

        return [
            {
                "id": str(r.id),
                "title": r.title,
                "category": r.category.value,
                "severity": r.severity.value,
                "company": r.company_name,
                "project": r.project_name,
                "score": r.risk_score,
                "evidence_count": len(r.evidence or []),
                "first_detected": r.first_detected_at.isoformat() if r.first_detected_at else None,
                "description": r.description,
            }
            for r in risks
        ]
