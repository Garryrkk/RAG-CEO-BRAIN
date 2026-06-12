
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.briefing import ExecutiveSignal, SignalType

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Signal configuration
# ─────────────────────────────────────────────────────────────

# Minimum % change to generate an activity spike/drop signal
ACTIVITY_SPIKE_THRESHOLD = 0.50    # 50% increase
COMMUNICATION_DROP_THRESHOLD = 0.40  # 40% decrease
RESPONSE_TIME_INCREASE_THRESHOLD = 0.50  # 50% slower


class ExecutiveSignalEngine:
    """
    Task 7: Transforms organizational data into executive intelligence signals.
    Signals are meaningful patterns, not raw metrics.
    """

    def __init__(self, db: Session):
        self.db = db

    def _create_signal(
        self,
        signal_type: SignalType,
        title: str,
        raw_observation: str,
        signal_insight: str,
        entity_name: str,
        entity_type: str,
        severity: str = "medium",
        magnitude: Optional[float] = None,
        direction: Optional[str] = None,
        baseline_value: Optional[float] = None,
        current_value: Optional[float] = None,
        change_percentage: Optional[float] = None,
        evidence: Optional[list] = None,
        description: Optional[str] = None,
        time_window_days: int = 30,
    ) -> ExecutiveSignal:
        """Create an executive signal, deduplicating active ones."""
        existing = (
            self.db.query(ExecutiveSignal)
            .filter(
                ExecutiveSignal.signal_type == signal_type,
                ExecutiveSignal.entity_name == entity_name,
                ExecutiveSignal.is_active == True,
            )
            .first()
        )
        if existing:
            existing.title = title
            existing.signal_insight = signal_insight
            existing.current_value = current_value
            existing.change_percentage = change_percentage
            existing.detected_at = datetime.utcnow()
            self.db.flush()
            return existing

        signal = ExecutiveSignal(
            signal_type=signal_type,
            title=title,
            description=description or signal_insight,
            raw_observation=raw_observation,
            signal_insight=signal_insight,
            entity_name=entity_name,
            entity_type=entity_type,
            severity=severity,
            magnitude=magnitude,
            direction=direction,
            baseline_value=baseline_value,
            current_value=current_value,
            change_percentage=change_percentage,
            evidence=evidence or [],
            time_window_days=time_window_days,
            detected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        self.db.add(signal)
        self.db.flush()
        logger.info(f"New signal [{signal_type.value}]: {title}")
        return signal

    # ─────────────────────────────────────────────────────────
    # SIGNAL 1: ACTIVITY SPIKE
    # ─────────────────────────────────────────────────────────

    def detect_activity_spikes(self) -> list[ExecutiveSignal]:
        """
        Detect when interaction volume with a company spikes significantly.
        Spike = 50%+ increase vs prior 30-day period.
        """
        from ..models.relationship import Interaction, Company

        signals = []
        now = datetime.utcnow()
        thirty_ago = now - timedelta(days=30)
        sixty_ago = now - timedelta(days=60)

        companies = self.db.query(Company).all()

        for company in companies:
            current = (
                self.db.query(func.count(Interaction.id))
                .filter(
                    Interaction.company_id == company.id,
                    Interaction.occurred_at >= thirty_ago,
                )
                .scalar() or 0
            )
            previous = (
                self.db.query(func.count(Interaction.id))
                .filter(
                    Interaction.company_id == company.id,
                    Interaction.occurred_at >= sixty_ago,
                    Interaction.occurred_at < thirty_ago,
                )
                .scalar() or 0
            )

            company.communication_frequency_30d = current
            company.communication_frequency_prev_30d = previous

            if previous == 0:
                continue

            change = (current - previous) / previous

            if change >= ACTIVITY_SPIKE_THRESHOLD:
                pct = round(change * 100)
                signal = self._create_signal(
                    signal_type=SignalType.ACTIVITY_SPIKE,
                    title=f"Activity spike with {company.name}: +{pct}%",
                    raw_observation=f"{current} interactions in last 30 days vs {previous} in prior 30 days",
                    signal_insight=f"Communication with {company.name} increased {pct}% — may indicate a critical phase or escalating issue.",
                    entity_name=company.name,
                    entity_type="company",
                    severity="high" if pct >= 100 else "medium",
                    magnitude=change,
                    direction="up",
                    baseline_value=float(previous),
                    current_value=float(current),
                    change_percentage=change * 100,
                    evidence=[{"company": company.name, "current": current, "previous": previous}],
                )
                signals.append(signal)

        self.db.commit()
        return signals

    # ─────────────────────────────────────────────────────────
    # SIGNAL 2: COMMUNICATION DROP
    # ─────────────────────────────────────────────────────────

    def detect_communication_drops(self) -> list[ExecutiveSignal]:
        """
        Detect when communication with a company drops significantly.
        Drop = 40%+ decrease vs prior period.
        """
        from ..models.relationship import Company

        signals = []
        companies = self.db.query(Company).filter(
            Company.communication_frequency_prev_30d > 0
        ).all()

        for company in companies:
            prev = company.communication_frequency_prev_30d or 0
            curr = company.communication_frequency_30d or 0

            if prev == 0:
                continue

            change = (curr - prev) / prev  # Negative = drop

            if change <= -COMMUNICATION_DROP_THRESHOLD:
                pct = round(abs(change) * 100)
                signal = self._create_signal(
                    signal_type=SignalType.COMMUNICATION_DROP,
                    title=f"Communication drop with {company.name}: -{pct}%",
                    raw_observation=f"{curr} interactions this month vs {prev} last month",
                    signal_insight=f"Vendor communication frequency dropped {pct}% — may indicate relationship deterioration.",
                    entity_name=company.name,
                    entity_type="company",
                    severity="critical" if pct >= 80 else ("high" if pct >= 60 else "medium"),
                    magnitude=abs(change),
                    direction="down",
                    baseline_value=float(prev),
                    current_value=float(curr),
                    change_percentage=change * 100,
                    evidence=[{"company": company.name, "current": curr, "previous": prev}],
                )
                signals.append(signal)

        self.db.commit()
        return signals

    # ─────────────────────────────────────────────────────────
    # SIGNAL 3: DELAYED RESPONSES
    # ─────────────────────────────────────────────────────────

    def detect_delayed_responses(self) -> list[ExecutiveSignal]:
        """
        Detect companies with significantly slower response times.
        """
        from ..models.relationship import Company
        signals = []
        companies = self.db.query(Company).filter(
            Company.avg_response_time_hours.isnot(None),
            Company.avg_response_time_hours > 0,
        ).all()

        for company in companies:
            if not company.avg_response_time_hours:
                continue
            # Flag if >72 hours average response time
            if company.avg_response_time_hours > 72:
                days = round(company.avg_response_time_hours / 24, 1)
                signal = self._create_signal(
                    signal_type=SignalType.DELAYED_RESPONSE,
                    title=f"Slow responses from {company.name}: avg {days} days",
                    raw_observation=f"Average response time: {company.avg_response_time_hours:.0f} hours",
                    signal_insight=f"{company.name} is taking an average of {days} days to respond — above acceptable threshold.",
                    entity_name=company.name,
                    entity_type="company",
                    severity="high" if days > 5 else "medium",
                    current_value=company.avg_response_time_hours,
                    baseline_value=24.0,  # Expected: 1 business day
                    change_percentage=((company.avg_response_time_hours - 24) / 24) * 100,
                    direction="up",
                )
                signals.append(signal)

        self.db.commit()
        return signals

    # ─────────────────────────────────────────────────────────
    # SIGNAL 4: INCREASED ESCALATIONS
    # ─────────────────────────────────────────────────────────

    def detect_escalation_increase(self) -> list[ExecutiveSignal]:
        """Detect when escalation count is rising."""
        from ..models.escalation import Escalation

        now = datetime.utcnow()
        signals = []

        # Active escalations by company
        active = (
            self.db.query(Escalation.company_name, func.count(Escalation.id).label("cnt"))
            .filter(Escalation.is_active == True)
            .group_by(Escalation.company_name)
            .all()
        )

        for company_name, count in active:
            if not company_name:
                continue
            if count >= 2:
                signal = self._create_signal(
                    signal_type=SignalType.INCREASED_ESCALATIONS,
                    title=f"{count} active escalations with {company_name}",
                    raw_observation=f"{count} simultaneous escalations detected",
                    signal_insight=f"Multiple concurrent escalations with {company_name} indicate systemic issues requiring immediate attention.",
                    entity_name=company_name,
                    entity_type="company",
                    severity="critical" if count >= 4 else ("high" if count >= 3 else "medium"),
                    current_value=float(count),
                )
                signals.append(signal)

        self.db.commit()
        return signals

    # ─────────────────────────────────────────────────────────
    # SIGNAL 5: NEW RISK
    # ─────────────────────────────────────────────────────────

    def detect_new_risks(self, lookback_hours: int = 24) -> list[ExecutiveSignal]:
        """Flag risks detected in the last N hours."""
        from ..models.risk import Risk, RiskSeverity
        signals = []
        since = datetime.utcnow() - timedelta(hours=lookback_hours)

        new_risks = (
            self.db.query(Risk)
            .filter(
                Risk.first_detected_at >= since,
                Risk.is_active == True,
                Risk.severity.in_([RiskSeverity.HIGH, RiskSeverity.CRITICAL]),
            )
            .all()
        )

        for risk in new_risks:
            signal = self._create_signal(
                signal_type=SignalType.NEW_RISK,
                title=f"New {risk.severity.value} risk: {risk.title[:80]}",
                raw_observation=f"Risk detected in category {risk.category.value}",
                signal_insight=f"A new {risk.severity.value}-severity {risk.category.value} risk was identified: {risk.title}",
                entity_name=risk.company_name or "Organization",
                entity_type="risk",
                severity=risk.severity.value,
                evidence=[{"risk_id": str(risk.id), "category": risk.category.value}],
            )
            signals.append(signal)

        self.db.commit()
        return signals

    # ─────────────────────────────────────────────────────────
    # SIGNAL 6: RELATIONSHIP DETERIORATION
    # ─────────────────────────────────────────────────────────

    def detect_relationship_deterioration(self) -> list[ExecutiveSignal]:
        """Flag companies whose health has declined to AT_RISK."""
        from ..models.relationship import Company, RelationshipHealth
        signals = []
        at_risk = self.db.query(Company).filter(
            Company.health == RelationshipHealth.AT_RISK
        ).all()

        for company in at_risk:
            signal = self._create_signal(
                signal_type=SignalType.RELATIONSHIP_DETERIORATION,
                title=f"Relationship at risk: {company.name}",
                raw_observation=f"Health score: {company.health_score:.0%}, {company.days_since_last_contact} days since contact",
                signal_insight=f"Relationship with {company.name} has deteriorated to AT RISK status (score: {company.health_score:.0%}).",
                entity_name=company.name,
                entity_type="company",
                severity="high",
                current_value=company.health_score,
                direction="down",
            )
            signals.append(signal)

        self.db.commit()
        return signals

    # ─────────────────────────────────────────────────────────
    # FULL SIGNAL GENERATION RUN
    # ─────────────────────────────────────────────────────────

    def run_full_signal_generation(self) -> dict:
        """
        Scheduled job: Run all signal detectors.
        Returns summary of generated signals.
        """
        activity_spikes = self.detect_activity_spikes()
        comm_drops = self.detect_communication_drops()
        delayed_responses = self.detect_delayed_responses()
        escalation_signals = self.detect_escalation_increase()
        new_risk_signals = self.detect_new_risks()
        deterioration_signals = self.detect_relationship_deterioration()

        total_active = self.db.query(ExecutiveSignal).filter(
            ExecutiveSignal.is_active == True
        ).count()

        return {
            "activity_spikes": len(activity_spikes),
            "communication_drops": len(comm_drops),
            "delayed_responses": len(delayed_responses),
            "escalation_increases": len(escalation_signals),
            "new_risks": len(new_risk_signals),
            "relationship_deterioration": len(deterioration_signals),
            "total_active_signals": total_active,
        }

    def get_active_signals(self) -> list[dict]:
        """Return all active executive signals ordered by severity."""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        signals = (
            self.db.query(ExecutiveSignal)
            .filter(ExecutiveSignal.is_active == True)
            .all()
        )
        signals.sort(key=lambda s: severity_order.get(s.severity, 4))
        return [
            {
                "id": str(s.id),
                "type": s.signal_type.value,
                "title": s.title,
                "insight": s.signal_insight,
                "entity": s.entity_name,
                "severity": s.severity,
                "direction": s.direction,
                "change_pct": s.change_percentage,
                "detected_at": s.detected_at.isoformat() if s.detected_at else None,
            }
            for s in signals
        ]
