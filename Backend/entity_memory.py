import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Person, Company, Project, EntityAlias, SourceAttribution
from app.core.logging import logger


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# PERSON MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class PersonMemoryService:
    """
    Manages the Person memory object lifecycle.
    Called every time a person is mentioned in any document.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create(
        self,
        canonical_name: str,
        primary_email: Optional[str] = None,
        company_id: Optional[UUID] = None,
        job_title: Optional[str] = None,
        source_document_id: Optional[str] = None,
        source_type: Optional[str] = None,
        source_timestamp: Optional[datetime] = None,
    ) -> Person:
        """
        Find existing canonical person or create a new one.
        Never creates a duplicate — always check email/name first.
        """
        # Try to find by email
        if primary_email:
            stmt = select(Person).where(
                Person.primary_email == primary_email.lower().strip(),
                Person.is_canonical == True,
            )
            existing = (await self.db.execute(stmt)).scalar_one_or_none()
            if existing:
                await self._enrich(existing, primary_email=primary_email,
                                   company_id=company_id, job_title=job_title,
                                   source_timestamp=source_timestamp)
                return existing

        # Try to find by canonical name (exact)
        norm_name = canonical_name.strip()
        stmt2 = select(Person).where(
            Person.canonical_name == norm_name,
            Person.is_canonical == True,
        )
        existing2 = (await self.db.execute(stmt2)).scalar_one_or_none()
        if existing2:
            await self._enrich(existing2, primary_email=primary_email,
                               company_id=company_id, job_title=job_title,
                               source_timestamp=source_timestamp)
            return existing2

        # Create new
        now = source_timestamp or utcnow()
        person = Person(
            canonical_name=norm_name,
            display_name=norm_name,
            primary_email=primary_email.lower().strip() if primary_email else None,
            primary_company_id=company_id,
            job_title=job_title,
            emails=[{"email": primary_email.lower(), "source": source_type}] if primary_email else [],
            communication_history={},
            related_projects=[],
            related_companies=[str(company_id)] if company_id else [],
            related_documents=[],
            related_conversations=[],
            related_commitments=[],
            related_risks=[],
            tags=[],
            custom_metadata={},
            first_seen_at=now,
            last_seen_at=now,
        )
        self.db.add(person)
        await self.db.flush()
        logger.info(f"Created person memory: {canonical_name} ({person.id})")
        return person

    async def _enrich(
        self,
        person: Person,
        primary_email: Optional[str] = None,
        company_id: Optional[UUID] = None,
        job_title: Optional[str] = None,
        source_timestamp: Optional[datetime] = None,
    ) -> None:
        """Update person with any newly discovered information."""
        changed = False
        now = source_timestamp or utcnow()

        if primary_email and not person.primary_email:
            person.primary_email = primary_email.lower().strip()
            changed = True

        if primary_email:
            emails = person.emails or []
            existing_emails = {e.get("email", "") for e in emails}
            if primary_email.lower() not in existing_emails:
                emails.append({"email": primary_email.lower(), "source": "enrichment"})
                person.emails = emails
                changed = True

        if company_id and not person.primary_company_id:
            person.primary_company_id = company_id
            changed = True

        if job_title and not person.job_title:
            person.job_title = job_title
            changed = True

        person.last_seen_at = max(person.last_seen_at or now, now)
        if changed:
            await self.db.flush()

    async def record_email(
        self, person_id: UUID, email_id: str, subject: str,
        occurred_at: datetime, other_participants: List[str]
    ) -> None:
        """Add an email interaction to person's communication history."""
        person = await self.db.get(Person, person_id)
        if not person:
            return

        history = person.communication_history or {}
        emails = history.get("emails", [])

        if email_id not in [e.get("id") for e in emails]:
            emails.append({
                "id": email_id,
                "subject": subject[:200],
                "occurred_at": occurred_at.isoformat(),
                "participants": other_participants[:20],
            })
            history["emails"] = emails[-500:]   # cap at 500 most recent
            person.communication_history = history
            person.email_count = (person.email_count or 0) + 1
            person.last_seen_at = max(person.last_seen_at or occurred_at, occurred_at)
            await self.db.flush()

    async def record_meeting(
        self, person_id: UUID, meeting_id: str, title: str,
        occurred_at: datetime, project_id: Optional[UUID] = None
    ) -> None:
        person = await self.db.get(Person, person_id)
        if not person:
            return

        history = person.communication_history or {}
        meetings = history.get("meetings", [])
        if meeting_id not in [m.get("id") for m in meetings]:
            meetings.append({
                "id": meeting_id,
                "title": title[:200],
                "occurred_at": occurred_at.isoformat(),
                "project_id": str(project_id) if project_id else None,
            })
            history["meetings"] = meetings[-200:]
            person.communication_history = history
            person.meeting_count = (person.meeting_count or 0) + 1
            person.last_seen_at = max(person.last_seen_at or occurred_at, occurred_at)
            await self.db.flush()

    async def link_project(self, person_id: UUID, project_id: UUID, role: str = "participant") -> None:
        person = await self.db.get(Person, person_id)
        if not person:
            return
        projects = person.related_projects or []
        entry = {"project_id": str(project_id), "role": role}
        if entry not in projects:
            projects.append(entry)
            person.related_projects = projects
            await self.db.flush()

    async def link_company(self, person_id: UUID, company_id: UUID, relationship: str = "works_at") -> None:
        person = await self.db.get(Person, person_id)
        if not person:
            return
        companies = person.related_companies or []
        entry = {"company_id": str(company_id), "relationship": relationship}
        if entry not in companies:
            companies.append(entry)
            person.related_companies = companies
            await self.db.flush()

    async def link_document(self, person_id: UUID, document_id: str, doc_type: str, title: str) -> None:
        person = await self.db.get(Person, person_id)
        if not person:
            return
        docs = person.related_documents or []
        if not any(d.get("id") == document_id for d in docs):
            docs.append({"id": document_id, "type": doc_type, "title": title[:200]})
            person.related_documents = docs[-200:]
            person.document_count = (person.document_count or 0) + 1
            await self.db.flush()

    async def add_commitment(
        self, person_id: UUID, commitment: Dict[str, Any]
    ) -> None:
        person = await self.db.get(Person, person_id)
        if not person:
            return
        commitments = person.related_commitments or []
        commitments.append(commitment)
        person.related_commitments = commitments
        person.commitment_count = (person.commitment_count or 0) + 1
        await self.db.flush()

    async def add_risk(self, person_id: UUID, risk: Dict[str, Any]) -> None:
        person = await self.db.get(Person, person_id)
        if not person:
            return
        risks = person.related_risks or []
        risks.append(risk)
        person.related_risks = risks
        person.risk_count = (person.risk_count or 0) + 1
        await self.db.flush()

    async def get_full_profile(self, person_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Returns the complete memory profile for a person.
        This is what the CEO sees when they click on a person's name.
        """
        person = await self.db.get(Person, person_id)
        if not person:
            return None

        return {
            "id": str(person.id),
            "canonical_name": person.canonical_name,
            "display_name": person.display_name,
            "contact": {
                "primary_email": person.primary_email,
                "emails": person.emails or [],
                "phone_numbers": person.phone_numbers or [],
                "linkedin_url": person.linkedin_url,
                "job_title": person.job_title,
                "department": person.department,
            },
            "company": str(person.primary_company_id) if person.primary_company_id else None,
            "communication_history": person.communication_history or {},
            "related_projects": person.related_projects or [],
            "related_companies": person.related_companies or [],
            "related_documents": person.related_documents or [],
            "related_conversations": person.related_conversations or [],
            "related_commitments": person.related_commitments or [],
            "related_risks": person.related_risks or [],
            "counters": {
                "emails": person.email_count,
                "meetings": person.meeting_count,
                "documents": person.document_count,
                "commitments": person.commitment_count,
                "risks": person.risk_count,
            },
            "first_seen_at": person.first_seen_at.isoformat() if person.first_seen_at else None,
            "last_seen_at": person.last_seen_at.isoformat() if person.last_seen_at else None,
            "tags": person.tags or [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class CompanyMemoryService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create(
        self,
        canonical_name: str,
        short_name: Optional[str] = None,
        domain: Optional[str] = None,
        industry: Optional[str] = None,
        country: Optional[str] = None,
        source_timestamp: Optional[datetime] = None,
    ) -> Company:
        # Try domain
        if domain:
            stmt = select(Company).where(
                Company.domain == domain.lower().strip(),
                Company.is_canonical == True,
            )
            existing = (await self.db.execute(stmt)).scalar_one_or_none()
            if existing:
                await self._enrich(existing, short_name=short_name, domain=domain,
                                   industry=industry, country=country)
                return existing

        # Try canonical name exact
        stmt2 = select(Company).where(
            Company.canonical_name == canonical_name.strip(),
            Company.is_canonical == True,
        )
        existing2 = (await self.db.execute(stmt2)).scalar_one_or_none()
        if existing2:
            await self._enrich(existing2, short_name=short_name, domain=domain,
                               industry=industry, country=country)
            return existing2

        now = source_timestamp or utcnow()
        company = Company(
            canonical_name=canonical_name.strip(),
            display_name=canonical_name.strip(),
            short_name=short_name,
            domain=domain.lower().strip() if domain else None,
            industry=industry,
            country=country,
            interactions=[], contracts=[], meetings=[], emails=[],
            commitments=[], risks=[], projects=[], key_contacts=[],
            tags=[], custom_metadata={},
            first_seen_at=now, last_seen_at=now,
        )
        self.db.add(company)
        await self.db.flush()
        logger.info(f"Created company memory: {canonical_name} ({company.id})")
        return company

    async def _enrich(self, company: Company, **kwargs) -> None:
        for key, value in kwargs.items():
            if value is not None:
                current = getattr(company, key, None)
                if current is None:
                    setattr(company, key, value)
        company.last_seen_at = utcnow()
        await self.db.flush()

    async def record_interaction(
        self, company_id: UUID, interaction: Dict[str, Any]
    ) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        interactions = company.interactions or []
        interactions.append(interaction)
        company.interactions = interactions[-500:]
        company.interaction_count = (company.interaction_count or 0) + 1
        company.last_seen_at = utcnow()
        await self.db.flush()

    async def record_contract(self, company_id: UUID, contract: Dict[str, Any]) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        contracts = company.contracts or []
        contracts.append(contract)
        company.contracts = contracts
        company.contract_count = (company.contract_count or 0) + 1
        await self.db.flush()

    async def record_meeting(self, company_id: UUID, meeting: Dict[str, Any]) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        meetings = company.meetings or []
        meetings.append(meeting)
        company.meetings = meetings[-200:]
        company.meeting_count = (company.meeting_count or 0) + 1
        await self.db.flush()

    async def add_commitment(self, company_id: UUID, commitment: Dict[str, Any]) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        commitments = company.commitments or []
        commitments.append(commitment)
        company.commitments = commitments
        company.commitment_count = (company.commitment_count or 0) + 1
        await self.db.flush()

    async def add_risk(self, company_id: UUID, risk: Dict[str, Any]) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        risks = company.risks or []
        risks.append(risk)
        company.risks = risks
        company.risk_count = (company.risk_count or 0) + 1
        await self.db.flush()

    async def link_project(self, company_id: UUID, project_id: UUID) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        projects = company.projects or []
        if str(project_id) not in projects:
            projects.append(str(project_id))
            company.projects = projects
            company.project_count = (company.project_count or 0) + 1
            await self.db.flush()

    async def link_key_contact(self, company_id: UUID, person_id: UUID) -> None:
        company = await self.db.get(Company, company_id)
        if not company:
            return
        contacts = company.key_contacts or []
        if str(person_id) not in contacts:
            contacts.append(str(person_id))
            company.key_contacts = contacts
            await self.db.flush()

    async def get_full_profile(self, company_id: UUID) -> Optional[Dict[str, Any]]:
        company = await self.db.get(Company, company_id)
        if not company:
            return None
        return {
            "id": str(company.id),
            "canonical_name": company.canonical_name,
            "short_name": company.short_name,
            "domain": company.domain,
            "industry": company.industry,
            "country": company.country,
            "interactions": company.interactions or [],
            "contracts": company.contracts or [],
            "meetings": company.meetings or [],
            "emails": company.emails or [],
            "commitments": company.commitments or [],
            "risks": company.risks or [],
            "projects": company.projects or [],
            "key_contacts": company.key_contacts or [],
            "counters": {
                "interactions": company.interaction_count,
                "contracts": company.contract_count,
                "meetings": company.meeting_count,
                "emails": company.email_count,
                "commitments": company.commitment_count,
                "risks": company.risk_count,
                "projects": company.project_count,
            },
            "first_seen_at": company.first_seen_at.isoformat() if company.first_seen_at else None,
            "last_seen_at": company.last_seen_at.isoformat() if company.last_seen_at else None,
            "tags": company.tags or [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class ProjectMemoryService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create(
        self,
        canonical_name: str,
        short_code: Optional[str] = None,
        owner_company_id: Optional[UUID] = None,
        owner_person_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        source_timestamp: Optional[datetime] = None,
    ) -> Project:
        # Try short code
        if short_code:
            stmt = select(Project).where(
                Project.short_code == short_code.upper(),
                Project.is_canonical == True,
            )
            existing = (await self.db.execute(stmt)).scalar_one_or_none()
            if existing:
                return existing

        # Try canonical name
        stmt2 = select(Project).where(
            Project.canonical_name == canonical_name.strip(),
            Project.is_canonical == True,
        )
        existing2 = (await self.db.execute(stmt2)).scalar_one_or_none()
        if existing2:
            return existing2

        now = source_timestamp or utcnow()
        project = Project(
            canonical_name=canonical_name.strip(),
            display_name=canonical_name.strip(),
            short_code=short_code.upper() if short_code else None,
            owner_company_id=owner_company_id,
            owner_person_id=owner_person_id,
            start_date=start_date,
            status="active",
            discussions=[], participants=[], deliverables=[],
            decisions=[], risks=[], dependencies=[],
            related_companies=[], related_documents=[],
            tags=[], custom_metadata={},
            first_seen_at=now, last_seen_at=now,
        )
        self.db.add(project)
        await self.db.flush()
        logger.info(f"Created project memory: {canonical_name} ({project.id})")
        return project

    async def add_discussion(
        self, project_id: UUID, discussion: Dict[str, Any]
    ) -> None:
        project = await self.db.get(Project, project_id)
        if not project:
            return
        discussions = project.discussions or []
        discussions.append(discussion)
        project.discussions = discussions[-500:]
        project.discussion_count = (project.discussion_count or 0) + 1
        project.last_activity_at = utcnow()
        await self.db.flush()

    async def add_participant(
        self, project_id: UUID, person_id: UUID, role: str = "participant"
    ) -> None:
        project = await self.db.get(Project, project_id)
        if not project:
            return
        participants = project.participants or []
        entry = {"person_id": str(person_id), "role": role}
        if entry not in participants:
            participants.append(entry)
            project.participants = participants
            project.participant_count = len(participants)
            await self.db.flush()

    async def add_deliverable(
        self, project_id: UUID, deliverable: Dict[str, Any]
    ) -> None:
        project = await self.db.get(Project, project_id)
        if not project:
            return
        deliverables = project.deliverables or []
        deliverables.append(deliverable)
        project.deliverables = deliverables
        await self.db.flush()

    async def add_decision(
        self, project_id: UUID, decision: Dict[str, Any]
    ) -> None:
        project = await self.db.get(Project, project_id)
        if not project:
            return
        decisions = project.decisions or []
        decisions.append(decision)
        project.decisions = decisions
        project.decision_count = (project.decision_count or 0) + 1
        await self.db.flush()

    async def add_risk(self, project_id: UUID, risk: Dict[str, Any]) -> None:
        project = await self.db.get(Project, project_id)
        if not project:
            return
        risks = project.risks or []
        risks.append(risk)
        project.risks = risks
        project.risk_count = (project.risk_count or 0) + 1
        await self.db.flush()

    async def add_dependency(
        self, project_id: UUID, depends_on_id: UUID, dependency_type: str
    ) -> None:
        project = await self.db.get(Project, project_id)
        if not project:
            return
        deps = project.dependencies or []
        entry = {
            "project_id": str(depends_on_id),
            "type": dependency_type,
        }
        if entry not in deps:
            deps.append(entry)
            project.dependencies = deps
            await self.db.flush()

    async def get_full_profile(self, project_id: UUID) -> Optional[Dict[str, Any]]:
        project = await self.db.get(Project, project_id)
        if not project:
            return None
        return {
            "id": str(project.id),
            "canonical_name": project.canonical_name,
            "short_code": project.short_code,
            "status": project.status,
            "priority": project.priority,
            "owner_person": str(project.owner_person_id) if project.owner_person_id else None,
            "owner_company": str(project.owner_company_id) if project.owner_company_id else None,
            "start_date": project.start_date.isoformat() if project.start_date else None,
            "end_date": project.end_date.isoformat() if project.end_date else None,
            "last_activity_at": project.last_activity_at.isoformat() if project.last_activity_at else None,
            "discussions": project.discussions or [],
            "participants": project.participants or [],
            "deliverables": project.deliverables or [],
            "decisions": project.decisions or [],
            "risks": project.risks or [],
            "dependencies": project.dependencies or [],
            "related_companies": project.related_companies or [],
            "related_documents": project.related_documents or [],
            "counters": {
                "discussions": project.discussion_count,
                "participants": project.participant_count,
                "documents": project.document_count,
                "decisions": project.decision_count,
                "risks": project.risk_count,
            },
            "tags": project.tags or [],
        }
