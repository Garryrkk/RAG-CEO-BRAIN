
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from uuid import UUID

from rapidfuzz import fuzz, process as rfprocess
from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Person, Company, Project, EntityAlias
from app.core.config import settings
from app.core.logging import logger
from app.db.redis_client import cache_get, cache_set


# ─── Common name suffixes to strip ───────────────────────────────────────────
LEGAL_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|pvt|private|gmbh|sa|sas|bv|"
    r"plc|ag|as|oy|ab|nv|co|company|group|holdings|enterprises|solutions|"
    r"technologies|tech|india|us|uk|global)\b\.?",
    re.IGNORECASE,
)

NICKNAME_MAP = {
    "john": ["johnny", "jon", "j"],
    "william": ["will", "bill", "willy", "liam"],
    "robert": ["rob", "bob", "bobby"],
    "james": ["jim", "jimmy", "jamie"],
    "michael": ["mike", "micky", "mich"],
    "richard": ["rick", "dick", "rich"],
    "christopher": ["chris", "kit"],
    "charles": ["charlie", "chuck", "chas"],
    "thomas": ["tom", "tommy"],
    "jennifer": ["jen", "jenny"],
    "katherine": ["kate", "kathy", "kat", "katie"],
    "elizabeth": ["liz", "beth", "eliza", "betty", "lisa"],
    "margaret": ["meg", "maggie", "marge", "peggy"],
    "patricia": ["pat", "patty", "trish"],
    "alexander": ["alex", "al", "xander"],
    "benjamin": ["ben", "benny"],
    "andrew": ["andy", "drew"],
    "daniel": ["dan", "danny"],
    "joseph": ["joe", "joey"],
    "matthew": ["matt", "matty"],
    "nicholas": ["nick", "nicky"],
    "samantha": ["sam", "sammy"],
    "stephanie": ["steph", "stevie"],
    "david": ["dave", "davy"],
}

# Reverse map: short form → canonical first name
_REVERSE_NICKNAME = {}
for canonical, nicknames in NICKNAME_MAP.items():
    for n in nicknames:
        _REVERSE_NICKNAME[n] = canonical


@dataclass
class ResolutionCandidate:
    entity_id: UUID
    entity_type: str   # person | company | project
    canonical_name: str
    confidence: float
    resolution_method: str   # email | name_exact | name_fuzzy | alias | context | domain


@dataclass
class ResolutionResult:
    resolved: bool
    entity_id: Optional[UUID]
    canonical_name: Optional[str]
    confidence: float
    method: str
    is_new_entity: bool = False
    candidates: List[ResolutionCandidate] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Lowercase, strip accents, remove punctuation, collapse whitespace.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s@.\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_email(email: str) -> str:
    return email.lower().strip()


def extract_email_domain(email: str) -> str:
    if "@" in email:
        return email.split("@")[1].lower().strip()
    return ""


def normalize_company_name(name: str) -> str:
    """
    Strip legal suffixes, normalize whitespace.
    'Schneider Electric India Pvt Ltd' → 'schneider electric'
    """
    n = normalize_text(name)
    n = LEGAL_SUFFIXES.sub("", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def tokenize_name(name: str) -> List[str]:
    return normalize_text(name).split()


def expand_nickname(first_name: str) -> List[str]:
    """Return all plausible forms of a first name."""
    fn = first_name.lower().strip()
    variants = {fn}
    if fn in NICKNAME_MAP:
        variants.update(NICKNAME_MAP[fn])
    if fn in _REVERSE_NICKNAME:
        canonical = _REVERSE_NICKNAME[fn]
        variants.add(canonical)
        variants.update(NICKNAME_MAP.get(canonical, []))
    return list(variants)


def initial_of(name: str) -> str:
    """'John' → 'J'"""
    return name[0].upper() if name else ""


# ─────────────────────────────────────────────────────────────────────────────
# PERSON RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class PersonResolver:
    """
    Resolves raw person references to canonical Person entities.

    Resolution pipeline (ordered by precision):
      1. Exact email match
      2. Alias table lookup
      3. Fuzzy name match (tokenized, initial-aware, nickname-aware)
      4. Context-based disambiguation
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.threshold_name = settings.ENTITY_RESOLUTION_NAME_THRESHOLD
        self.threshold_context = settings.ENTITY_RESOLUTION_CONTEXT_WEIGHT

    async def resolve(
        self,
        raw_name: Optional[str] = None,
        email: Optional[str] = None,
        company_id: Optional[UUID] = None,
        project_ids: Optional[List[UUID]] = None,
        source_document_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ResolutionResult:
        """
        Main entry point. Returns the best matching canonical Person.
        """
        # 1. Email match — highest confidence
        if email:
            result = await self._resolve_by_email(email)
            if result.resolved:
                return result

        # 2. Alias table lookup
        if raw_name or email:
            result = await self._resolve_by_alias(raw_name, email)
            if result.resolved:
                return result

        # 3. Fuzzy name match
        if raw_name:
            result = await self._resolve_by_name(raw_name, company_id)
            if result.resolved:
                # Apply context boost if company or project context available
                if company_id or project_ids:
                    result = await self._apply_context_boost(result, company_id, project_ids)
                return result

        # 4. Context-only (same company + partial name fragments)
        if raw_name and (company_id or project_ids):
            result = await self._resolve_by_context(raw_name, company_id, project_ids)
            if result.resolved:
                return result

        return ResolutionResult(
            resolved=False,
            entity_id=None,
            canonical_name=raw_name,
            confidence=0.0,
            method="unresolved",
        )

    async def _resolve_by_email(self, email: str) -> ResolutionResult:
        norm_email = normalize_email(email)

        # Cache check
        cache_key = f"person_email:{norm_email}"
        cached = await cache_get(cache_key)
        if cached:
            return ResolutionResult(
                resolved=True,
                entity_id=UUID(cached["entity_id"]),
                canonical_name=cached["canonical_name"],
                confidence=1.0,
                method="email_cache",
            )

        # Check primary email
        stmt = select(Person).where(
            Person.primary_email == norm_email,
            Person.is_canonical == True,
        )
        row = (await self.db.execute(stmt)).scalar_one_or_none()
        if row:
            await cache_set(cache_key, {
                "entity_id": str(row.id),
                "canonical_name": row.canonical_name,
            }, ttl=3600)
            return ResolutionResult(
                resolved=True,
                entity_id=row.id,
                canonical_name=row.canonical_name,
                confidence=1.0,
                method="email_exact",
            )

        # Check emails JSONB array
        stmt2 = select(Person).where(
            Person.emails.contains([{"email": norm_email}]),
            Person.is_canonical == True,
        )
        row2 = (await self.db.execute(stmt2)).scalar_one_or_none()
        if row2:
            return ResolutionResult(
                resolved=True,
                entity_id=row2.id,
                canonical_name=row2.canonical_name,
                confidence=0.98,
                method="email_array",
            )

        # Alias table
        stmt3 = select(EntityAlias).where(
            EntityAlias.raw_value == norm_email,
            EntityAlias.entity_type == "person",
            EntityAlias.alias_type == "email",
        )
        alias = (await self.db.execute(stmt3)).scalar_one_or_none()
        if alias:
            person = await self.db.get(Person, alias.entity_id)
            if person and person.is_canonical:
                return ResolutionResult(
                    resolved=True,
                    entity_id=person.id,
                    canonical_name=person.canonical_name,
                    confidence=alias.confidence,
                    method="email_alias",
                )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="email_miss")

    async def _resolve_by_alias(
        self, raw_name: Optional[str], email: Optional[str]
    ) -> ResolutionResult:
        """Look up pre-registered aliases."""
        lookups = []
        if raw_name:
            lookups.append(normalize_text(raw_name))
        if email:
            lookups.append(normalize_email(email))

        for lookup in lookups:
            stmt = select(EntityAlias).where(
                EntityAlias.raw_value == lookup,
                EntityAlias.entity_type == "person",
            )
            alias = (await self.db.execute(stmt)).scalar_one_or_none()
            if alias:
                person = await self.db.get(Person, alias.entity_id)
                if person and person.is_canonical:
                    return ResolutionResult(
                        resolved=True,
                        entity_id=person.id,
                        canonical_name=person.canonical_name,
                        confidence=alias.confidence,
                        method="alias_lookup",
                    )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="alias_miss")

    async def _resolve_by_name(
        self, raw_name: str, company_id: Optional[UUID] = None
    ) -> ResolutionResult:
        """
        Multi-strategy name matching:
        - Exact normalized match
        - Token overlap (handles middle names, initials)
        - Nickname expansion
        - Initial matching ("J. Smith" → "John Smith")
        """
        tokens = tokenize_name(raw_name)
        if not tokens:
            return ResolutionResult(resolved=False, entity_id=None,
                                    canonical_name=None, confidence=0.0, method="name_empty")

        # Load candidate persons (filter by company if available)
        stmt = select(Person).where(Person.is_canonical == True)
        if company_id:
            stmt = stmt.where(Person.primary_company_id == company_id)
        persons = (await self.db.execute(stmt)).scalars().all()

        if not persons:
            # Retry without company filter
            stmt2 = select(Person).where(Person.is_canonical == True)
            persons = (await self.db.execute(stmt2)).scalars().all()

        best_score = 0.0
        best_person = None
        best_method = "name_fuzzy"

        for person in persons:
            score, method = self._score_name_match(raw_name, tokens, person.canonical_name)
            if score > best_score:
                best_score = score
                best_person = person
                best_method = method

        if best_person and best_score >= self.threshold_name:
            return ResolutionResult(
                resolved=True,
                entity_id=best_person.id,
                canonical_name=best_person.canonical_name,
                confidence=best_score,
                method=best_method,
            )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=raw_name, confidence=best_score, method="name_miss")

    def _score_name_match(
        self, raw: str, raw_tokens: List[str], canonical: str
    ) -> Tuple[float, str]:
        """
        Returns (best_score, method_name).
        Tries multiple strategies and returns the highest scoring one.
        """
        can_tokens = tokenize_name(canonical)
        scores = []

        # 1. Direct fuzzy ratio
        direct = fuzz.ratio(normalize_text(raw), normalize_text(canonical)) / 100.0
        scores.append((direct, "fuzzy_ratio"))

        # 2. Token sort ratio (handles name ordering differences)
        token_sort = fuzz.token_sort_ratio(normalize_text(raw), normalize_text(canonical)) / 100.0
        scores.append((token_sort, "token_sort"))

        # 3. Token set ratio (handles middle names, extra tokens)
        token_set = fuzz.token_set_ratio(normalize_text(raw), normalize_text(canonical)) / 100.0
        scores.append((token_set, "token_set"))

        # 4. Initial-aware matching ("J. Smith" vs "John Smith")
        initial_score = self._initial_match_score(raw_tokens, can_tokens)
        scores.append((initial_score, "initial_match"))

        # 5. Nickname expansion
        nick_score = self._nickname_match_score(raw_tokens, can_tokens)
        scores.append((nick_score, "nickname"))

        return max(scores, key=lambda x: x[0])

    def _initial_match_score(
        self, raw_tokens: List[str], can_tokens: List[str]
    ) -> float:
        """
        'J. Smith' → tokens = ['j', 'smith']
        'John Smith' → tokens = ['john', 'smith']
        Score based on initial match for single-char tokens.
        """
        if not raw_tokens or not can_tokens:
            return 0.0

        matched = 0
        total = max(len(raw_tokens), len(can_tokens))

        for rt in raw_tokens:
            for ct in can_tokens:
                if len(rt) == 1 and ct.startswith(rt):
                    matched += 0.8
                    break
                elif rt == ct:
                    matched += 1.0
                    break

        return matched / total if total > 0 else 0.0

    def _nickname_match_score(
        self, raw_tokens: List[str], can_tokens: List[str]
    ) -> float:
        """
        Expand all token variants and find best overlap.
        """
        if not raw_tokens or not can_tokens:
            return 0.0

        raw_expanded = set()
        for t in raw_tokens:
            raw_expanded.add(t)
            raw_expanded.update(expand_nickname(t))

        can_expanded = set()
        for t in can_tokens:
            can_expanded.add(t)
            can_expanded.update(expand_nickname(t))

        intersection = raw_expanded & can_expanded
        union = raw_expanded | can_expanded

        return len(intersection) / len(union) if union else 0.0

    async def _resolve_by_context(
        self,
        raw_name: str,
        company_id: Optional[UUID],
        project_ids: Optional[List[UUID]],
    ) -> ResolutionResult:
        """
        Context-only resolution: find person who shares company/project context
        and has a plausible name fragment match (lower threshold).
        """
        stmt = select(Person).where(Person.is_canonical == True)
        persons = (await self.db.execute(stmt)).scalars().all()

        raw_tokens = tokenize_name(raw_name)
        best = None
        best_score = 0.0

        for person in persons:
            name_score, method = self._score_name_match(
                raw_name, raw_tokens, person.canonical_name
            )

            # Check company context
            context_bonus = 0.0
            if company_id and person.primary_company_id == company_id:
                context_bonus += self.threshold_context

            # Check project context
            if project_ids:
                person_projects = {r for r in (person.related_projects or [])}
                overlap = len(set(str(p) for p in project_ids) & person_projects)
                if overlap:
                    context_bonus += self.threshold_context * min(overlap, 2)

            total = min(name_score + context_bonus, 1.0)
            if total > best_score:
                best_score = total
                best = person

        # Lower threshold when context is available
        context_threshold = self.threshold_name - self.threshold_context
        if best and best_score >= context_threshold:
            return ResolutionResult(
                resolved=True,
                entity_id=best.id,
                canonical_name=best.canonical_name,
                confidence=best_score,
                method="context",
            )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=raw_name, confidence=0.0, method="context_miss")

    async def _apply_context_boost(
        self,
        result: ResolutionResult,
        company_id: Optional[UUID],
        project_ids: Optional[List[UUID]],
    ) -> ResolutionResult:
        """Boost confidence of an existing resolution using context signals."""
        if not result.resolved or not result.entity_id:
            return result

        person = await self.db.get(Person, result.entity_id)
        if not person:
            return result

        boost = 0.0
        if company_id and person.primary_company_id == company_id:
            boost += 0.05

        if project_ids:
            person_projects = set(person.related_projects or [])
            overlap = len(set(str(p) for p in project_ids) & person_projects)
            if overlap:
                boost += 0.03 * min(overlap, 3)

        result.confidence = min(result.confidence + boost, 1.0)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_ABBREVIATIONS = {
    # Add known abbreviations here; loaded from DB at runtime too
    "se": "schneider electric",
    "ge": "general electric",
    "ibm": "international business machines",
    "ms": "microsoft",
    "amzn": "amazon",
    "goog": "google",
    "fb": "meta",
}


class CompanyResolver:
    """
    Resolves raw company references to canonical Company entities.

    Strategies:
      1. Domain matching (exact)
      2. Alias table
      3. Abbreviation expansion
      4. Normalized name fuzzy match
      5. Context (shared contacts, projects)
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.threshold = settings.ENTITY_RESOLUTION_COMPANY_THRESHOLD

    async def resolve(
        self,
        raw_name: Optional[str] = None,
        domain: Optional[str] = None,
        abbreviation: Optional[str] = None,
        context_person_ids: Optional[List[UUID]] = None,
    ) -> ResolutionResult:
        # 1. Domain match
        if domain:
            result = await self._resolve_by_domain(domain)
            if result.resolved:
                return result

        # 2. Alias lookup
        if raw_name or abbreviation:
            result = await self._resolve_by_alias(raw_name, abbreviation)
            if result.resolved:
                return result

        # 3. Abbreviation expansion
        if abbreviation:
            result = await self._resolve_by_abbreviation(abbreviation)
            if result.resolved:
                return result

        # 4. Fuzzy name match
        if raw_name:
            result = await self._resolve_by_name(raw_name)
            if result.resolved:
                return result

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=raw_name, confidence=0.0, method="company_unresolved")

    async def _resolve_by_domain(self, domain: str) -> ResolutionResult:
        norm_domain = domain.lower().strip().lstrip("www.")
        stmt = select(Company).where(
            Company.domain == norm_domain,
            Company.is_canonical == True,
        )
        row = (await self.db.execute(stmt)).scalar_one_or_none()
        if row:
            return ResolutionResult(
                resolved=True,
                entity_id=row.id,
                canonical_name=row.canonical_name,
                confidence=1.0,
                method="domain_exact",
            )
        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="domain_miss")

    async def _resolve_by_alias(
        self, raw_name: Optional[str], abbreviation: Optional[str]
    ) -> ResolutionResult:
        lookups = []
        if raw_name:
            lookups.append(normalize_company_name(raw_name))
        if abbreviation:
            lookups.append(abbreviation.lower().strip())

        for lookup in lookups:
            stmt = select(EntityAlias).where(
                EntityAlias.raw_value == lookup,
                EntityAlias.entity_type == "company",
            )
            alias = (await self.db.execute(stmt)).scalar_one_or_none()
            if alias:
                company = await self.db.get(Company, alias.entity_id)
                if company and company.is_canonical:
                    return ResolutionResult(
                        resolved=True,
                        entity_id=company.id,
                        canonical_name=company.canonical_name,
                        confidence=alias.confidence,
                        method="company_alias",
                    )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="company_alias_miss")

    async def _resolve_by_abbreviation(self, abbreviation: str) -> ResolutionResult:
        abbr_lower = abbreviation.lower().strip()
        expanded = COMPANY_ABBREVIATIONS.get(abbr_lower)
        if expanded:
            # Try to find by normalized name
            stmt = select(Company).where(Company.is_canonical == True)
            companies = (await self.db.execute(stmt)).scalars().all()
            for company in companies:
                if normalize_company_name(company.canonical_name) == expanded:
                    return ResolutionResult(
                        resolved=True,
                        entity_id=company.id,
                        canonical_name=company.canonical_name,
                        confidence=0.9,
                        method="abbreviation_map",
                    )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="abbreviation_miss")

    async def _resolve_by_name(self, raw_name: str) -> ResolutionResult:
        norm_raw = normalize_company_name(raw_name)
        if not norm_raw:
            return ResolutionResult(resolved=False, entity_id=None,
                                    canonical_name=raw_name, confidence=0.0, method="company_name_empty")

        stmt = select(Company).where(Company.is_canonical == True)
        companies = (await self.db.execute(stmt)).scalars().all()

        best_score = 0.0
        best_company = None

        for company in companies:
            norm_can = normalize_company_name(company.canonical_name)
            # Try multiple fuzzy scores
            scores = [
                fuzz.ratio(norm_raw, norm_can) / 100.0,
                fuzz.token_sort_ratio(norm_raw, norm_can) / 100.0,
                fuzz.token_set_ratio(norm_raw, norm_can) / 100.0,
                fuzz.partial_ratio(norm_raw, norm_can) / 100.0 * 0.85,  # penalize partial
            ]
            score = max(scores)
            if score > best_score:
                best_score = score
                best_company = company

        if best_company and best_score >= self.threshold:
            return ResolutionResult(
                resolved=True,
                entity_id=best_company.id,
                canonical_name=best_company.canonical_name,
                confidence=best_score,
                method="company_name_fuzzy",
            )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=raw_name, confidence=best_score, method="company_name_miss")


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class ProjectResolver:
    """
    Resolves raw project references to canonical Project entities.

    Strategies:
      1. Short-code exact match
      2. Alias table
      3. Fuzzy name match
      4. Context matching (shared company, participants)
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.threshold = settings.ENTITY_RESOLUTION_PROJECT_THRESHOLD

    async def resolve(
        self,
        raw_name: Optional[str] = None,
        short_code: Optional[str] = None,
        company_id: Optional[UUID] = None,
        person_ids: Optional[List[UUID]] = None,
    ) -> ResolutionResult:
        # 1. Short code
        if short_code:
            result = await self._resolve_by_short_code(short_code)
            if result.resolved:
                return result

        # 2. Alias lookup
        if raw_name:
            result = await self._resolve_by_alias(raw_name)
            if result.resolved:
                return result

        # 3. Fuzzy name
        if raw_name:
            result = await self._resolve_by_name(raw_name, company_id)
            if result.resolved:
                return result

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=raw_name, confidence=0.0, method="project_unresolved")

    async def _resolve_by_short_code(self, short_code: str) -> ResolutionResult:
        norm_code = short_code.upper().strip()
        stmt = select(Project).where(
            Project.short_code == norm_code,
            Project.is_canonical == True,
        )
        row = (await self.db.execute(stmt)).scalar_one_or_none()
        if row:
            return ResolutionResult(
                resolved=True,
                entity_id=row.id,
                canonical_name=row.canonical_name,
                confidence=1.0,
                method="short_code_exact",
            )
        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="short_code_miss")

    async def _resolve_by_alias(self, raw_name: str) -> ResolutionResult:
        norm = normalize_text(raw_name)
        stmt = select(EntityAlias).where(
            EntityAlias.raw_value == norm,
            EntityAlias.entity_type == "project",
        )
        alias = (await self.db.execute(stmt)).scalar_one_or_none()
        if alias:
            project = await self.db.get(Project, alias.entity_id)
            if project and project.is_canonical:
                return ResolutionResult(
                    resolved=True,
                    entity_id=project.id,
                    canonical_name=project.canonical_name,
                    confidence=alias.confidence,
                    method="project_alias",
                )
        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=None, confidence=0.0, method="project_alias_miss")

    async def _resolve_by_name(
        self, raw_name: str, company_id: Optional[UUID] = None
    ) -> ResolutionResult:
        norm_raw = normalize_text(raw_name)
        stmt = select(Project).where(Project.is_canonical == True)
        if company_id:
            stmt = stmt.where(Project.owner_company_id == company_id)
        projects = (await self.db.execute(stmt)).scalars().all()

        if not projects and company_id:
            stmt2 = select(Project).where(Project.is_canonical == True)
            projects = (await self.db.execute(stmt2)).scalars().all()

        best_score = 0.0
        best_project = None

        for project in projects:
            norm_can = normalize_text(project.canonical_name)
            scores = [
                fuzz.ratio(norm_raw, norm_can) / 100.0,
                fuzz.token_sort_ratio(norm_raw, norm_can) / 100.0,
                fuzz.token_set_ratio(norm_raw, norm_can) / 100.0,
            ]
            score = max(scores)
            if score > best_score:
                best_score = score
                best_project = project

        if best_project and best_score >= self.threshold:
            return ResolutionResult(
                resolved=True,
                entity_id=best_project.id,
                canonical_name=best_project.canonical_name,
                confidence=best_score,
                method="project_name_fuzzy",
            )

        return ResolutionResult(resolved=False, entity_id=None,
                                canonical_name=raw_name, confidence=best_score, method="project_name_miss")


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED ENTITY RESOLVER (orchestrator for all three types)
# ─────────────────────────────────────────────────────────────────────────────

class EntityResolutionEngine:
    """
    Top-level resolver. Called by the processing pipeline for every document.
    Coordinates PersonResolver, CompanyResolver, ProjectResolver.
    Also handles alias registration when a new resolution is confirmed.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.person_resolver = PersonResolver(db)
        self.company_resolver = CompanyResolver(db)
        self.project_resolver = ProjectResolver(db)

    async def resolve_person(self, **kwargs) -> ResolutionResult:
        return await self.person_resolver.resolve(**kwargs)

    async def resolve_company(self, **kwargs) -> ResolutionResult:
        return await self.company_resolver.resolve(**kwargs)

    async def resolve_project(self, **kwargs) -> ResolutionResult:
        return await self.project_resolver.resolve(**kwargs)

    async def register_alias(
        self,
        entity_id: UUID,
        entity_type: str,
        raw_value: str,
        alias_type: str,
        confidence: float = 1.0,
        source_document_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> EntityAlias:
        """
        Persist a newly discovered alias so future lookups are instant.
        """
        norm = normalize_text(raw_value)

        # Check if alias already exists
        stmt = select(EntityAlias).where(
            EntityAlias.entity_id == entity_id,
            EntityAlias.entity_type == entity_type,
            EntityAlias.raw_value == norm,
        )
        existing = (await self.db.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing

        alias = EntityAlias(
            entity_id=entity_id,
            entity_type=entity_type,
            raw_value=norm,
            alias_type=alias_type,
            confidence=confidence,
            source_document_id=source_document_id,
            source_type=source_type,
        )
        self.db.add(alias)
        await self.db.flush()
        logger.debug(f"Registered alias: {norm!r} → {entity_type}:{entity_id}")
        return alias

    async def merge_persons(
        self,
        canonical_id: UUID,
        duplicate_ids: List[UUID],
        reason: str = "resolution",
    ) -> Person:
        """
        Merge duplicate person records into the canonical one.
        Transfers all aliases, updates merged_from_ids.
        """
        canonical = await self.db.get(Person, canonical_id)
        if not canonical:
            raise ValueError(f"Canonical person {canonical_id} not found")

        merged_ids = list(canonical.merged_from_ids or [])

        for dup_id in duplicate_ids:
            dup = await self.db.get(Person, dup_id)
            if not dup:
                continue

            # Transfer aliases
            stmt = select(EntityAlias).where(
                EntityAlias.entity_id == dup_id,
                EntityAlias.entity_type == "person",
            )
            aliases = (await self.db.execute(stmt)).scalars().all()
            for alias in aliases:
                alias.entity_id = canonical_id
                # Register original name as alias
            await self.register_alias(
                canonical_id, "person", dup.canonical_name, "name", 0.95
            )

            # Merge email lists
            dup_emails = dup.emails or []
            can_emails = canonical.emails or []
            email_set = {e["email"]: e for e in can_emails}
            for e in dup_emails:
                if e["email"] not in email_set:
                    email_set[e["email"]] = e
            canonical.emails = list(email_set.values())

            if dup.primary_email and not canonical.primary_email:
                canonical.primary_email = dup.primary_email

            merged_ids.append(str(dup_id))
            dup.is_canonical = False

            logger.info(f"Merged person {dup.canonical_name} ({dup_id}) into {canonical.canonical_name} ({canonical_id}). Reason: {reason}")

        canonical.merged_from_ids = merged_ids
        await self.db.flush()
        return canonical

    async def merge_companies(
        self, canonical_id: UUID, duplicate_ids: List[UUID]
    ) -> Company:
        """Merge duplicate company records."""
        canonical = await self.db.get(Company, canonical_id)
        if not canonical:
            raise ValueError(f"Canonical company {canonical_id} not found")

        merged_ids = list(canonical.merged_from_ids or [])
        for dup_id in duplicate_ids:
            dup = await self.db.get(Company, dup_id)
            if not dup:
                continue
            await self.register_alias(
                canonical_id, "company", dup.canonical_name, "name", 0.95
            )
            if dup.short_name:
                await self.register_alias(
                    canonical_id, "company", dup.short_name, "abbreviation", 0.9
                )
            merged_ids.append(str(dup_id))
            dup.is_canonical = False
            logger.info(f"Merged company {dup.canonical_name} into {canonical.canonical_name}")

        canonical.merged_from_ids = merged_ids
        await self.db.flush()
        return canonical
