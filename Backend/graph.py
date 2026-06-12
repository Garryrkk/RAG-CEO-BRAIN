from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Relationship
from app.core.logging import logger


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Allowed relationship type registry
# ─────────────────────────────────────────────────────────────────────────────

VALID_RELATIONSHIPS: Dict[Tuple[str, str], List[str]] = {
    ("person", "company"):  ["works_at", "contracted_with", "consulted_for", "founded", "leads"],
    ("person", "project"):  ["leads", "participates_in", "sponsors", "manages", "reviews"],
    ("company", "project"): ["owns", "contracted_for", "involved_in", "funds"],
    ("project", "document"):["produced", "references", "governed_by"],
    ("person", "commitment"):["committed_to", "responsible_for", "accountable_for"],
    ("company", "commitment"):["committed_to", "contracted_for"],
    ("project", "risk"):    ["has_risk", "depends_on_mitigating"],
    ("person", "person"):   ["reports_to", "collaborates_with", "managed_by"],
}


def is_valid_relationship(
    source_type: str, target_type: str, rel_type: str
) -> bool:
    return rel_type in VALID_RELATIONSHIPS.get((source_type, target_type), [])


# ─────────────────────────────────────────────────────────────────────────────
# Relationship Graph Service
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipGraphService:
    """
    Creates and queries relationship edges between entities.
    Edges are weighted by frequency (how many times observed)
    and strength (confidence of the relationship).
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Write operations ─────────────────────────────────────────────────────

    async def create_or_update(
        self,
        source_entity_id: UUID,
        source_entity_type: str,
        target_entity_id: UUID,
        target_entity_type: str,
        relationship_type: str,
        relationship_label: Optional[str] = None,
        strength: float = 1.0,
        evidence_document_id: Optional[str] = None,
        evidence_event_id: Optional[UUID] = None,
        observed_at: Optional[datetime] = None,
    ) -> Relationship:
        """
        Upsert a relationship edge.
        If edge already exists → increment frequency, update last_observed_at.
        """
        observed_at = observed_at or utcnow()

        stmt = select(Relationship).where(
            and_(
                Relationship.source_entity_id == source_entity_id,
                Relationship.target_entity_id == target_entity_id,
                Relationship.relationship_type == relationship_type,
            )
        )
        existing = (await self.db.execute(stmt)).scalar_one_or_none()

        if existing:
            existing.frequency += 1
            existing.strength = max(existing.strength, strength)
            existing.last_observed_at = observed_at

            if evidence_document_id:
                docs = existing.evidence_document_ids or []
                if evidence_document_id not in docs:
                    docs.append(evidence_document_id)
                    existing.evidence_document_ids = docs[-100:]

            if evidence_event_id:
                events = existing.evidence_event_ids or []
                if str(evidence_event_id) not in events:
                    events.append(str(evidence_event_id))
                    existing.evidence_event_ids = events[-100:]

            await self.db.flush()
            return existing

        rel = Relationship(
            source_entity_id=source_entity_id,
            source_entity_type=source_entity_type,
            target_entity_id=target_entity_id,
            target_entity_type=target_entity_type,
            relationship_type=relationship_type,
            relationship_label=relationship_label,
            strength=strength,
            frequency=1,
            evidence_document_ids=[evidence_document_id] if evidence_document_id else [],
            evidence_event_ids=[str(evidence_event_id)] if evidence_event_id else [],
            first_observed_at=observed_at,
            last_observed_at=observed_at,
            is_active=True,
        )
        self.db.add(rel)
        await self.db.flush()
        logger.debug(
            f"Relationship: {source_entity_type}:{source_entity_id} "
            f"—[{relationship_type}]→ {target_entity_type}:{target_entity_id}"
        )
        return rel

    # ── Convenience methods for each relationship type ────────────────────────

    async def link_person_to_company(
        self, person_id: UUID, company_id: UUID,
        rel_type: str = "works_at", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            person_id, "person", company_id, "company", rel_type, **kwargs
        )

    async def link_person_to_project(
        self, person_id: UUID, project_id: UUID,
        role: str = "participates_in", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            person_id, "person", project_id, "project", role, **kwargs
        )

    async def link_company_to_project(
        self, company_id: UUID, project_id: UUID,
        rel_type: str = "involved_in", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            company_id, "company", project_id, "project", rel_type, **kwargs
        )

    async def link_project_to_document(
        self, project_id: UUID, document_id: str, rel_type: str = "produced"
    ) -> Relationship:
        # document_id is a string so we store it as a synthetic UUID based on hash
        import hashlib, uuid as uuidlib
        doc_uuid = uuidlib.UUID(hashlib.md5(document_id.encode()).hexdigest())
        return await self.create_or_update(
            project_id, "project", doc_uuid, "document", rel_type
        )

    async def link_person_to_commitment(
        self, person_id: UUID, commitment_id: UUID,
        rel_type: str = "committed_to", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            person_id, "person", commitment_id, "commitment", rel_type, **kwargs
        )

    async def link_company_to_commitment(
        self, company_id: UUID, commitment_id: UUID,
        rel_type: str = "committed_to", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            company_id, "company", commitment_id, "commitment", rel_type, **kwargs
        )

    async def link_project_to_risk(
        self, project_id: UUID, risk_id: UUID,
        rel_type: str = "has_risk", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            project_id, "project", risk_id, "risk", rel_type, **kwargs
        )

    async def link_persons(
        self, person_a_id: UUID, person_b_id: UUID,
        rel_type: str = "collaborates_with", **kwargs
    ) -> Relationship:
        return await self.create_or_update(
            person_a_id, "person", person_b_id, "person", rel_type, **kwargs
        )

    # ── Query operations ─────────────────────────────────────────────────────

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        entity_type: str,
        direction: str = "both",   # outbound | inbound | both
        relationship_types: Optional[List[str]] = None,
        min_strength: float = 0.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all relationships for an entity.
        Supports directional queries and type filtering.
        """
        conditions = []

        if direction in ("outbound", "both"):
            out_cond = and_(
                Relationship.source_entity_id == entity_id,
                Relationship.source_entity_type == entity_type,
            )
            conditions.append(out_cond)

        if direction in ("inbound", "both"):
            in_cond = and_(
                Relationship.target_entity_id == entity_id,
                Relationship.target_entity_type == entity_type,
            )
            conditions.append(in_cond)

        from sqlalchemy import or_
        stmt = select(Relationship).where(or_(*conditions))

        if relationship_types:
            stmt = stmt.where(Relationship.relationship_type.in_(relationship_types))

        if min_strength > 0:
            stmt = stmt.where(Relationship.strength >= min_strength)

        stmt = stmt.where(Relationship.is_active == True)
        stmt = stmt.order_by(Relationship.frequency.desc()).limit(limit)

        rows = (await self.db.execute(stmt)).scalars().all()

        results = []
        for r in rows:
            is_outbound = r.source_entity_id == entity_id
            results.append({
                "id": str(r.id),
                "direction": "outbound" if is_outbound else "inbound",
                "source": {
                    "id": str(r.source_entity_id),
                    "type": r.source_entity_type,
                },
                "target": {
                    "id": str(r.target_entity_id),
                    "type": r.target_entity_type,
                },
                "relationship_type": r.relationship_type,
                "label": r.relationship_label,
                "strength": r.strength,
                "frequency": r.frequency,
                "first_observed_at": r.first_observed_at.isoformat() if r.first_observed_at else None,
                "last_observed_at": r.last_observed_at.isoformat() if r.last_observed_at else None,
                "evidence_documents": r.evidence_document_ids or [],
            })

        return results

    async def get_projects_by_company(
        self, company_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Answer: "Show me all projects involving Schneider."
        """
        stmt = select(Relationship).where(
            Relationship.source_entity_id == company_id,
            Relationship.source_entity_type == "company",
            Relationship.target_entity_type == "project",
            Relationship.is_active == True,
        ).order_by(Relationship.last_observed_at.desc())

        rows = (await self.db.execute(stmt)).scalars().all()
        return [
            {
                "project_id": str(r.target_entity_id),
                "relationship_type": r.relationship_type,
                "strength": r.strength,
                "frequency": r.frequency,
                "last_seen": r.last_observed_at.isoformat() if r.last_observed_at else None,
            }
            for r in rows
        ]

    async def get_persons_by_project(
        self, project_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Answer: "Which people worked on this project?"
        """
        stmt = select(Relationship).where(
            Relationship.target_entity_id == project_id,
            Relationship.target_entity_type == "project",
            Relationship.source_entity_type == "person",
            Relationship.is_active == True,
        ).order_by(Relationship.frequency.desc())

        rows = (await self.db.execute(stmt)).scalars().all()
        return [
            {
                "person_id": str(r.source_entity_id),
                "role": r.relationship_type,
                "strength": r.strength,
                "frequency": r.frequency,
                "first_seen": r.first_observed_at.isoformat() if r.first_observed_at else None,
            }
            for r in rows
        ]

    async def get_companies_by_project(
        self, project_id: UUID
    ) -> List[Dict[str, Any]]:
        stmt = select(Relationship).where(
            Relationship.target_entity_id == project_id,
            Relationship.target_entity_type == "project",
            Relationship.source_entity_type == "company",
            Relationship.is_active == True,
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return [
            {"company_id": str(r.source_entity_id), "type": r.relationship_type}
            for r in rows
        ]

    async def get_relationship_graph(
        self,
        entity_id: UUID,
        entity_type: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Build a traversal graph up to `depth` hops from the anchor entity.
        Returns nodes + edges suitable for frontend graph rendering.
        """
        visited_nodes = set()
        nodes = []
        edges = []
        queue = [(entity_id, entity_type, 0)]

        while queue:
            current_id, current_type, current_depth = queue.pop(0)
            key = f"{current_type}:{current_id}"
            if key in visited_nodes:
                continue
            visited_nodes.add(key)

            nodes.append({"id": str(current_id), "type": current_type})

            if current_depth >= depth:
                continue

            rels = await self.get_entity_relationships(
                current_id, current_type, direction="both", limit=50
            )
            for rel in rels:
                edges.append(rel)
                # Add neighbor to queue
                if rel["direction"] == "outbound":
                    neighbor_id = UUID(rel["target"]["id"])
                    neighbor_type = rel["target"]["type"]
                else:
                    neighbor_id = UUID(rel["source"]["id"])
                    neighbor_type = rel["source"]["type"]

                neighbor_key = f"{neighbor_type}:{neighbor_id}"
                if neighbor_key not in visited_nodes:
                    queue.append((neighbor_id, neighbor_type, current_depth + 1))

        return {
            "anchor": {"id": str(entity_id), "type": entity_type},
            "depth": depth,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    async def deactivate_stale_relationships(
        self, stale_days: int = 180
    ) -> int:
        """Mark relationships as inactive if not observed in N days."""
        from datetime import timedelta
        cutoff = utcnow() - timedelta(days=stale_days)
        stmt = (
            select(Relationship)
            .where(
                Relationship.last_observed_at < cutoff,
                Relationship.is_active == True,
                Relationship.frequency < 3,   # don't deactivate strong edges
            )
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        for r in rows:
            r.is_active = False
        await self.db.flush()
        return len(rows)
