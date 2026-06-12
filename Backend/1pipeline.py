

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

import httpx
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.db.qdrant import get_qdrant
from app.models.entities import DocumentChunk


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Embedding Client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaEmbeddingClient:
    """
    Calls Ollama's /api/embeddings endpoint with BGE-M3.
    Handles batching, retries, and rate limiting.
    """

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text string. Returns vector of dimension 1024."""
        client = await self._get_client()
        response = await client.post(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", [])
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, got {len(embedding)}"
            )
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts. Processes in sub-batches of EMBEDDING_BATCH_SIZE.
        Returns embeddings in same order as input.
        """
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await asyncio.gather(
                *[self._embed_with_retry(text) for text in batch]
            )
            all_embeddings.extend(batch_embeddings)
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.05)  # small pause between batches

        return all_embeddings

    async def _embed_with_retry(
        self, text: str, max_retries: int = 3
    ) -> List[float]:
        """Embed with exponential backoff on failure."""
        for attempt in range(max_retries):
            try:
                return await self.embed_single(text)
            except httpx.HTTPStatusError as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to embed after {max_retries} attempts")

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton embedding client
_embedding_client: Optional[OllamaEmbeddingClient] = None


def get_embedding_client() -> OllamaEmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OllamaEmbeddingClient()
    return _embedding_client


# ─────────────────────────────────────────────────────────────────────────────
# Vector Metadata Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chunk_payload(
    chunk: DocumentChunk,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Qdrant payload for a document chunk.
    Metadata discipline: every vector must carry full context
    so it can be filtered precisely during retrieval.
    """
    payload = {
        # ── Source Attribution ────────────────────────────────────────────
        "source": chunk.source_type,
        "source_type": chunk.source_type,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "chunk_db_id": str(chunk.id),
        "chunking_strategy": chunk.chunking_strategy,

        # ── Temporal ─────────────────────────────────────────────────────
        "timestamp": chunk.source_timestamp.isoformat() if chunk.source_timestamp else None,
        "ingested_at": utcnow().isoformat(),

        # ── Entity References (critical for metadata filtering) ───────────
        "person_ids": chunk.person_ids or [],
        "company_ids": chunk.company_ids or [],
        "project_ids": chunk.project_ids or [],

        # ── Position ─────────────────────────────────────────────────────
        "section_title": chunk.section_title,
        "page_number": chunk.page_number,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,

        # ── Content ──────────────────────────────────────────────────────
        "content": chunk.content[:2000],   # truncate for payload; full text in DB
        "content_hash": chunk.content_hash,

        # ── Model ─────────────────────────────────────────────────────────
        "embedding_model": settings.EMBEDDING_MODEL,
    }

    if extra_metadata:
        payload.update(extra_metadata)

    # Remove None values to keep payload clean
    return {k: v for k, v in payload.items() if v is not None}


def build_entity_payload(
    entity_id: UUID,
    entity_type: str,
    canonical_name: str,
    source_ids: Optional[List[str]] = None,
    company_ids: Optional[List[str]] = None,
    project_ids: Optional[List[str]] = None,
    person_ids: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Qdrant payload for an entity embedding (person/company/project).
    """
    payload = {
        "entity_id": str(entity_id),
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "source_ids": source_ids or [],
        "company_ids": company_ids or [],
        "project_ids": project_ids or [],
        "person_ids": person_ids or [],
        "embedding_model": settings.EMBEDDING_MODEL,
        "indexed_at": utcnow().isoformat(),
    }
    if extra:
        payload.update(extra)
    return {k: v for k, v in payload.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Pipeline Service
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Orchestrates the full embedding workflow:
      Content → Chunk (already done) → Embed → Store in Qdrant → Update DB state

    Handles:
    - Batch processing of unembedded chunks
    - Entity embedding (person/company/project summaries)
    - Deduplication via content hash
    - Failure tracking and retry support
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.client = get_embedding_client()

    # ── Document Chunk Embedding ──────────────────────────────────────────────

    async def embed_chunk(self, chunk: DocumentChunk) -> str:
        """
        Embed a single DocumentChunk and store in Qdrant.
        Returns the Qdrant vector ID.
        """
        if chunk.is_embedded and chunk.qdrant_vector_id:
            return chunk.qdrant_vector_id

        embedding = await self.client.embed_single(chunk.content)
        vector_id = str(uuid.uuid4())
        payload = build_chunk_payload(chunk)

        qdrant = await get_qdrant()
        await qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_DOCUMENTS,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        # Update DB state
        chunk.is_embedded = True
        chunk.embedding_model = settings.EMBEDDING_MODEL
        chunk.qdrant_vector_id = vector_id
        chunk.embedded_at = utcnow()
        await self.db.flush()

        return vector_id

    async def embed_pending_chunks(
        self,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
        batch_size: int = 50,
        limit: int = 1000,
    ) -> Dict[str, int]:
        """
        Process all unembedded chunks in batches.
        Returns stats: {"processed": N, "failed": N, "skipped": N}
        """
        stmt = select(DocumentChunk).where(DocumentChunk.is_embedded == False)
        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)
        if source_type:
            stmt = stmt.where(DocumentChunk.source_type == source_type)
        stmt = stmt.limit(limit)

        chunks = (await self.db.execute(stmt)).scalars().all()
        if not chunks:
            return {"processed": 0, "failed": 0, "skipped": 0}

        logger.info(f"Embedding {len(chunks)} pending chunks...")
        processed = failed = skipped = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            try:
                embeddings = await self.client.embed_batch(texts)
                points = []
                for chunk, embedding in zip(batch, embeddings):
                    vector_id = str(uuid.uuid4())
                    payload = build_chunk_payload(chunk)
                    points.append(PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload=payload,
                    ))
                    chunk.is_embedded = True
                    chunk.embedding_model = settings.EMBEDDING_MODEL
                    chunk.qdrant_vector_id = vector_id
                    chunk.embedded_at = utcnow()

                qdrant = await get_qdrant()
                await qdrant.upsert(
                    collection_name=settings.QDRANT_COLLECTION_DOCUMENTS,
                    points=points,
                )
                await self.db.flush()
                processed += len(batch)
                logger.info(
                    f"Embedded batch {i // batch_size + 1}: {len(batch)} chunks"
                )

            except Exception as e:
                failed += len(batch)
                logger.error(f"Batch embedding failed at offset {i}: {e}", exc_info=True)

        return {"processed": processed, "failed": failed, "skipped": skipped}

    # ── Entity Embedding ──────────────────────────────────────────────────────

    async def embed_person(
        self,
        person_id: UUID,
        canonical_name: str,
        bio_text: str,
        company_ids: Optional[List[str]] = None,
        project_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Embed a person entity summary for semantic search.
        The bio_text should summarize who this person is + key context.
        """
        embedding = await self.client.embed_single(bio_text)
        vector_id = str(person_id)  # Use person UUID as vector ID for easy lookup

        payload = build_entity_payload(
            entity_id=person_id,
            entity_type="person",
            canonical_name=canonical_name,
            company_ids=company_ids,
            project_ids=project_ids,
            extra={"bio_summary": bio_text[:500]},
        )

        qdrant = await get_qdrant()
        await qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_PERSONS,
            points=[PointStruct(id=vector_id, vector=embedding, payload=payload)],
        )
        logger.debug(f"Embedded person: {canonical_name}")
        return vector_id

    async def embed_company(
        self,
        company_id: UUID,
        canonical_name: str,
        summary_text: str,
        project_ids: Optional[List[str]] = None,
    ) -> str:
        embedding = await self.client.embed_single(summary_text)
        vector_id = str(company_id)

        payload = build_entity_payload(
            entity_id=company_id,
            entity_type="company",
            canonical_name=canonical_name,
            project_ids=project_ids,
            extra={"summary": summary_text[:500]},
        )

        qdrant = await get_qdrant()
        await qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_COMPANIES,
            points=[PointStruct(id=vector_id, vector=embedding, payload=payload)],
        )
        return vector_id

    async def embed_project(
        self,
        project_id: UUID,
        canonical_name: str,
        summary_text: str,
        company_ids: Optional[List[str]] = None,
        person_ids: Optional[List[str]] = None,
    ) -> str:
        embedding = await self.client.embed_single(summary_text)
        vector_id = str(project_id)

        payload = build_entity_payload(
            entity_id=project_id,
            entity_type="project",
            canonical_name=canonical_name,
            company_ids=company_ids,
            person_ids=person_ids,
            extra={"summary": summary_text[:500]},
        )

        qdrant = await get_qdrant()
        await qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_PROJECTS,
            points=[PointStruct(id=vector_id, vector=embedding, payload=payload)],
        )
        return vector_id

    # ── Semantic Search ───────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional metadata filters.
        This is Phase 4's foundation — but we wire it up now.
        """
        query_vector = await self.client.embed_single(query)
        qdrant = await get_qdrant()

        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_qdrant_filter(filters)

        results = await qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Convert simple key=value filters to Qdrant Filter object.
        Supports: exact match on keyword fields, array contains.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None

    async def delete_document_vectors(self, document_id: str) -> int:
        """Remove all vectors for a document (used on re-ingestion)."""
        qdrant = await get_qdrant()
        result = await qdrant.delete(
            collection_name=settings.QDRANT_COLLECTION_DOCUMENTS,
            points_selector=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            ),
        )
        logger.info(f"Deleted vectors for document: {document_id}")
        return result.operation_id if result else 0

    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Return embedding coverage stats across all collections."""
        qdrant = await get_qdrant()
        stats = {}
        for name in [
            settings.QDRANT_COLLECTION_DOCUMENTS,
            settings.QDRANT_COLLECTION_PERSONS,
            settings.QDRANT_COLLECTION_COMPANIES,
            settings.QDRANT_COLLECTION_PROJECTS,
            settings.QDRANT_COLLECTION_EVENTS,
        ]:
            try:
                info = await qdrant.get_collection(name)
                stats[name] = {
                    "vector_count": info.vectors_count,
                    "indexed_vector_count": info.indexed_vectors_count,
                    "status": info.status,
                }
            except Exception as e:
                stats[name] = {"error": str(e)}

        # DB stats
        stmt = select(DocumentChunk.is_embedded,
                      DocumentChunk.source_type,
                      ).add_columns(
            __import__("sqlalchemy").func.count().label("count")
        ).group_by(DocumentChunk.is_embedded, DocumentChunk.source_type)
        rows = (await self.db.execute(stmt)).all()
        db_stats = {}
        for row in rows:
            key = f"{row.source_type}_{'embedded' if row.is_embedded else 'pending'}"
            db_stats[key] = row.count

        return {"qdrant": stats, "db_chunks": db_stats}
