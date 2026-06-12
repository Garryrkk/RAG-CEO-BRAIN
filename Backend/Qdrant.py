

from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

from app.core.config import settings
from app.core.logging import logger


_qdrant_client: Optional[AsyncQdrantClient] = None


async def get_qdrant() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
    return _qdrant_client


async def init_collections():
    """
    Initialize all Qdrant collections required by Phase 3.
    Idempotent — safe to call on every startup.
    """
    client = await get_qdrant()

    collections = {
        settings.QDRANT_COLLECTION_DOCUMENTS: {
            "description": "Chunked document embeddings with full metadata"
        },
        settings.QDRANT_COLLECTION_PERSONS: {
            "description": "Person entity embeddings"
        },
        settings.QDRANT_COLLECTION_COMPANIES: {
            "description": "Company entity embeddings"
        },
        settings.QDRANT_COLLECTION_PROJECTS: {
            "description": "Project entity embeddings"
        },
        settings.QDRANT_COLLECTION_EVENTS: {
            "description": "Timeline event embeddings"
        },
    }

    existing = {c.name for c in (await client.get_collections()).collections}

    for name, meta in collections.items():
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for fast metadata filtering
            await _create_payload_indexes(client, name)
            logger.info(f"Created Qdrant collection: {name} — {meta['description']}")
        else:
            logger.info(f"Qdrant collection already exists: {name}")


async def _create_payload_indexes(client: AsyncQdrantClient, collection_name: str):
    """Create keyword and text indexes on common metadata fields."""
    keyword_fields = [
        "source_type",
        "entity_type",
        "company_id",
        "person_id",
        "project_id",
        "document_id",
        "conversation_id",
    ]
    for field in keyword_fields:
        try:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Index may already exist

    # Text index for full-text search on content
    try:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="content",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                lowercase=True,
            ),
        )
    except Exception:
        pass
