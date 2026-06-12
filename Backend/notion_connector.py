
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import structlog

from app.connectors.base.connector import BaseConnector, ConnectorConfig, SourceType
from app.connectors.base.registry import register_connector
from app.connectors.base.auth import APIKeyProvider
from app.models.canonical import (
    CanonicalDocument,
    ContentType,
    PersonReference,
    RelationshipReference,
)

logger = structlog.get_logger(__name__)

NOTION_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
PAGE_SIZE = 100


@register_connector(SourceType.NOTION)
class NotionConnector(BaseConnector):
    """
    Complete Notion workspace connector.
    Preserves structural relationships — does not flatten everything to text.
    """

    SOURCE_TYPE = SourceType.NOTION

    def __init__(self, config: ConnectorConfig, redis, db):
        super().__init__(config, redis, db)
        self._token = config.credentials["integration_token"]

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        try:
            await self._notion_get("/users/me")
            return True
        except Exception as e:
            logger.error("Notion auth failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self, checkpoint: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Discover all pages and databases accessible to the integration."""
        # Search for all pages
        async for item in self._search_all(filter_type="page", since=checkpoint):
            yield item

        # Search for all databases
        async for item in self._search_all(filter_type="database", since=checkpoint):
            yield item

    async def _search_all(
        self,
        filter_type: str,
        since: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Paginate through all accessible items using Notion search API."""
        body: Dict[str, Any] = {
            "filter": {"value": filter_type, "property": "object"},
            "page_size": PAGE_SIZE,
            "sort": {"direction": "ascending", "timestamp": "last_edited_time"},
        }

        if since:
            # Notion search doesn't support date filter directly; filter post-fetch
            pass

        cursor = None
        while True:
            if cursor:
                body["start_cursor"] = cursor

            data = await self._notion_post("/search", body)
            for item in data.get("results", []):
                last_edited = item.get("last_edited_time", "")
                if since and last_edited and last_edited < since:
                    continue
                yield {
                    "id": item["id"],
                    "type": item["object"],  # "page" or "database"
                    "last_edited": last_edited,
                    "parent": item.get("parent", {}),
                    "_raw": item,
                }

            next_cursor = data.get("next_cursor")
            if not data.get("has_more") or not next_cursor:
                break
            cursor = next_cursor

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract(self, item_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        item_id = item_descriptor["id"]
        item_type = item_descriptor["type"]

        if item_type == "database":
            return await self._extract_database(item_id, item_descriptor.get("_raw", {}))
        else:
            return await self._extract_page(item_id, item_descriptor.get("_raw", {}))

    async def _extract_page(self, page_id: str, raw_page: Dict[str, Any]) -> Dict[str, Any]:
        """Extract full page content including all blocks."""
        page = raw_page or await self._notion_get(f"/pages/{page_id}")
        blocks = await self._fetch_blocks(page_id)
        comments = await self._fetch_comments(page_id)

        return {
            "_type": "page",
            "page": page,
            "blocks": blocks,
            "comments": comments,
        }

    async def _extract_database(self, db_id: str, raw_db: Dict[str, Any]) -> Dict[str, Any]:
        """Extract database schema and all records."""
        database = raw_db or await self._notion_get(f"/databases/{db_id}")
        records = await self._fetch_database_records(db_id)

        return {
            "_type": "database",
            "database": database,
            "records": records,
        }

    async def _fetch_blocks(self, block_id: str, depth: int = 0) -> List[Dict[str, Any]]:
        """Recursively fetch all blocks (content) of a page."""
        if depth > 5:  # Prevent infinite recursion on deeply nested pages
            return []

        blocks = []
        cursor = None
        while True:
            params = f"?page_size={PAGE_SIZE}"
            if cursor:
                params += f"&start_cursor={cursor}"
            data = await self._notion_get(f"/blocks/{block_id}/children{params}")

            for block in data.get("results", []):
                blocks.append(block)
                # Recursively fetch children
                if block.get("has_children"):
                    block["children"] = await self._fetch_blocks(block["id"], depth + 1)

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        return blocks

    async def _fetch_database_records(self, db_id: str) -> List[Dict[str, Any]]:
        """Fetch all records from a database with full properties."""
        records = []
        cursor = None
        while True:
            body: Dict[str, Any] = {"page_size": PAGE_SIZE}
            if cursor:
                body["start_cursor"] = cursor

            data = await self._notion_post(f"/databases/{db_id}/query", body)
            records.extend(data.get("results", []))

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        return records

    async def _fetch_comments(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetch all comments on a page."""
        try:
            data = await self._notion_get(f"/comments?block_id={page_id}&page_size=100")
            return data.get("results", [])
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    async def normalize(self, raw: Dict[str, Any]) -> CanonicalDocument:
        if raw["_type"] == "database":
            return await self._normalize_database(raw)
        else:
            return await self._normalize_page(raw)

    async def _normalize_page(self, raw: Dict[str, Any]) -> CanonicalDocument:
        """Convert a Notion page to canonical format."""
        page = raw["page"]
        blocks = raw.get("blocks", [])
        comments = raw.get("comments", [])

        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = page["id"].replace("-", "")
        doc.source_type = SourceType.NOTION.value
        doc.content_type = ContentType.PAGE

        # Title from properties
        doc.title = self._extract_page_title(page)

        # Content from blocks
        doc.content = self._blocks_to_text(blocks)
        doc.content_raw = {"blocks": blocks, "properties": page.get("properties", {})}

        # Author
        created_by = page.get("created_by", {})
        doc.author = PersonReference(user_id=created_by.get("id"))

        # Timestamps
        doc.created_at = page.get("created_time")
        doc.modified_at = page.get("last_edited_time")

        # Relationships — preserve hierarchy
        parent = page.get("parent", {})
        parent_type = parent.get("type")
        if parent_type == "page_id":
            doc.parent_id = parent["page_id"]
            doc.relationships.append(RelationshipReference(
                target_id=parent["page_id"],
                relationship_type="child_of",
                source_type="notion_page",
            ))
        elif parent_type == "database_id":
            doc.parent_id = parent["database_id"]
            doc.relationships.append(RelationshipReference(
                target_id=parent["database_id"],
                relationship_type="record_in",
                source_type="notion_database",
            ))

        # Extract relation properties
        for prop_name, prop_value in page.get("properties", {}).items():
            if prop_value.get("type") == "relation":
                for rel in prop_value.get("relation", []):
                    doc.relationships.append(RelationshipReference(
                        target_id=rel["id"],
                        relationship_type=f"relates_to:{prop_name}",
                        source_type="notion_page",
                    ))

        # Comments
        if comments:
            comment_texts = []
            for comment in comments:
                text = self._rich_text_to_str(comment.get("rich_text", []))
                if text:
                    comment_texts.append(f"[Comment by {comment.get('created_by', {}).get('id', 'unknown')}]: {text}")
            if comment_texts:
                doc.content += "\n\n--- Comments ---\n" + "\n".join(comment_texts)

        # Metadata
        doc.source_metadata = {
            "notion_id": page["id"],
            "url": page.get("url"),
            "cover": page.get("cover"),
            "icon": page.get("icon"),
            "archived": page.get("archived", False),
            "properties": self._extract_all_properties(page.get("properties", {})),
        }

        doc.compute_content_hash()
        return doc

    async def _normalize_database(self, raw: Dict[str, Any]) -> CanonicalDocument:
        """Convert a Notion database to canonical format."""
        database = raw["database"]
        records = raw.get("records", [])

        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = database["id"].replace("-", "")
        doc.source_type = SourceType.NOTION.value
        doc.content_type = ContentType.DATABASE_RECORD

        doc.title = self._rich_text_to_str(database.get("title", []))
        doc.content = f"Database: {doc.title}\n\nSchema:\n{self._db_schema_to_text(database)}\n\nRecords: {len(records)}"

        doc.created_at = database.get("created_time")
        doc.modified_at = database.get("last_edited_time")

        doc.source_metadata = {
            "notion_id": database["id"],
            "url": database.get("url"),
            "schema": database.get("properties", {}),
            "record_count": len(records),
        }

        # Store record summaries
        record_summaries = []
        for record in records[:100]:  # Cap at 100 for the parent doc
            record_summaries.append({
                "id": record["id"],
                "title": self._extract_page_title(record),
                "last_edited": record.get("last_edited_time"),
            })
        doc.content_raw = {"schema": database.get("properties", {}), "record_summaries": record_summaries}

        doc.compute_content_hash()
        return doc

    # ------------------------------------------------------------------
    # Block content extraction
    # ------------------------------------------------------------------

    def _blocks_to_text(self, blocks: List[Dict[str, Any]], indent: int = 0) -> str:
        """Convert Notion blocks to plain text, preserving structure."""
        lines = []
        prefix = "  " * indent

        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            if block_type in ("paragraph", "quote", "callout"):
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                if text:
                    lines.append(f"{prefix}{text}")

            elif block_type.startswith("heading_"):
                level = int(block_type[-1])
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                lines.append(f"{prefix}{'#' * level} {text}")

            elif block_type in ("bulleted_list_item", "numbered_list_item", "to_do"):
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                checked = block_data.get("checked", False)
                prefix_sym = "- [x]" if checked else "- [ ]" if block_type == "to_do" else "-"
                lines.append(f"{prefix}{prefix_sym} {text}")

            elif block_type == "toggle":
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                lines.append(f"{prefix}▶ {text}")

            elif block_type == "code":
                lang = block_data.get("language", "")
                text = self._rich_text_to_str(block_data.get("rich_text", []))
                lines.append(f"{prefix}```{lang}\n{text}\n```")

            elif block_type == "table":
                lines.append(f"{prefix}[Table]")

            elif block_type == "table_row":
                cells = block_data.get("cells", [])
                row = " | ".join(self._rich_text_to_str(cell) for cell in cells)
                lines.append(f"{prefix}{row}")

            elif block_type == "divider":
                lines.append(f"{prefix}---")

            elif block_type in ("image", "file", "pdf"):
                caption = self._rich_text_to_str(block_data.get("caption", []))
                lines.append(f"{prefix}[{block_type.upper()}: {caption}]")

            elif block_type == "bookmark":
                url = block_data.get("url", "")
                lines.append(f"{prefix}[Bookmark: {url}]")

            # Recursively process children
            children = block.get("children", [])
            if children:
                lines.append(self._blocks_to_text(children, indent + 1))

        return "\n".join(filter(None, lines))

    def _rich_text_to_str(self, rich_text: List[Dict[str, Any]]) -> str:
        return "".join(rt.get("plain_text", "") for rt in rich_text)

    def _extract_page_title(self, page: Dict[str, Any]) -> str:
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                return self._rich_text_to_str(prop.get("title", []))
        return "Untitled"

    def _extract_all_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all property values in a simplified format."""
        result = {}
        for name, prop in properties.items():
            ptype = prop.get("type", "")
            try:
                if ptype == "title":
                    result[name] = self._rich_text_to_str(prop.get("title", []))
                elif ptype == "rich_text":
                    result[name] = self._rich_text_to_str(prop.get("rich_text", []))
                elif ptype in ("select", "status"):
                    sel = prop.get(ptype, {}) or {}
                    result[name] = sel.get("name")
                elif ptype == "multi_select":
                    result[name] = [s.get("name") for s in prop.get("multi_select", [])]
                elif ptype == "date":
                    date_val = prop.get("date") or {}
                    result[name] = date_val.get("start")
                elif ptype == "checkbox":
                    result[name] = prop.get("checkbox")
                elif ptype in ("number", "url", "email", "phone_number"):
                    result[name] = prop.get(ptype)
                elif ptype == "people":
                    result[name] = [p.get("id") for p in prop.get("people", [])]
                elif ptype == "relation":
                    result[name] = [r.get("id") for r in prop.get("relation", [])]
                elif ptype == "formula":
                    formula = prop.get("formula", {})
                    result[name] = formula.get(formula.get("type", "string"), None)
            except Exception:
                result[name] = None
        return result

    def _db_schema_to_text(self, database: Dict[str, Any]) -> str:
        lines = []
        for name, prop in database.get("properties", {}).items():
            lines.append(f"- {name} ({prop.get('type', 'unknown')})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Tuple[bool, str]:
        try:
            user = await self._notion_get("/users/me")
            name = user.get("name") or user.get("id", "unknown")
            return True, f"Connected as {name}"
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    async def _notion_get(self, path: str) -> Dict[str, Any]:
        url = f"{NOTION_BASE}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self._headers())
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                await asyncio.sleep(retry_after)
                return await self._notion_get(path)
            response.raise_for_status()
            return response.json()

    async def _notion_post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{NOTION_BASE}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=self._headers(), json=body)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                await asyncio.sleep(retry_after)
                return await self._notion_post(path, body)
            response.raise_for_status()
            return response.json()
