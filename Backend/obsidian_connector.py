

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import httpx
import structlog
import yaml

from app.connectors.base.connector import BaseConnector, ConnectorConfig, SourceType
from app.connectors.base.registry import register_connector
from app.models.canonical import (
    CanonicalDocument,
    ContentType,
    PersonReference,
    RelationshipReference,
)

logger = structlog.get_logger(__name__)

# Supported attachment extensions in Obsidian vaults
ATTACHMENT_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".ogg",
    ".docx", ".xlsx", ".pptx", ".csv",
}

NOTE_EXTENSIONS = {".md", ".markdown"}
CANVAS_EXTENSIONS = {".canvas"}

# Regex patterns
WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]+)?(?:\|([^\[\]]+))?\]\]")
EMBED_PATTERN = re.compile(r"!\[\[([^\[\]]+)\]\]")
TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z_][a-zA-Z0-9_/-]*)")
CALLOUT_PATTERN = re.compile(r"^> \[!(\w+)\]", re.MULTILINE)
DATAVIEW_INLINE = re.compile(r"\[([^\[\]]+)::\s*([^\[\]]+)\]")
FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@register_connector(SourceType.OBSIDIAN)
class ObsidianConnector(BaseConnector):
    """
    Obsidian vault connector.
    Supports both direct filesystem access and Local REST API plugin.
    """

    SOURCE_TYPE = SourceType.OBSIDIAN

    def __init__(self, config: ConnectorConfig, redis, db):
        super().__init__(config, redis, db)
        self.vault_path: Optional[str] = config.settings.get("vault_path")
        self.api_url: Optional[str] = config.settings.get("api_url")  # Local REST API
        self.api_key: Optional[str] = config.credentials.get("api_key")
        self._link_index: Dict[str, str] = {}  # filename → canonical_id mapping
        self._use_api = bool(self.api_url and self.api_key)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        if self._use_api:
            try:
                await self._api_get("/vault/")
                return True
            except Exception as e:
                logger.error("Obsidian REST API auth failed", error=str(e))
                return False
        elif self.vault_path:
            return os.path.isdir(self.vault_path)
        return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self, checkpoint: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Discover all notes, canvases, and attachments in the vault."""
        if self._use_api:
            async for item in self._discover_via_api(checkpoint):
                yield item
        else:
            async for item in self._discover_via_filesystem(checkpoint):
                yield item

    async def _discover_via_filesystem(
        self,
        checkpoint: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Walk the vault directory and yield file descriptors."""
        vault = Path(self.vault_path)

        # Skip hidden folders and Obsidian config
        skip_dirs = {".obsidian", ".git", ".trash", ".DS_Store", "node_modules"}

        for root, dirs, files in os.walk(vault):
            # Filter out skip dirs in-place (prevents descending into them)
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            root_path = Path(root)
            rel_root = root_path.relative_to(vault)

            for filename in files:
                file_path = root_path / filename
                suffix = file_path.suffix.lower()

                stat = file_path.stat()
                modified_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

                if checkpoint and modified_time < checkpoint:
                    continue

                if suffix in NOTE_EXTENSIONS:
                    yield {
                        "id": str(file_path),
                        "type": "note",
                        "path": str(file_path),
                        "relative_path": str(rel_root / filename),
                        "filename": filename,
                        "stem": file_path.stem,
                        "modified_at": modified_time,
                        "size": stat.st_size,
                    }
                elif suffix in CANVAS_EXTENSIONS:
                    yield {
                        "id": str(file_path),
                        "type": "canvas",
                        "path": str(file_path),
                        "relative_path": str(rel_root / filename),
                        "filename": filename,
                        "stem": file_path.stem,
                        "modified_at": modified_time,
                    }
                elif suffix in ATTACHMENT_EXTENSIONS:
                    yield {
                        "id": str(file_path),
                        "type": "attachment",
                        "path": str(file_path),
                        "relative_path": str(rel_root / filename),
                        "filename": filename,
                        "modified_at": modified_time,
                        "size": stat.st_size,
                    }

    async def _discover_via_api(self, checkpoint: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """Discover files via Obsidian Local REST API."""
        data = await self._api_get("/vault/")
        for file_info in data.get("files", []):
            path = file_info.get("path", "")
            suffix = Path(path).suffix.lower()

            yield {
                "id": path,
                "type": "note" if suffix in NOTE_EXTENSIONS else "attachment",
                "path": path,
                "relative_path": path,
                "filename": Path(path).name,
                "stem": Path(path).stem,
                "modified_at": file_info.get("lastModified"),
            }

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract(self, item_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        item_type = item_descriptor["type"]

        if item_type == "note":
            return await self._extract_note(item_descriptor)
        elif item_type == "canvas":
            return await self._extract_canvas(item_descriptor)
        elif item_type == "attachment":
            return await self._extract_attachment(item_descriptor)
        else:
            return item_descriptor

    async def _extract_note(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        """Read and parse a markdown note file."""
        if self._use_api:
            path = desc["path"]
            data = await self._api_get(f"/vault/{path}")
            content = data.get("content", "")
        else:
            with open(desc["path"], "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        frontmatter, body = self._parse_frontmatter(content)

        return {
            "_type": "note",
            "descriptor": desc,
            "raw_content": content,
            "frontmatter": frontmatter,
            "body": body,
            "wikilinks": self._extract_wikilinks(content),
            "embeds": self._extract_embeds(content),
            "tags": self._extract_tags(content, frontmatter),
            "callouts": self._extract_callouts(content),
            "dataview_fields": self._extract_dataview_fields(content),
        }

    async def _extract_canvas(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a canvas JSON file."""
        if self._use_api:
            data = await self._api_get(f"/vault/{desc['path']}")
            content = data.get("content", "{}")
        else:
            with open(desc["path"], "r", encoding="utf-8") as f:
                content = f.read()

        try:
            canvas_data = json.loads(content)
        except json.JSONDecodeError:
            canvas_data = {}

        return {
            "_type": "canvas",
            "descriptor": desc,
            "canvas": canvas_data,
        }

    async def _extract_attachment(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "_type": "attachment",
            "descriptor": desc,
        }

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    async def normalize(self, raw: Dict[str, Any]) -> CanonicalDocument:
        raw_type = raw["_type"]
        if raw_type == "note":
            return self._normalize_note(raw)
        elif raw_type == "canvas":
            return self._normalize_canvas(raw)
        else:
            return self._normalize_obsidian_attachment(raw)

    def _normalize_note(self, raw: Dict[str, Any]) -> CanonicalDocument:
        desc = raw["descriptor"]
        frontmatter = raw.get("frontmatter", {})
        body = raw.get("body", "")

        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = desc["relative_path"]
        doc.source_type = "obsidian"
        doc.content_type = ContentType.NOTE

        # Title: frontmatter title > filename
        doc.title = (
            frontmatter.get("title")
            or frontmatter.get("aliases", [None])[0]
            or desc.get("stem", "Untitled")
        )

        doc.content = body
        doc.content_raw = {
            "frontmatter": frontmatter,
            "wikilinks": raw.get("wikilinks", []),
            "tags": raw.get("tags", []),
            "dataview_fields": raw.get("dataview_fields", {}),
        }

        # Author from frontmatter
        author_name = frontmatter.get("author") or frontmatter.get("created_by")
        if author_name:
            doc.author = PersonReference(name=str(author_name))

        # Timestamps
        doc.created_at = (
            frontmatter.get("date")
            or frontmatter.get("created")
            or desc.get("modified_at")
        )
        doc.modified_at = (
            frontmatter.get("modified")
            or frontmatter.get("updated")
            or desc.get("modified_at")
        )

        # Folder path
        rel_path = desc.get("relative_path", "")
        doc.folder_path = str(Path(rel_path).parent)

        # Tags
        doc.tags = raw.get("tags", [])

        # Relationships from wikilinks
        for link in raw.get("wikilinks", []):
            doc.relationships.append(RelationshipReference(
                target_id=link["target"],
                relationship_type="wikilink",
                source_type="obsidian_note",
            ))

        # Relationships from embeds
        for embed in raw.get("embeds", []):
            doc.relationships.append(RelationshipReference(
                target_id=embed,
                relationship_type="embeds",
                source_type="obsidian_note",
            ))

        # Additional metadata from frontmatter
        doc.source_metadata = {
            "vault_path": self.vault_path,
            "relative_path": rel_path,
            "frontmatter": frontmatter,
            "wikilinks": raw.get("wikilinks", []),
            "embeds": raw.get("embeds", []),
            "callouts": raw.get("callouts", []),
            "dataview_fields": raw.get("dataview_fields", {}),
            "is_daily_note": self._is_daily_note(desc.get("stem", "")),
            "is_template": "template" in doc.folder_path.lower(),
        }

        doc.compute_content_hash()
        return doc

    def _normalize_canvas(self, raw: Dict[str, Any]) -> CanonicalDocument:
        desc = raw["descriptor"]
        canvas = raw.get("canvas", {})
        nodes = canvas.get("nodes", [])
        edges = canvas.get("edges", [])

        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = desc["relative_path"]
        doc.source_type = "obsidian"
        doc.content_type = ContentType.NOTE  # Canvas = visual note

        doc.title = desc.get("stem", "Canvas")
        doc.content = self._canvas_to_text(nodes, edges)
        doc.modified_at = desc.get("modified_at")

        # Link to referenced notes
        for node in nodes:
            if node.get("type") == "file":
                doc.relationships.append(RelationshipReference(
                    target_id=node.get("file", ""),
                    relationship_type="canvas_references",
                    source_type="obsidian_canvas",
                ))

        doc.source_metadata = {
            "canvas_nodes": len(nodes),
            "canvas_edges": len(edges),
            "canvas_data": canvas,
        }

        doc.compute_content_hash()
        return doc

    def _normalize_obsidian_attachment(self, raw: Dict[str, Any]) -> CanonicalDocument:
        desc = raw["descriptor"]
        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = desc["relative_path"]
        doc.source_type = "obsidian"
        doc.content_type = ContentType.ATTACHMENT
        doc.title = desc.get("filename", "")
        doc.content = ""
        doc.modified_at = desc.get("modified_at")
        doc.folder_path = str(Path(desc.get("relative_path", "")).parent)
        doc.compute_content_hash()
        return doc

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        match = FRONTMATTER_PATTERN.match(content)
        if match:
            try:
                fm = yaml.safe_load(match.group(1)) or {}
                body = content[match.end():]
                return fm, body
            except yaml.YAMLError:
                pass
        return {}, content

    def _extract_wikilinks(self, content: str) -> List[Dict[str, str]]:
        """Extract all [[wikilinks]] with optional aliases."""
        links = []
        for match in WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            alias = match.group(2)
            links.append({"target": target, "alias": alias or target})
        return links

    def _extract_embeds(self, content: str) -> List[str]:
        """Extract all ![[embeds]]."""
        return [m.group(1).strip() for m in EMBED_PATTERN.finditer(content)]

    def _extract_tags(
        self,
        content: str,
        frontmatter: Dict[str, Any],
    ) -> List[str]:
        """Extract tags from both frontmatter and inline."""
        tags: Set[str] = set()
        # Frontmatter tags
        fm_tags = frontmatter.get("tags", [])
        if isinstance(fm_tags, list):
            tags.update(str(t).lower() for t in fm_tags)
        elif isinstance(fm_tags, str):
            tags.update(t.lower() for t in fm_tags.split(","))
        # Inline tags (avoid code blocks)
        for match in TAG_PATTERN.finditer(content):
            tags.add(match.group(1).lower())
        return list(tags)

    def _extract_callouts(self, content: str) -> List[str]:
        return [m.group(1).lower() for m in CALLOUT_PATTERN.finditer(content)]

    def _extract_dataview_fields(self, content: str) -> Dict[str, str]:
        fields = {}
        for match in DATAVIEW_INLINE.finditer(content):
            fields[match.group(1).strip()] = match.group(2).strip()
        return fields

    def _is_daily_note(self, stem: str) -> bool:
        """Heuristic: check if filename looks like a daily note (YYYY-MM-DD)."""
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", stem))

    def _canvas_to_text(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Convert canvas nodes/edges to readable text."""
        lines = ["[Canvas]"]
        for node in nodes:
            if node.get("type") == "text":
                lines.append(node.get("text", ""))
            elif node.get("type") == "file":
                lines.append(f"[Linked: {node.get('file', '')}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Tuple[bool, str]:
        if self._use_api:
            try:
                data = await self._api_get("/vault/")
                return True, f"Connected. {len(data.get('files', []))} files visible."
            except Exception as e:
                return False, str(e)
        elif self.vault_path and os.path.isdir(self.vault_path):
            count = sum(1 for _ in Path(self.vault_path).rglob("*.md"))
            return True, f"Vault accessible. ~{count} notes found."
        return False, "No vault path or API configured"

    # ------------------------------------------------------------------
    # REST API helpers
    # ------------------------------------------------------------------

    async def _api_get(self, path: str) -> Dict[str, Any]:
        url = f"{self.api_url.rstrip('/')}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return response.json()
