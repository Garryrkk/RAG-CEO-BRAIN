

import asyncio
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
import structlog

from app.connectors.base.connector import BaseConnector, ConnectorConfig, SourceType
from app.connectors.base.registry import register_connector
from app.connectors.base.auth import MicrosoftOAuthProvider
from app.models.canonical import (
    AttachmentReference,
    CanonicalDocument,
    ContentType,
    EmailThreadNode,
    PersonReference,
    RelationshipReference,
)
from app.storage.minio_client import MinIOClient

logger = structlog.get_logger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
PAGE_SIZE = 100  # Max allowed by Graph API
ATTACHMENT_MAX_SIZE = 25 * 1024 * 1024  # 25 MB — skip larger inline

# Folders to sync — add more as needed
WELL_KNOWN_FOLDERS = [
    "inbox",
    "sentitems",
    "drafts",
    "archive",
    "deleteditems",
    "junkemail",
    "outbox",
]

# Metadata fields to request from Graph API
MESSAGE_SELECT_FIELDS = ",".join([
    "id", "internetMessageId", "subject", "bodyPreview", "body",
    "from", "toRecipients", "ccRecipients", "bccRecipients",
    "sentDateTime", "receivedDateTime", "createdDateTime", "lastModifiedDateTime",
    "conversationId", "conversationIndex", "parentFolderId",
    "importance", "sensitivity", "isRead", "isDraft",
    "hasAttachments", "attachments", "internetMessageHeaders",
    "flag", "categories", "inferenceClassification",
])


@register_connector(SourceType.OUTLOOK)
class OutlookConnector(BaseConnector):
    """
    Complete Outlook/Exchange connector via Microsoft Graph API.
    """

    SOURCE_TYPE = SourceType.OUTLOOK

    def __init__(self, config: ConnectorConfig, redis, db):
        super().__init__(config, redis, db)
        self.minio = MinIOClient()
        self._auth_provider = MicrosoftOAuthProvider(
            connector_id=config.connector_id,
            redis=redis,
            client_id=config.credentials["client_id"],
            client_secret=config.credentials["client_secret"],
            tenant_id=config.credentials.get("tenant_id", "common"),
            scopes=[
                "https://graph.microsoft.com/Mail.Read",
                "https://graph.microsoft.com/Mail.ReadBasic",
                "offline_access",
            ],
            redirect_uri=config.credentials["redirect_uri"],
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        try:
            await self._auth_provider.get_token()
            return True
        except Exception as e:
            logger.error("Outlook auth failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Discovery — enumerate every email in the mailbox
    # ------------------------------------------------------------------

    async def discover(self, checkpoint: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield email descriptors for every message across all folders.
        checkpoint can be an ISO timestamp (for incremental) or delta token (for delta sync).
        """
        # Check if we have a saved delta token for incremental sync
        delta_token = await self._load_delta_token()

        if delta_token and checkpoint:
            # True incremental: only get changes since last sync
            async for item in self._discover_via_delta(delta_token):
                yield item
        else:
            # Full discovery across all folders
            folders = await self._discover_folders()
            for folder in folders:
                async for item in self._discover_folder_messages(
                    folder_id=folder["id"],
                    folder_name=folder.get("displayName", folder["id"]),
                    since=checkpoint,
                ):
                    yield item

    async def _discover_folders(self) -> List[Dict[str, Any]]:
        """Discover all mail folders including nested custom folders."""
        folders = []

        # Well-known folders first
        for folder_name in WELL_KNOWN_FOLDERS:
            try:
                folder = await self._graph_get(f"/me/mailFolders/{folder_name}")
                folders.append(folder)
            except Exception:
                pass  # Some folders may not exist

        # All other folders
        url = "/me/mailFolders?$top=100&includeHiddenFolders=true"
        while url:
            data = await self._graph_get(url)
            for folder in data.get("value", []):
                if not any(f["id"] == folder["id"] for f in folders):
                    folders.append(folder)
            url = data.get("@odata.nextLink", "").replace(GRAPH_BASE, "") or None

            # Get child folders recursively
            for folder in data.get("value", []):
                child_url = f"/me/mailFolders/{folder['id']}/childFolders?$top=100"
                while child_url:
                    child_data = await self._graph_get(child_url)
                    folders.extend(child_data.get("value", []))
                    child_url = child_data.get("@odata.nextLink", "").replace(GRAPH_BASE, "") or None

        logger.info("Folders discovered", count=len(folders))
        return folders

    async def _discover_folder_messages(
        self,
        folder_id: str,
        folder_name: str,
        since: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Paginate through all messages in a folder."""
        filter_param = ""
        if since:
            # ISO timestamp filter for incremental
            filter_param = f"&$filter=receivedDateTime ge {since}"

        # Order by received date ascending (oldest first, important for thread reconstruction)
        url = (
            f"/me/mailFolders/{folder_id}/messages"
            f"?$top={PAGE_SIZE}&$select={MESSAGE_SELECT_FIELDS}"
            f"&$orderby=receivedDateTime asc"
            f"{filter_param}"
        )

        page_count = 0
        while url:
            data = await self._graph_get(url)
            messages = data.get("value", [])

            for msg in messages:
                yield {
                    "id": msg["id"],
                    "type": "email",
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    "conversation_id": msg.get("conversationId"),
                    "has_attachments": msg.get("hasAttachments", False),
                    "received_at": msg.get("receivedDateTime"),
                    "_raw": msg,  # Pass raw data through to avoid double API call
                }

            next_link = data.get("@odata.nextLink", "")
            url = next_link.replace(GRAPH_BASE, "") if next_link else None
            page_count += 1

            if page_count % 10 == 0:
                logger.info(
                    "Folder page progress",
                    folder=folder_name,
                    page=page_count,
                    messages_so_far=page_count * PAGE_SIZE,
                )

    async def _discover_via_delta(self, delta_token: str) -> AsyncIterator[Dict[str, Any]]:
        """Use Graph delta API for true incremental sync (only changed items)."""
        url = f"/me/mailFolders/inbox/messages/delta?$deltatoken={delta_token}"
        while url:
            data = await self._graph_get(url)
            for msg in data.get("value", []):
                if "@removed" not in msg:
                    yield {
                        "id": msg["id"],
                        "type": "email",
                        "has_attachments": msg.get("hasAttachments", False),
                        "_raw": msg,
                    }
            # New delta token after full traversal
            next_delta = data.get("@odata.deltaLink", "")
            if next_delta:
                await self._save_delta_token(next_delta.split("deltatoken=")[-1])
            url = data.get("@odata.nextLink", "").replace(GRAPH_BASE, "") or None

    # ------------------------------------------------------------------
    # Extraction — fetch full email data
    # ------------------------------------------------------------------

    async def extract(self, item_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract complete email data.
        If the raw data was passed in discovery, use it directly to avoid extra API calls.
        """
        if "_raw" in item_descriptor:
            raw = item_descriptor["_raw"]
        else:
            raw = await self._graph_get(
                f"/me/messages/{item_descriptor['id']}"
                f"?$select={MESSAGE_SELECT_FIELDS}&$expand=attachments"
            )

        # Fetch attachments metadata if needed
        if raw.get("hasAttachments") and "attachments" not in raw:
            attachments_data = await self._graph_get(
                f"/me/messages/{raw['id']}/attachments"
            )
            raw["attachments"] = attachments_data.get("value", [])

        raw["_folder_name"] = item_descriptor.get("folder_name", "unknown")
        return raw

    # ------------------------------------------------------------------
    # Normalization — raw Graph API response → CanonicalDocument
    # ------------------------------------------------------------------

    async def normalize(self, raw: Dict[str, Any]) -> CanonicalDocument:
        """Transform Microsoft Graph email into canonical format."""
        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = raw["id"]
        doc.source_type = SourceType.OUTLOOK.value
        doc.content_type = ContentType.EMAIL

        # Title / subject
        doc.title = raw.get("subject", "(No Subject)") or "(No Subject)"

        # Content — prefer text body, fall back to HTML
        body = raw.get("body", {})
        if body.get("contentType", "").lower() == "text":
            doc.content = body.get("content", "")
        else:
            doc.content = self._strip_html(body.get("content", ""))
            doc.content_html = body.get("content")

        # Author
        from_recipient = raw.get("from", {}).get("emailAddress", {})
        doc.author = PersonReference(
            email=from_recipient.get("address"),
            name=from_recipient.get("name"),
        )

        # Recipients
        doc.recipients = [
            PersonReference(
                email=r["emailAddress"]["address"],
                name=r["emailAddress"].get("name"),
            )
            for r in raw.get("toRecipients", [])
            if "emailAddress" in r
        ]
        doc.cc = [
            PersonReference(
                email=r["emailAddress"]["address"],
                name=r["emailAddress"].get("name"),
            )
            for r in raw.get("ccRecipients", [])
            if "emailAddress" in r
        ]
        doc.bcc = [
            PersonReference(
                email=r["emailAddress"]["address"],
                name=r["emailAddress"].get("name"),
            )
            for r in raw.get("bccRecipients", [])
            if "emailAddress" in r
        ]
        doc.participants = list({
            p.email: p
            for p in ([doc.author] if doc.author else []) + doc.recipients + doc.cc
            if p.email
        }.values())

        # Timestamps
        doc.created_at = raw.get("sentDateTime") or raw.get("receivedDateTime")
        doc.modified_at = raw.get("lastModifiedDateTime")

        # Thread information
        doc.thread_id = raw.get("conversationId")
        doc.source_metadata = {
            "internet_message_id": raw.get("internetMessageId"),
            "conversation_id": raw.get("conversationId"),
            "conversation_index": raw.get("conversationIndex"),
            "parent_folder_id": raw.get("parentFolderId"),
            "folder_name": raw.get("_folder_name", "unknown"),
            "importance": raw.get("importance", "normal"),
            "sensitivity": raw.get("sensitivity", "normal"),
            "is_read": raw.get("isRead", False),
            "is_draft": raw.get("isDraft", False),
            "categories": raw.get("categories", []),
            "flag": raw.get("flag", {}),
            "inference_classification": raw.get("inferenceClassification"),
            "internet_headers": self._extract_internet_headers(raw),
        }
        doc.folder_path = raw.get("_folder_name", "unknown")

        # Attachments
        doc.attachments = await self._process_attachments(
            message_id=raw["id"],
            raw_attachments=raw.get("attachments", []),
            canonical_id=doc.id,
        )

        # Compute content hash for deduplication
        doc.compute_content_hash()

        return doc

    # ------------------------------------------------------------------
    # Thread reconstruction
    # ------------------------------------------------------------------

    async def reconstruct_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Reconstruct the full conversation tree for an email thread.
        Returns a tree structure with parent/child relationships.
        This is called post-sync when building the knowledge layer.
        """
        # Fetch all messages in conversation
        url = (
            f"/me/messages?$filter=conversationId eq '{thread_id}'"
            f"&$select=id,internetMessageId,subject,from,sentDateTime,conversationIndex"
            f"&$orderby=sentDateTime asc&$top=250"
        )
        data = await self._graph_get(url)
        messages = data.get("value", [])

        if not messages:
            return None

        # Build thread tree using conversationIndex
        # The conversationIndex bytes encode parent-child relationships
        nodes: Dict[str, EmailThreadNode] = {}
        for msg in messages:
            node = EmailThreadNode(
                message_id=msg.get("internetMessageId", msg["id"]),
                canonical_id=msg["id"],  # Will be updated to canonical UUID later
                subject=msg.get("subject", ""),
                sender=PersonReference(
                    email=msg.get("from", {}).get("emailAddress", {}).get("address"),
                    name=msg.get("from", {}).get("emailAddress", {}).get("name"),
                ),
                timestamp=msg.get("sentDateTime", ""),
            )
            nodes[msg["id"]] = node

        # Simple thread tree: first message is root, rest are children
        # (Full conversation index parsing is complex; this is production-ready approximation)
        messages_sorted = sorted(messages, key=lambda m: m.get("sentDateTime", ""))
        if not messages_sorted:
            return None

        root = nodes[messages_sorted[0]["id"]]
        for msg in messages_sorted[1:]:
            root.children.append(nodes[msg["id"]])

        return root.to_dict()

    # ------------------------------------------------------------------
    # Attachment handling
    # ------------------------------------------------------------------

    async def _process_attachments(
        self,
        message_id: str,
        raw_attachments: List[Dict[str, Any]],
        canonical_id: str,
    ) -> List[AttachmentReference]:
        """
        Download and store all attachments.
        Queues them for content extraction automatically.
        """
        refs = []
        for attachment in raw_attachments:
            if attachment.get("@odata.type") == "#microsoft.graph.itemAttachment":
                # Embedded item (another email/event) — skip binary download
                continue

            attachment_id = str(uuid.uuid4())
            filename = attachment.get("name", f"attachment_{attachment_id}")
            content_type = attachment.get("contentType", "application/octet-stream")
            size_bytes = attachment.get("size", 0)

            ref = AttachmentReference(
                attachment_id=attachment_id,
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                storage_path=f"attachments/{canonical_id}/{attachment_id}/{filename}",
                processing_status="pending",
            )

            if size_bytes > ATTACHMENT_MAX_SIZE:
                logger.warning(
                    "Attachment too large, skipping download",
                    filename=filename,
                    size_mb=size_bytes / 1024 / 1024,
                )
                ref.processing_status = "skipped_too_large"
                refs.append(ref)
                continue

            # Download and store in MinIO
            try:
                if "contentBytes" in attachment:
                    # Inline attachment
                    import base64
                    content = base64.b64decode(attachment["contentBytes"])
                else:
                    # Fetch attachment bytes from Graph
                    content = await self._download_attachment(message_id, attachment["id"])

                await self.minio.put_bytes(
                    bucket="attachments",
                    key=ref.storage_path,
                    data=content,
                    content_type=content_type,
                )
                ref.downloaded = True
                ref.processing_status = "queued"

                # Queue for content extraction (Task 5 — File Processing Pipeline)
                await self._queue_attachment_processing(ref)

            except Exception as e:
                logger.error("Attachment download failed", filename=filename, error=str(e))
                ref.processing_status = "failed"

            refs.append(ref)
        return refs

    async def _download_attachment(self, message_id: str, attachment_id: str) -> bytes:
        """Download attachment content from Graph API."""
        headers = await self._auth_provider.get_headers()
        url = f"{GRAPH_BASE}/me/messages/{message_id}/attachments/{attachment_id}/$value"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.content

    async def _queue_attachment_processing(self, ref: AttachmentReference) -> None:
        """Enqueue attachment for content extraction."""
        from app.tasks.file_processing import process_attachment
        process_attachment.delay(ref.to_dict())

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Tuple[bool, str]:
        try:
            profile = await self._graph_get("/me?$select=displayName,mail")
            return True, f"Connected as {profile.get('mail', profile.get('displayName', 'unknown'))}"
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # Graph API helpers
    # ------------------------------------------------------------------

    async def _graph_get(self, path: str) -> Dict[str, Any]:
        """Make authenticated GET request to Microsoft Graph."""
        headers = await self._auth_provider.get_headers()
        url = f"{GRAPH_BASE}{path}" if path.startswith("/") else path
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 429:
                # Rate limited — respect Retry-After header
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning("Graph API rate limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
                return await self._graph_get(path)
            response.raise_for_status()
            return response.json()

    def _strip_html(self, html: str) -> str:
        """Basic HTML stripping. For production, use html2text or bleach."""
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def _extract_internet_headers(self, raw: Dict[str, Any]) -> Dict[str, str]:
        """Extract internet message headers for forensic metadata."""
        headers = {}
        for header in raw.get("internetMessageHeaders", []):
            name = header.get("name", "").lower()
            if name in {"message-id", "in-reply-to", "references", "x-mailer", "x-originating-ip"}:
                headers[name] = header.get("value", "")
        return headers

    async def _save_delta_token(self, token: str) -> None:
        await self.redis.set(
            f"outlook:delta_token:{self.connector_id}",
            token,
            ex=86400 * 30,
        )

    async def _load_delta_token(self) -> Optional[str]:
        val = await self.redis.get(f"outlook:delta_token:{self.connector_id}")
        return val.decode() if val else None
