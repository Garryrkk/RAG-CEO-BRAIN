
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Optional
from uuid import UUID, uuid4

import httpx

from app.pipeline.canonical import CanonicalDocument

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Connector State
# ─────────────────────────────────────────────────────────────────────────────

class ConnectorStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    AUTH_EXPIRED = "auth_expired"
    NOT_CONFIGURED = "not_configured"


@dataclass
class ConnectorCredentials:
    """Credentials for a connector. Stored encrypted, loaded at runtime."""
    connector_id: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None             # Microsoft
    workspace_id: Optional[str] = None          # Slack
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.token_expires_at is None:
            return False
        return datetime.utcnow() >= self.token_expires_at

    @property
    def needs_refresh(self) -> bool:
        if self.token_expires_at is None:
            return False
        return datetime.utcnow() >= self.token_expires_at - timedelta(minutes=5)


@dataclass
class SyncState:
    """Tracks incremental sync position for a connector."""
    connector_id: str
    last_sync_at: Optional[datetime] = None
    last_delta_token: Optional[str] = None     # Microsoft delta tokens
    last_cursor: Optional[str] = None          # Slack cursors
    last_page_token: Optional[str] = None      # Google Drive
    synced_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class ConnectorResult:
    """Result of a connector fetch operation."""
    connector_id: str
    documents: list[CanonicalDocument]
    next_sync_state: SyncState
    has_more: bool = False
    error: Optional[str] = None
    fetched_count: int = 0
    skipped_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Base Connector
# ─────────────────────────────────────────────────────────────────────────────

class BaseConnector(ABC):
    """
    Every connector inherits from this base.

    The contract:
      - authenticate() → verify/refresh credentials
      - fetch() → yield CanonicalDocument objects
      - sync_incremental() → fetch only new/changed items
      - health_check() → verify the connection is live

    The framework handles:
      - Retry with backoff
      - Rate limit detection and pause
      - Error state tracking
      - Incremental sync coordination
    """

    CONNECTOR_ID: str = "base"
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 2.0
    RATE_LIMIT_PAUSE_SECONDS: int = 60

    def __init__(self, credentials: ConnectorCredentials):
        self.credentials = credentials
        self.status = ConnectorStatus.ACTIVE
        self._http: Optional[httpx.AsyncClient] = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=30.0,
                headers=self._default_headers(),
            )
        return self._http

    def _default_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.credentials.access_token:
            headers["Authorization"] = f"Bearer {self.credentials.access_token}"
        return headers

    # ── Must implement ─────────────────────────────────────────────────────────

    @abstractmethod
    async def authenticate(self) -> bool:
        """Verify and refresh credentials. Returns True if auth is valid."""

    @abstractmethod
    async def fetch_all(
        self,
        since: Optional[datetime] = None,
    ) -> AsyncIterator[CanonicalDocument]:
        """Fetch all documents (or since a given datetime). Yields CanonicalDocuments."""

    @abstractmethod
    async def fetch_incremental(
        self,
        sync_state: SyncState,
    ) -> ConnectorResult:
        """Fetch only new/changed items since last sync. Uses delta tokens or cursors."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the connector is live and credentials are valid."""

    # ── Framework methods ──────────────────────────────────────────────────────

    async def _get_with_retry(self, url: str, params: Optional[dict] = None) -> dict[str, Any]:
        """GET request with retry and rate-limit handling."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.http.get(url, params=params or {})

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.RATE_LIMIT_PAUSE_SECONDS))
                    logger.warning(f"{self.CONNECTOR_ID}: rate limited, pausing {retry_after}s")
                    self.status = ConnectorStatus.RATE_LIMITED
                    await asyncio.sleep(retry_after)
                    self.status = ConnectorStatus.ACTIVE
                    continue

                if response.status_code == 401:
                    logger.info(f"{self.CONNECTOR_ID}: token expired, refreshing")
                    refreshed = await self._refresh_token()
                    if not refreshed:
                        self.status = ConnectorStatus.AUTH_EXPIRED
                        raise ConnectionError(f"{self.CONNECTOR_ID}: auth refresh failed")
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"{self.CONNECTOR_ID}: HTTP {e.response.status_code} on {url}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self.RETRY_DELAY_SECONDS * (2 ** attempt))

            except Exception as e:
                logger.error(f"{self.CONNECTOR_ID}: request failed: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(self.RETRY_DELAY_SECONDS * (2 ** attempt))

        raise RuntimeError(f"{self.CONNECTOR_ID}: max retries exceeded for {url}")

    async def _refresh_token(self) -> bool:
        """Override in OAuth connectors to implement token refresh."""
        return False

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTOR: Microsoft Outlook
# ─────────────────────────────────────────────────────────────────────────────

class OutlookConnector(BaseConnector):
    """
    Microsoft Outlook Email connector via Microsoft Graph API.

    Authentication: OAuth2 with refresh token (MSAL)
    Incremental sync: Delta query tokens
    Data: Emails, calendar events, meeting invites
    """

    CONNECTOR_ID = "outlook"
    GRAPH_BASE = "https://graph.microsoft.com/v1.0"
    MAIL_FOLDERS = ["inbox", "sentItems"]
    PAGE_SIZE = 50

    async def authenticate(self) -> bool:
        if self.credentials.needs_refresh:
            return await self._refresh_token()
        return bool(self.credentials.access_token)

    async def _refresh_token(self) -> bool:
        try:
            response = await self.http.post(
                f"https://login.microsoftonline.com/{self.credentials.tenant_id}/oauth2/v2.0/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.credentials.refresh_token,
                    "client_id": self.credentials.client_id,
                    "client_secret": self.credentials.client_secret,
                    "scope": "https://graph.microsoft.com/.default",
                },
            )
            response.raise_for_status()
            token_data = response.json()
            self.credentials.access_token = token_data["access_token"]
            self.credentials.token_expires_at = datetime.utcnow() + timedelta(
                seconds=token_data.get("expires_in", 3600)
            )
            self._http = None  # Force header refresh
            return True
        except Exception as e:
            logger.error(f"Outlook token refresh failed: {e}")
            return False

    async def fetch_all(
        self,
        since: Optional[datetime] = None,
    ) -> AsyncIterator[CanonicalDocument]:
        from app.pipeline.canonical import OutlookNormalizer
        normalizer = OutlookNormalizer()

        for folder in self.MAIL_FOLDERS:
            url = f"{self.GRAPH_BASE}/me/mailFolders/{folder}/messages"
            params = {
                "$top": self.PAGE_SIZE,
                "$select": "id,subject,from,toRecipients,ccRecipients,sentDateTime,body,webLink,conversationId,hasAttachments,importance,internetMessageId",
                "$orderby": "sentDateTime desc",
            }
            if since:
                params["$filter"] = f"sentDateTime ge {since.isoformat()}Z"

            while url:
                data = await self._get_with_retry(url, params if "graph.microsoft.com" in url else {})
                for raw_email in data.get("value", []):
                    yield normalizer.normalize(raw_email)
                url = data.get("@odata.nextLink")

    async def fetch_incremental(self, sync_state: SyncState) -> ConnectorResult:
        from app.pipeline.canonical import OutlookNormalizer
        normalizer = OutlookNormalizer()
        documents = []

        for folder in self.MAIL_FOLDERS:
            if sync_state.last_delta_token:
                url = sync_state.last_delta_token
            else:
                url = f"{self.GRAPH_BASE}/me/mailFolders/{folder}/messages/delta"

            params = {"$top": self.PAGE_SIZE}
            while True:
                data = await self._get_with_retry(url, params if url.endswith("delta") else {})
                for raw_email in data.get("value", []):
                    documents.append(normalizer.normalize(raw_email))

                if "@odata.deltaLink" in data:
                    sync_state.last_delta_token = data["@odata.deltaLink"]
                    break
                url = data.get("@odata.nextLink", "")
                if not url:
                    break

        sync_state.last_sync_at = datetime.utcnow()
        sync_state.synced_count += len(documents)
        return ConnectorResult(
            connector_id=self.CONNECTOR_ID,
            documents=documents,
            next_sync_state=sync_state,
            fetched_count=len(documents),
        )

    async def health_check(self) -> bool:
        try:
            await self._get_with_retry(f"{self.GRAPH_BASE}/me")
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTOR: Microsoft Teams
# ─────────────────────────────────────────────────────────────────────────────

class TeamsConnector(BaseConnector):
    """
    Microsoft Teams connector via Microsoft Graph API.
    Fetches channel messages and direct messages.
    """

    CONNECTOR_ID = "teams"
    GRAPH_BASE = "https://graph.microsoft.com/v1.0"

    async def authenticate(self) -> bool:
        if self.credentials.needs_refresh:
            return await self._refresh_token()
        return bool(self.credentials.access_token)

    async def _refresh_token(self) -> bool:
        # Same MSAL flow as Outlook
        try:
            response = await self.http.post(
                f"https://login.microsoftonline.com/{self.credentials.tenant_id}/oauth2/v2.0/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.credentials.refresh_token,
                    "client_id": self.credentials.client_id,
                    "client_secret": self.credentials.client_secret,
                    "scope": "https://graph.microsoft.com/.default",
                },
            )
            response.raise_for_status()
            token_data = response.json()
            self.credentials.access_token = token_data["access_token"]
            self.credentials.token_expires_at = datetime.utcnow() + timedelta(
                seconds=token_data.get("expires_in", 3600)
            )
            self._http = None
            return True
        except Exception as e:
            logger.error(f"Teams token refresh failed: {e}")
            return False

    async def fetch_all(self, since: Optional[datetime] = None) -> AsyncIterator[CanonicalDocument]:
        from app.pipeline.canonical import TeamsNormalizer
        normalizer = TeamsNormalizer()

        teams_data = await self._get_with_retry(f"{self.GRAPH_BASE}/me/joinedTeams")
        for team in teams_data.get("value", []):
            team_id = team["id"]
            channels_data = await self._get_with_retry(
                f"{self.GRAPH_BASE}/teams/{team_id}/channels"
            )
            for channel in channels_data.get("value", []):
                channel_id = channel["id"]
                url = f"{self.GRAPH_BASE}/teams/{team_id}/channels/{channel_id}/messages"
                params = {"$top": 50}
                if since:
                    params["$filter"] = f"lastModifiedDateTime ge {since.isoformat()}Z"

                while url:
                    data = await self._get_with_retry(url, params if "channels" in url else {})
                    for msg in data.get("value", []):
                        doc = normalizer.normalize(msg)
                        doc.meta["team_id"] = team_id
                        doc.meta["team_name"] = team.get("displayName", "")
                        doc.meta["channel_name"] = channel.get("displayName", "")
                        yield doc
                    url = data.get("@odata.nextLink", "")

    async def fetch_incremental(self, sync_state: SyncState) -> ConnectorResult:
        documents = []
        since = sync_state.last_sync_at or (datetime.utcnow() - timedelta(days=7))
        async for doc in self.fetch_all(since=since):
            documents.append(doc)
        sync_state.last_sync_at = datetime.utcnow()
        sync_state.synced_count += len(documents)
        return ConnectorResult(
            connector_id=self.CONNECTOR_ID,
            documents=documents,
            next_sync_state=sync_state,
            fetched_count=len(documents),
        )

    async def health_check(self) -> bool:
        try:
            await self._get_with_retry(f"{self.GRAPH_BASE}/me/joinedTeams")
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTOR: Slack
# ─────────────────────────────────────────────────────────────────────────────

class SlackConnector(BaseConnector):
    """
    Slack connector via Web API.
    Fetches channel history, DMs, and thread replies.
    """

    CONNECTOR_ID = "slack"
    SLACK_BASE = "https://slack.com/api"
    PAGE_SIZE = 200

    async def authenticate(self) -> bool:
        return bool(self.credentials.access_token)

    async def _get_channels(self) -> list[dict]:
        data = await self._get_with_retry(
            f"{self.SLACK_BASE}/conversations.list",
            {"types": "public_channel,private_channel", "limit": 200},
        )
        if not data.get("ok"):
            raise ValueError(f"Slack API error: {data.get('error')}")
        return data.get("channels", [])

    async def fetch_all(self, since: Optional[datetime] = None) -> AsyncIterator[CanonicalDocument]:
        from app.pipeline.canonical import SlackNormalizer
        normalizer = SlackNormalizer()

        channels = await self._get_channels()
        oldest = str(since.timestamp()) if since else None

        for channel in channels:
            channel_id = channel["id"]
            channel_name = channel.get("name", channel_id)
            params = {
                "channel": channel_id,
                "limit": self.PAGE_SIZE,
            }
            if oldest:
                params["oldest"] = oldest

            while True:
                data = await self._get_with_retry(
                    f"{self.SLACK_BASE}/conversations.history", params
                )
                if not data.get("ok"):
                    logger.warning(f"Slack: channel {channel_name} error: {data.get('error')}")
                    break
                for msg in data.get("messages", []):
                    if msg.get("type") == "message" and not msg.get("subtype"):
                        yield normalizer.normalize(msg, channel_name)

                if data.get("has_more") and data.get("response_metadata", {}).get("next_cursor"):
                    params["cursor"] = data["response_metadata"]["next_cursor"]
                else:
                    break

    async def fetch_incremental(self, sync_state: SyncState) -> ConnectorResult:
        documents = []
        since = sync_state.last_sync_at or (datetime.utcnow() - timedelta(days=7))
        async for doc in self.fetch_all(since=since):
            documents.append(doc)
        sync_state.last_sync_at = datetime.utcnow()
        sync_state.synced_count += len(documents)
        return ConnectorResult(
            connector_id=self.CONNECTOR_ID,
            documents=documents,
            next_sync_state=sync_state,
            fetched_count=len(documents),
        )

    async def health_check(self) -> bool:
        try:
            data = await self._get_with_retry(f"{self.SLACK_BASE}/auth.test")
            return data.get("ok", False)
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTOR: Notion
# ─────────────────────────────────────────────────────────────────────────────

class NotionConnector(BaseConnector):
    """
    Notion connector via Notion API v1.
    Fetches pages, databases, and their content blocks.
    """

    CONNECTOR_ID = "notion"
    NOTION_BASE = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"
    PAGE_SIZE = 100

    def _default_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Notion-Version": self.NOTION_VERSION,
            "Content-Type": "application/json",
        }

    async def authenticate(self) -> bool:
        return bool(self.credentials.access_token)

    async def _get_blocks(self, block_id: str) -> list[dict]:
        url = f"{self.NOTION_BASE}/blocks/{block_id}/children"
        data = await self._get_with_retry(url)
        return data.get("results", [])

    async def fetch_all(self, since: Optional[datetime] = None) -> AsyncIterator[CanonicalDocument]:
        from app.pipeline.canonical import NotionNormalizer
        normalizer = NotionNormalizer()

        # Search all pages
        url = f"{self.NOTION_BASE}/search"
        payload: dict[str, Any] = {
            "filter": {"object": "page"},
            "page_size": self.PAGE_SIZE,
        }
        if since:
            payload["filter"]["last_edited_time"] = {"after": since.isoformat()}

        cursor = None
        while True:
            if cursor:
                payload["start_cursor"] = cursor
            response = await self.http.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            for page in data.get("results", []):
                blocks = await self._get_blocks(page["id"])
                yield normalizer.normalize(page, blocks)

            if data.get("has_more"):
                cursor = data.get("next_cursor")
            else:
                break

    async def fetch_incremental(self, sync_state: SyncState) -> ConnectorResult:
        documents = []
        since = sync_state.last_sync_at or (datetime.utcnow() - timedelta(days=30))
        async for doc in self.fetch_all(since=since):
            documents.append(doc)
        sync_state.last_sync_at = datetime.utcnow()
        sync_state.synced_count += len(documents)
        return ConnectorResult(
            connector_id=self.CONNECTOR_ID,
            documents=documents,
            next_sync_state=sync_state,
            fetched_count=len(documents),
        )

    async def health_check(self) -> bool:
        try:
            response = await self.http.get(f"{self.NOTION_BASE}/users/me")
            return response.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTOR: Google Drive
# ─────────────────────────────────────────────────────────────────────────────

class GoogleDriveConnector(BaseConnector):
    """
    Google Drive connector via Google Drive API v3.
    Fetches documents, sheets, presentations, and PDFs.
    """

    CONNECTOR_ID = "google_drive"
    DRIVE_BASE = "https://www.googleapis.com/drive/v3"
    DOCS_BASE = "https://docs.googleapis.com/v1"
    PAGE_SIZE = 100
    SUPPORTED_MIME_TYPES = [
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
        "application/pdf",
    ]

    async def authenticate(self) -> bool:
        if self.credentials.needs_refresh:
            return await self._refresh_token()
        return bool(self.credentials.access_token)

    async def _refresh_token(self) -> bool:
        try:
            response = await self.http.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.credentials.refresh_token,
                    "client_id": self.credentials.client_id,
                    "client_secret": self.credentials.client_secret,
                },
            )
            response.raise_for_status()
            token_data = response.json()
            self.credentials.access_token = token_data["access_token"]
            self.credentials.token_expires_at = datetime.utcnow() + timedelta(
                seconds=token_data.get("expires_in", 3600)
            )
            self._http = None
            return True
        except Exception as e:
            logger.error(f"Google Drive token refresh failed: {e}")
            return False

    async def fetch_all(self, since: Optional[datetime] = None) -> AsyncIterator[CanonicalDocument]:
        from app.pipeline.canonical import CanonicalDocument, ConnectorSource, ContentType, Participant

        mime_query = " or ".join(f"mimeType='{m}'" for m in self.SUPPORTED_MIME_TYPES)
        query = f"({mime_query}) and trashed=false"
        if since:
            query += f" and modifiedTime >= '{since.isoformat()}Z'"

        params = {
            "q": query,
            "pageSize": self.PAGE_SIZE,
            "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,createdTime,webViewLink,owners,size)",
        }

        while True:
            data = await self._get_with_retry(f"{self.DRIVE_BASE}/files", params)
            for file_meta in data.get("files", []):
                text = await self._extract_text(file_meta)
                if text:
                    doc = CanonicalDocument(
                        source_id=file_meta["id"],
                        connector_source=ConnectorSource.GOOGLE_DRIVE,
                        content_type=self._mime_to_content_type(file_meta["mimeType"]),
                        source_url=file_meta.get("webViewLink"),
                        title=file_meta.get("name"),
                        body_text=text,
                        occurred_at=self._parse_date(file_meta.get("modifiedTime")),
                        meta={
                            "mime_type": file_meta["mimeType"],
                            "file_size": file_meta.get("size"),
                            "owners": [o.get("emailAddress") for o in file_meta.get("owners", [])],
                        },
                    )
                    yield doc

            next_token = data.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token

    async def _extract_text(self, file_meta: dict) -> Optional[str]:
        mime = file_meta.get("mimeType", "")
        file_id = file_meta["id"]

        if mime == "application/vnd.google-apps.document":
            try:
                data = await self._get_with_retry(
                    f"{self.DOCS_BASE}/documents/{file_id}"
                )
                return self._extract_doc_text(data)
            except Exception:
                return None

        elif mime == "application/pdf":
            try:
                response = await self.http.get(
                    f"{self.DRIVE_BASE}/files/{file_id}?alt=media"
                )
                response.raise_for_status()
                import io
                import fitz  # PyMuPDF
                doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
                return "\n".join(page.get_text() for page in doc)
            except Exception:
                return None

        return None

    def _extract_doc_text(self, doc_data: dict) -> str:
        lines = []
        for element in doc_data.get("body", {}).get("content", []):
            paragraph = element.get("paragraph", {})
            for el in paragraph.get("elements", []):
                text_run = el.get("textRun", {})
                content = text_run.get("content", "")
                if content.strip():
                    lines.append(content)
        return " ".join(lines)

    def _mime_to_content_type(self, mime: str):
        from app.pipeline.canonical import ContentType
        mapping = {
            "application/vnd.google-apps.document": ContentType.DOCUMENT,
            "application/vnd.google-apps.spreadsheet": ContentType.SPREADSHEET,
            "application/vnd.google-apps.presentation": ContentType.PRESENTATION,
            "application/pdf": ContentType.DOCUMENT,
        }
        return mapping.get(mime, ContentType.DOCUMENT)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        from dateutil import parser
        try:
            return parser.parse(date_str)
        except Exception:
            return None

    async def fetch_incremental(self, sync_state: SyncState) -> ConnectorResult:
        documents = []
        since = sync_state.last_sync_at or (datetime.utcnow() - timedelta(days=30))
        async for doc in self.fetch_all(since=since):
            documents.append(doc)
        sync_state.last_sync_at = datetime.utcnow()
        sync_state.synced_count += len(documents)
        return ConnectorResult(
            connector_id=self.CONNECTOR_ID,
            documents=documents,
            next_sync_state=sync_state,
            fetched_count=len(documents),
        )

    async def health_check(self) -> bool:
        try:
            await self._get_with_retry(f"{self.DRIVE_BASE}/about?fields=user")
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Connector Registry
# ─────────────────────────────────────────────────────────────────────────────

class ConnectorRegistry:
    """
    Manages all registered connectors.
    Single source of truth for connector instances.
    """

    CONNECTOR_CLASSES = {
        "outlook": OutlookConnector,
        "teams": TeamsConnector,
        "slack": SlackConnector,
        "notion": NotionConnector,
        "google_drive": GoogleDriveConnector,
    }

    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, credentials: ConnectorCredentials) -> BaseConnector:
        cls = self.CONNECTOR_CLASSES.get(credentials.connector_id)
        if not cls:
            raise ValueError(f"Unknown connector: {credentials.connector_id}")
        connector = cls(credentials)
        self._connectors[credentials.connector_id] = connector
        return connector

    def get(self, connector_id: str) -> Optional[BaseConnector]:
        return self._connectors.get(connector_id)

    def all_active(self) -> list[BaseConnector]:
        return [c for c in self._connectors.values() if c.status == ConnectorStatus.ACTIVE]

    async def health_report(self) -> dict[str, bool]:
        report = {}
        for cid, connector in self._connectors.items():
            report[cid] = await connector.health_check()
        return report
