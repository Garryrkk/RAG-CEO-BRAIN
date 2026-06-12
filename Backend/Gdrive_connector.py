

import asyncio
import json
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import structlog

from app.connectors.base.connector import BaseConnector, ConnectorConfig, SourceType
from app.connectors.base.registry import register_connector
from app.connectors.base.auth import ServiceAccountProvider, OAuthProvider
from app.models.canonical import (
    AttachmentReference,
    CanonicalDocument,
    ContentType,
    PersonReference,
    RelationshipReference,
)
from app.storage.minio_client import MinIOClient

logger = structlog.get_logger(__name__)

DRIVE_BASE = "https://www.googleapis.com/drive/v3"
FILES_BASE = "https://www.googleapis.com/drive/v3/files"
PAGE_SIZE = 100

# Google Workspace MIME types → export format mapping
GWORKSPACE_EXPORT_MAP = {
    "application/vnd.google-apps.document": {
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "extension": ".docx",
        "content_type": ContentType.DOCUMENT,
    },
    "application/vnd.google-apps.spreadsheet": {
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "extension": ".xlsx",
        "content_type": ContentType.SPREADSHEET,
    },
    "application/vnd.google-apps.presentation": {
        "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "extension": ".pptx",
        "content_type": ContentType.PRESENTATION,
    },
    "application/vnd.google-apps.drawing": {
        "mime": "application/pdf",
        "extension": ".pdf",
        "content_type": ContentType.PDF,
    },
}

# Standard (non-Google) downloadable types
DOWNLOADABLE_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "text/plain",
    "text/csv",
    "text/markdown",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}

FILE_SELECT_FIELDS = (
    "id,name,mimeType,description,starred,trashed,"
    "createdTime,modifiedTime,viewedByMeTime,"
    "owners,lastModifyingUser,sharingUser,"
    "parents,webViewLink,webContentLink,"
    "size,version,md5Checksum,sha256Checksum,"
    "permissions,capabilities,folderColorRgb,"
    "contentHints,imageMediaMetadata,videoMediaMetadata,"
    "headRevisionId"
)


@register_connector(SourceType.GOOGLE_DRIVE)
class GoogleDriveConnector(BaseConnector):
    """
    Google Drive connector supporting OAuth2 and Service Account auth.
    """

    SOURCE_TYPE = SourceType.GOOGLE_DRIVE

    def __init__(self, config: ConnectorConfig, redis, db):
        super().__init__(config, redis, db)
        self.minio = MinIOClient()

        # Support both OAuth (user-level) and Service Account (org-level)
        if "service_account_key" in config.credentials:
            self._auth_provider = ServiceAccountProvider(
                connector_id=config.connector_id,
                redis=redis,
                service_account_key=config.credentials["service_account_key"],
                scopes=[
                    "https://www.googleapis.com/auth/drive.readonly",
                    "https://www.googleapis.com/auth/drive.metadata.readonly",
                ],
            )
        else:
            self._auth_provider = OAuthProvider(
                connector_id=config.connector_id,
                redis=redis,
                client_id=config.credentials["client_id"],
                client_secret=config.credentials["client_secret"],
                token_url="https://oauth2.googleapis.com/token",
                authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
                scopes=[
                    "https://www.googleapis.com/auth/drive.readonly",
                    "openid",
                    "email",
                ],
                redirect_uri=config.credentials.get("redirect_uri", ""),
            )

        self._folder_path_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        try:
            await self._auth_provider.get_token()
            return True
        except Exception as e:
            logger.error("Google Drive auth failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover(self, checkpoint: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Discover all files in Drive.
        Uses pageToken for incremental sync, or full scan for initial sync.
        """
        page_token = None
        if checkpoint:
            # Load saved Drive change token for incremental
            page_token = await self._load_page_token()

        if page_token:
            # Incremental via Changes API
            async for item in self._discover_via_changes(page_token):
                yield item
        else:
            # Full file listing
            async for item in self._discover_all_files():
                yield item

    async def _discover_all_files(self) -> AsyncIterator[Dict[str, Any]]:
        """Full file listing using Files.list with pagination."""
        page_token = None
        while True:
            params = {
                "pageSize": PAGE_SIZE,
                "fields": f"nextPageToken,files({FILE_SELECT_FIELDS})",
                "q": "trashed = false",
                "orderBy": "createdTime asc",
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token

            data = await self._drive_get("/files", params=params)

            for file in data.get("files", []):
                mime = file.get("mimeType", "")
                if mime == "application/vnd.google-apps.folder":
                    continue  # Don't yield folders as items; handle folder path separately

                yield {
                    "id": file["id"],
                    "type": "file",
                    "name": file.get("name", ""),
                    "mime_type": mime,
                    "parents": file.get("parents", []),
                    "version": file.get("version"),
                    "modified_at": file.get("modifiedTime"),
                    "_raw": file,
                }

            page_token = data.get("nextPageToken")
            if not page_token:
                # Save the start page token for next incremental sync
                await self._save_start_page_token()
                break

    async def _discover_via_changes(self, page_token: str) -> AsyncIterator[Dict[str, Any]]:
        """Incremental sync via Drive Changes API."""
        while page_token:
            params = {
                "pageToken": page_token,
                "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file)",
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
            }
            data = await self._drive_get("/changes", params=params)

            for change in data.get("changes", []):
                if change.get("removed"):
                    continue  # Deleted file — skip for now
                file = change.get("file", {})
                if not file:
                    continue
                mime = file.get("mimeType", "")
                if mime == "application/vnd.google-apps.folder":
                    continue

                yield {
                    "id": file["id"],
                    "type": "file",
                    "name": file.get("name", ""),
                    "mime_type": mime,
                    "parents": file.get("parents", []),
                    "version": file.get("version"),
                    "modified_at": file.get("modifiedTime"),
                    "_raw": file,
                }

            new_token = data.get("newStartPageToken")
            if new_token:
                await self._save_page_token(new_token)
            page_token = data.get("nextPageToken")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract(self, item_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Extract file metadata and download content to MinIO."""
        raw = item_descriptor.get("_raw") or await self._drive_get(
            f"/files/{item_descriptor['id']}",
            params={"fields": FILE_SELECT_FIELDS, "supportsAllDrives": "true"},
        )

        mime = raw.get("mimeType", "")
        file_id = raw["id"]
        file_name = raw.get("name", "unknown")

        # Resolve full folder path
        parents = raw.get("parents", [])
        folder_path = await self._resolve_folder_path(parents[0]) if parents else "/"

        # Download file content
        storage_path, downloaded = await self._download_file(file_id, file_name, mime)

        # Fetch version history
        versions = await self._fetch_versions(file_id)

        return {
            "_type": "file",
            "file": raw,
            "folder_path": folder_path,
            "storage_path": storage_path,
            "downloaded": downloaded,
            "versions": versions,
            "mime_type": mime,
        }

    async def _download_file(
        self, file_id: str, file_name: str, mime: str
    ) -> Tuple[str, bool]:
        """Download file to MinIO. Returns (storage_path, success)."""
        storage_path = f"gdrive/{file_id}/{file_name}"

        try:
            if mime in GWORKSPACE_EXPORT_MAP:
                export_info = GWORKSPACE_EXPORT_MAP[mime]
                content = await self._export_google_file(file_id, export_info["mime"])
                storage_path = f"gdrive/{file_id}/{file_name}{export_info['extension']}"
                content_type = export_info["mime"]
            elif mime in DOWNLOADABLE_TYPES:
                content = await self._download_raw_file(file_id)
                content_type = mime
            else:
                logger.debug("Skipping unsupported mime type", mime=mime, name=file_name)
                return storage_path, False

            await self.minio.put_bytes(
                bucket="documents",
                key=storage_path,
                data=content,
                content_type=content_type,
            )

            # Queue for content processing
            from app.tasks.file_processing import process_file
            process_file.delay({
                "storage_path": storage_path,
                "file_id": file_id,
                "file_name": file_name,
                "mime_type": content_type,
            })

            return storage_path, True

        except Exception as e:
            logger.error("File download failed", file_id=file_id, name=file_name, error=str(e))
            return storage_path, False

    async def _export_google_file(self, file_id: str, export_mime: str) -> bytes:
        """Export a Google Workspace file (Doc/Sheet/Slide) to a standard format."""
        headers = await self._auth_provider.get_headers()
        url = f"{DRIVE_BASE}/files/{file_id}/export"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(
                url,
                params={"mimeType": export_mime},
                headers=headers,
            )
            response.raise_for_status()
            return response.content

    async def _download_raw_file(self, file_id: str) -> bytes:
        """Download raw file content."""
        headers = await self._auth_provider.get_headers()
        url = f"{DRIVE_BASE}/files/{file_id}"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(
                url,
                params={"alt": "media", "supportsAllDrives": "true"},
                headers=headers,
            )
            response.raise_for_status()
            return response.content

    async def _fetch_versions(self, file_id: str) -> List[Dict[str, Any]]:
        """Fetch file revision history."""
        try:
            data = await self._drive_get(
                f"/files/{file_id}/revisions",
                params={"fields": "revisions(id,modifiedTime,lastModifyingUser,size,keepForever)"},
            )
            return data.get("revisions", [])
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    async def normalize(self, raw: Dict[str, Any]) -> CanonicalDocument:
        file = raw["file"]
        mime = raw.get("mime_type", file.get("mimeType", ""))

        doc = CanonicalDocument()
        doc.id = str(uuid.uuid4())
        doc.source_id = file["id"]
        doc.source_type = SourceType.GOOGLE_DRIVE.value
        doc.content_type = self._mime_to_content_type(mime)
        doc.title = file.get("name", "Untitled")
        doc.content = file.get("description", "")  # Full text added after processing

        # Author / ownership
        owners = file.get("owners", [])
        if owners:
            doc.author = PersonReference(
                email=owners[0].get("emailAddress"),
                name=owners[0].get("displayName"),
            )

        last_modifier = file.get("lastModifyingUser", {})
        if last_modifier:
            doc.participants = [PersonReference(
                email=last_modifier.get("emailAddress"),
                name=last_modifier.get("displayName"),
            )]

        doc.created_at = file.get("createdTime")
        doc.modified_at = file.get("modifiedTime")
        doc.folder_path = raw.get("folder_path", "/")

        # Version information
        versions = raw.get("versions", [])
        doc.source_metadata = {
            "drive_id": file["id"],
            "mime_type": mime,
            "web_view_link": file.get("webViewLink"),
            "head_revision_id": file.get("headRevisionId"),
            "size": file.get("size"),
            "md5_checksum": file.get("md5Checksum"),
            "sha256_checksum": file.get("sha256Checksum"),
            "version": file.get("version"),
            "starred": file.get("starred", False),
            "capabilities": file.get("capabilities", {}),
            "version_history": [
                {
                    "revision_id": v.get("id"),
                    "modified_at": v.get("modifiedTime"),
                    "modified_by": v.get("lastModifyingUser", {}).get("displayName"),
                    "size": v.get("size"),
                }
                for v in versions
            ],
            "storage_path": raw.get("storage_path"),
            "downloaded": raw.get("downloaded", False),
        }

        # Relationship to parent folder
        parents = file.get("parents", [])
        for parent_id in parents:
            doc.relationships.append(RelationshipReference(
                target_id=parent_id,
                relationship_type="in_folder",
                source_type="google_drive_folder",
            ))

        doc.compute_content_hash()
        return doc

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def _resolve_folder_path(self, folder_id: str, depth: int = 0) -> str:
        """Recursively resolve full folder path string."""
        if depth > 10 or not folder_id:
            return "/"

        if folder_id in self._folder_path_cache:
            return self._folder_path_cache[folder_id]

        try:
            folder = await self._drive_get(
                f"/files/{folder_id}",
                params={"fields": "id,name,parents", "supportsAllDrives": "true"},
            )
            folder_name = folder.get("name", folder_id)
            parents = folder.get("parents", [])

            if parents:
                parent_path = await self._resolve_folder_path(parents[0], depth + 1)
                full_path = f"{parent_path}/{folder_name}".replace("//", "/")
            else:
                full_path = f"/{folder_name}"

            self._folder_path_cache[folder_id] = full_path
            return full_path
        except Exception:
            return f"/{folder_id}"

    def _mime_to_content_type(self, mime: str) -> ContentType:
        if mime in GWORKSPACE_EXPORT_MAP:
            return GWORKSPACE_EXPORT_MAP[mime]["content_type"]
        if "pdf" in mime:
            return ContentType.PDF
        if "spreadsheet" in mime or "excel" in mime or "csv" in mime:
            return ContentType.SPREADSHEET
        if "presentation" in mime or "powerpoint" in mime:
            return ContentType.PRESENTATION
        if "word" in mime or "document" in mime:
            return ContentType.DOCUMENT
        if "image" in mime:
            return ContentType.IMAGE
        return ContentType.DOCUMENT

    async def _save_start_page_token(self) -> None:
        data = await self._drive_get("/changes/startPageToken")
        token = data.get("startPageToken")
        if token:
            await self.redis.set(
                f"gdrive:page_token:{self.connector_id}",
                token,
                ex=86400 * 30,
            )

    async def _save_page_token(self, token: str) -> None:
        await self.redis.set(
            f"gdrive:page_token:{self.connector_id}",
            token,
            ex=86400 * 30,
        )

    async def _load_page_token(self) -> Optional[str]:
        val = await self.redis.get(f"gdrive:page_token:{self.connector_id}")
        return val.decode() if val else None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Tuple[bool, str]:
        try:
            data = await self._drive_get("/about", params={"fields": "user,storageQuota"})
            user = data.get("user", {})
            return True, f"Connected as {user.get('emailAddress', 'unknown')}"
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _drive_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headers = await self._auth_provider.get_headers()
        url = f"{DRIVE_BASE}{path}"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, headers=headers, params=params or {})
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                await asyncio.sleep(retry_after)
                return await self._drive_get(path, params)
            response.raise_for_status()
            return response.json()
