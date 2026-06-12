

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ContentType(str, Enum):
    # Communication
    EMAIL = "email"
    EMAIL_THREAD = "email_thread"
    CHAT_MESSAGE = "chat_message"
    CHAT_THREAD = "chat_thread"
    MEETING_NOTE = "meeting_note"
    CALENDAR_EVENT = "calendar_event"

    # Documents
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    PDF = "pdf"
    NOTE = "note"
    PAGE = "page"
    DATABASE_RECORD = "database_record"

    # Attachments
    ATTACHMENT = "attachment"
    IMAGE = "image"

    # Organizational
    TASK = "task"
    PROJECT = "project"
    COMMENT = "comment"


@dataclass
class PersonReference:
    """Reference to a person mentioned in or associated with a document."""
    email: Optional[str] = None
    name: Optional[str] = None
    display_name: Optional[str] = None
    user_id: Optional[str] = None  # Source-specific ID

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonReference":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AttachmentReference:
    """Reference to an attachment stored in MinIO."""
    attachment_id: str
    filename: str
    content_type: str
    size_bytes: int
    storage_path: str           # MinIO path
    source_url: Optional[str] = None  # Original source URL
    downloaded: bool = False
    processing_status: str = "pending"  # pending | processing | complete | failed
    extracted_text: Optional[str] = None  # Populated after file processing

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RelationshipReference:
    """Represents a relationship between documents."""
    target_id: str                # ID of the related CanonicalDocument
    relationship_type: str        # "reply_to" | "references" | "child_of" | "attachment_of"
    source_type: Optional[str] = None  # Original source type of target

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EmailThreadNode:
    """A single node in an email thread tree."""
    message_id: str
    canonical_id: str
    subject: str
    sender: PersonReference
    timestamp: str
    children: List["EmailThreadNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "canonical_id": self.canonical_id,
            "subject": self.subject,
            "sender": self.sender.to_dict(),
            "timestamp": self.timestamp,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ExtractedMetadata:
    """
    Metadata extracted by the enrichment pipeline.
    Populated in Phase 2 Task 7 (metadata enrichment).
    """
    people_mentioned: List[str] = field(default_factory=list)
    organizations_mentioned: List[str] = field(default_factory=list)
    projects_mentioned: List[str] = field(default_factory=list)
    locations_mentioned: List[str] = field(default_factory=list)
    dates_mentioned: List[str] = field(default_factory=list)
    deadlines: List[Dict[str, Any]] = field(default_factory=list)
    meetings_referenced: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # External URLs/docs mentioned
    key_topics: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None  # positive | neutral | negative
    language: Optional[str] = None
    word_count: int = 0
    has_action_items: bool = False
    has_commitments: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CanonicalDocument:
    """
    The universal representation of any piece of information.

    This is the single canonical data model (Task 6).
    A normalized Outlook email and a normalized Notion page
    must be indistinguishable at this level.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""              # Original ID in source system
    source_type: str = ""            # "outlook" | "notion" | "google_drive" | etc.
    content_type: ContentType = ContentType.DOCUMENT

    # Core content
    title: str = ""
    content: str = ""                # Extracted plain text
    content_html: Optional[str] = None  # Rich content if available
    content_raw: Optional[Dict[str, Any]] = None  # Original raw structure

    # People
    author: Optional[PersonReference] = None
    participants: List[PersonReference] = field(default_factory=list)
    recipients: List[PersonReference] = field(default_factory=list)  # For email
    cc: List[PersonReference] = field(default_factory=list)
    bcc: List[PersonReference] = field(default_factory=list)

    # Timestamps
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    synced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Source-specific metadata
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    # Relationships
    relationships: List[RelationshipReference] = field(default_factory=list)
    parent_id: Optional[str] = None     # For threaded content
    thread_id: Optional[str] = None     # For email threads
    thread_position: Optional[int] = None  # Position in thread
    thread_tree: Optional[Dict[str, Any]] = None  # Full thread tree (root only)

    # Attachments
    attachments: List[AttachmentReference] = field(default_factory=list)

    # Enriched metadata (populated post-extraction)
    extracted_metadata: Optional[ExtractedMetadata] = None

    # Storage
    folder_path: Optional[str] = None   # Original folder / hierarchy
    tags: List[str] = field(default_factory=list)

    # Deduplication
    content_hash: Optional[str] = None  # SHA-256 of normalized content
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    # Validation
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = True

    # Processing state
    processing_status: str = "pending"  # pending | processing | complete | failed
    embedding_status: str = "not_ready"  # not_ready | ready | embedded

    def compute_content_hash(self) -> str:
        """Compute a deterministic hash of the document's core content."""
        hashable = json.dumps({
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content[:1000],  # First 1k chars for near-dup detection
            "created_at": self.created_at,
        }, sort_keys=True)
        self.content_hash = hashlib.sha256(hashable.encode()).hexdigest()
        return self.content_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "content_type": self.content_type.value,
            "title": self.title,
            "content": self.content,
            "content_html": self.content_html,
            "author": self.author.to_dict() if self.author else None,
            "participants": [p.to_dict() for p in self.participants],
            "recipients": [r.to_dict() for r in self.recipients],
            "cc": [c.to_dict() for c in self.cc],
            "bcc": [b.to_dict() for b in self.bcc],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "synced_at": self.synced_at,
            "source_metadata": self.source_metadata,
            "relationships": [r.to_dict() for r in self.relationships],
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "thread_position": self.thread_position,
            "thread_tree": self.thread_tree,
            "attachments": [a.to_dict() for a in self.attachments],
            "extracted_metadata": self.extracted_metadata.to_dict() if self.extracted_metadata else None,
            "folder_path": self.folder_path,
            "tags": self.tags,
            "content_hash": self.content_hash,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "validation_errors": self.validation_errors,
            "is_valid": self.is_valid,
            "processing_status": self.processing_status,
            "embedding_status": self.embedding_status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalDocument":
        doc = cls()
        for key, value in d.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
        if isinstance(doc.content_type, str):
            doc.content_type = ContentType(doc.content_type)
        if doc.author and isinstance(doc.author, dict):
            doc.author = PersonReference.from_dict(doc.author)
        return doc
