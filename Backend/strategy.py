

import re
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from app.core.config import settings
from app.core.logging import logger


@dataclass
class Chunk:
    content: str
    chunk_index: int
    chunking_strategy: str
    source_type: str

    # Position
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ─────────────────────────────────────────────────────────────────────────────
# Base chunker
# ─────────────────────────────────────────────────────────────────────────────

class BaseChunker:
    """Sliding-window fallback used when source-specific chunking isn't possible."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source_type: str = "document") -> List[Chunk]:
        if not text or not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        start = 0
        idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            content = " ".join(chunk_words)

            if content.strip():
                char_start = len(" ".join(words[:start]))
                char_end = char_start + len(content)
                chunks.append(Chunk(
                    content=content,
                    chunk_index=idx,
                    chunking_strategy="sliding_window",
                    source_type=source_type,
                    char_start=char_start,
                    char_end=char_end,
                ))
                idx += 1

            start += step

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL CHUNKER — conversation-aware
# ─────────────────────────────────────────────────────────────────────────────

# Patterns to detect email headers and quoted text
EMAIL_HEADER_PATTERNS = [
    re.compile(r"^(From|To|CC|BCC|Subject|Date|Sent):.*$", re.M | re.I),
    re.compile(r"^-{3,}.*?Original Message.*?-{3,}$", re.M | re.I),
    re.compile(r"^On .+wrote:$", re.M),
    re.compile(r"^>{1,3}\s?", re.M),   # quoted lines
]

FORWARDED_PATTERN = re.compile(
    r"------+\s*Forwarded\s+message\s*------+", re.I
)


class EmailChunker:
    """
    Conversation-aware email chunking.
    Each turn in the email thread becomes a distinct chunk.
    Preserves who-said-what context.
    """

    def __init__(self, max_chunk_size: int = None):
        self.max_chunk_size = max_chunk_size or settings.CHUNK_SIZE_EMAILS
        self.base_chunker = BaseChunker(self.max_chunk_size, settings.CHUNK_OVERLAP)

    def chunk(self, email_content: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Split email thread into individual conversation turns.
        """
        if not email_content or not email_content.strip():
            return []

        # Split on forwarded message boundaries first
        parts = FORWARDED_PATTERN.split(email_content)

        # Split each part into reply turns
        all_turns = []
        for part in parts:
            turns = self._split_into_turns(part)
            all_turns.extend(turns)

        chunks = []
        idx = 0
        for turn_idx, turn in enumerate(all_turns):
            turn_text = turn.get("body", "").strip()
            if not turn_text or len(turn_text) < 20:
                continue

            # If turn is too long, apply sliding window within it
            word_count = len(turn_text.split())
            if word_count > self.max_chunk_size:
                sub_chunks = self.base_chunker.chunk(turn_text, "email")
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.chunking_strategy = "email_conversation_window"
                    sc.metadata = {
                        **(metadata or {}),
                        "turn_index": turn_idx,
                        "sender": turn.get("sender"),
                        "date": turn.get("date"),
                    }
                    chunks.append(sc)
                    idx += 1
            else:
                chunk = Chunk(
                    content=turn_text,
                    chunk_index=idx,
                    chunking_strategy="email_conversation",
                    source_type="email",
                    metadata={
                        **(metadata or {}),
                        "turn_index": turn_idx,
                        "sender": turn.get("sender"),
                        "date": turn.get("date"),
                    },
                )
                chunks.append(chunk)
                idx += 1

        return chunks

    def _split_into_turns(self, text: str) -> List[Dict[str, str]]:
        """
        Split email text into individual turns (reply chain).
        """
        turns = []
        # Detect turn separators: "On [date] [name] wrote:"
        turn_separator = re.compile(
            r"(On\s+.{10,60}\s+wrote:)", re.M
        )
        parts = turn_separator.split(text)

        current_body = ""
        current_header_info = {}

        for part in parts:
            if turn_separator.match(part.strip()):
                # This is a turn header
                if current_body.strip():
                    turns.append({**current_header_info, "body": current_body.strip()})
                current_header_info = {"separator": part.strip()}
                current_body = ""
            else:
                current_body += part

        if current_body.strip():
            turns.append({**current_header_info, "body": current_body.strip()})

        # If no turn separators found, treat whole email as one turn
        if not turns:
            turns = [{"body": text.strip()}]

        return turns


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT CHUNKER — section-aware
# ─────────────────────────────────────────────────────────────────────────────

# Heading detection
HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+(.+)$", re.M),           # Markdown headings
    re.compile(r"^([A-Z][A-Z\s]{4,60})$", re.M),     # ALL CAPS headings
    re.compile(r"^(\d+\.?\d*\.?\d*\s+[A-Z].{3,80})$", re.M),  # Numbered sections
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\s*\n[=\-]{3,}", re.M),  # Underlined
]


class DocumentChunker:
    """
    Section-aware document chunking.
    Splits on detected headings, preserves section context.
    Each chunk includes the section title as metadata.
    """

    def __init__(self, max_chunk_size: int = None, overlap: int = None):
        self.max_chunk_size = max_chunk_size or settings.CHUNK_SIZE_DOCUMENTS
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self.base_chunker = BaseChunker(self.max_chunk_size, self.overlap)

    def chunk(self, content: str, source_type: str = "document",
              metadata: Optional[Dict] = None) -> List[Chunk]:
        if not content or not content.strip():
            return []

        sections = self._split_by_sections(content)
        chunks = []
        idx = 0

        for section_title, section_content in sections:
            if not section_content.strip() or len(section_content.strip()) < 30:
                continue

            word_count = len(section_content.split())

            if word_count <= self.max_chunk_size:
                chunks.append(Chunk(
                    content=section_content.strip(),
                    chunk_index=idx,
                    chunking_strategy="section_boundary",
                    source_type=source_type,
                    section_title=section_title,
                    metadata={**(metadata or {}), "section": section_title},
                ))
                idx += 1
            else:
                # Section too long → sliding window within section
                sub_chunks = self.base_chunker.chunk(section_content, source_type)
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.chunking_strategy = "section_window"
                    sc.section_title = section_title
                    sc.metadata = {**(metadata or {}), "section": section_title}
                    chunks.append(sc)
                    idx += 1

        if not chunks:
            # Fallback to sliding window
            return self.base_chunker.chunk(content, source_type)

        return chunks

    def _split_by_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """
        Detect section boundaries and split accordingly.
        Returns [(section_title, section_content), ...]
        """
        # Try to find headings
        heading_positions = []
        for pattern in HEADING_PATTERNS:
            for m in pattern.finditer(text):
                heading_positions.append((m.start(), m.group(0).strip(), m.end()))

        if not heading_positions:
            return [(None, text)]

        # Sort by position
        heading_positions.sort(key=lambda x: x[0])

        sections = []
        # Content before first heading
        if heading_positions[0][0] > 0:
            preamble = text[:heading_positions[0][0]].strip()
            if preamble:
                sections.append(("Introduction", preamble))

        for i, (pos, title, end_pos) in enumerate(heading_positions):
            if i + 1 < len(heading_positions):
                next_pos = heading_positions[i + 1][0]
                content = text[end_pos:next_pos].strip()
            else:
                content = text[end_pos:].strip()

            # Clean title
            clean_title = re.sub(r"^#+\s*", "", title).strip()
            sections.append((clean_title, content))

        return sections


# ─────────────────────────────────────────────────────────────────────────────
# MEETING NOTES CHUNKER — agenda-aware
# ─────────────────────────────────────────────────────────────────────────────

AGENDA_PATTERNS = [
    re.compile(r"^agenda\s*item\s*\d+", re.I | re.M),
    re.compile(r"^\d+\.\s+[A-Z]", re.M),
    re.compile(r"^(discussion|decision|action\s+item|follow.up|next\s+steps|attendees|participants|notes)\s*:?\s*$", re.I | re.M),
]


class MeetingNotesChunker:
    """
    Agenda-aware meeting notes chunking.
    Splits on agenda items, decisions, action items.
    Preserves the agenda structure as section titles.
    """

    def __init__(self, max_chunk_size: int = None):
        self.max_chunk_size = max_chunk_size or settings.CHUNK_SIZE_MEETING_NOTES
        self.base_chunker = BaseChunker(self.max_chunk_size, settings.CHUNK_OVERLAP)

    def chunk(self, content: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not content or not content.strip():
            return []

        sections = self._split_by_agenda(content)
        chunks = []
        idx = 0

        for section_title, section_content in sections:
            if not section_content.strip() or len(section_content.strip()) < 20:
                continue

            word_count = len(section_content.split())
            if word_count <= self.max_chunk_size:
                chunks.append(Chunk(
                    content=section_content.strip(),
                    chunk_index=idx,
                    chunking_strategy="agenda_section",
                    source_type="meeting_notes",
                    section_title=section_title,
                    metadata={
                        **(metadata or {}),
                        "section": section_title,
                        "is_action_item": "action" in (section_title or "").lower(),
                        "is_decision": "decision" in (section_title or "").lower(),
                    },
                ))
                idx += 1
            else:
                sub_chunks = self.base_chunker.chunk(section_content, "meeting_notes")
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.chunking_strategy = "agenda_window"
                    sc.section_title = section_title
                    chunks.append(sc)
                    idx += 1

        return chunks or self.base_chunker.chunk(content, "meeting_notes")

    def _split_by_agenda(self, text: str) -> List[Tuple[str, str]]:
        positions = []
        for pattern in AGENDA_PATTERNS:
            for m in pattern.finditer(text):
                positions.append((m.start(), m.group(0).strip(), m.end()))

        if not positions:
            return [("Notes", text)]

        positions.sort(key=lambda x: x[0])
        sections = []

        if positions[0][0] > 0:
            preamble = text[:positions[0][0]].strip()
            if preamble:
                sections.append(("Meeting Header", preamble))

        for i, (pos, title, end_pos) in enumerate(positions):
            next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            content = text[end_pos:next_pos].strip()
            sections.append((title, content))

        return sections


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT CHUNKER — clause-aware
# ─────────────────────────────────────────────────────────────────────────────

CLAUSE_PATTERNS = [
    re.compile(r"^(ARTICLE|SECTION|CLAUSE|SCHEDULE|ANNEX|EXHIBIT|APPENDIX)\s+[\dIVXivx]+", re.M | re.I),
    re.compile(r"^\d+\.\s+[A-Z][A-Z\s]{3,50}$", re.M),
    re.compile(r"^(\d+\.\d+)\s+[A-Z]", re.M),   # 2.1 Sub-clauses
]


class ContractChunker:
    """
    Clause-aware contract chunking.
    Each clause / article / section = one chunk.
    Critical: obligations, definitions, payment terms preserved intact.
    """

    def __init__(self, max_chunk_size: int = None):
        self.max_chunk_size = max_chunk_size or settings.CHUNK_SIZE_CONTRACTS
        self.base_chunker = BaseChunker(self.max_chunk_size, settings.CHUNK_OVERLAP // 2)

    def chunk(self, content: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not content or not content.strip():
            return []

        clauses = self._split_by_clauses(content)
        chunks = []
        idx = 0

        for clause_title, clause_content in clauses:
            if not clause_content.strip() or len(clause_content.strip()) < 30:
                continue

            word_count = len(clause_content.split())
            if word_count <= self.max_chunk_size:
                is_key = any(kw in (clause_title or "").lower() for kw in
                             ["payment", "termination", "obligation", "liability",
                              "warranty", "indemnif", "definition", "penalty"])
                chunks.append(Chunk(
                    content=clause_content.strip(),
                    chunk_index=idx,
                    chunking_strategy="clause_boundary",
                    source_type="contract",
                    section_title=clause_title,
                    metadata={
                        **(metadata or {}),
                        "clause": clause_title,
                        "is_key_clause": is_key,
                    },
                ))
                idx += 1
            else:
                sub_chunks = self.base_chunker.chunk(clause_content, "contract")
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.chunking_strategy = "clause_window"
                    sc.section_title = clause_title
                    chunks.append(sc)
                    idx += 1

        return chunks or self.base_chunker.chunk(content, "contract")

    def _split_by_clauses(self, text: str) -> List[Tuple[str, str]]:
        positions = []
        for pattern in CLAUSE_PATTERNS:
            for m in pattern.finditer(text):
                positions.append((m.start(), m.group(0).strip(), m.end()))

        if not positions:
            return [("Contract Body", text)]

        positions.sort(key=lambda x: x[0])

        sections = []
        if positions[0][0] > 0:
            preamble = text[:positions[0][0]].strip()
            if preamble:
                sections.append(("Preamble / Definitions", preamble))

        for i, (pos, title, end_pos) in enumerate(positions):
            next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            content = text[end_pos:next_pos].strip()
            sections.append((title, content))

        return sections


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING ROUTER — selects strategy based on source type
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingRouter:
    """
    Routes documents to the correct chunker based on source type.
    This is the single entry point for all chunking operations.
    """

    def __init__(self):
        self._email_chunker = EmailChunker()
        self._document_chunker = DocumentChunker()
        self._meeting_chunker = MeetingNotesChunker()
        self._contract_chunker = ContractChunker()
        self._base_chunker = BaseChunker()

    def chunk(
        self,
        content: str,
        source_type: str,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Route to the correct chunker and return chunks.
        source_type: email | document | meeting_notes | contract | message | note
        """
        if not content or not content.strip():
            return []

        logger.debug(f"Chunking {source_type} ({len(content)} chars)")

        if source_type == "email":
            return self._email_chunker.chunk(content, metadata)

        elif source_type == "meeting_notes":
            return self._meeting_chunker.chunk(content, metadata)

        elif source_type == "contract":
            return self._contract_chunker.chunk(content, metadata)

        elif source_type in ("document", "report", "presentation", "spreadsheet"):
            return self._document_chunker.chunk(content, source_type, metadata)

        else:
            # message, note, unknown → sliding window
            return self._base_chunker.chunk(content, source_type)

    def get_strategy_info(self, source_type: str) -> Dict[str, Any]:
        """Return metadata about which chunking strategy will be used."""
        strategy_map = {
            "email": {
                "strategy": "conversation_aware",
                "chunk_size": settings.CHUNK_SIZE_EMAILS,
                "rationale": "Each email turn is a semantic unit"
            },
            "meeting_notes": {
                "strategy": "agenda_aware",
                "chunk_size": settings.CHUNK_SIZE_MEETING_NOTES,
                "rationale": "Agenda items, decisions, and action items are semantic units"
            },
            "contract": {
                "strategy": "clause_aware",
                "chunk_size": settings.CHUNK_SIZE_CONTRACTS,
                "rationale": "Legal clauses must not be split mid-obligation"
            },
            "document": {
                "strategy": "section_aware",
                "chunk_size": settings.CHUNK_SIZE_DOCUMENTS,
                "rationale": "Section headings mark topic boundaries"
            },
        }
        return strategy_map.get(source_type, {
            "strategy": "sliding_window",
            "chunk_size": settings.CHUNK_SIZE_DOCUMENTS,
            "rationale": "Default fallback"
        })


# Singleton router
chunking_router = ChunkingRouter()
