

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from task6_source_attribution.source_attribution import AttributedAnswer, Attribution
from task8_context_window.context_window import ManagedContext


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnswerQuality(str, Enum):
    GROUNDED      = "grounded"       # all statements have citations
    PARTIAL       = "partial"        # some statements have citations
    UNGROUNDED    = "ungrounded"     # no citations — should not happen
    UNCERTAIN     = "uncertain"      # evidence was insufficient


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GeneratedAnswer:
    """The final answer delivered to the CEO."""
    query: str
    answer_text: str                  # clean, readable answer
    quality: AnswerQuality
    attribution: Optional[AttributedAnswer]
    evidence_count: int
    uncertainty_flags: list[str] = field(default_factory=list)
    generated_at: str = ""
    model_used: str = ""
    latency_ms: float = 0.0

    def to_executive_display(self) -> str:
        """Format for CEO-facing display."""
        lines = [
            f"{'─' * 60}",
            f"ANSWER",
            f"{'─' * 60}",
            self.answer_text,
            "",
        ]
        if self.uncertainty_flags:
            lines += ["⚠ CAVEATS:", *[f"  • {f}" for f in self.uncertainty_flags], ""]

        if self.attribution:
            lines += [f"{'─' * 60}", "SOURCES"]
            seen: set[str] = set()
            for attr in self.attribution.all_attributions:
                if attr.attribution_id in seen:
                    continue
                seen.add(attr.attribution_id)
                lines.append(
                    f"  [{attr.source_type.upper()}] "
                    f"{attr.source_date}  "
                    f"{'| ' + ', '.join(attr.participants[:2]) if attr.participants else ''}"
                )
                lines.append(f'    "{attr.excerpt[:100]}"')

        lines += [
            "",
            f"Quality: {self.quality.value}  |  "
            f"Evidence: {self.evidence_count} sources  |  "
            f"Generated: {self.generated_at}  |  "
            f"Latency: {self.latency_ms:.0f}ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an executive intelligence assistant.
Your ONLY job is to organise the retrieved organisational memory provided below into
a clear, concise answer for the CEO.

STRICT RULES:
1. NEVER invent facts. Use ONLY what is in the CONTEXT block.
2. NEVER speculate. If information is missing, say so explicitly.
3. NEVER answer without evidence. If you cannot find evidence, say "Insufficient evidence."
4. ALWAYS cite your source using the format: [Source Type · Date · Person if available]
5. Keep your answer factual, structured, and under 300 words.
6. If sources conflict, note the conflict explicitly.
7. Begin each factual claim on a new line.

OUTPUT FORMAT:
Current Status: <one sentence summary>

Key Facts:
- <fact 1> [Source · Date]
- <fact 2> [Source · Date]
- ...

Open Items:
- <item> [Source · Date]

Risks:
- <risk> [Source · Date]

(Omit any section for which there is no evidence in the context.)
"""


def _build_user_prompt(query: str, context_block: str) -> str:
    return f"""QUESTION: {query}

CONTEXT:
{context_block}

Answer the question using ONLY the information in the CONTEXT above.
Do not add information from outside the context.
"""


# ---------------------------------------------------------------------------
# LLM adapter interface
# ---------------------------------------------------------------------------

class LLMAdapter:
    """
    Abstract interface for the LLM.
    Swap this out for any provider (OpenAI, Anthropic, local, etc.)
    """

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class StubLLMAdapter(LLMAdapter):
    """
    Stub adapter for testing / development.
    Returns a structured synthetic answer based on the context.
    Replace with a real LLM call in production.
    """

    def complete(self, system: str, user: str) -> str:
        # Extract key phrases from context to build a stub answer
        context_lower = user.lower()

        lines = ["Current Status: Information retrieved from organisational memory.\n"]

        if "delayed" in context_lower or "delay" in context_lower:
            lines.append("Key Facts:")
            lines.append("- Approval is delayed pending regulator response. [Email · Recent]")

        if "timeline" in context_lower or "shifted" in context_lower:
            lines.append("- Project timeline has shifted due to the approval hold. [Meeting · Recent]")

        if "commitment" in context_lower or "follow" in context_lower:
            lines.append("\nOpen Items:")
            lines.append("- Follow-up with regulator is pending. [Commitment · Recent]")

        if "risk" in context_lower:
            lines.append("\nRisks:")
            lines.append("- Q2 delivery is at risk if approval is not received soon. [Risk Record · Recent]")

        if len(lines) == 1:
            lines.append("Insufficient evidence found in the provided context to answer this query.")

        return "\n".join(lines)


class AnthropicAdapter(LLMAdapter):
    """
    Production adapter using Anthropic Claude.
    Requires the `anthropic` package: pip install anthropic
    """

    def __init__(self, model: str = "claude-opus-4-5", max_tokens: int = 800) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic()
            self._model = model
            self._max_tokens = max_tokens
        except ImportError:
            raise RuntimeError("Install anthropic: pip install anthropic")

    def complete(self, system: str, user: str) -> str:
        import anthropic
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text


class OpenAIAdapter(LLMAdapter):
    """
    Production adapter using OpenAI GPT models.
    Requires the `openai` package: pip install openai
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 800) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI()
            self._model = model
            self._max_tokens = max_tokens
        except ImportError:
            raise RuntimeError("Install openai: pip install openai")

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Quality assessor
# ---------------------------------------------------------------------------

class AnswerQualityAssessor:
    """Assess the quality of a generated answer."""

    def assess(self, answer_text: str, evidence_count: int) -> tuple[AnswerQuality, list[str]]:
        flags: list[str] = []

        if evidence_count == 0:
            return AnswerQuality.UNGROUNDED, ["No evidence sources were provided."]

        # Check for hallucination signals
        hallucination_phrases = [
            "i believe", "i think", "probably", "i assume", "in my opinion",
            "it seems likely", "it's possible that", "generally speaking",
        ]
        for phrase in hallucination_phrases:
            if phrase in answer_text.lower():
                flags.append(f"Potential speculation detected: '{phrase}'")

        # Check for citation presence
        has_citations = bool(re.search(r"\[.+?\]", answer_text))
        if not has_citations:
            flags.append("Answer lacks inline citations.")

        insufficient = "insufficient evidence" in answer_text.lower()
        if insufficient:
            return AnswerQuality.UNCERTAIN, flags

        if flags:
            return AnswerQuality.PARTIAL, flags

        return AnswerQuality.GROUNDED, flags


# ---------------------------------------------------------------------------
# Main answer generation engine
# ---------------------------------------------------------------------------

class AnswerGenerationEngine:
    """
    Full answer generation pipeline:
      1. Build grounded prompt from managed context
      2. Call LLM (real or stub)
      3. Post-process: strip hallucinations, add attribution shell
      4. Assess quality
      5. Return GeneratedAnswer

    Usage
    -----
    engine = AnswerGenerationEngine(llm=AnthropicAdapter())
    answer = engine.generate(
        query="What is happening with Schneider?",
        managed_context=managed_ctx,
        attributions=attribution_list,
    )
    print(answer.to_executive_display())
    """

    def __init__(
        self,
        llm: Optional[LLMAdapter] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm or StubLLMAdapter()
        self._system_prompt = system_prompt
        self._assessor = AnswerQualityAssessor()

    def generate(
        self,
        query: str,
        managed_context: ManagedContext,
        attributed_answer: Optional[AttributedAnswer] = None,
    ) -> GeneratedAnswer:
        t0 = time.time()

        user_prompt = _build_user_prompt(query, managed_context.prompt_block)

        try:
            raw_answer = self._llm.complete(self._system_prompt, user_prompt)
        except Exception as exc:
            raw_answer = (
                f"Insufficient evidence — LLM call failed: {exc}. "
                "Please retry or contact support."
            )

        latency_ms = (time.time() - t0) * 1000

        quality, flags = self._assessor.assess(
            raw_answer,
            evidence_count=managed_context.total_tokens_used // 100,  # rough proxy
        )

        return GeneratedAnswer(
            query=query,
            answer_text=raw_answer,
            quality=quality,
            attribution=attributed_answer,
            evidence_count=len(managed_context.sections_included),
            uncertainty_flags=flags,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            model_used=type(self._llm).__name__,
            latency_ms=latency_ms,
        )

    def generate_from_scratch(
        self,
        query: str,
        raw_context_text: str,
        attributions: Optional[list[Attribution]] = None,
    ) -> GeneratedAnswer:
        """
        Minimal entry point: provide query + raw context text directly.
        No ManagedContext required.
        """
        t0 = time.time()
        user_prompt = _build_user_prompt(query, raw_context_text)

        try:
            raw_answer = self._llm.complete(self._system_prompt, user_prompt)
        except Exception as exc:
            raw_answer = f"Insufficient evidence — error: {exc}"

        latency_ms = (time.time() - t0) * 1000
        quality, flags = self._assessor.assess(raw_answer, evidence_count=1)

        return GeneratedAnswer(
            query=query,
            answer_text=raw_answer,
            quality=quality,
            attribution=None,
            evidence_count=1,
            uncertainty_flags=flags,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            model_used=type(self._llm).__name__,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from task3_hybrid_search.hybrid_search import MemoryChunk, HybridSearchEngine
    from task4_context_assembly.context_assembly import ContextAssemblyEngine
    from task8_context_window.context_window import ContextWindowManager
    from task6_source_attribution.source_attribution import SourceAttributionEngine

    now = _time.time()

    corpus = [
        MemoryChunk("email_001", "email",
                    "Approval delayed. Waiting for regulator sign-off on Smart Meter.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter",
                              "person": "John Smith"},
                    timestamp=now - 86400 * 2),
        MemoryChunk("meeting_002", "meeting",
                    "Smart Meter timeline shifted 2 weeks due to approval delay.",
                    metadata={"company": "Schneider Electric", "project": "Smart Meter"},
                    timestamp=now - 86400 * 5),
        MemoryChunk("risk_003", "risk",
                    "Risk: Q2 delivery at risk. Regulator response overdue.",
                    metadata={"company": "Schneider Electric"},
                    timestamp=now - 86400 * 1),
        MemoryChunk("commitment_004", "commitment",
                    "John Smith to follow up with regulator by March 25.",
                    metadata={"person": "John Smith", "status": "open"},
                    timestamp=now - 86400 * 3),
    ]

    query = "What is happening with Schneider Electric?"

    # Pipeline
    results  = HybridSearchEngine().search(query, corpus, filters={"company": "Schneider Electric"})
    assembled = ContextAssemblyEngine().assemble(query, results, strategy="company_360")
    managed   = ContextWindowManager(budget_tokens=3000).compress(assembled)
    attributed = SourceAttributionEngine().attribute(query, "", results=results)

    engine = AnswerGenerationEngine(llm=StubLLMAdapter())
    answer = engine.generate(query, managed, attributed)

    print("=" * 70)
    print("PHASE 4 - TASK 9: ANSWER GENERATION LAYER — DEMO")
    print("=" * 70)
    print(answer.to_executive_display())
