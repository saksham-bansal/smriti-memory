"""Core data schemas for Smriti's 4-Layer Memory Stack and Claim-Level Provenance."""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ClaimStatus(str, Enum):
    """Bi-temporal lifecycle state of an extracted memory note/claim."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class Turn:
    """Layer 1 (Episodic Store): A single immutable dialogue turn with a unique ID."""

    turn_id: str
    session_id: str
    conversation_id: str
    speaker: str
    text: str
    timestamp: str
    turn_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionTranscript:
    """A single chronological conversation session containing ordered turns."""

    session_id: str
    conversation_id: str
    timestamp: str
    turns: List[Turn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicClaim:
    """
    Layer 2 (Memory Notes): A de-contextualized, temporally anchored atomic proposition
    with mandatory foreign-key provenance to Layer 1 source turns.
    """

    claim_id: str
    conversation_id: str
    subject: str
    predicate: str
    object_value: str
    statement: str
    source_turn_ids: List[str]
    source_session_ids: List[str]
    evidence_quotes: List[str]
    valid_from: str
    normalized_date: Optional[str] = None
    valid_until: Optional[str] = None
    status: str = ClaimStatus.ACTIVE.value
    superseded_by: Optional[str] = None
    is_dream_synthesis: bool = False
    salience: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityEdge:
    """Directed edge in the entity-relationship graph with turn-level provenance."""

    edge_id: str
    conversation_id: str
    source_entity: str
    relation: str
    target_entity: str
    claim_id: str
    source_turn_ids: List[str]
    status: str = ClaimStatus.ACTIVE.value


@dataclass
class WikiPage:
    """
    Layer 3 (Structured Representation): An offline-compiled ('Dreamed') entity or topic
    wiki page with cross-links [[Entity]] and strict inline [turn_id] provenance.
    """

    page_id: str
    conversation_id: str
    title: str
    page_type: str  # "entity" | "topic" | "timeline"
    markdown_body: str
    linked_entities: List[str]
    source_claim_ids: List[str]
    source_turn_ids: List[str]
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalMemoryHeader:
    """
    Layer 4 (Always-Loaded Context): Compact top-level index ('global.md' / 'MEMORY.md')
    injected into every session prompt as a map of known entities, timeline, and core truths.
    """

    conversation_id: str
    header_markdown: str
    token_count: int
    source_turn_ids: List[str]
    updated_at: str


@dataclass
class ClaimProvenance:
    """Inspectable claim-level grounding emitted by the agent for every factual claim."""

    claim_text: str
    source_turn_ids: List[str]
    source_session_ids: List[str] = field(default_factory=list)
    supporting_quotes: List[str] = field(default_factory=list)
    verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedContextBundle:
    """Budget-packed multi-layer memory context prepared for agent generation."""

    conversation_id: str
    query: str
    global_header: str
    wiki_snippets: List[WikiPage]
    claims: List[AtomicClaim]
    hydrated_turns: List[Turn]
    total_tokens: int
    budget_tokens: int
    formatted_prompt_context: str


@dataclass
class AgentResponse:
    """Complete agent output with claim-level provenance and diagnostic metadata."""

    query: str
    answer: str
    abstained: bool
    claims: List[ClaimProvenance]
    retrieved_turn_ids: List[str]
    retrieved_claim_ids: List[str]
    retrieved_wiki_titles: List[str]
    prompt_tokens: int
    latency_ms: float = 0.0

    @property
    def all_cited_turn_ids(self) -> List[str]:
        """Deduplicated list of all verified source turn IDs cited across claims."""
        seen = []
        for c in self.claims:
            for tid in c.source_turn_ids:
                if tid not in seen:
                    seen.append(tid)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["all_cited_turn_ids"] = self.all_cited_turn_ids
        return d
