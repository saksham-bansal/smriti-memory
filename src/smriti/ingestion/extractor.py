"""
Layer 1 & Layer 2 Online Session Ingestion Pipeline.

Transforms raw conversational turns into:
  - Layer 1: Immutable Episodic Turns in SQLite + FTS5 + Dense Vector Index
  - Layer 2: De-contextualized, pronoun-resolved, temporally anchored AtomicClaims
    with mandatory foreign-key provenance (`source_turn_ids`, `evidence_quotes`)
  - Layer 3 (Edges): Entity-Relationship graph triples linking subjects, attributes, and objects
"""
import hashlib
import re
from typing import Dict, List, Optional, Tuple

from smriti.config import SmritiConfig
from smriti.ingestion.temporal import TemporalNormalizer
from smriti.models import (
    AtomicClaim,
    ClaimStatus,
    EntityEdge,
    SessionTranscript,
    Turn,
)
from smriti.storage.sqlite_store import SQLiteMemoryStore
from smriti.storage.vector_index import VectorIndex

# Patterns for identifying mutable state slots that can be superseded in later sessions
_ADV = r"(?:also\s+|recently\s+|just\s+|finally\s+|now\s+|currently\s+)?"
_MUTABLE_SLOT_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "residence_location",
        re.compile(
            rf"\b(?:i|we)\s+{_ADV}(?:moved\s+to|relocated\s+to|now\s+live\s+in|currently\s+live\s+in|live\s+in|am\s+living\s+in)\s+([A-Z][A-Za-z\s,\-]{{1,35}}?)(?:[\.\!\?,;]|$|\s+(?:last|this|yesterday|for|with|and|because|since|right))",
            re.IGNORECASE,
        ),
        "lives_in",
    ),
    (
        "dietary_preference",
        re.compile(
            rf"\b(?:i|we)\s+{_ADV}(?:switched\s+to|am\s+now|went|became|eat|follow)\s+(?:a\s+|an\s+|the\s+)?(vegan|vegetarian|pescatarian|keto|paleo|gluten-free|dairy-free|carnivore)(?:\s+diet|\s+option|\s+lifestyle)?\b",
            re.IGNORECASE,
        ),
        "dietary_preference",
    ),
    (
        "employment_role",
        re.compile(
            rf"\b(?:i|we)\s+{_ADV}(?:started\s+working\s+as|got\s+a\s+job\s+as|work\s+as|am\s+now\s+a|became\s+a|joined\s+as)\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z\s\-]{{2,35}}?)(?:[\.\!\?,;]|$|\s+(?:at|in|last|this|since))",
            re.IGNORECASE,
        ),
        "works_as",
    ),
    (
        "workplace",
        re.compile(
            rf"\b(?:i|we)\s+{_ADV}(?:started\s+working\s+at|got\s+a\s+job\s+at|joined|now\s+work\s+at|work\s+at)\s+([A-Z][A-Za-z0-9\s&\.\-]{{1,35}}?)(?:[\.\!\?,;]|$|\s+(?:as|in|last|this|since))",
            re.IGNORECASE,
        ),
        "works_at",
    ),
    (
        "relationship_status",
        re.compile(
            rf"\b(?:i|we)\s+{_ADV}(?:am|are|got|became)\s+(?:currently\s+)?(single|engaged|married|divorced|dating\s+[A-Z][a-z]+)\b",
            re.IGNORECASE,
        ),
        "relationship_status",
    ),
    (
        "favorite_preference",
        re.compile(
            r"\bmy\s+favorite\s+([a-z][a-z\s]{1,25}?)\s+is\s+(?:now\s+)?([A-Za-z0-9_\-\.\s]{2,30}?)(?:[\.\!\?,;]|$|\s+(?:because|since|and|right))",
            re.IGNORECASE,
        ),
        "favorite",
    ),
    (
        "project_database",
        re.compile(
            r"\b(?:we|i)\s+(?:switched\s+from\s+[A-Za-z0-9_\-]+\s+to|migrated\s+to|use|are\s+using|chose)\s+([A-Za-z0-9_\-\.]+)\s+(?:for|as)\s+(?:our|the|my)\s+([a-z][a-z\s_]{2,25}?)(?:[\.\!\?,;]|$)",
            re.IGNORECASE,
        ),
        "tech_choice",
    ),
]

# General topic/predicate classifier for non-slot factual turns
_TOPIC_PREDICATES: List[Tuple[str, List[str]]] = [
    ("career_and_education", ["job", "work", "career", "counseling", "school", "college", "degree", "study", "office", "company", "interview", "agency"]),
    ("family_and_relationships", ["sister", "brother", "mom", "mother", "dad", "father", "kids", "children", "son", "daughter", "husband", "wife", "partner", "friend", "adopt", "adoption", "family", "married", "wedding"]),
    ("hobbies_and_arts", ["paint", "painting", "pottery", "clay", "ceramics", "book", "read", "reading", "music", "guitar", "piano", "violin", "dance", "dancing", "photo", "art", "class", "studio", "craft", "museum"]),
    ("sports_and_outdoors", ["run", "running", "race", "5k", "marathon", "hike", "hiking", "camp", "camping", "swim", "yoga", "workout", "gym", "bike", "mountain", "park", "trail", "beach"]),
    ("pets_and_animals", ["dog", "puppy", "cat", "kitten", "pet", "pets", "shelter", "vet"]),
    ("identity_and_community", ["lgbtq", "transgender", "support group", "pride", "parade", "mentor", "volunteer", "community", "counselor", "therapy"]),
    ("travel_and_events", ["trip", "travel", "visit", "visited", "vacation", "concert", "festival", "birthday", "party", "move", "moved", "sweden", "country", "hometown"]),
    ("diet_and_lifestyle", ["vegan", "vegetarian", "pescatarian", "diet", "food", "cook", "cooking", "restaurant", "coffee", "tea", "allergy"]),
]


def _classify_predicate(text: str) -> str:
    lower = text.lower()
    for pred, keywords in _TOPIC_PREDICATES:
        for kw in keywords:
            if kw in lower:
                return pred
    return "personal_experience"


def _extract_named_entities(text: str, speaker: str, other_speakers: List[str]) -> List[str]:
    """Extract capitalized entity/concept mentions from a turn."""
    stopwords_cap = {
        "I", "Im", "Ive", "Id", "Ill", "My", "Me", "We", "Weve", "Were", "Our",
        "You", "Your", "Youre", "He", "She", "They", "It", "Its", "That", "This",
        "There", "Here", "What", "When", "Where", "Who", "Why", "How", "Yes",
        "Yeah", "Yep", "No", "Nope", "Oh", "Hey", "Hi", "Hello", "Well", "So",
        "But", "And", "Or", "If", "Because", "Thanks", "Thank", "Wow", "Cool",
        "Great", "Nice", "Good", "Sure", "Definitely", "Totally", "Awesome",
        "Sounds", "Haha", "Lol", "Hmm", "Maybe", "Actually", "Anyway", "Plus",
    }
    entities = []
    for sp in [speaker, *other_speakers]:
        if sp and sp not in entities:
            entities.append(sp)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9\-]{1,25}(?:\s+[A-Z][a-zA-Z0-9\-]{1,25})?)\b", text):
        cand = m.group(1).strip()
        first_tok = cand.split()[0]
        if first_tok not in stopwords_cap and len(cand) > 2:
            if cand not in entities:
                entities.append(cand)
    return entities[:8]


def _decontextualize_turn_text(
    turn: Turn,
    prev_turn: Optional[Turn],
    other_speaker: str,
    temporal_anchors: List[str],
    enable_temporal: bool = True,
) -> str:
    """
    Rewrites a raw dialogue turn into a self-contained, pronoun-grounded,
    and temporally anchored statement while preserving exact factual fidelity.
    """
    speaker = turn.speaker
    text = turn.text.strip()

    # Replace leading/standalone first-person markers with explicit speaker attribution
    resolved = re.sub(r"\bI'm\b", f"{speaker} is", text)
    resolved = re.sub(r"\bI am\b", f"{speaker} is", resolved)
    resolved = re.sub(r"\bI've\b", f"{speaker} has", resolved)
    resolved = re.sub(r"\bI have\b", f"{speaker} has", resolved)
    resolved = re.sub(r"\bI'd\b", f"{speaker} would", resolved)
    resolved = re.sub(r"\bI'll\b", f"{speaker} will", resolved)
    resolved = re.sub(r"\bI\b", speaker, resolved)
    resolved = re.sub(r"\b[Mm]y\b", f"{speaker}'s", resolved)
    resolved = re.sub(r"\b[Mm]e\b", speaker, resolved)
    if other_speaker:
        resolved = re.sub(r"\b[Yy]ou're\b", f"{other_speaker} is", resolved)
        resolved = re.sub(r"\b[Yy]ou are\b", f"{other_speaker} is", resolved)
        resolved = re.sub(r"\b[Yy]our\b", f"{other_speaker}'s", resolved)

    # Include preceding turn context if the previous turn asked a question (carrying antecedent nouns)
    context_prefix = ""
    if prev_turn and "?" in prev_turn.text:
        full_prev = prev_turn.text.strip()
        if len(full_prev) > 170:
            full_prev = "..." + full_prev[-165:]
        context_prefix = f"[In response to {prev_turn.speaker}: \"{full_prev}\"] "

    temporal_suffix = ""
    if enable_temporal and temporal_anchors:
        temporal_suffix = f" [Temporal Anchor: {'; '.join(temporal_anchors)}]"

    return f"({turn.timestamp}) {speaker}: {context_prefix}{resolved}{temporal_suffix}"


class SessionIngestor:
    """Ingests session transcripts into Layer 1 (Episodic) and Layer 2 (Atomic Claims)."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        vector_index: VectorIndex,
        config: Optional[SmritiConfig] = None,
    ):
        self.store = store
        self.vector_index = vector_index
        self.config = config or SmritiConfig()

    def ingest_session(self, session: SessionTranscript) -> List[AtomicClaim]:
        """
        1. Persists immutable session & turns to Layer 1 (`sessions`, `turns`, `turns_fts`).
        2. Indexes raw turns into Dense Vector Index.
        3. Extracts de-contextualized, temporally anchored `AtomicClaim`s (Layer 2) with
           strict foreign-key provenance (`source_turn_ids`) and entity graph edges.
        4. Runs immediate slot supersession check if `enable_supersession` is active.
        """
        # Step 1: Layer 1 Persistence
        self.store.insert_session(session)

        speakers = []
        for t in session.turns:
            if t.speaker not in speakers:
                speakers.append(t.speaker)

        extracted_claims: List[AtomicClaim] = []

        for idx, turn in enumerate(session.turns):
            prev_turn = session.turns[idx - 1] if idx > 0 else None
            other_speakers = [s for s in speakers if s != turn.speaker]
            other_speaker = other_speakers[0] if other_speakers else ""

            # Index raw turn into vector store (with speaker + timestamp header)
            raw_embed_text = f"[{turn.timestamp}] {turn.speaker}: {turn.text}"
            self.vector_index.index_item(
                session.conversation_id, "turn", turn.turn_id, raw_embed_text
            )

            # Skip pure trivial acknowledgments (< 3 words without numbers or proper nouns)
            words = re.findall(r"[A-Za-z0-9]+", turn.text)
            if len(words) < 3 and not any(w[0].isupper() or w.isdigit() for w in words):
                continue

            # Step 2: Temporal Anchor Extraction
            if self.config.enable_temporal_anchoring:
                anchors, norm_date = TemporalNormalizer.extract_temporal_anchors(
                    turn.text, turn.timestamp or session.timestamp
                )
            else:
                anchors, norm_date = [], None

            # Step 3: De-contextualized Statement Construction
            statement = _decontextualize_turn_text(
                turn=turn,
                prev_turn=prev_turn,
                other_speaker=other_speaker,
                temporal_anchors=anchors,
                enable_temporal=self.config.enable_temporal_anchoring,
            )

            source_tids = [turn.turn_id]
            evidence_quotes = [f"{turn.turn_id} ({turn.timestamp}, {turn.speaker}): {turn.text}"]
            if prev_turn and "?" in prev_turn.text and len(words) <= 10:
                # Short elliptical answer depends on previous question turn for complete provenance
                source_tids = [prev_turn.turn_id, turn.turn_id]
                evidence_quotes.insert(
                    0,
                    f"{prev_turn.turn_id} ({prev_turn.timestamp}, {prev_turn.speaker}): {prev_turn.text}",
                )

            # Step 4: Check for explicit mutable state-slot claims (for bi-temporal supersession)
            slot_matched = False
            for slot_category, pattern, pred_name in _MUTABLE_SLOT_PATTERNS:
                for m in pattern.finditer(turn.text):
                    slot_matched = True
                    if m.lastindex and m.lastindex >= 2:
                        sub_key = m.group(1).strip().lower()
                        obj_val = m.group(2).strip()
                        predicate = f"{pred_name}:{sub_key}"
                    else:
                        obj_val = m.group(1).strip()
                        predicate = pred_name

                    cid_seed = f"{session.conversation_id}|{turn.turn_id}|{predicate}|{obj_val}"
                    claim_id = "clm_" + hashlib.md5(cid_seed.encode("utf-8")).hexdigest()[:12]

                    slot_statement = (
                        f"[State Fact] {turn.speaker} {predicate.replace('_', ' ')} = '{obj_val}' "
                        f"(as of {turn.timestamp}). Full context: {statement}"
                    )
                    slot_claim = AtomicClaim(
                        claim_id=claim_id,
                        conversation_id=session.conversation_id,
                        subject=turn.speaker,
                        predicate=predicate,
                        object_value=obj_val,
                        statement=slot_statement,
                        source_turn_ids=list(source_tids),
                        source_session_ids=[session.session_id],
                        evidence_quotes=list(evidence_quotes),
                        valid_from=turn.timestamp or session.timestamp,
                        normalized_date=norm_date,
                        status=ClaimStatus.ACTIVE.value,
                        salience=0.95,
                    )
                    self._reconcile_and_save_claim(slot_claim)
                    extracted_claims.append(slot_claim)

            # Step 5: General atomic claim for the turn
            pred = _classify_predicate(turn.text)
            entities = _extract_named_entities(turn.text, turn.speaker, other_speakers)
            obj_summary = ", ".join(entities[1:4]) if len(entities) > 1 else pred

            cid_seed = f"{session.conversation_id}|{turn.turn_id}|general"
            claim_id = "clm_" + hashlib.md5(cid_seed.encode("utf-8")).hexdigest()[:12]

            salience = min(1.0, 0.55 + 0.05 * min(len(entities), 5) + (0.15 if anchors else 0.0))
            claim = AtomicClaim(
                claim_id=claim_id,
                conversation_id=session.conversation_id,
                subject=turn.speaker,
                predicate=pred,
                object_value=obj_summary,
                statement=statement,
                source_turn_ids=source_tids,
                source_session_ids=[session.session_id],
                evidence_quotes=evidence_quotes,
                valid_from=turn.timestamp or session.timestamp,
                normalized_date=norm_date,
                status=ClaimStatus.ACTIVE.value,
                salience=salience,
            )
            self.store.upsert_claim(claim)
            self.vector_index.index_item(
                session.conversation_id, "claim", claim.claim_id, claim.statement
            )
            extracted_claims.append(claim)

            # Step 6: Layer 3 Entity Graph Edges
            if self.config.enable_entity_graph:
                for ent in entities:
                    if ent.lower() != turn.speaker.lower():
                        edge_id = "edg_" + hashlib.md5(
                            f"{claim.claim_id}|{turn.speaker}|{ent}".encode("utf-8")
                        ).hexdigest()[:12]
                        edge = EntityEdge(
                            edge_id=edge_id,
                            conversation_id=session.conversation_id,
                            source_entity=turn.speaker,
                            relation=pred,
                            target_entity=ent,
                            claim_id=claim.claim_id,
                            source_turn_ids=source_tids,
                            status=ClaimStatus.ACTIVE.value,
                        )
                        self.store.upsert_entity_edge(edge)

        return extracted_claims

    def _reconcile_and_save_claim(self, new_claim: AtomicClaim) -> None:
        """
        Bi-temporal supersession resolver: if an older ACTIVE claim exists for the same
        (conversation_id, subject, mutable predicate) with a different object_value,
        mark the older claim SUPERSEDED (`valid_until = new_claim.valid_from`).
        """
        if self.config.enable_supersession:
            existing = self.store.get_claims(
                conversation_id=new_claim.conversation_id,
                status=ClaimStatus.ACTIVE.value,
                subject=new_claim.subject,
                predicate=new_claim.predicate,
            )
            for old in existing:
                if (
                    old.claim_id != new_claim.claim_id
                    and old.object_value.strip().lower() != new_claim.object_value.strip().lower()
                ):
                    self.store.mark_claim_superseded(
                        old_claim_id=old.claim_id,
                        new_claim_id=new_claim.claim_id,
                        valid_until=new_claim.valid_from,
                    )
        self.store.upsert_claim(new_claim)
        self.vector_index.index_item(
            new_claim.conversation_id, "claim", new_claim.claim_id, new_claim.statement
        )
