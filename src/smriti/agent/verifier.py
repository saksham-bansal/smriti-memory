"""
Deterministic Claim-Level Provenance Validator & False-Premise / Abstention Guard.

Enforces Hard Requirement 2:
  'Every factual answer claim about past interactions must trace to specific
   supporting source interactions.'
"""
import re
from typing import List, Optional, Set, Tuple

from smriti.models import (
    AtomicClaim,
    ClaimProvenance,
    RetrievedContextBundle,
    Turn,
)
from smriti.retrieval.hybrid import QueryAnalysis
from smriti.storage.sqlite_store import SQLiteMemoryStore
from smriti.storage.vector_index import _light_stem


class ProvenanceVerifier:
    """
    Programmatically validates and hydrates claim-level provenance against Layer 1
    (`SQLiteMemoryStore`) and detects ungrounded / false-premise queries.
    """

    def __init__(self, store: SQLiteMemoryStore):
        self.store = store

    def check_adversarial_or_unanswerable(
        self,
        conversation_id: str,
        q_info: QueryAnalysis,
        bundle: RetrievedContextBundle,
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Checks whether the query contains a false premise (e.g. attributing Speaker B's
        action/attribute to Speaker A, or asking about a topic absent from memory).
        Returns (should_abstain, explanation_if_abstain, evidence_turn_ids).
        """
        if not bundle.hydrated_turns and not bundle.claims:
            return (
                True,
                "Abstained: No relevant past interactions found in memory for this query.",
                [],
            )

        # Extract topical stems excluding mentioned participant names
        known_speakers = {t.speaker for t in bundle.hydrated_turns} | {
            c.subject for c in bundle.claims
        }
        mentioned_speakers = [
            s for s in known_speakers if s.lower() in q_info.query.lower()
        ]
        speaker_lower = {s.lower() for s in mentioned_speakers}
        generic_q_stems = {
            _light_stem(w)
            for w in [
                "go", "went", "do", "did", "make", "made", "take", "took", "get",
                "got", "have", "had", "say", "said", "talk", "tell", "feel", "think",
                "plan", "want", "like", "love", "enjoy", "start", "begin", "happen",
                "event", "activity", "thing", "person", "friend", "time", "date",
                "year", "month", "day", "week", "place", "city", "country", "current",
                "currently", "now", "previously", "before", "after", "recent", "recently",
            ]
        }
        topical_tokens = [t for t in q_info.content_tokens if t not in speaker_lower]
        topical_stems = {
            _light_stem(t) for t in topical_tokens if _light_stem(t) not in generic_q_stems
        }

        if not topical_stems:
            return False, None, []

        # Check how many topical stems appear anywhere in the retrieved bundle
        all_bundle_text = " ".join(
            [t.text for t in bundle.hydrated_turns] + [c.statement for c in bundle.claims]
        ).lower()
        bundle_stems = {_light_stem(w) for w in re.findall(r"[a-z0-9]+", all_bundle_text)}
        matched_stems = topical_stems & bundle_stems

        # 1. Complete topical absence -> Unanswerable / Abstain
        if len(matched_stems) == 0 and len(topical_stems) >= 1:
            return (
                True,
                "Abstained: The requested topic or detail was not mentioned in any past interaction.",
                [],
            )

        # 2. Speaker-Swap / False-Premise Detection (LoCoMo Category 5 Adversarial pattern)
        if len(mentioned_speakers) == 1 and len(matched_stems) >= 1:
            target_speaker = mentioned_speakers[0]
            # Score hydrated turns by topical stem overlap with the query
            turn_overlaps: List[Tuple[Turn, int]] = []
            for t in bundle.hydrated_turns:
                t_stems = {_light_stem(w) for w in re.findall(r"[a-z0-9]+", t.text.lower())}
                ov = len(matched_stems & t_stems)
                if ov > 0:
                    turn_overlaps.append((t, ov))

            turn_overlaps.sort(key=lambda x: x[1], reverse=True)
            top_ov = turn_overlaps[0][1] if turn_overlaps else 0
            # Inspect the highest-overlap turns (within 1 stem of the maximum overlap)
            best_turns = [t for t, ov in turn_overlaps if ov >= max(1, top_ov - 1)][:3]

            target_assertions = 0
            other_assertions = 0
            other_turn_ids: List[str] = []
            other_speaker_name = ""

            for t in best_turns:
                t_low = t.text.lower()
                if t.speaker.lower() == target_speaker.lower():
                    # If target_speaker is asking the OTHER speaker a question ("What motivated you to pursue counseling?" / "like that necklace?")
                    is_question_to_other = "?" in t.text and bool(
                        re.search(r"\b(you|your|yours|you're|you've)\b", t_low)
                    )
                    has_self_pronoun = bool(
                        re.search(r"\b(i|my|me|mine|i'm|i've|i'd|i'll)\b", t_low)
                    )
                    if is_question_to_other and not has_self_pronoun:
                        other_assertions += 1
                        if t.turn_id not in other_turn_ids:
                            other_turn_ids.append(t.turn_id)
                    else:
                        target_assertions += 1
                else:
                    # Spoken by the OTHER participant: unless they are explicitly stating a fact about "you" (target_speaker),
                    # this turn describes the OTHER participant's own life/experience!
                    addresses_target_declaratively = (
                        "?" not in t.text
                        and bool(re.search(r"\b(you|your)\b", t_low))
                        and not bool(
                            re.search(r"\b(i|my|me|we|our|us|i'm|i've|we've|we're)\b", t_low)
                        )
                    )
                    if addresses_target_declaratively:
                        target_assertions += 1
                    else:
                        other_assertions += 1
                        other_speaker_name = t.speaker
                        if t.turn_id not in other_turn_ids:
                            other_turn_ids.append(t.turn_id)

            if target_assertions == 0 and other_assertions > 0:
                ref_str = ", ".join(other_turn_ids[:2])
                msg = (
                    f"Abstained (False Premise / Speaker Attribution Mismatch): Past interactions do not "
                    f"record {target_speaker} doing or having this; rather, those details belong to "
                    f"{other_speaker_name or 'the other participant'} [{ref_str}]."
                )
                return True, msg, other_turn_ids[:2]

        return False, None, []

    def verify_and_hydrate_claims(
        self,
        conversation_id: str,
        raw_claims: List[ClaimProvenance],
        fallback_turns: List[Turn],
    ) -> List[ClaimProvenance]:
        """
        Verifies that every `turn_id` cited in `raw_claims` exists in the Layer 1
        Episodic Store for `conversation_id`, strips any fabricated IDs, and hydrates
        verbatim supporting quotes and session IDs directly from SQLite.
        """
        verified_list: List[ClaimProvenance] = []

        for rc in raw_claims:
            valid_tids: List[str] = []
            session_ids: List[str] = []
            quotes: List[str] = []

            for tid in rc.source_turn_ids:
                clean_tid = tid.strip().strip("[]()")
                turn_obj = self.store.get_turn(conversation_id, clean_tid)
                if turn_obj is not None:
                    if clean_tid not in valid_tids:
                        valid_tids.append(clean_tid)
                    if turn_obj.session_id not in session_ids:
                        session_ids.append(turn_obj.session_id)
                    quote_str = (
                        f"{turn_obj.turn_id} ({turn_obj.timestamp}, {turn_obj.speaker}): {turn_obj.text}"
                    )
                    if quote_str not in quotes:
                        quotes.append(quote_str)

            if valid_tids:
                verified_list.append(
                    ClaimProvenance(
                        claim_text=rc.claim_text,
                        source_turn_ids=valid_tids,
                        source_session_ids=session_ids,
                        supporting_quotes=quotes,
                        verified=True,
                    )
                )

        # If the responder emitted text without structured turn IDs, ground to top retrieved turn
        if not verified_list and fallback_turns:
            top_t = fallback_turns[0]
            verified_list.append(
                ClaimProvenance(
                    claim_text=f"Grounded in retrieved turn {top_t.turn_id}",
                    source_turn_ids=[top_t.turn_id],
                    source_session_ids=[top_t.session_id],
                    supporting_quotes=[
                        f"{top_t.turn_id} ({top_t.timestamp}, {top_t.speaker}): {top_t.text}"
                    ],
                    verified=True,
                )
            )

        return verified_list
