"""
Unified Memory-Augmented Agent (`SmritiMemoryAgent`).

Integrates:
  - Layer 1–4 Ingestion (`SessionIngestor`)
  - Offline 'Dream' Consolidation & Linked Wiki Compilation (`DreamEngine`)
  - Budgeted Tri-Path Hybrid Retrieval (`HybridRetriever` + `ContextBudgetPacker`)
  - Provenance-Enforced Generation (`LLMClient` + Deterministic Extractive Reasoner)
  - Post-Generation Claim Provenance Verification (`ProvenanceVerifier`)
"""
import json
import re
import time
from typing import Dict, List, Optional, Tuple

from smriti.agent.verifier import ProvenanceVerifier
from smriti.config import SmritiConfig
from smriti.ingestion.dreamer import DreamEngine
from smriti.ingestion.extractor import SessionIngestor
from smriti.llm.client import LLMClient, estimate_tokens
from smriti.models import (
    AgentResponse,
    AtomicClaim,
    ClaimProvenance,
    ClaimStatus,
    RetrievedContextBundle,
    SessionTranscript,
    Turn,
)
from smriti.retrieval.hybrid import HybridRetriever, QueryAnalysis, _lexical_overlap_score
from smriti.retrieval.packer import ContextBudgetPacker
from smriti.storage.sqlite_store import SQLiteMemoryStore
from smriti.storage.vector_index import VectorIndex, _light_stem

_SYSTEM_PROMPT = """You are Smriti, a precision memory-augmented assistant with strict claim-level provenance.
Answer the user's question using ONLY the provided 4-Layer Memory Context.

CRITICAL RULES:
1. PROVENANCE REQUIRED: Every factual claim you make MUST cite the exact supporting Turn ID(s) in square brackets (e.g., [D1:3]).
2. TEMPORAL PRECISION: When answering 'When' or 'What date/year' questions, use the resolved absolute dates from the [Temporal Anchor: ...] annotations or session dates rather than vague words like 'yesterday' or 'last year'.
3. BI-TEMPORAL STATE UPDATES: If a fact was SUPERSEDED (marked [STATUS: SUPERSEDED/HISTORICAL] or in a [Dream State-Evolution Synthesis]), report the CURRENT ACTIVE value for present-tense questions, and only report the superseded value if the user asks about the past/previous state (or mention both clearly with dates).
4. ABSTENTION ON FALSE PREMISE / UNANSWERABLE: If the question asks about something not in the context, or attributes Person B's action/characteristic to Person A, set "abstained": true and explain briefly.
5. CONCISENESS: Keep your answer concise, direct, and factual (1-2 sentences).

Respond in strict JSON format:
{
  "answer": "Concise factual answer with inline [turn_id] citations.",
  "abstained": false,
  "claims": [
    {
      "claim_text": "Atomic factual statement from the answer.",
      "source_turn_ids": ["D1:3"]
    }
  ]
}"""


class SmritiMemoryAgent:
    """End-to-end persistent cross-session memory agent."""

    def __init__(self, config: Optional[SmritiConfig] = None):
        self.config = config or SmritiConfig()
        self.store = SQLiteMemoryStore(self.config.db_path)
        self.vector_index = VectorIndex(self.store)
        self.ingestor = SessionIngestor(self.store, self.vector_index, self.config)
        self.dreamer = DreamEngine(self.store, self.vector_index, self.config)
        self.retriever = HybridRetriever(self.store, self.vector_index, self.config)
        self.packer = ContextBudgetPacker(self.store, self.config)
        self.verifier = ProvenanceVerifier(self.store)
        self.llm = LLMClient(self.config)

    def close(self) -> None:
        self.store.close()

    def ingest_session(self, session: SessionTranscript, run_dream: bool = False) -> Dict[str, int]:
        claims = self.ingestor.ingest_session(session)
        dream_stats = {"synthesis_claims": 0, "wiki_pages": 0, "superseded_count": 0}
        if run_dream and self.config.enable_dreaming:
            dream_stats = self.dreamer.dream(session.conversation_id)
        return {
            "ingested_turns": len(session.turns),
            "extracted_claims": len(claims),
            **dream_stats,
        }

    def dream(self, conversation_id: str) -> Dict[str, int]:
        return self.dreamer.dream(conversation_id)

    def answer_query(
        self,
        conversation_id: str,
        query: str,
        budget_override: Optional[int] = None,
    ) -> AgentResponse:
        t0 = time.perf_counter()

        # 1. Hybrid Retrieval across Layers 1, 2, 3
        ranked_claims, ranked_turns, ranked_wikis, q_info = self.retriever.retrieve(
            conversation_id=conversation_id,
            query=query,
        )

        # 2. Strict Token-Budget Context Packing (Layers 1–4)
        bundle = self.packer.pack(
            conversation_id=conversation_id,
            query=query,
            ranked_claims=ranked_claims,
            ranked_turns=ranked_turns,
            ranked_wikis=ranked_wikis,
            q_info=q_info,
            budget_override=budget_override,
        )

        # 3. False-Premise / Unanswerable Abstention Check
        should_abstain, abstain_reason, abstain_tids = (
            self.verifier.check_adversarial_or_unanswerable(conversation_id, q_info, bundle)
        )
        if should_abstain and abstain_reason:
            verified_claims = []
            if abstain_tids:
                verified_claims = self.verifier.verify_and_hydrate_claims(
                    conversation_id,
                    [ClaimProvenance(claim_text=abstain_reason, source_turn_ids=abstain_tids)],
                    fallback_turns=[],
                )
            latency = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                query=query,
                answer=abstain_reason,
                abstained=True,
                claims=verified_claims,
                retrieved_turn_ids=[t.turn_id for t in bundle.hydrated_turns],
                retrieved_claim_ids=[c.claim_id for c in bundle.claims],
                retrieved_wiki_titles=[w.title for w in bundle.wiki_snippets],
                prompt_tokens=bundle.total_tokens,
                latency_ms=latency,
            )

        # 4. Generate Grounded Response (Live LLM if configured, else Deterministic Extractive Synthesizer)
        answer_text = ""
        abstained = False
        raw_provenance: List[ClaimProvenance] = []

        if self.llm.has_live_api:
            user_prompt = (
                f"### 4-Layer Retrieved Memory Context (Budget: {bundle.total_tokens}/{bundle.budget_tokens} tokens)\n"
                f"{bundle.formatted_prompt_context}\n\n"
                f"### User Question\n{query}\n\n"
                f"Return JSON with `answer`, `abstained`, and `claims` (each with `claim_text` and `source_turn_ids`)."
            )
            llm_out = self.llm.generate(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=self.config.temperature,
                json_mode=True,
            )
            parsed = self._parse_llm_json(llm_out)
            if parsed:
                answer_text = str(parsed.get("answer", "")).strip()
                abstained = bool(parsed.get("abstained", False))
                for item in parsed.get("claims", []):
                    if isinstance(item, dict) and item.get("source_turn_ids"):
                        raw_provenance.append(
                            ClaimProvenance(
                                claim_text=str(item.get("claim_text", answer_text)),
                                source_turn_ids=[str(x) for x in item.get("source_turn_ids", [])],
                            )
                        )

        # Fallback or deterministic extractive synthesis
        if not answer_text:
            answer_text, abstained, raw_provenance = self._extractive_synthesize(
                query, q_info, bundle
            )

        # Also extract any inline [turn_id] mentions from answer_text if raw_provenance is empty
        if not raw_provenance and answer_text:
            inline_ids = re.findall(r"\[([A-Za-z0-9_:\-]+)\]", answer_text)
            if inline_ids:
                raw_provenance.append(
                    ClaimProvenance(claim_text=answer_text, source_turn_ids=inline_ids)
                )

        # 5. Deterministic Post-Generation Provenance Verification
        verified_claims = (
            []
            if abstained and not raw_provenance
            else self.verifier.verify_and_hydrate_claims(
                conversation_id=conversation_id,
                raw_claims=raw_provenance,
                fallback_turns=bundle.hydrated_turns[:2],
            )
        )

        # Ensure inline citation appears in answer_text for transparency
        if verified_claims and not re.search(r"\[[A-Za-z0-9_:\-]+\]", answer_text):
            primary_refs = ", ".join(verified_claims[0].source_turn_ids[:2])
            answer_text = f"{answer_text.rstrip('.')} [{primary_refs}]."

        latency = (time.perf_counter() - t0) * 1000.0
        return AgentResponse(
            query=query,
            answer=answer_text,
            abstained=abstained,
            claims=verified_claims,
            retrieved_turn_ids=[t.turn_id for t in bundle.hydrated_turns],
            retrieved_claim_ids=[c.claim_id for c in bundle.claims],
            retrieved_wiki_titles=[w.title for w in bundle.wiki_snippets],
            prompt_tokens=bundle.total_tokens,
            latency_ms=latency,
        )

    @staticmethod
    def _parse_llm_json(raw: str) -> Optional[Dict]:
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
        return None

    def _extractive_synthesize(
        self,
        query: str,
        q_info: QueryAnalysis,
        bundle: RetrievedContextBundle,
    ) -> Tuple[str, bool, List[ClaimProvenance]]:
        """
        High-precision deterministic answer synthesizer that uses:
          - Layer 2 resolved temporal anchors for Temporal ('When' / 'What date' / 'How long') queries
          - Bi-temporal state-evolution synthesis claims for State-Update / Historical queries
          - Multi-claim synthesis for Multi-Hop queries
          - Sentence-level lexical & entity alignment with exact Layer 1 turn provenance
        """
        if not bundle.claims and not bundle.hydrated_turns:
            return (
                "Abstained: Insufficient evidence in memory to answer this request.",
                True,
                [],
            )

        # 1. Check if top claims include Dream State-Evolution Syntheses matching the query
        trans_parts: List[str] = []
        trans_prov: List[ClaimProvenance] = []
        q_low = query.lower()
        for c in bundle.claims[:6]:
            if "drm_trans_" in c.claim_id:
                # Require subject match if query mentions a specific participant
                if q_info.mentioned_entities and not any(
                    e.lower() == c.subject.lower() for e in q_info.mentioned_entities
                ):
                    continue
                # Check predicate/slot alignment with query
                pred_words = set(re.findall(r"[a-z]+", c.predicate.lower()))
                pred_stems = {_light_stem(w) for w in pred_words}
                slot_aligned = bool(pred_stems & q_info.content_stems)
                if c.predicate == "lives_in" and any(
                    w in q_low for w in ("where", "live", "living", "city", "move", "moved", "relocate")
                ):
                    slot_aligned = True
                elif c.predicate == "dietary_preference" and any(
                    w in q_low for w in ("diet", "eat", "food", "vegan", "vegetarian")
                ):
                    slot_aligned = True
                elif c.predicate in ("works_as", "works_at") and any(
                    w in q_low for w in ("job", "work", "role", "career", "company")
                ):
                    slot_aligned = True
                elif c.predicate == "relationship_status" and any(
                    w in q_low for w in ("relationship", "status", "married", "single", "engaged")
                ):
                    slot_aligned = True

                if slot_aligned:
                    if q_info.is_historical:
                        m_prev = re.search(
                            r"previous\s+([a-z_:\s]+?)\s+of\s+'([^']+)'\s+\(valid\s+from\s+([^\)]+)\)",
                            c.statement,
                        )
                        if m_prev:
                            old_tids = c.source_turn_ids[:-1] or c.source_turn_ids
                            part = (
                                f"Previously, {c.subject}'s {m_prev.group(1).strip()} was "
                                f"'{m_prev.group(2)}' ({m_prev.group(3)}) [{', '.join(old_tids)}]."
                            )
                            trans_parts.append(part)
                            trans_prov.append(
                                ClaimProvenance(claim_text=part, source_turn_ids=old_tids)
                            )
                    else:
                        m_curr = re.search(
                            r"is\s+CURRENTLY\s+'([^']+)'\s+\(updated\s+on\s+([^,\)]+).*?previous\s+[a-z_:\s]+?\s+of\s+'([^']+)'",
                            c.statement,
                        )
                        if m_curr:
                            part = (
                                f"{c.subject}'s {c.predicate.replace('_', ' ')} is currently "
                                f"'{m_curr.group(1)}' (updated on {m_curr.group(2)}; previously '{m_curr.group(3)}') "
                                f"[{', '.join(c.source_turn_ids)}]."
                            )
                            trans_parts.append(part)
                            trans_prov.append(
                                ClaimProvenance(claim_text=part, source_turn_ids=c.source_turn_ids)
                            )
        if trans_parts:
            return " ".join(trans_parts), False, trans_prov

        # 2. Temporal Queries ('When...', 'What date...', 'How long...')
        if q_info.is_temporal and bundle.claims:
            best_c = bundle.claims[0]
            for c in bundle.claims[:5]:
                if c.status == ClaimStatus.SUPERSEDED.value and not q_info.is_historical:
                    continue
                if c.normalized_date or "[Temporal Anchor:" in c.statement:
                    best_c = c
                    break

            tids = best_c.source_turn_ids
            refs = ", ".join(tids)
            m_anchor = re.search(r"\[Temporal Anchor:\s*([^\]]+)\]", best_c.statement)
            clean_stmt = re.sub(r"\s*\[Temporal Anchor:[^\]]+\]", "", best_c.statement)
            clean_stmt = re.sub(r"^\([^\)]+\)\s*", "", clean_stmt)
            if best_c.normalized_date and m_anchor:
                ans = (
                    f"{best_c.normalized_date} ({m_anchor.group(1)}). "
                    f"Supporting interaction: {clean_stmt} [{refs}]."
                )
            elif best_c.normalized_date:
                ans = f"{best_c.normalized_date} — {clean_stmt} [{refs}]."
            else:
                ans = f"{clean_stmt} [{refs}]."
            return (
                ans,
                False,
                [ClaimProvenance(claim_text=ans, source_turn_ids=tids)],
            )

        # 3. General Single-Hop & Multi-Hop Synthesis from Top Claims & Turns
        selected_claims: List[AtomicClaim] = []
        for c in bundle.claims:
            if (
                self.config.enable_supersession
                and c.status == ClaimStatus.SUPERSEDED.value
                and not q_info.is_historical
            ):
                continue
            selected_claims.append(c)
            if len(selected_claims) >= 3:
                break

        if selected_claims:
            prov_list: List[ClaimProvenance] = []
            ans_parts: List[str] = []
            used_tids: List[str] = []
            for idx, c in enumerate(selected_claims):
                clean_stmt = re.sub(r"\s*\[Temporal Anchor:[^\]]+\]", "", c.statement)
                clean_stmt = re.sub(r"^\([^\)]+\)\s*", "", clean_stmt)
                clean_stmt = re.sub(r"^\[In response to [^\]]+\]\s*", "", clean_stmt)
                refs = ", ".join(c.source_turn_ids)
                if idx == 0 or not any(tid in used_tids for tid in c.source_turn_ids):
                    part = f"{clean_stmt} [{refs}]"
                    ans_parts.append(part)
                    prov_list.append(
                        ClaimProvenance(claim_text=clean_stmt, source_turn_ids=c.source_turn_ids)
                    )
                    used_tids.extend(c.source_turn_ids)

            full_ans = " Additionally, ".join(ans_parts)
            # Check if query is a Yes/No question
            if re.match(r"^(did|does|do|is|are|was|were|has|have|had|would|could|can)\b", q_low):
                if re.search(r"\b(yes|yeah|yep|yup|definitely|sure|certainly|loved|love|always)\b", full_ans.lower()):
                    full_ans = f"Yes — {full_ans}"
                elif re.search(r"\b(no|nope|never|not|didn't|don't|wouldn't)\b", full_ans.lower()):
                    full_ans = f"No — {full_ans}"
            return full_ans, False, prov_list

        # 4. Direct Turn Fallback
        top_t = bundle.hydrated_turns[0]
        ans = f"{top_t.speaker} ({top_t.timestamp}): {top_t.text} [{top_t.turn_id}]"
        return (
            ans,
            False,
            [ClaimProvenance(claim_text=ans, source_turn_ids=[top_t.turn_id])],
        )
