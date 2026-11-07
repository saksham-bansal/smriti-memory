"""
Tri-Path Hybrid Retriever over Smriti's 4-Layer Memory Stack:
  - Path A: Dense Vector Search (Claims, Episodic Turns, Compiled Wikis)
  - Path B: SQLite FTS5 BM25 Lexical Search (Claims, Episodic Turns, Compiled Wikis)
  - Path C: 1-Hop Entity Graph Traversal & Bi-Temporal State Filtering
  - Fusion: Reciprocal Rank Fusion (RRF) + Lexical Specificity + Temporal/Supersession Reranking
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from smriti.config import SmritiConfig
from smriti.models import AtomicClaim, ClaimStatus, Turn, WikiPage
from smriti.storage.sqlite_store import SQLiteMemoryStore
from smriti.storage.vector_index import VectorIndex, _light_stem

_QUERY_STOPWORDS = {
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how",
    "did", "do", "does", "is", "are", "was", "were", "has", "have", "had",
    "will", "would", "could", "should", "can", "may", "might", "the", "a",
    "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "after", "before", "during", "between", "and", "or", "but", "if",
    "that", "this", "these", "those", "it", "they", "them", "their", "he",
    "she", "his", "her", "hers", "we", "us", "our", "you", "your", "i", "my",
    "me", "be", "been", "being", "there", "here", "any", "some", "many",
    "much", "kind", "type", "sort", "name", "named", "mention", "mentioned",
}


@dataclass
class QueryAnalysis:
    query: str
    content_tokens: List[str] = field(default_factory=list)
    content_stems: Set[str] = field(default_factory=set)
    mentioned_entities: List[str] = field(default_factory=list)
    is_temporal: bool = False
    is_historical: bool = False
    is_current_state: bool = False
    top_match_score: float = 0.0


def analyze_query(query: str, known_entities: List[str]) -> QueryAnalysis:
    lower = query.lower()
    raw_tokens = re.findall(r"[a-z0-9]+", lower)
    content_tokens = [t for t in raw_tokens if t not in _QUERY_STOPWORDS and len(t) > 1]
    content_stems = {_light_stem(t) for t in content_tokens}

    mentioned = []
    for ent in known_entities:
        if ent.lower() in lower:
            mentioned.append(ent)

    # Also detect capitalized words inside query (excluding sentence-initial WH words)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9\-]{2,25})\b", query):
        w = m.group(1)
        if w.lower() not in _QUERY_STOPWORDS and w not in mentioned:
            mentioned.append(w)

    is_temporal = bool(
        re.search(
            r"\b(when|what (?:\w+ )?(?:date|time|year|month|day)|how long|how many (?:years|months|days|weeks)|which (?:year|month|date|day))\b",
            lower,
        )
    )
    is_historical = bool(
        re.search(
            r"\b(previously|used to|before moving|before switching|prior to|originally|formerly|previous|first|old)\b",
            lower,
        )
    )
    is_current_state = bool(
        re.search(r"\b(current|currently|now|latest|presently|still|today)\b", lower)
    )

    return QueryAnalysis(
        query=query,
        content_tokens=content_tokens,
        content_stems=content_stems,
        mentioned_entities=mentioned,
        is_temporal=is_temporal,
        is_historical=is_historical,
        is_current_state=is_current_state,
    )


def _lexical_overlap_score(
    query_tokens: List[str],
    query_stems: Set[str],
    mentioned_entities: List[str],
    candidate_text: str,
) -> float:
    if not query_tokens or not candidate_text:
        return 0.0
    cand_lower = candidate_text.lower()
    cand_tokens = set(re.findall(r"[a-z0-9]+", cand_lower))
    cand_stems = {_light_stem(t) for t in cand_tokens}

    # Filter out speaker/entity tokens when computing topical overlap so a turn by "Caroline"
    # doesn't score high just for containing "Caroline"
    ent_lower = {e.lower() for e in mentioned_entities}
    topical_q_tokens = [t for t in query_tokens if t not in ent_lower]
    topical_q_stems = {_light_stem(t) for t in topical_q_tokens}

    if not topical_q_stems:
        topical_q_stems = query_stems
        topical_q_tokens = query_tokens

    exact_hits = sum(1 for t in topical_q_tokens if t in cand_tokens)
    stem_hits = sum(1 for s in topical_q_stems if s in cand_stems)

    exact_ratio = exact_hits / max(len(topical_q_tokens), 1)
    stem_ratio = stem_hits / max(len(topical_q_stems), 1)

    entity_bonus = 0.0
    for ent in mentioned_entities:
        if ent.lower() in cand_lower:
            entity_bonus += 0.15

    return 0.45 * exact_ratio + 0.55 * stem_ratio + min(entity_bonus, 0.30)


class HybridRetriever:
    """Executes Tri-Path retrieval across Layers 1, 2, and 3 with RRF and Bi-Temporal Reranking."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        vector_index: VectorIndex,
        config: Optional[SmritiConfig] = None,
    ):
        self.store = store
        self.vector_index = vector_index
        self.config = config or SmritiConfig()

    def retrieve(
        self,
        conversation_id: str,
        query: str,
        top_k_claims: int = 15,
        top_k_turns: int = 12,
        top_k_wikis: int = 3,
    ) -> Tuple[List[AtomicClaim], List[Turn], List[WikiPage], QueryAnalysis]:
        # Discover known entities in conversation
        all_claims = self.store.get_claims(conversation_id)
        known_entities = sorted({c.subject for c in all_claims})
        q_info = analyze_query(query, known_entities)

        # ------------------------------------------------------------------
        # 1. Retrieve & Rank Layer 2 Atomic Claims
        # ------------------------------------------------------------------
        rrf_scores: Dict[str, float] = defaultdict(float)
        claims_map: Dict[str, AtomicClaim] = {c.claim_id: c for c in all_claims}
        k_const = self.config.rrf_k

        # Path A: Dense Vector Search on Claims
        dense_claim_hits = self.vector_index.search(
            conversation_id, "claim", query, top_k=top_k_claims * 3
        )
        for rank, (cid, sim) in enumerate(dense_claim_hits):
            if cid in claims_map:
                if not self.config.enable_dreaming and claims_map[cid].is_dream_synthesis:
                    continue
                rrf_scores[cid] += 1.2 * (1.0 / (k_const + rank + 1)) + 0.25 * max(sim, 0.0)

        # Path B: Lexical BM25 Search on Claims
        if self.config.enable_hybrid_bm25:
            bm25_claim_hits = self.store.search_claims_bm25(
                conversation_id, query, top_k=top_k_claims * 3
            )
            for rank, (c_obj, bm25_val) in enumerate(bm25_claim_hits):
                if not self.config.enable_dreaming and c_obj.is_dream_synthesis:
                    continue
                claims_map[c_obj.claim_id] = c_obj
                rrf_scores[c_obj.claim_id] += 1.6 * (1.0 / (k_const + rank + 1)) + 0.02 * min(
                    bm25_val, 15.0
                )

        # Path C: 1-Hop Entity Graph Expansion
        if self.config.enable_entity_graph and q_info.mentioned_entities:
            graph_claims = self.store.get_entity_Claims_1hop(
                conversation_id, q_info.mentioned_entities, limit=30
            )
            for rank, c_obj in enumerate(graph_claims):
                if not self.config.enable_dreaming and c_obj.is_dream_synthesis:
                    continue
                claims_map[c_obj.claim_id] = c_obj
                rrf_scores[c_obj.claim_id] += 0.65 * (1.0 / (k_const + rank + 1))

        # Also evaluate all claims for high lexical specificity so rare detail mentions aren't missed
        for cid, c_obj in claims_map.items():
            if not self.config.enable_dreaming and c_obj.is_dream_synthesis:
                continue
            lex = _lexical_overlap_score(
                q_info.content_tokens,
                q_info.content_stems,
                q_info.mentioned_entities,
                f"{c_obj.statement} {' '.join(c_obj.evidence_quotes)}",
            )
            if lex > 0.15:
                rrf_scores[cid] += 0.85 * lex

        # Bi-Temporal & Intent-Aware Reranking on candidate claims
        final_claim_scores: List[Tuple[AtomicClaim, float]] = []
        for cid, base_score in rrf_scores.items():
            c_obj = claims_map.get(cid)
            if not c_obj:
                continue
            score = base_score

            # Subject alignment bonus if query explicitly asks about a speaker
            if q_info.mentioned_entities:
                if any(e.lower() == c_obj.subject.lower() for e in q_info.mentioned_entities):
                    score *= 1.22

            # Temporal query boost
            if q_info.is_temporal and self.config.enable_temporal_anchoring:
                if "[Temporal Anchor:" in c_obj.statement or c_obj.normalized_date:
                    score *= 1.25

            # Bi-Temporal Supersession Adjustment
            if self.config.enable_supersession:
                if c_obj.status == ClaimStatus.SUPERSEDED.value:
                    if q_info.is_historical:
                        score *= 1.35
                    else:
                        score *= 0.42  # Demote outdated/superseded facts on normal queries
                elif "drm_trans_" in c_obj.claim_id:
                    # Dream state-evolution synthesis explains both current and historical values!
                    if q_info.is_historical or q_info.is_current_state or score > 0.25:
                        score *= 1.45

            final_claim_scores.append((c_obj, score))

        final_claim_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_claims = [c for c, _ in final_claim_scores[:top_k_claims]]

        # ------------------------------------------------------------------
        # 2. Retrieve & Rank Layer 1 Episodic Turns (Direct + Hydrated from Claims)
        # ------------------------------------------------------------------
        turn_scores: Dict[str, float] = defaultdict(float)
        turns_map: Dict[str, Turn] = {}

        # Direct Dense Turn Search
        dense_turn_hits = self.vector_index.search(
            conversation_id, "turn", query, top_k=top_k_turns * 3
        )
        for rank, (tid, sim) in enumerate(dense_turn_hits):
            turn_scores[tid] += 1.1 * (1.0 / (k_const + rank + 1)) + 0.25 * max(sim, 0.0)

        # Direct BM25 Turn Search
        if self.config.enable_hybrid_bm25:
            bm25_turn_hits = self.store.search_turns_bm25(
                conversation_id, query, top_k=top_k_turns * 3
            )
            for rank, (t_obj, bm25_val) in enumerate(bm25_turn_hits):
                turns_map[t_obj.turn_id] = t_obj
                turn_scores[t_obj.turn_id] += 1.5 * (1.0 / (k_const + rank + 1)) + 0.02 * min(
                    bm25_val, 15.0
                )

        # Boost turns that are the verified provenance backing of top ranked_claims
        for rank, c_obj in enumerate(ranked_claims):
            claim_weight = 1.8 * (1.0 / (10 + rank))
            if c_obj.status == ClaimStatus.SUPERSEDED.value and not q_info.is_historical:
                claim_weight *= 0.35
            for tid in c_obj.source_turn_ids:
                turn_scores[tid] += claim_weight

        # Hydrate candidate turns and apply lexical specificity scoring
        candidate_tids = list(turn_scores.keys())
        hydrated_list = self.store.get_turns_by_ids(conversation_id, candidate_tids)
        for t_obj in hydrated_list:
            turns_map[t_obj.turn_id] = t_obj

        scored_turns: List[Tuple[Turn, float]] = []
        for tid, base_s in turn_scores.items():
            t_obj = turns_map.get(tid)
            if not t_obj:
                continue
            lex = _lexical_overlap_score(
                q_info.content_tokens,
                q_info.content_stems,
                q_info.mentioned_entities,
                f"{t_obj.speaker}: {t_obj.text}",
            )
            total_s = base_s + 0.75 * lex
            if q_info.mentioned_entities and any(
                e.lower() == t_obj.speaker.lower() for e in q_info.mentioned_entities
            ):
                total_s *= 1.15
            scored_turns.append((t_obj, total_s))

        scored_turns.sort(key=lambda x: x[1], reverse=True)
        ranked_turns = [t for t, _ in scored_turns[:top_k_turns]]

        # ------------------------------------------------------------------
        # 3. Retrieve Layer 3 Linked Wiki Pages (if Dreaming is enabled)
        # ------------------------------------------------------------------
        ranked_wikis: List[WikiPage] = []
        if self.config.enable_dreaming:
            wiki_scores: Dict[str, float] = defaultdict(float)
            all_wikis = {w.page_id: w for w in self.store.get_wiki_pages(conversation_id)}

            for rank, (pid, sim) in enumerate(
                self.vector_index.search(conversation_id, "wiki", query, top_k=top_k_wikis * 2)
            ):
                if pid in all_wikis:
                    wiki_scores[pid] += 1.0 / (k_const + rank + 1) + 0.2 * max(sim, 0.0)

            if self.config.enable_hybrid_bm25:
                for rank, (w_obj, bm25_v) in enumerate(
                    self.store.search_wiki_bm25(conversation_id, query, top_k=top_k_wikis * 2)
                ):
                    all_wikis[w_obj.page_id] = w_obj
                    wiki_scores[w_obj.page_id] += 1.2 / (k_const + rank + 1) + 0.01 * min(
                        bm25_v, 10.0
                    )

            for pid, w_obj in all_wikis.items():
                if any(e.lower() == w_obj.title.lower() for e in q_info.mentioned_entities):
                    wiki_scores[pid] += 0.35

            sorted_w = sorted(
                [(all_wikis[pid], sc) for pid, sc in wiki_scores.items() if pid in all_wikis],
                key=lambda x: x[1],
                reverse=True,
            )
            ranked_wikis = [w for w, _ in sorted_w[:top_k_wikis]]

        return ranked_claims, ranked_turns, ranked_wikis, q_info
