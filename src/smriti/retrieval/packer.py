"""
Token-Budget Context Packer (`ContextBudgetPacker`).

Enforces a strict token budget `B` across Smriti's 4-Layer Hierarchy:
  1. Layer 4: Always-Loaded Global Memory Index (`global.md`)
  2. Layer 3: Linked Entity/Topic Wiki Excerpts (query-filtered lines with `[turn_id]` refs)
  3. Layer 2: Ranked De-contextualized & Temporally Anchored Atomic Claims
  4. Layer 1: Verbatim Hydrated Source Turns (`[turn_id | session_id | timestamp | speaker]: text`)
"""
import re
from typing import List, Optional, Set

from smriti.config import SmritiConfig
from smriti.llm.client import estimate_tokens
from smriti.models import (
    AtomicClaim,
    RetrievedContextBundle,
    Turn,
    WikiPage,
)
from smriti.retrieval.hybrid import QueryAnalysis
from smriti.storage.sqlite_store import SQLiteMemoryStore


class ContextBudgetPacker:
    """Packs retrieved 4-layer memories into a strictly budgeted prompt context."""

    def __init__(self, store: SQLiteMemoryStore, config: Optional[SmritiConfig] = None):
        self.store = store
        self.config = config or SmritiConfig()

    def pack(
        self,
        conversation_id: str,
        query: str,
        ranked_claims: List[AtomicClaim],
        ranked_turns: List[Turn],
        ranked_wikis: List[WikiPage],
        q_info: Optional[QueryAnalysis] = None,
        budget_override: Optional[int] = None,
    ) -> RetrievedContextBundle:
        budget = budget_override or self.config.token_budget
        used_tokens = 0
        sections: List[str] = []

        # ------------------------------------------------------------------
        # 1. Layer 4: Always-Loaded Global Memory Index (`global.md`)
        # ------------------------------------------------------------------
        global_header_str = ""
        if self.config.enable_dreaming:
            gh = self.store.get_global_header(conversation_id)
            if gh and gh.header_markdown:
                gh_tokens = estimate_tokens(gh.header_markdown)
                max_gh = min(self.config.global_header_max_tokens, int(budget * 0.20))
                if gh_tokens <= max_gh:
                    global_header_str = gh.header_markdown
                    used_tokens += gh_tokens
                    sections.append(global_header_str)

        # ------------------------------------------------------------------
        # 2. Layer 2: De-Contextualized & Temporally-Anchored Atomic Claims
        # ------------------------------------------------------------------
        claims_budget = int(budget * self.config.claims_budget_fraction)
        packed_claims: List[AtomicClaim] = []
        claim_lines: List[str] = ["### Layer 2: Retrieved Atomic Memory Claims & Temporal Notes"]
        claims_tokens = estimate_tokens(claim_lines[0])
        referenced_turn_ids: List[str] = []

        for c in ranked_claims:
            refs = ", ".join(c.source_turn_ids)
            status_badge = (
                " [STATUS: SUPERSEDED/HISTORICAL]"
                if c.status == "SUPERSEDED"
                else " [STATUS: ACTIVE]"
            )
            line = f"- ({c.claim_id}){status_badge} {c.statement} [Source Turns: {refs}]"
            t_cost = estimate_tokens(line)
            if claims_tokens + t_cost > claims_budget and len(packed_claims) >= 3:
                break
            if used_tokens + claims_tokens + t_cost > budget * 0.65:
                break
            claim_lines.append(line)
            claims_tokens += t_cost
            packed_claims.append(c)
            for tid in c.source_turn_ids:
                if tid not in referenced_turn_ids:
                    referenced_turn_ids.append(tid)

        if len(claim_lines) > 1:
            used_tokens += claims_tokens
            sections.append("\n".join(claim_lines))

        # ------------------------------------------------------------------
        # 3. Layer 3: Linked Entity/Topic Wiki Snippets (Dream Layer)
        # ------------------------------------------------------------------
        packed_wikis: List[WikiPage] = []
        if self.config.enable_dreaming and ranked_wikis:
            wiki_budget = int(budget * self.config.wiki_budget_fraction)
            wiki_lines: List[str] = ["### Layer 3: Compiled Entity & Topic Wiki Excerpts"]
            wiki_tokens = estimate_tokens(wiki_lines[0])
            q_terms = set(q_info.content_tokens) if q_info else set()

            for w in ranked_wikis:
                # Select the most query-relevant bullet lines from the wiki page
                raw_b_lines = w.markdown_body.splitlines()
                relevant_bullets = []
                for bl in raw_b_lines:
                    if bl.startswith("#"):
                        continue
                    bl_low = bl.lower()
                    if any(qt in bl_low for qt in q_terms) or "~~" in bl or "**" in bl:
                        relevant_bullets.append(bl)
                if not relevant_bullets:
                    relevant_bullets = [bl for bl in raw_b_lines if bl.startswith("- ")][:4]

                snippet = f"**Wiki [[{w.title}]]**:\n" + "\n".join(relevant_bullets[:5])
                t_cost = estimate_tokens(snippet)
                if wiki_tokens + t_cost > wiki_budget:
                    break
                if used_tokens + wiki_tokens + t_cost > budget * 0.78:
                    break
                wiki_lines.append(snippet)
                wiki_tokens += t_cost
                packed_wikis.append(w)

            if len(wiki_lines) > 1:
                used_tokens += wiki_tokens
                sections.append("\n".join(wiki_lines))

        # ------------------------------------------------------------------
        # 4. Layer 1: Verbatim Hydrated Source Turns (Ground-Truth Provenance)
        # ------------------------------------------------------------------
        # Interleave top directly-ranked turns with turns referenced by top claims
        ordered_turn_candidates: List[Turn] = []
        seen_tids: Set[str] = set()

        for t in ranked_turns[:6]:
            if t.turn_id not in seen_tids:
                ordered_turn_candidates.append(t)
                seen_tids.add(t.turn_id)

        claim_turns = self.store.get_turns_by_ids(conversation_id, referenced_turn_ids)
        for t in claim_turns:
            if t.turn_id not in seen_tids:
                ordered_turn_candidates.append(t)
                seen_tids.add(t.turn_id)

        for t in ranked_turns[6:]:
            if t.turn_id not in seen_tids:
                ordered_turn_candidates.append(t)
                seen_tids.add(t.turn_id)

        packed_turns: List[Turn] = []
        turn_lines: List[str] = [
            "### Layer 1: Verbatim Source Dialogue Turns (Ground-Truth Provenance)"
        ]
        turn_tokens = estimate_tokens(turn_lines[0])

        for t in ordered_turn_candidates:
            line = f"- [{t.turn_id}] (Session: {t.session_id} | Date: {t.timestamp}) {t.speaker}: {t.text}"
            t_cost = estimate_tokens(line)
            if used_tokens + turn_tokens + t_cost > budget and len(packed_turns) >= 2:
                break
            turn_lines.append(line)
            turn_tokens += t_cost
            packed_turns.append(t)

        if len(turn_lines) > 1:
            used_tokens += turn_tokens
            sections.append("\n".join(turn_lines))

        formatted_context = "\n\n".join(sections)
        return RetrievedContextBundle(
            conversation_id=conversation_id,
            query=query,
            global_header=global_header_str,
            wiki_snippets=packed_wikis,
            claims=packed_claims,
            hydrated_turns=packed_turns,
            total_tokens=estimate_tokens(formatted_context),
            budget_tokens=budget,
            formatted_prompt_context=formatted_context,
        )
