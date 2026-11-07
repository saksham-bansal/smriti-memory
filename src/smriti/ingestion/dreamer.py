"""
Offline 'Dreaming' Consolidation Engine (Sleep-Time Compute).

Inspired by Anthropic's 'The Components of Agent Memory', `DreamEngine` runs an
offline consolidation pass across accumulated sessions to:
  1. Audit and synthesize bi-temporal state transitions (ACTIVE vs SUPERSEDED claims)
  2. Connect fragmented multi-session facts into cross-session synthesis claims
     (preserving the strict set-union of `source_turn_ids` for 100% provenance integrity)
  3. Compile Layer 3 Linked Entity & Topic Wiki Pages (`[[Entity]]` links + `[turn_id]` citations)
  4. Promote high-salience facts and the Wiki index map into Layer 4 (`global.md` header)
"""
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from smriti.config import SmritiConfig
from smriti.llm.client import estimate_tokens
from smriti.models import (
    AtomicClaim,
    ClaimStatus,
    GlobalMemoryHeader,
    WikiPage,
)
from smriti.storage.sqlite_store import SQLiteMemoryStore
from smriti.storage.vector_index import VectorIndex


def _merge_unique(lists: List[List[str]]) -> List[str]:
    seen: List[str] = []
    for sub in lists:
        for item in sub:
            if item not in seen:
                seen.append(item)
    return seen


class DreamEngine:
    """Offline memory consolidation and provenance-preserving Wiki compiler."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        vector_index: VectorIndex,
        config: Optional[SmritiConfig] = None,
    ):
        self.store = store
        self.vector_index = vector_index
        self.config = config or SmritiConfig()

    def dream(self, conversation_id: str) -> Dict[str, int]:
        """
        Executes a full offline Dream consolidation cycle for `conversation_id`.
        Returns diagnostic counts of synthesized claims, superseded links, and compiled wiki pages.
        """
        all_claims = self.store.get_claims(conversation_id)
        if not all_claims:
            return {"synthesis_claims": 0, "wiki_pages": 0, "superseded_count": 0}

        claims_by_id: Dict[str, AtomicClaim] = {c.claim_id: c for c in all_claims}
        base_claims = [c for c in all_claims if not c.is_dream_synthesis]

        # ------------------------------------------------------------------
        # 1. Bi-Temporal State Transition Synthesis
        # ------------------------------------------------------------------
        synthesis_count = 0
        superseded_claims = [
            c for c in base_claims if c.status == ClaimStatus.SUPERSEDED.value and c.superseded_by
        ]
        for old_c in superseded_claims:
            new_c = claims_by_id.get(old_c.superseded_by or "")
            if not new_c:
                continue
            # Follow chain to latest active claim if superseded multiple times
            curr = new_c
            visited = {old_c.claim_id}
            while curr.superseded_by and curr.superseded_by in claims_by_id and curr.superseded_by not in visited:
                visited.add(curr.claim_id)
                curr = claims_by_id[curr.superseded_by]

            syn_id = "drm_trans_" + hashlib.md5(
                f"{conversation_id}|{old_c.claim_id}|{curr.claim_id}".encode("utf-8")
            ).hexdigest()[:10]
            merged_turns = _merge_unique([old_c.source_turn_ids, curr.source_turn_ids])
            merged_sessions = _merge_unique([old_c.source_session_ids, curr.source_session_ids])
            merged_quotes = _merge_unique([old_c.evidence_quotes, curr.evidence_quotes])

            statement = (
                f"[Dream State-Evolution Synthesis] {curr.subject}'s {curr.predicate.replace('_', ' ')} "
                f"is CURRENTLY '{curr.object_value}' (updated on {curr.valid_from}, ref: {', '.join(curr.source_turn_ids)}), "
                f"which SUPERSEDED their previous {old_c.predicate.replace('_', ' ')} of '{old_c.object_value}' "
                f"(valid from {old_c.valid_from} until {old_c.valid_until}, ref: {', '.join(old_c.source_turn_ids)})."
            )
            syn_claim = AtomicClaim(
                claim_id=syn_id,
                conversation_id=conversation_id,
                subject=curr.subject,
                predicate=curr.predicate,
                object_value=f"{curr.object_value} (previously {old_c.object_value})",
                statement=statement,
                source_turn_ids=merged_turns,
                source_session_ids=merged_sessions,
                evidence_quotes=merged_quotes,
                valid_from=curr.valid_from,
                normalized_date=curr.normalized_date,
                status=ClaimStatus.ACTIVE.value,
                is_dream_synthesis=True,
                salience=1.0,
            )
            self.store.upsert_claim(syn_claim)
            self.vector_index.index_item(
                conversation_id, "claim", syn_claim.claim_id, syn_claim.statement
            )
            synthesis_count += 1

        # ------------------------------------------------------------------
        # 2. Cross-Session Entity-Specific Link Synthesis (Max 3 tightly-linked claims)
        # ------------------------------------------------------------------
        # Group active claims by (subject, specific object_value) when object_value has a concrete entity
        by_subj_entity: Dict[Tuple[str, str], List[AtomicClaim]] = defaultdict(list)
        for c in base_claims:
            if c.status == ClaimStatus.ACTIVE.value and c.object_value != c.predicate:
                for ent_part in c.object_value.split(","):
                    ent_clean = ent_part.strip()
                    if len(ent_clean) >= 4 and ent_clean.lower() not in {c.subject.lower(), "personal_experience"}:
                        by_subj_entity[(c.subject, ent_clean)].append(c)

        for (subj, ent_key), group in by_subj_entity.items():
            distinct_sessions = _merge_unique([c.source_session_ids for c in group])
            if len(distinct_sessions) < 2 or len(group) < 2 or len(group) > 4:
                continue
            selected = sorted(group, key=lambda x: x.salience, reverse=True)[:3]
            merged_turns = _merge_unique([c.source_turn_ids for c in selected])
            merged_sessions = _merge_unique([c.source_session_ids for c in selected])
            merged_quotes = _merge_unique([c.evidence_quotes for c in selected])[:4]

            snippets = []
            for c in selected:
                t_refs = ", ".join(c.source_turn_ids)
                snippets.append(f"{c.statement} [{t_refs}]")

            syn_id = "drm_link_" + hashlib.md5(
                f"{conversation_id}|{subj}|{ent_key}".encode("utf-8")
            ).hexdigest()[:10]
            syn_statement = (
                f"[Dream Cross-Session Link: {subj} & {ent_key}] "
                + " | ".join(snippets)
            )
            syn_claim = AtomicClaim(
                claim_id=syn_id,
                conversation_id=conversation_id,
                subject=subj,
                predicate=selected[0].predicate,
                object_value=ent_key,
                statement=syn_statement,
                source_turn_ids=merged_turns,
                source_session_ids=merged_sessions,
                evidence_quotes=merged_quotes,
                valid_from=selected[-1].valid_from,
                normalized_date=selected[-1].normalized_date,
                status=ClaimStatus.ACTIVE.value,
                is_dream_synthesis=True,
                salience=0.88,
            )
            self.store.upsert_claim(syn_claim)
            self.vector_index.index_item(
                conversation_id, "claim", syn_claim.claim_id, syn_claim.statement
            )
            synthesis_count += 1

        # ------------------------------------------------------------------
        # 3. Compile Layer 3 Linked Entity & Topic Wiki Pages
        # ------------------------------------------------------------------
        refreshed_claims = self.store.get_claims(conversation_id)
        wiki_pages = self._compile_wiki_pages(conversation_id, refreshed_claims)
        for page in wiki_pages:
            self.store.upsert_wiki_page(page)
            self.vector_index.index_item(
                conversation_id, "wiki", page.page_id, f"{page.title}\n{page.markdown_body}"
            )

        # ------------------------------------------------------------------
        # 4. Promote Layer 4 Always-Loaded Global Context Header (global.md)
        # ------------------------------------------------------------------
        header = self._compile_global_header(conversation_id, refreshed_claims, wiki_pages)
        self.store.upsert_global_header(header)

        return {
            "synthesis_claims": synthesis_count,
            "wiki_pages": len(wiki_pages),
            "superseded_count": len(superseded_claims),
        }

    def _compile_wiki_pages(
        self, conversation_id: str, claims: List[AtomicClaim]
    ) -> List[WikiPage]:
        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        pages: List[WikiPage] = []

        # Group claims by primary speaker/subject
        by_subject: Dict[str, List[AtomicClaim]] = defaultdict(list)
        for c in claims:
            if not c.is_dream_synthesis:
                by_subject[c.subject].append(c)

        all_subjects = sorted(by_subject.keys())

        for subj in all_subjects:
            subj_claims = by_subject[subj]
            active_slots = [
                c for c in subj_claims
                if c.status == ClaimStatus.ACTIVE.value and c.salience >= 0.90
            ]
            superseded_slots = [
                c for c in subj_claims if c.status == ClaimStatus.SUPERSEDED.value
            ]

            # Group general active claims by topic predicate
            by_topic: Dict[str, List[AtomicClaim]] = defaultdict(list)
            for c in subj_claims:
                if c.status == ClaimStatus.ACTIVE.value and c.predicate != "personal_experience":
                    by_topic[c.predicate].append(c)

            lines: List[str] = [f"# Wiki Page: [[{subj}]]"]
            linked_ents: Set[str] = {s for s in all_subjects if s != subj}
            used_claim_ids: List[str] = []
            used_turn_ids: List[str] = []

            if active_slots:
                lines.append("\n## Current Active States & Preferences")
                for c in active_slots[:12]:
                    refs = ", ".join(c.source_turn_ids)
                    lines.append(
                        f"- **{c.predicate.replace('_', ' ')}**: {c.object_value} (since {c.valid_from}) [{refs}]"
                    )
                    used_claim_ids.append(c.claim_id)
                    used_turn_ids.extend(c.source_turn_ids)

            if superseded_slots:
                lines.append("\n## Historical / Superseded States (Bi-Temporal Log)")
                for c in superseded_slots[:10]:
                    refs = ", ".join(c.source_turn_ids)
                    lines.append(
                        f"- ~~{c.predicate.replace('_', ' ')}: {c.object_value}~~ (valid {c.valid_from} -> {c.valid_until}) [{refs}]"
                    )
                    used_claim_ids.append(c.claim_id)
                    used_turn_ids.extend(c.source_turn_ids)

            for topic, t_claims in sorted(by_topic.items()):
                lines.append(f"\n## {topic.replace('_', ' ').title()}")
                for c in t_claims[:10]:
                    refs = ", ".join(c.source_turn_ids)
                    lines.append(f"- {c.statement} [{refs}]")
                    used_claim_ids.append(c.claim_id)
                    used_turn_ids.extend(c.source_turn_ids)

            if linked_ents:
                links_str = ", ".join(f"[[{e}]]" for e in sorted(linked_ents))
                lines.append(f"\n## Linked Entities\n{links_str}")

            page_id = "wiki_" + hashlib.md5(
                f"{conversation_id}|entity|{subj.lower()}".encode("utf-8")
            ).hexdigest()[:10]
            pages.append(
                WikiPage(
                    page_id=page_id,
                    conversation_id=conversation_id,
                    title=subj,
                    page_type="entity",
                    markdown_body="\n".join(lines),
                    linked_entities=sorted(linked_ents),
                    source_claim_ids=_merge_unique([used_claim_ids]),
                    source_turn_ids=_merge_unique([used_turn_ids]),
                    updated_at=now_str,
                )
            )

        # Also compile Topic Wiki Pages across all participants
        by_global_topic: Dict[str, List[AtomicClaim]] = defaultdict(list)
        for c in claims:
            if not c.is_dream_synthesis and c.predicate != "personal_experience":
                by_global_topic[c.predicate].append(c)

        for topic, t_claims in sorted(by_global_topic.items()):
            if len(t_claims) < 2:
                continue
            lines = [f"# Topic Wiki: [[{topic.replace('_', ' ').title()}]]"]
            used_cids: List[str] = []
            used_tids: List[str] = []
            for c in t_claims[:18]:
                refs = ", ".join(c.source_turn_ids)
                status_tag = " (SUPERSEDED)" if c.status == ClaimStatus.SUPERSEDED.value else ""
                lines.append(f"- **[[{c.subject}]]**{status_tag}: {c.statement} [{refs}]")
                used_cids.append(c.claim_id)
                used_tids.extend(c.source_turn_ids)

            page_id = "wiki_" + hashlib.md5(
                f"{conversation_id}|topic|{topic}".encode("utf-8")
            ).hexdigest()[:10]
            pages.append(
                WikiPage(
                    page_id=page_id,
                    conversation_id=conversation_id,
                    title=topic.replace("_", " ").title(),
                    page_type="topic",
                    markdown_body="\n".join(lines),
                    linked_entities=all_subjects,
                    source_claim_ids=_merge_unique([used_cids]),
                    source_turn_ids=_merge_unique([used_tids]),
                    updated_at=now_str,
                )
            )

        return pages

    def _compile_global_header(
        self,
        conversation_id: str,
        claims: List[AtomicClaim],
        wiki_pages: List[WikiPage],
    ) -> GlobalMemoryHeader:
        """
        Compiles Layer 4 ('global.md' / 'MEMORY.md'): a compact (~150-220 token)
        always-loaded context header providing a high-level map of the conversation.
        """
        sessions = self.store.get_sessions(conversation_id)
        participants = sorted({c.subject for c in claims if not c.is_dream_synthesis})
        first_ts = sessions[0]["timestamp"] if sessions else "N/A"
        last_ts = sessions[-1]["timestamp"] if sessions else "N/A"
        wiki_titles = [f"[[{p.title}]]" for p in wiki_pages]

        lines = [
            f"### Global Memory Index (`global.md` — {conversation_id})",
            f"- **Participants**: {', '.join(participants)} | **Sessions**: {len(sessions)} ({first_ts} -> {last_ts})",
            f"- **Compiled Wiki Pages**: {', '.join(wiki_titles[:12])}",
        ]

        # Promote top state-evolution & high-salience active slot facts into global header
        high_salience = [
            c for c in claims
            if c.status == ClaimStatus.ACTIVE.value and (c.is_dream_synthesis or c.salience >= 0.93)
        ]
        # Prioritize state transitions first
        high_salience.sort(key=lambda x: (1 if "drm_trans_" in x.claim_id else 0, x.salience), reverse=True)

        header_turns: List[str] = []
        if high_salience:
            lines.append("- **Promoted Core Facts & State Updates**:")
            for c in high_salience[:6]:
                refs = ", ".join(c.source_turn_ids[:3])
                short_stmt = c.statement[:180]
                candidate_line = f"  * {short_stmt} [{refs}]"
                if estimate_tokens("\n".join(lines + [candidate_line])) > self.config.global_header_max_tokens:
                    break
                lines.append(candidate_line)
                header_turns.extend(c.source_turn_ids)

        md = "\n".join(lines)
        return GlobalMemoryHeader(
            conversation_id=conversation_id,
            header_markdown=md,
            token_count=estimate_tokens(md),
            source_turn_ids=_merge_unique([header_turns]),
            updated_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
