"""
Persistent SQLite + FTS5 storage engine implementing Smriti's 4-Layer Memory Stack:
  - Layer 1: Immutable Episodic Transcripts (sessions, turns, turns_fts)
  - Layer 2: Bi-Temporal Atomic Memory Notes (claims, claims_fts)
  - Layer 3: Linked Entity/Topic Wiki Pages & Entity Graph (wiki_pages, wiki_fts, entity_edges)
  - Layer 4: Always-Loaded Global Context Header (global_headers)
  - Vector Store: Persisted float32 dense embeddings (embeddings)
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smriti.models import (
    AtomicClaim,
    ClaimStatus,
    EntityEdge,
    GlobalMemoryHeader,
    SessionTranscript,
    Turn,
    WikiPage,
)


def _sanitize_fts_query(query: str) -> str:
    """Convert an arbitrary natural language query into a safe FTS5 OR query."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "what", "when", "where",
        "who", "why", "how", "did", "do", "does", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "about", "as", "into", "like", "through",
        "after", "over", "between", "out", "against", "during", "without",
        "before", "under", "around", "among", "and", "or", "but", "if", "then",
        "that", "this", "these", "those", "it", "they", "them", "their", "his",
        "her", "he", "she", "we", "you", "your", "my", "me", "i", "have", "has",
        "had", "been", "being", "be", "can", "could", "would", "should", "will",
    }
    raw_tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    meaningful = [t for t in raw_tokens if t not in stopwords and len(t) > 1]
    if not meaningful:
        meaningful = [t for t in raw_tokens if len(t) > 1]
    if not meaningful:
        return ""
    # Deduplicate while preserving order
    seen = []
    for tok in meaningful:
        if tok not in seen:
            seen.append(tok)
    return " OR ".join(f'"{tok}"' for tok in seen[:16])


class SQLiteMemoryStore:
    """ACID-compliant persistent store across process restarts using SQLite WAL + FTS5."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")

        # Layer 1: Sessions & Immutable Episodic Turns
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (conversation_id, session_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                turn_index INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (conversation_id, turn_id)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_turns_conv_session
            ON turns (conversation_id, session_id, turn_index);
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                conversation_id UNINDEXED,
                turn_id UNINDEXED,
                speaker,
                text,
                timestamp
            );
            """
        )

        # Layer 2: Bi-Temporal Atomic Memory Claims / Notes
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_value TEXT NOT NULL,
                statement TEXT NOT NULL,
                source_turn_ids_json TEXT NOT NULL,
                source_session_ids_json TEXT NOT NULL,
                evidence_quotes_json TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                normalized_date TEXT,
                valid_until TEXT,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                superseded_by TEXT,
                is_dream_synthesis INTEGER NOT NULL DEFAULT 0,
                salience REAL NOT NULL DEFAULT 0.8
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claims_conv_status
            ON claims (conversation_id, status);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claims_conv_subj_pred
            ON claims (conversation_id, lower(subject), lower(predicate));
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                conversation_id UNINDEXED,
                claim_id UNINDEXED,
                subject,
                predicate,
                object_value,
                statement,
                normalized_date,
                evidence_quotes
            );
            """
        )

        # Layer 3: Linked Entity/Topic Wiki Pages & Entity Graph
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_pages (
                page_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                page_type TEXT NOT NULL,
                markdown_body TEXT NOT NULL,
                linked_entities_json TEXT NOT NULL,
                source_claim_ids_json TEXT NOT NULL,
                source_turn_ids_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wiki_conv_title
            ON wiki_pages (conversation_id, lower(title));
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
                conversation_id UNINDEXED,
                page_id UNINDEXED,
                title,
                markdown_body
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_edges (
                edge_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                source_entity TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                source_turn_ids_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE'
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_conv_entities
            ON entity_edges (conversation_id, lower(source_entity), lower(target_entity));
            """
        )

        # Layer 4: Always-Loaded Global Context Header (MEMORY.md / global.md)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS global_headers (
                conversation_id TEXT PRIMARY KEY,
                header_markdown TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                source_turn_ids_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        # Persisted Dense Embeddings
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                conversation_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY (conversation_id, item_type, item_id)
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Layer 1: Episodic Transcript Operations
    # ------------------------------------------------------------------
    def insert_session(self, session: SessionTranscript) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO sessions (session_id, conversation_id, timestamp, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.conversation_id,
                session.timestamp,
                json.dumps(session.metadata),
            ),
        )
        for idx, turn in enumerate(session.turns):
            t_idx = turn.turn_index if turn.turn_index else idx
            cur.execute(
                """
                INSERT OR REPLACE INTO turns
                (turn_id, session_id, conversation_id, speaker, text, timestamp, turn_index, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    session.session_id,
                    session.conversation_id,
                    turn.speaker,
                    turn.text,
                    turn.timestamp or session.timestamp,
                    t_idx,
                    json.dumps(turn.metadata),
                ),
            )
            cur.execute(
                "DELETE FROM turns_fts WHERE conversation_id = ? AND turn_id = ?",
                (session.conversation_id, turn.turn_id),
            )
            cur.execute(
                """
                INSERT INTO turns_fts (conversation_id, turn_id, speaker, text, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.conversation_id,
                    turn.turn_id,
                    turn.speaker,
                    turn.text,
                    turn.timestamp or session.timestamp,
                ),
            )
        self.conn.commit()

    def get_turn(self, conversation_id: str, turn_id: str) -> Optional[Turn]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM turns WHERE conversation_id = ? AND turn_id = ?",
            (conversation_id, turn_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_turn(row)

    def get_turns_by_ids(self, conversation_id: str, turn_ids: List[str]) -> List[Turn]:
        if not turn_ids:
            return []
        placeholders = ",".join("?" for _ in turn_ids)
        cur = self.conn.cursor()
        rows = cur.execute(
            f"""
            SELECT * FROM turns
            WHERE conversation_id = ? AND turn_id IN ({placeholders})
            ORDER BY session_id ASC, turn_index ASC
            """,
            [conversation_id, *turn_ids],
        ).fetchall()
        by_id = {r["turn_id"]: self._row_to_turn(r) for r in rows}
        return [by_id[tid] for tid in turn_ids if tid in by_id]

    def get_all_turns(self, conversation_id: str) -> List[Turn]:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT * FROM turns
            WHERE conversation_id = ?
            ORDER BY rowid ASC
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def get_sessions(self, conversation_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM sessions WHERE conversation_id = ? ORDER BY rowid ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_turns_bm25(
        self, conversation_id: str, query: str, top_k: int = 20
    ) -> List[Tuple[Turn, float]]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return []
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT turn_id, bm25(turns_fts, 0.0, 0.0, 1.0, 2.5, 1.5) AS rank_score
                FROM turns_fts
                WHERE turns_fts MATCH ? AND conversation_id = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (fts_q, conversation_id, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        results: List[Tuple[Turn, float]] = []
        for r in rows:
            t = self.get_turn(conversation_id, r["turn_id"])
            if t is not None:
                # bm25() in SQLite FTS5 returns negative values (more negative = better)
                results.append((t, float(-r["rank_score"])))
        return results

    # ------------------------------------------------------------------
    # Layer 2: Atomic Memory Claims Operations
    # ------------------------------------------------------------------
    def upsert_claim(self, claim: AtomicClaim) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO claims (
                claim_id, conversation_id, subject, predicate, object_value,
                statement, source_turn_ids_json, source_session_ids_json,
                evidence_quotes_json, valid_from, normalized_date, valid_until,
                status, superseded_by, is_dream_synthesis, salience
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.claim_id,
                claim.conversation_id,
                claim.subject,
                claim.predicate,
                claim.object_value,
                claim.statement,
                json.dumps(claim.source_turn_ids),
                json.dumps(claim.source_session_ids),
                json.dumps(claim.evidence_quotes),
                claim.valid_from,
                claim.normalized_date,
                claim.valid_until,
                claim.status,
                claim.superseded_by,
                1 if claim.is_dream_synthesis else 0,
                claim.salience,
            ),
        )
        cur.execute(
            "DELETE FROM claims_fts WHERE conversation_id = ? AND claim_id = ?",
            (claim.conversation_id, claim.claim_id),
        )
        cur.execute(
            """
            INSERT INTO claims_fts (
                conversation_id, claim_id, subject, predicate, object_value,
                statement, normalized_date, evidence_quotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.conversation_id,
                claim.claim_id,
                claim.subject,
                claim.predicate,
                claim.object_value,
                claim.statement,
                claim.normalized_date or "",
                " ".join(claim.evidence_quotes),
            ),
        )
        self.conn.commit()

    def mark_claim_superseded(
        self, old_claim_id: str, new_claim_id: str, valid_until: str
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE claims
            SET status = ?, superseded_by = ?, valid_until = ?
            WHERE claim_id = ?
            """,
            (ClaimStatus.SUPERSEDED.value, new_claim_id, valid_until, old_claim_id),
        )
        cur.execute(
            """
            UPDATE entity_edges
            SET status = ?
            WHERE claim_id = ?
            """,
            (ClaimStatus.SUPERSEDED.value, old_claim_id),
        )
        self.conn.commit()

    def get_claim(self, claim_id: str) -> Optional[AtomicClaim]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        return self._row_to_claim(row) if row else None

    def get_claims(
        self,
        conversation_id: str,
        status: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[AtomicClaim]:
        cur = self.conn.cursor()
        clauses = ["conversation_id = ?"]
        params: List[Any] = [conversation_id]
        if status:
            clauses.append("status = ?")
            params.append(status)
        if subject:
            clauses.append("lower(subject) = lower(?)")
            params.append(subject)
        if predicate:
            clauses.append("lower(predicate) = lower(?)")
            params.append(predicate)
        where_sql = " AND ".join(clauses)
        rows = cur.execute(
            f"SELECT * FROM claims WHERE {where_sql} ORDER BY rowid ASC", params
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def search_claims_bm25(
        self, conversation_id: str, query: str, top_k: int = 25
    ) -> List[Tuple[AtomicClaim, float]]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return []
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT claim_id, bm25(claims_fts, 0.0, 0.0, 2.0, 1.5, 2.0, 3.0, 2.0, 1.0) AS rank_score
                FROM claims_fts
                WHERE claims_fts MATCH ? AND conversation_id = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (fts_q, conversation_id, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        results: List[Tuple[AtomicClaim, float]] = []
        for r in rows:
            c = self.get_claim(r["claim_id"])
            if c is not None:
                results.append((c, float(-r["rank_score"])))
        return results

    # ------------------------------------------------------------------
    # Layer 3: Linked Entity/Topic Wikis & Entity Graph Operations
    # ------------------------------------------------------------------
    def upsert_entity_edge(self, edge: EntityEdge) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO entity_edges (
                edge_id, conversation_id, source_entity, relation,
                target_entity, claim_id, source_turn_ids_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.conversation_id,
                edge.source_entity,
                edge.relation,
                edge.target_entity,
                edge.claim_id,
                json.dumps(edge.source_turn_ids),
                edge.status,
            ),
        )
        self.conn.commit()

    def get_entity_Claims_1hop(
        self, conversation_id: str, entity_names: List[str], limit: int = 25
    ) -> List[AtomicClaim]:
        """1-hop spreading activation over the entity graph and subject/object slots."""
        if not entity_names:
            return []
        cur = self.conn.cursor()
        lowered = [e.lower().strip() for e in entity_names if e.strip()]
        if not lowered:
            return []
        placeholders = ",".join("?" for _ in lowered)
        edge_rows = cur.execute(
            f"""
            SELECT DISTINCT claim_id, source_entity, target_entity
            FROM entity_edges
            WHERE conversation_id = ?
              AND (lower(source_entity) IN ({placeholders})
                   OR lower(target_entity) IN ({placeholders}))
            LIMIT ?
            """,
            [conversation_id, *lowered, *lowered, limit * 2],
        ).fetchall()

        claim_ids = [r["claim_id"] for r in edge_rows]
        # Also match claims where subject or object contains the entity
        direct_rows = cur.execute(
            f"""
            SELECT claim_id FROM claims
            WHERE conversation_id = ?
              AND (lower(subject) IN ({placeholders})
                   OR lower(object_value) IN ({placeholders}))
            LIMIT ?
            """,
            [conversation_id, *lowered, *lowered, limit * 2],
        ).fetchall()
        for r in direct_rows:
            if r["claim_id"] not in claim_ids:
                claim_ids.append(r["claim_id"])

        results: List[AtomicClaim] = []
        for cid in claim_ids[:limit]:
            c = self.get_claim(cid)
            if c is not None:
                results.append(c)
        return results

    def upsert_wiki_page(self, page: WikiPage) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO wiki_pages (
                page_id, conversation_id, title, page_type, markdown_body,
                linked_entities_json, source_claim_ids_json, source_turn_ids_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                page.page_id,
                page.conversation_id,
                page.title,
                page.page_type,
                page.markdown_body,
                json.dumps(page.linked_entities),
                json.dumps(page.source_claim_ids),
                json.dumps(page.source_turn_ids),
                page.updated_at,
            ),
        )
        cur.execute(
            "DELETE FROM wiki_fts WHERE conversation_id = ? AND page_id = ?",
            (page.conversation_id, page.page_id),
        )
        cur.execute(
            """
            INSERT INTO wiki_fts (conversation_id, page_id, title, markdown_body)
            VALUES (?, ?, ?, ?)
            """,
            (page.conversation_id, page.page_id, page.title, page.markdown_body),
        )
        self.conn.commit()

    def get_wiki_pages(self, conversation_id: str) -> List[WikiPage]:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM wiki_pages WHERE conversation_id = ? ORDER BY title ASC",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_wiki(r) for r in rows]

    def search_wiki_bm25(
        self, conversation_id: str, query: str, top_k: int = 5
    ) -> List[Tuple[WikiPage, float]]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return []
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT page_id, bm25(wiki_fts, 0.0, 0.0, 3.0, 1.5) AS rank_score
                FROM wiki_fts
                WHERE wiki_fts MATCH ? AND conversation_id = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (fts_q, conversation_id, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        pages_by_id = {p.page_id: p for p in self.get_wiki_pages(conversation_id)}
        out: List[Tuple[WikiPage, float]] = []
        for r in rows:
            pid = r["page_id"]
            if pid in pages_by_id:
                out.append((pages_by_id[pid], float(-r["rank_score"])))
        return out

    # ------------------------------------------------------------------
    # Layer 4: Global Memory Header Operations
    # ------------------------------------------------------------------
    def upsert_global_header(self, header: GlobalMemoryHeader) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO global_headers (
                conversation_id, header_markdown, token_count, source_turn_ids_json, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                header.conversation_id,
                header.header_markdown,
                header.token_count,
                json.dumps(header.source_turn_ids),
                header.updated_at,
            ),
        )
        self.conn.commit()

    def get_global_header(self, conversation_id: str) -> Optional[GlobalMemoryHeader]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM global_headers WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not row:
            return None
        return GlobalMemoryHeader(
            conversation_id=row["conversation_id"],
            header_markdown=row["header_markdown"],
            token_count=int(row["token_count"]),
            source_turn_ids=json.loads(row["source_turn_ids_json"]),
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> Turn:
        return Turn(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            conversation_id=row["conversation_id"],
            speaker=row["speaker"],
            text=row["text"],
            timestamp=row["timestamp"],
            turn_index=int(row["turn_index"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> AtomicClaim:
        return AtomicClaim(
            claim_id=row["claim_id"],
            conversation_id=row["conversation_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            statement=row["statement"],
            source_turn_ids=json.loads(row["source_turn_ids_json"]),
            source_session_ids=json.loads(row["source_session_ids_json"]),
            evidence_quotes=json.loads(row["evidence_quotes_json"]),
            valid_from=row["valid_from"],
            normalized_date=row["normalized_date"],
            valid_until=row["valid_until"],
            status=row["status"],
            superseded_by=row["superseded_by"],
            is_dream_synthesis=bool(row["is_dream_synthesis"]),
            salience=float(row["salience"]),
        )

    @staticmethod
    def _row_to_wiki(row: sqlite3.Row) -> WikiPage:
        return WikiPage(
            page_id=row["page_id"],
            conversation_id=row["conversation_id"],
            title=row["title"],
            page_type=row["page_type"],
            markdown_body=row["markdown_body"],
            linked_entities=json.loads(row["linked_entities_json"]),
            source_claim_ids=json.loads(row["source_claim_ids_json"]),
            source_turn_ids=json.loads(row["source_turn_ids_json"]),
            updated_at=row["updated_at"],
        )
