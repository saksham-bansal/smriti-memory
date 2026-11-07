"""
Command-Line Interface for Smriti Agent Memory Across Sessions.

Supports:
  - `ingest`: Ingest a session JSON file into persistent SQLite + Vector storage
  - `dream`: Run offline Dream consolidation & compile linked Entity/Topic Wikis + global.md
  - `ask`: Query the persistent memory store with inspectable claim-level provenance
  - `inspect`: Inspect Layer 1-4 counts, global.md index, Wiki pages, and Superseded vs Active claims
  - `demo-persistence`: End-to-end multi-process demonstration proving persistence across OS process restarts
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from smriti.agent.responder import SmritiMemoryAgent
from smriti.config import SmritiConfig
from smriti.models import SessionTranscript, Turn


def _load_sessions_from_json(path: str) -> List[SessionTranscript]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "sessions" in data:
        conv_id = data.get("conversation_id", "default_conv")
        out: List[SessionTranscript] = []
        for s in data["sessions"]:
            s_id = s["session_id"]
            ts = s["timestamp"]
            turns = [
                Turn(
                    turn_id=t["turn_id"],
                    session_id=s_id,
                    conversation_id=conv_id,
                    speaker=t["speaker"],
                    text=t["text"],
                    timestamp=t.get("timestamp", ts),
                    turn_index=idx,
                )
                for idx, t in enumerate(s.get("turns", []))
            ]
            out.append(SessionTranscript(session_id=s_id, conversation_id=conv_id, timestamp=ts, turns=turns))
        return out
    elif isinstance(data, dict) and "session_id" in data:
        conv_id = data.get("conversation_id", "default_conv")
        s_id = data["session_id"]
        ts = data["timestamp"]
        turns = [
            Turn(
                turn_id=t["turn_id"],
                session_id=s_id,
                conversation_id=conv_id,
                speaker=t["speaker"],
                text=t["text"],
                timestamp=t.get("timestamp", ts),
                turn_index=idx,
            )
            for idx, t in enumerate(data.get("turns", []))
        ]
        return [SessionTranscript(session_id=s_id, conversation_id=conv_id, timestamp=ts, turns=turns)]
    raise ValueError(f"Unsupported session JSON schema in {path}")


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = SmritiConfig(db_path=args.db)
    agent = SmritiMemoryAgent(cfg)
    try:
        sessions = _load_sessions_from_json(args.input)
        totals: Dict[str, int] = {"sessions": 0, "turns": 0, "claims": 0}
        last_conv = ""
        for s in sessions:
            res = agent.ingest_session(s, run_dream=False)
            totals["sessions"] += 1
            totals["turns"] += res["ingested_turns"]
            totals["claims"] += res["extracted_claims"]
            last_conv = s.conversation_id
        dream_res = {}
        if args.dream and last_conv:
            dream_res = agent.dream(last_conv)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "db_path": args.db,
                    "conversation_id": last_conv,
                    **totals,
                    "dream": dream_res,
                },
                indent=2,
            )
        )
    finally:
        agent.close()


def cmd_dream(args: argparse.Namespace) -> None:
    cfg = SmritiConfig(db_path=args.db)
    agent = SmritiMemoryAgent(cfg)
    try:
        stats = agent.dream(args.conversation_id)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "db_path": args.db,
                    "conversation_id": args.conversation_id,
                    "dream_stats": stats,
                },
                indent=2,
            )
        )
    finally:
        agent.close()


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = SmritiConfig(db_path=args.db, token_budget=args.budget)
    agent = SmritiMemoryAgent(cfg)
    try:
        resp = agent.answer_query(
            conversation_id=args.conversation_id,
            query=args.query,
            budget_override=args.budget,
        )
        print(json.dumps(resp.to_dict(), indent=2))
    finally:
        agent.close()


def cmd_inspect(args: argparse.Namespace) -> None:
    cfg = SmritiConfig(db_path=args.db)
    agent = SmritiMemoryAgent(cfg)
    try:
        conv_id = args.conversation_id
        sessions = agent.store.get_sessions(conv_id)
        turns = agent.store.get_all_turns(conv_id)
        claims = agent.store.get_claims(conv_id)
        active_c = [c for c in claims if c.status == "ACTIVE"]
        superseded_c = [c for c in claims if c.status == "SUPERSEDED"]
        wikis = agent.store.get_wiki_pages(conv_id)
        gh = agent.store.get_global_header(conv_id)

        out: Dict[str, Any] = {
            "conversation_id": conv_id,
            "db_path": args.db,
            "layer1_episodic": {"sessions": len(sessions), "turns": len(turns)},
            "layer2_claims": {
                "total": len(claims),
                "active": len(active_c),
                "superseded": len(superseded_c),
                "dream_syntheses": sum(1 for c in claims if c.is_dream_synthesis),
            },
            "layer3_wikis": [
                {
                    "title": w.title,
                    "type": w.page_type,
                    "linked_entities": w.linked_entities,
                    "source_turn_ids": w.source_turn_ids,
                }
                for w in wikis
            ],
            "layer4_global_header": gh.header_markdown if gh else None,
        }
        print(json.dumps(out, indent=2))
    finally:
        agent.close()


def cmd_demo_persistence(args: argparse.Namespace) -> None:
    """
    Demonstrates cross-session memory persistence across separate OS processes:
      - Process 1: Ingests Session 1 (May 2023) and exits
      - Process 2: Ingests Session 2 (July 2023, with state updates & relative time) and exits
      - Process 3: Runs offline Dream consolidation & Wiki compilation and exits
      - Process 4: Queries the persisted database on:
          (a) Current vs Historical superseded state (Bi-temporal supersession)
          (b) Relative-to-absolute temporal reasoning
          (c) False-premise / adversarial abstention
    """
    db_path = args.db or "artifacts/demo_persistence.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(db_path).exists():
        Path(db_path).unlink()

    conv_id = "sarvam_demo_user"
    session_1 = {
        "conversation_id": conv_id,
        "session_id": "session_1",
        "timestamp": "10:15 am on 8 May, 2023",
        "turns": [
            {
                "turn_id": "S1:1",
                "speaker": "Aarav",
                "text": "Hey Meera! I currently live in Bangalore and I work as a Backend Engineer at Flipkart.",
            },
            {
                "turn_id": "S1:2",
                "speaker": "Meera",
                "text": "Awesome! I adopted a golden retriever puppy yesterday and my sister Ananya loves pottery.",
            },
            {
                "turn_id": "S1:3",
                "speaker": "Aarav",
                "text": "I follow a vegetarian diet right now, and we use PostgreSQL for our user-profile service.",
            },
        ],
    }

    session_2 = {
        "conversation_id": conv_id,
        "session_id": "session_2",
        "timestamp": "3:30 pm on 17 August, 2023",
        "turns": [
            {
                "turn_id": "S2:1",
                "speaker": "Aarav",
                "text": "Big update! I moved to Hyderabad last Friday and I switched to a vegan diet.",
            },
            {
                "turn_id": "S2:2",
                "speaker": "Meera",
                "text": "Congrats on the move! I'm looking for a birthday gift for my sister Ananya next week.",
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        s1_file = Path(tmpdir) / "s1.json"
        s2_file = Path(tmpdir) / "s2.json"
        s1_file.write_text(json.dumps(session_1), encoding="utf-8")
        s2_file.write_text(json.dumps(session_2), encoding="utf-8")

        py = sys.executable
        env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}

        print("=== [Process 1] Ingesting Session 1 (8 May 2023) & terminating process ===")
        r1 = subprocess.run(
            [py, "-m", "smriti.cli", "ingest", "--db", db_path, "--input", str(s1_file)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        print(r1.stdout.strip())

        print("\n=== [Process 2] New OS Process: Ingesting Session 2 (17 August 2023) & terminating ===")
        r2 = subprocess.run(
            [py, "-m", "smriti.cli", "ingest", "--db", db_path, "--input", str(s2_file)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        print(r2.stdout.strip())

        print("\n=== [Process 3] New OS Process: Running Offline 'Dream' Consolidation & Wiki Compilation ===")
        r3 = subprocess.run(
            [py, "-m", "smriti.cli", "dream", "--db", db_path, "--conversation-id", conv_id],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        print(r3.stdout.strip())

        queries = [
            ("Bi-Temporal State Update (Current)", "Where does Aarav live currently and what diet does he follow?"),
            ("Bi-Temporal Historical Recall", "Where did Aarav live previously before moving?"),
            ("Relative-to-Absolute Temporal Reasoning", "On what exact date did Meera adopt her golden retriever puppy?"),
            ("Adversarial / Speaker-Swap Trap", "When did Aarav adopt a golden retriever puppy?"),
        ]

        for label, q in queries:
            print(f"\n=== [Process 4+] New OS Process Query — {label} ===")
            print(f"Query: {q}")
            rq = subprocess.run(
                [
                    py, "-m", "smriti.cli", "ask",
                    "--db", db_path,
                    "--conversation-id", conv_id,
                    "--query", q,
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            print(rq.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smriti",
        description="Smriti: Bi-Temporal 4-Layer Agent Memory with Dream Consolidation & Claim-Level Provenance",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest session transcript JSON into persistent store")
    p_ing.add_argument("--db", required=True, help="Path to SQLite memory database")
    p_ing.add_argument("--input", required=True, help="Path to session JSON file")
    p_ing.add_argument("--dream", action="store_true", help="Run offline Dream pass after ingestion")
    p_ing.set_defaults(func=cmd_ingest)

    p_drm = sub.add_parser("dream", help="Run offline Dream consolidation and Wiki compilation")
    p_drm.add_argument("--db", required=True, help="Path to SQLite memory database")
    p_drm.add_argument("--conversation-id", required=True, help="Conversation ID to consolidate")
    p_drm.set_defaults(func=cmd_dream)

    p_ask = sub.add_parser("ask", help="Query persistent memory with claim-level provenance")
    p_ask.add_argument("--db", required=True, help="Path to SQLite memory database")
    p_ask.add_argument("--conversation-id", required=True, help="Conversation ID")
    p_ask.add_argument("--query", required=True, help="User question")
    p_ask.add_argument("--budget", type=int, default=1500, help="Token budget for memory context")
    p_ask.set_defaults(func=cmd_ask)

    p_ins = sub.add_parser("inspect", help="Inspect 4-layer memory store contents")
    p_ins.add_argument("--db", required=True, help="Path to SQLite memory database")
    p_ins.add_argument("--conversation-id", required=True, help="Conversation ID")
    p_ins.set_defaults(func=cmd_inspect)

    p_demo = sub.add_parser(
        "demo-persistence",
        help="Run automated multi-process cross-session persistence and provenance demo",
    )
    p_demo.add_argument("--db", default="artifacts/demo_persistence.db", help="Demo DB path")
    p_demo.set_defaults(func=cmd_demo_persistence)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
