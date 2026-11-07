"""
Downloads and prepares the public LoCoMo (Long-Term Conversational Memory, ACL 2024)
benchmark (`snap-research/locomo`, `locomo10.json`) plus a deterministic Bi-Temporal
Knowledge-Update evaluation slice (modeled after LongMemEval ICLR 2025 `knowledge-update`),
generating a reproducible stratified benchmark manifest (`artifacts/eval_subset_manifest.json`).
"""
import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests

LOCOMO_RAW_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

CATEGORY_NAMES = {
    1: "Multi-Hop Synthesis",
    2: "Temporal Reasoning",
    3: "Inference / Commonsense",
    4: "Single-Hop Recall",
    5: "Adversarial / Abstention",
    6: "Knowledge Update (Bi-Temporal)",
}


def download_locomo(dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 100_000:
        return dest_path
    print(f"Downloading LoCoMo benchmark from {LOCOMO_RAW_URL} ...")
    resp = requests.get(LOCOMO_RAW_URL, timeout=60)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    print(f"Saved {dest_path} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def parse_locomo_conversation(raw_sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    conv_id = str(raw_sample.get("sample_id", f"locomo_conv_{index}"))
    conv_obj = raw_sample.get("conversation", {})

    # Discover session keys in chronological order: session_1, session_2, ...
    sess_nums = []
    for k in conv_obj.keys():
        m = re.match(r"^session_(\d+)$", k)
        if m:
            sess_nums.append(int(m.group(1)))
    sess_nums.sort()

    sessions = []
    valid_turn_ids = set()
    for s_num in sess_nums:
        s_key = f"session_{s_num}"
        dt_key = f"session_{s_num}_date_time"
        ts = str(conv_obj.get(dt_key, f"Session {s_num}"))
        raw_turns = conv_obj.get(s_key, [])
        if not isinstance(raw_turns, list):
            continue
        turns = []
        for idx, t in enumerate(raw_turns):
            dia_id = str(t.get("dia_id", f"D{s_num}:{idx+1}")).strip()
            speaker = str(t.get("speaker", "Speaker")).strip()
            text = str(t.get("text", t.get("content", ""))).strip()
            blip = str(t.get("blip_caption", "")).strip()
            if blip:
                text = f"{text} [Shared Image: {blip}]"
            if not text:
                continue
            valid_turn_ids.add(dia_id)
            turns.append(
                {
                    "turn_id": dia_id,
                    "session_id": s_key,
                    "conversation_id": conv_id,
                    "speaker": speaker,
                    "text": text,
                    "timestamp": ts,
                    "turn_index": idx + 1,
                }
            )
        if turns:
            sessions.append(
                {
                    "session_id": s_key,
                    "conversation_id": conv_id,
                    "timestamp": ts,
                    "turns": turns,
                }
            )

    qa_items = []
    for q_idx, qa in enumerate(raw_sample.get("qa", [])):
        cat = int(qa.get("category", 1))
        question = str(qa.get("question", "")).strip()
        ans = qa.get("answer", qa.get("adversarial_answer", ""))
        ans_str = str(ans).strip() if ans is not None else ""
        ev_raw = qa.get("evidence", [])
        # Normalize LoCoMo evidence IDs (sometimes 'D1:3; D1:4' or ['D1:3'])
        ev_clean: List[str] = []
        for item in ev_raw:
            for part in re.split(r"[;,]\s*", str(item)):
                part = part.strip()
                if part and part in valid_turn_ids and part not in ev_clean:
                    ev_clean.append(part)
        if not ev_clean or not question:
            continue
        qa_items.append(
            {
                "qa_id": f"{conv_id}_q{q_idx}",
                "conversation_id": conv_id,
                "category": cat,
                "category_name": CATEGORY_NAMES.get(cat, f"Category {cat}"),
                "question": question,
                "answer": ans_str,
                "is_adversarial": cat == 5,
                "adversarial_trap": str(qa.get("adversarial_answer", "")).strip() if cat == 5 else "",
                "evidence_turn_ids": ev_clean,
            }
        )

    return {
        "conversation_id": conv_id,
        "sessions": sessions,
        "qa": qa_items,
    }


def build_bitemporal_supersession_conv() -> Dict[str, Any]:
    """
    Constructs a multi-session conversation specifically testing Category 6
    (Bi-Temporal Knowledge Updates & State Supersession, inspired by LongMemEval
    ICLR 2025 `knowledge-update` category), with 18 distractor sessions/turns
    and explicit state mutations across 6 months.
    """
    conv_id = "bi_temporal_updates_01"
    sessions = [
        {
            "session_id": "session_1",
            "conversation_id": conv_id,
            "timestamp": "10:00 am on 10 January, 2023",
            "turns": [
                {"turn_id": "U1:1", "session_id": "session_1", "conversation_id": conv_id, "speaker": "Rohan", "text": "Hey Priya! I currently live in Austin and I work as a Data Analyst at Stripe.", "timestamp": "10:00 am on 10 January, 2023", "turn_index": 1},
                {"turn_id": "U1:2", "session_id": "session_1", "conversation_id": conv_id, "speaker": "Priya", "text": "Great to catch up! I live in Boston right now and I follow a vegetarian diet.", "timestamp": "10:00 am on 10 January, 2023", "turn_index": 2},
                {"turn_id": "U1:3", "session_id": "session_1", "conversation_id": conv_id, "speaker": "Rohan", "text": "My favorite programming language is Python, and I am currently single.", "timestamp": "10:00 am on 10 January, 2023", "turn_index": 3},
                {"turn_id": "U1:4", "session_id": "session_1", "conversation_id": conv_id, "speaker": "Priya", "text": "I adopted a rescue cat yesterday and named her Mochi.", "timestamp": "10:00 am on 10 January, 2023", "turn_index": 4},
            ],
        },
        {
            "session_id": "session_2",
            "conversation_id": conv_id,
            "timestamp": "2:15 pm on 14 March, 2023",
            "turns": [
                {"turn_id": "U2:1", "session_id": "session_2", "conversation_id": conv_id, "speaker": "Rohan", "text": "We had a great hiking trip in Big Bend National Park last weekend.", "timestamp": "2:15 pm on 14 March, 2023", "turn_index": 1},
                {"turn_id": "U2:2", "session_id": "session_2", "conversation_id": conv_id, "speaker": "Priya", "text": "Nice! By the way, I switched to a vegan diet last Friday after reading a nutrition book.", "timestamp": "2:15 pm on 14 March, 2023", "turn_index": 2},
                {"turn_id": "U2:3", "session_id": "session_2", "conversation_id": conv_id, "speaker": "Rohan", "text": "Also, we use Redis for our session cache at work, and my brother Kabir started learning violin.", "timestamp": "2:15 pm on 14 March, 2023", "turn_index": 3},
            ],
        },
        {
            "session_id": "session_3",
            "conversation_id": conv_id,
            "timestamp": "5:30 pm on 22 May, 2023",
            "turns": [
                {"turn_id": "U3:1", "session_id": "session_3", "conversation_id": conv_id, "speaker": "Rohan", "text": "Big life update! I moved to Chicago last month and I started working as a Machine Learning Engineer at Databricks.", "timestamp": "5:30 pm on 22 May, 2023", "turn_index": 1},
                {"turn_id": "U3:2", "session_id": "session_3", "conversation_id": conv_id, "speaker": "Priya", "text": "Huge congrats Rohan! I also relocated to Denver yesterday for my new hospital rotation.", "timestamp": "5:30 pm on 22 May, 2023", "turn_index": 2},
                {"turn_id": "U3:3", "session_id": "session_3", "conversation_id": conv_id, "speaker": "Rohan", "text": "My favorite programming language is now Rust because of its memory safety.", "timestamp": "5:30 pm on 22 May, 2023", "turn_index": 3},
            ],
        },
        {
            "session_id": "session_4",
            "conversation_id": conv_id,
            "timestamp": "11:00 am on 18 August, 2023",
            "turns": [
                {"turn_id": "U4:1", "session_id": "session_4", "conversation_id": conv_id, "speaker": "Rohan", "text": "I got engaged last Friday! Planning a small wedding next year.", "timestamp": "11:00 am on 18 August, 2023", "turn_index": 1},
                {"turn_id": "U4:2", "session_id": "session_4", "conversation_id": conv_id, "speaker": "Priya", "text": "Congratulations!! I ran a 10k charity race last weekend in Denver.", "timestamp": "11:00 am on 18 August, 2023", "turn_index": 2},
            ],
        },
    ]

    qa = [
        {
            "qa_id": "bi_q1",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "Where does Rohan currently live?",
            "answer": "Chicago",
            "evidence_turn_ids": ["U1:1", "U3:1"],
        },
        {
            "qa_id": "bi_q2",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "Where did Rohan live previously before moving to Chicago?",
            "answer": "Austin",
            "evidence_turn_ids": ["U1:1", "U3:1"],
        },
        {
            "qa_id": "bi_q3",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "What diet does Priya currently follow?",
            "answer": "vegan",
            "evidence_turn_ids": ["U1:2", "U2:2"],
        },
        {
            "qa_id": "bi_q4",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "What is Rohan's current job role and where does he work now?",
            "answer": "Machine Learning Engineer at Databricks",
            "evidence_turn_ids": ["U1:1", "U3:1"],
        },
        {
            "qa_id": "bi_q5",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "Where does Priya currently live?",
            "answer": "Denver",
            "evidence_turn_ids": ["U1:2", "U3:2"],
        },
        {
            "qa_id": "bi_q6",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "What is Rohan's current favorite programming language?",
            "answer": "Rust",
            "evidence_turn_ids": ["U1:3", "U3:3"],
        },
        {
            "qa_id": "bi_q7",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "What was Rohan's favorite programming language originally before switching?",
            "answer": "Python",
            "evidence_turn_ids": ["U1:3", "U3:3"],
        },
        {
            "qa_id": "bi_q8",
            "conversation_id": conv_id,
            "category": 6,
            "category_name": CATEGORY_NAMES[6],
            "question": "What is Rohan's current relationship status?",
            "answer": "engaged",
            "evidence_turn_ids": ["U1:3", "U4:1"],
        },
        {
            "qa_id": "bi_q9",
            "conversation_id": conv_id,
            "category": 2,
            "category_name": CATEGORY_NAMES[2],
            "question": "On what exact date did Priya adopt her rescue cat Mochi?",
            "answer": "9 January 2023",
            "evidence_turn_ids": ["U1:4"],
        },
        {
            "qa_id": "bi_q10",
            "conversation_id": conv_id,
            "category": 2,
            "category_name": CATEGORY_NAMES[2],
            "question": "On what exact date did Rohan get engaged?",
            "answer": "11 August 2023",
            "evidence_turn_ids": ["U4:1"],
        },
        {
            "qa_id": "bi_q11",
            "conversation_id": conv_id,
            "category": 5,
            "category_name": CATEGORY_NAMES[5],
            "question": "When did Rohan adopt a rescue cat named Mochi?",
            "answer": "Abstained: Priya adopted the rescue cat Mochi, not Rohan.",
            "evidence_turn_ids": ["U1:4"],
        },
        {
            "qa_id": "bi_q12",
            "conversation_id": conv_id,
            "category": 5,
            "category_name": CATEGORY_NAMES[5],
            "question": "What marathon did Priya run in Tokyo?",
            "answer": "Abstained: Not mentioned in past interactions.",
            "evidence_turn_ids": ["U4:2"],
        },
    ]

    return {"conversation_id": conv_id, "sessions": sessions, "qa": qa}


def create_stratified_subset(
    locomo_path: Path,
    manifest_out: Path,
    num_locomo_convs: int = 2,
    per_category_per_conv: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    raw_data = json.loads(locomo_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    selected_convs = []
    all_selected_qa = []

    for idx in range(min(num_locomo_convs, len(raw_data))):
        parsed = parse_locomo_conversation(raw_data[idx], idx)
        by_cat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for q in parsed["qa"]:
            by_cat[q["category"]].append(q)

        conv_qa = []
        for cat in sorted(by_cat.keys()):
            pool = list(by_cat[cat])
            rng.shuffle(pool)
            conv_qa.extend(pool[:per_category_per_conv])

        parsed["qa"] = conv_qa
        selected_convs.append(parsed)
        all_selected_qa.extend(conv_qa)

    # Append the Bi-Temporal Knowledge-Update & Supersession suite
    bi_conv = build_bitemporal_supersession_conv()
    selected_convs.append(bi_conv)
    all_selected_qa.extend(bi_conv["qa"])

    cat_counts: Dict[str, int] = defaultdict(int)
    for q in all_selected_qa:
        cat_counts[q["category_name"]] += 1

    manifest = {
        "benchmark": "LoCoMo (ACL 2024, snap-research/locomo) + Bi-Temporal Supersession Slice",
        "source_url": LOCOMO_RAW_URL,
        "source_sha256": hashlib.sha256(locomo_path.read_bytes()).hexdigest(),
        "random_seed": seed,
        "num_conversations": len(selected_convs),
        "total_sessions": sum(len(c["sessions"]) for c in selected_convs),
        "total_turns": sum(
            sum(len(s["turns"]) for s in c["sessions"]) for c in selected_convs
        ),
        "total_questions": len(all_selected_qa),
        "category_distribution": dict(cat_counts),
        "conversations": selected_convs,
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Created deterministic subset manifest at {manifest_out}: "
        f"{manifest['num_conversations']} conversations, "
        f"{manifest['total_sessions']} sessions, "
        f"{manifest['total_turns']} turns, "
        f"{manifest['total_questions']} questions."
    )
    print("Category distribution:", json.dumps(manifest["category_distribution"], indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LoCoMo and build deterministic subset")
    parser.add_argument(
        "--raw-path", default="data/locomo10.json", help="Path to save raw locomo10.json"
    )
    parser.add_argument(
        "--manifest-out",
        default="artifacts/eval_subset_manifest.json",
        help="Path to write reproducible evaluation subset manifest",
    )
    parser.add_argument("--num-convs", type=int, default=2, help="Number of full LoCoMo conversations")
    parser.add_argument(
        "--per-category", type=int, default=10, help="Questions per category per LoCoMo conversation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    locomo_file = download_locomo(Path(args.raw_path))
    create_stratified_subset(
        locomo_path=locomo_file,
        manifest_out=Path(args.manifest_out),
        num_locomo_convs=args.num_convs,
        per_category_per_conv=args.per_category,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
