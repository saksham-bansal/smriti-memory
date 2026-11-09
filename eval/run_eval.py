"""
Single-Command Reproducible Evaluation Pipeline (`eval/run_eval.py`).

Satisfies Hard Requirement 3 & 4:
  - Evaluates 3 standard baselines (`baseline_recency`, `baseline_summary`, `baseline_naive_rag`)
  - Evaluates `smriti_full` (4-Layer Bi-Temporal Memory + Dreaming + Linked Wikis + Hybrid RRF)
  - Evaluates 3 architectural ablations (`ablation_no_dream_wiki`, `ablation_no_temporal`, `ablation_dense_only`)
  - Writes raw per-question outputs to `artifacts/eval_results/raw_predictions.jsonl`
  - Computes aggregate & per-category tables in `artifacts/eval_results/summary_metrics.json`
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from eval.baselines import (
    NaiveDenseRAGBaseline,
    RecencyBufferBaseline,
    RollingSummaryBaseline,
)
from eval.metrics import evaluate_single_prediction
from smriti.agent.responder import SmritiMemoryAgent
from smriti.config import SmritiConfig
from smriti.llm.client import estimate_tokens
from smriti.models import SessionTranscript, Turn

SYSTEM_CONFIGS = [
    ("baseline_recency", "1. Sliding-Window Recency Buffer"),
    ("baseline_summary", "2. Rolling Session Summarization"),
    ("baseline_naive_rag", "3. Standard Dense Vector RAG (Raw Turns)"),
    ("smriti_full", "4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)"),
    ("ablation_no_dream_wiki", "5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession"),
    ("ablation_no_temporal", "6. Ablation: w/o Relative-to-Absolute Temporal Anchoring"),
    ("ablation_dense_only", "7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only)"),
]


def _dict_to_sessions(conv_dict: Dict[str, Any]) -> List[SessionTranscript]:
    conv_id = conv_dict["conversation_id"]
    sessions: List[SessionTranscript] = []
    for s in conv_dict["sessions"]:
        turns = [
            Turn(
                turn_id=t["turn_id"],
                session_id=s["session_id"],
                conversation_id=conv_id,
                speaker=t["speaker"],
                text=t["text"],
                timestamp=t["timestamp"],
                turn_index=t.get("turn_index", i + 1),
            )
            for i, t in enumerate(s["turns"])
        ]
        sessions.append(
            SessionTranscript(
                session_id=s["session_id"],
                conversation_id=conv_id,
                timestamp=s["timestamp"],
                turns=turns,
            )
        )
    return sessions


def _build_smriti_variant(variant_id: str, db_path: str, token_budget: int) -> SmritiMemoryAgent:
    if variant_id == "smriti_full":
        cfg = SmritiConfig(
            db_path=db_path,
            token_budget=token_budget,
            enable_dreaming=True,
            enable_temporal_anchoring=True,
            enable_supersession=True,
            enable_hybrid_bm25=True,
            enable_entity_graph=True,
        )
    elif variant_id == "ablation_no_dream_wiki":
        cfg = SmritiConfig(
            db_path=db_path,
            token_budget=token_budget,
            enable_dreaming=False,
            enable_temporal_anchoring=True,
            enable_supersession=False,
            enable_hybrid_bm25=True,
            enable_entity_graph=True,
        )
    elif variant_id == "ablation_no_temporal":
        cfg = SmritiConfig(
            db_path=db_path,
            token_budget=token_budget,
            enable_dreaming=True,
            enable_temporal_anchoring=False,
            enable_supersession=True,
            enable_hybrid_bm25=True,
            enable_entity_graph=True,
        )
    elif variant_id == "ablation_dense_only":
        cfg = SmritiConfig(
            db_path=db_path,
            token_budget=token_budget,
            enable_dreaming=True,
            enable_temporal_anchoring=True,
            enable_supersession=True,
            enable_hybrid_bm25=False,
            enable_entity_graph=False,
        )
    else:
        raise ValueError(f"Unknown Smriti variant: {variant_id}")
    return SmritiMemoryAgent(cfg)


def aggregate_records(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    metric_keys = [
        "judge_score",
        "answer_recall",
        "token_f1",
        "citation_recall",
        "citation_precision",
        "citation_f1",
        "retrieval_recall",
        "session_provenance_hit",
        "citation_validity",
        "prompt_tokens",
        "token_savings_pct",
        "latency_ms",
    ]
    overall = {k: round(sum(r[k] for r in records) / n, 4) for k in metric_keys}

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cat[r["category_name"]].append(r)

    per_category = {}
    for cat_name, cat_recs in sorted(by_cat.items()):
        cn = len(cat_recs)
        per_category[cat_name] = {
            "count": cn,
            "judge_score": round(sum(r["judge_score"] for r in cat_recs) / cn, 4),
            "answer_recall": round(sum(r["answer_recall"] for r in cat_recs) / cn, 4),
            "citation_f1": round(sum(r["citation_f1"] for r in cat_recs) / cn, 4),
            "citation_recall": round(sum(r["citation_recall"] for r in cat_recs) / cn, 4),
            "retrieval_recall": round(sum(r["retrieval_recall"] for r in cat_recs) / cn, 4),
        }

    return {
        "label": label,
        "num_questions": n,
        "overall": overall,
        "per_category": per_category,
    }


def run_evaluation(manifest_path: Path, output_dir: Path, token_budget: int = 1500) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = output_dir / "raw_predictions.jsonl"
    summary_json_path = output_dir / "summary_metrics.json"

    all_raw_records: List[Dict[str, Any]] = []
    system_summaries: Dict[str, Any] = {}

    # Compute full-context token counts per conversation
    full_tokens_by_conv: Dict[str, int] = {}
    for conv in manifest["conversations"]:
        cid = conv["conversation_id"]
        full_str = "\n".join(
            f"[{t['turn_id']}] ({t['timestamp']}) {t['speaker']}: {t['text']}"
            for s in conv["sessions"]
            for t in s["turns"]
        )
        full_tokens_by_conv[cid] = max(estimate_tokens(full_str), 2500)

    with tempfile.TemporaryDirectory() as tmpdir:
        for sys_id, sys_label in SYSTEM_CONFIGS:
            print(f"Running evaluation for {sys_label} ({sys_id}) ...")
            sys_records: List[Dict[str, Any]] = []

            for conv in manifest["conversations"]:
                conv_id = conv["conversation_id"]
                sessions = _dict_to_sessions(conv)
                qa_list = conv["qa"]
                full_tok = full_tokens_by_conv[conv_id]

                if sys_id == "baseline_recency":
                    runner = RecencyBufferBaseline(token_budget=token_budget)
                    for s in sessions:
                        runner.ingest_session(s)

                    def _eval_recency(qa: Dict[str, Any]) -> Dict[str, Any]:
                        resp = runner.answer_query(conv_id, qa["question"])
                        rec = evaluate_single_prediction(qa, resp, full_tok)
                        rec["system_id"] = sys_id
                        rec["system_label"] = sys_label
                        return rec

                    with ThreadPoolExecutor(max_workers=4) as pool:
                        sys_records.extend(list(pool.map(_eval_recency, qa_list)))

                elif sys_id == "baseline_summary":
                    runner = RollingSummaryBaseline(token_budget=token_budget)
                    for s in sessions:
                        runner.ingest_session(s)

                    def _eval_summary(qa: Dict[str, Any]) -> Dict[str, Any]:
                        resp = runner.answer_query(conv_id, qa["question"])
                        rec = evaluate_single_prediction(qa, resp, full_tok)
                        rec["system_id"] = sys_id
                        rec["system_label"] = sys_label
                        return rec

                    with ThreadPoolExecutor(max_workers=4) as pool:
                        sys_records.extend(list(pool.map(_eval_summary, qa_list)))

                elif sys_id == "baseline_naive_rag":
                    runner = NaiveDenseRAGBaseline(token_budget=token_budget)
                    for s in sessions:
                        runner.ingest_session(s)
                    rag_lock = threading.Lock()

                    def _eval_rag(qa: Dict[str, Any]) -> Dict[str, Any]:
                        with rag_lock:
                             hits = runner.vector_index.search(conv_id, "turn", qa["question"], top_k=10)
                             turn_ids = [tid for tid, _ in hits]
                             turns = runner.store.get_turns_by_ids(conv_id, turn_ids)
                             all_turns = runner.store.get_all_turns(conv_id)
                        packed = []
                        used = 0
                        for t in turns:
                            cost = estimate_tokens(f"[{t.turn_id}] ({t.timestamp}) {t.speaker}: {t.text}")
                            if used + cost > token_budget and packed:
                                break
                            packed.append(t)
                            used += cost
                        known_entities = sorted({t.speaker for t in all_turns})
                        from eval.baselines import _baseline_synthesize_from_turns
                        resp = _baseline_synthesize_from_turns(qa["question"], packed, known_entities, runner.llm)
                        rec = evaluate_single_prediction(qa, resp, full_tok)
                        rec["system_id"] = sys_id
                        rec["system_label"] = sys_label
                        return rec

                    with ThreadPoolExecutor(max_workers=4) as pool:
                        sys_records.extend(list(pool.map(_eval_rag, qa_list)))

                else:
                    # Smriti Full or Ablation Variant: test real persistence across close & reopen!
                    db_file = str(Path(tmpdir) / f"{sys_id}_{conv_id}.db")
                    ingestor_agent = _build_smriti_variant(sys_id, db_file, token_budget)
                    for s in sessions:
                        ingestor_agent.ingest_session(s, run_dream=False)
                    if ingestor_agent.config.enable_dreaming:
                        ingestor_agent.dream(conv_id)
                    ingestor_agent.close()

                    # Re-open from persisted SQLite file per worker to verify cross-restart persistence
                    def _eval_smriti(qa: Dict[str, Any]) -> Dict[str, Any]:
                        q_agent = _build_smriti_variant(sys_id, db_file, token_budget)
                        try:
                            resp = q_agent.answer_query(conv_id, qa["question"])
                            rec = evaluate_single_prediction(qa, resp, full_tok)
                            rec["system_id"] = sys_id
                            rec["system_label"] = sys_label
                            return rec
                        finally:
                            q_agent.close()

                    with ThreadPoolExecutor(max_workers=4) as pool:
                        sys_records.extend(list(pool.map(_eval_smriti, qa_list)))

            all_raw_records.extend(sys_records)
            agg = aggregate_records(sys_records, sys_label)
            system_summaries[sys_id] = agg
            ov = agg["overall"]
            print(
                f"  -> Judge Acc: {ov['judge_score']*100:.1f}% | "
                f"Citation F1: {ov['citation_f1']*100:.1f}% | "
                f"Retrieval Recall: {ov['retrieval_recall']*100:.1f}% | "
                f"Mean Tokens: {ov['prompt_tokens']:.0f}"
            )

    with raw_jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_raw_records:
            f.write(json.dumps(r) + "\n")

    summary_payload = {
        "benchmark": manifest["benchmark"],
        "source_sha256": manifest["source_sha256"],
        "random_seed": manifest["random_seed"],
        "num_conversations": manifest["num_conversations"],
        "total_sessions": manifest["total_sessions"],
        "total_turns": manifest["total_turns"],
        "total_questions": manifest["total_questions"],
        "token_budget": token_budget,
        "full_context_tokens_by_conv": full_tokens_by_conv,
        "systems": system_summaries,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nSaved raw predictions ({len(all_raw_records)} rows) to {raw_jsonl_path}")
    print(f"Saved summary metrics to {summary_json_path}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible LoCoMo + Bi-Temporal evaluation")
    parser.add_argument(
        "--subset",
        default="artifacts/eval_subset_manifest.json",
        help="Path to subset manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval_results",
        help="Directory to write raw_predictions.jsonl and summary_metrics.json",
    )
    parser.add_argument("--budget", type=int, default=1500, help="Memory context token budget")
    args = parser.parse_args()

    run_evaluation(
        manifest_path=Path(args.subset),
        output_dir=Path(args.output_dir),
        token_budget=args.budget,
    )


if __name__ == "__main__":
    main()
