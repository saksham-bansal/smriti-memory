"""
Deterministic Report & Table Builder (`eval/build_report.py`).

Rebuilds summary metrics and Markdown tables directly from `raw_predictions.jsonl`
(satisfying Hard Requirement 4: 'One command must rerun the evaluation and rebuild
the results from raw outputs').
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from eval.run_eval import SYSTEM_CONFIGS, aggregate_records


def rebuild_from_raw(results_dir: Path, tables_md_out: Path) -> Dict[str, Any]:
    raw_path = results_dir / "raw_predictions.jsonl"
    summary_path = results_dir / "summary_metrics.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}; run `eval/run_eval.py` first.")

    by_sys: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_sys[rec["system_id"]].append(rec)

    rebuilt_systems: Dict[str, Any] = {}
    for sys_id, sys_label in SYSTEM_CONFIGS:
        if sys_id in by_sys:
            rebuilt_systems[sys_id] = aggregate_records(by_sys[sys_id], sys_label)

    existing_meta = {}
    if summary_path.exists():
        existing_meta = json.loads(summary_path.read_text(encoding="utf-8"))
    existing_meta["systems"] = rebuilt_systems
    summary_path.write_text(json.dumps(existing_meta, indent=2), encoding="utf-8")

    # Build Markdown Summary Tables
    lines: List[str] = [
        "## Rebuilt Evaluation Results (Generated from `artifacts/eval_results/raw_predictions.jsonl`)\n",
        "### Table 1: Overall Benchmark Performance Across Baselines, `Smriti`, and Ablations\n",
        "| System / Configuration | Answer Judge Score (%) | Answer Token Recall (%) | Provenance Citation F1 (%) | Provenance Citation Recall (%) | Context Retrieval Recall@k (%) | Citation Validity (%) | Mean Prompt Tokens | Token Savings vs. Full Replay (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for sys_id, sys_label in SYSTEM_CONFIGS:
        if sys_id not in rebuilt_systems:
            continue
        ov = rebuilt_systems[sys_id]["overall"]
        bold = "**" if sys_id == "smriti_full" else ""
        lines.append(
            f"| {bold}{sys_label}{bold} | "
            f"{bold}{ov['judge_score']*100:.1f}%{bold} | "
            f"{bold}{ov['answer_recall']*100:.1f}%{bold} | "
            f"{bold}{ov['citation_f1']*100:.1f}%{bold} | "
            f"{bold}{ov['citation_recall']*100:.1f}%{bold} | "
            f"{bold}{ov['retrieval_recall']*100:.1f}%{bold} | "
            f"{bold}{ov['citation_validity']*100:.1f}%{bold} | "
            f"{ov['prompt_tokens']:.0f} | "
            f"{ov['token_savings_pct']:.1f}% |"
        )

    lines.append("\n### Table 2: Per-Category Judge Accuracy Breakdown (%)\n")
    cat_order = [
        "Single-Hop Recall",
        "Multi-Hop Synthesis",
        "Temporal Reasoning",
        "Inference / Commonsense",
        "Adversarial / Abstention",
        "Knowledge Update (Bi-Temporal)",
    ]
    lines.append(
        "| System / Configuration | Single-Hop | Multi-Hop | Temporal | Inference | Adversarial Abstention | Bi-Temporal Update |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sys_id, sys_label in SYSTEM_CONFIGS:
        if sys_id not in rebuilt_systems:
            continue
        pcat = rebuilt_systems[sys_id]["per_category"]
        bold = "**" if sys_id == "smriti_full" else ""
        cells = []
        for cname in cat_order:
            val = pcat.get(cname, {}).get("judge_score", 0.0) * 100.0
            cells.append(f"{bold}{val:.1f}%{bold}")
        lines.append(f"| {bold}{sys_label}{bold} | " + " | ".join(cells) + " |")

    lines.append("\n### Table 3: Per-Category Ground-Truth Provenance Citation Recall (%)\n")
    lines.append(
        "| System / Configuration | Single-Hop | Multi-Hop | Temporal | Inference | Adversarial | Bi-Temporal Update |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sys_id, sys_label in SYSTEM_CONFIGS:
        if sys_id not in rebuilt_systems:
            continue
        pcat = rebuilt_systems[sys_id]["per_category"]
        bold = "**" if sys_id == "smriti_full" else ""
        cells = []
        for cname in cat_order:
            val = pcat.get(cname, {}).get("citation_recall", 0.0) * 100.0
            cells.append(f"{bold}{val:.1f}%{bold}")
        lines.append(f"| {bold}{sys_label}{bold} | " + " | ".join(cells) + " |")

    md_text = "\n".join(lines) + "\n"
    tables_md_out.parent.mkdir(parents=True, exist_ok=True)
    tables_md_out.write_text(md_text, encoding="utf-8")
    print(f"Rebuilt summary tables at {tables_md_out}")
    return existing_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild evaluation tables from raw_predictions.jsonl")
    parser.add_argument(
        "--results-dir",
        default="artifacts/eval_results",
        help="Directory containing raw_predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/eval_results/rebuilt_tables.md",
        help="Path to write rebuilt markdown tables",
    )
    args = parser.parse_args()
    rebuild_from_raw(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
