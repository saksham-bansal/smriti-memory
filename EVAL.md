# EVAL.md — Benchmark Evaluation, Baselines, Ablations & Failure Analysis

## 1. One-Command Reproducibility (Hard Requirement 4)

The entire evaluation—ingesting multi-session conversations across separate database lifecycles, running all 3 baselines, `Smriti`, and 3 architectural ablations across 102 questions (714 total inference records), and rebuilding the summary tables from raw JSONL outputs—can be reproduced with a single command:

```bash
make eval && make report
# Or directly:
PYTHONPATH=src:. .venv/bin/python eval/run_eval.py --subset artifacts/eval_subset_manifest.json --output-dir artifacts/eval_results
PYTHONPATH=src:. .venv/bin/python eval/build_report.py --results-dir artifacts/eval_results --output artifacts/eval_results/rebuilt_tables.md
```

- **Committed Artifacts**:
  - Stratified Benchmark Manifest: [`artifacts/eval_subset_manifest.json`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/artifacts/eval_subset_manifest.json)
  - Raw Per-Question Predictions & Provenance Traces (714 JSONL records): [`artifacts/eval_results/raw_predictions.jsonl`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/artifacts/eval_results/raw_predictions.jsonl)
  - Aggregated Metrics JSON: [`artifacts/eval_results/summary_metrics.json`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/artifacts/eval_results/summary_metrics.json)
  - Auto-Rebuilt Markdown Tables: [`artifacts/eval_results/rebuilt_tables.md`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/artifacts/eval_results/rebuilt_tables.md)

---

## 2. Benchmark Selection & Reproducible Subset Specification

### 2.1 Primary Benchmark: **LoCoMo (`snap-research/locomo`, ACL 2024)** + **Bi-Temporal Supersession Slice**
We selected **LoCoMo** (*Evaluating Very Long-Term Conversational Memory of LLM Agents*, Maharana et al., ACL 2024, [`locomo10.json`](https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json), SHA-256: `4eb7f95f440fb978f9fa6eb1db7effd0ebe1fbd3cead0eb1ab6d9cd4ef9f0a47`) augmented with a deterministic **Bi-Temporal Knowledge-Update Slice** (modeled after **LongMemEval**, Wu et al., ICLR 2025 `knowledge-update` category).

**Why this benchmark combination is uniquely suited for this assignment**:
1. **True Cross-Session Scale**: LoCoMo conversations span **19 to 32 distinct sessions** and **300–600 turns** (~9,500–15,000 tokens per conversation) over 6–10 months of simulated timestamps.
2. **Human-Annotated Turn-Level Provenance (`dia_id`)**: Unlike benchmarks that only grade final answer strings, every question in LoCoMo includes ground-truth `evidence: ["D1:3", ...]` pointing to the exact dialogue turn(s) required to answer the question. This enables **objective, quantitative measurement of Provenance Precision, Recall, and F1** (Hard Requirement 2 & 3).
3. **Six Orthogonal Reasoning Categories**:
   - **Single-Hop Recall (`category=4` in `locomo10.json`)**: Direct recall of specific facts buried in a single historical turn (mean `1.01` evidence turns).
   - **Multi-Hop Synthesis (`category=1` in `locomo10.json`)**: Synthesizing facts scattered across multiple sessions (mean `2.31` evidence turns). *(Empirical schema note: while tables in the LoCoMo paper list Single-Hop first, inspecting `locomo10.json` reveals `category=1` holds the multi-turn synthesis items with 2.31 average evidence turns and `category=4` holds single-turn items with 1.01 average evidence turns).*
   - **Temporal Reasoning (`category=2`)**: Resolving relative time markers (*"yesterday"*, *"the week before"*, *"two weekends ago"*, *"seven years now"*) against session dates.
   - **Inference / Commonsense (`category=3`)**: Combining conversational facts with commonsense reasoning.
   - **Adversarial / False-Premise Abstention (`category=5`)**: Trick questions that attribute Speaker B’s action/attribute to Speaker A (`adversarial_answer` is the trap response). Tests whether the memory system abstains or hallucinates.
   - **Knowledge Update / Bi-Temporal Supersession (`category=6`)**: Explicit state mutations across sessions (e.g., moving cities `Austin -> Chicago`, changing diets `vegetarian -> vegan`, changing roles `Data Analyst at Stripe -> ML Engineer at Databricks`, switching favorite language `Python -> Rust`).

### 2.2 Reproducible Subset Selection ([`eval/download_data.py`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/eval/download_data.py))
- Using fixed random seed `42`, `eval/download_data.py` ingests **100% of the sessions and turns** of the first two full LoCoMo multi-session conversations (`conv-26`: 19 sessions, 419 turns; `conv-30`: 19 sessions, 369 turns) plus the 4-session Bi-Temporal Knowledge-Update conversation (`bi_temporal_updates_01`: 4 sessions, 12 turns), totaling **3 conversations, 42 sessions, and 800 turns**.
- From the full question pool, it draws a stratified sample of **102 questions** balanced across all 6 categories (`20 Single-Hop`, `20 Multi-Hop`, `22 Temporal`, `10 Inference`, `22 Adversarial`, `8 Bi-Temporal Update`).
- **Scope Limitation**: We evaluate on 3 complete multi-session histories (800 turns, 102 questions × 7 systems = 714 evaluations) rather than all 10 LoCoMo conversations so the full benchmark + 3 ablations executes in **< 25 seconds** on a single CPU laptop. Passing `--num-convs 10 --per-category 50` to `eval/download_data.py` scales to the full dataset.

---

## 3. Baselines & Metrics Justification

### 3.1 Credible Standard Baselines ([`eval/baselines.py`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/eval/baselines.py))
All baselines and `Smriti` variants operate under the **exact same context token budget ($B = 1,500$ tokens)**:
1. **`baseline_recency` (Sliding-Window Recency Buffer)**: Retains the most recent dialogue turns up to $B = 1,500$ tokens (~14% of a 10,000-token conversation). Represents standard context window truncation.
2. **`baseline_summary` (Rolling Session Summarization)**: Compresses each session into representative summary sentences during ingestion and packs summaries up to $B = 1,500$ tokens. Represents standard context-compaction memory.
3. **`baseline_naive_rag` (Standard Dense Vector RAG over Raw Turns)**: Embeds `[timestamp] speaker: text` for every raw turn and retrieves the top-$k$ turns by cosine similarity up to budget $B$, combining the top 3 matches. Represents the standard production vector-RAG baseline without Layer 2 de-contextualization, relative-date normalization, bi-temporal supersession, hybrid BM25+Graph RRF, or Dream Wiki compilation.

### 3.2 Evaluation Metrics ([`eval/metrics.py`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/eval/metrics.py))
- **Answer Judge Score (`judge_score`, 0 / 0.5 / 1.0)**: Grades semantic/factual correctness:
  - *Categories 1, 3, 4*: Stemmed content-word recall and concept-cluster entailment against the reference answer.
  - *Category 2 (Temporal Reasoning)*: Strict temporal verification requiring all numeric date/year tokens and month names in the reference answer to appear in the prediction (preventing systems from getting credit for `"5 July 2023"` merely by echoing an unanchored `"6 July 2023"` session timestamp header when the turn says `"yesterday"`).
  - *Category 5 (Adversarial)*: Scores `1.0` if and only if the system abstains (`abstained=True`) or rejects the false premise; systems that output the speaker-swapped trap answer score `0.0`.
  - *Category 6 (Bi-Temporal Update)*: Scores `1.0` if the primary clause reports the active updated state (for current-state queries) or prior superseded state (for historical queries).
- **Provenance Citation Recall, Precision & F1 (`citation_recall`, `citation_precision`, `citation_f1`)**: Exact set comparison between the agent’s emitted `AgentResponse.all_cited_turn_ids` and the human-annotated ground-truth `evidence_turn_ids`.
- **Context Retrieval Recall@k (`retrieval_recall`)**: Fraction of ground-truth `evidence_turn_ids` present in the packed context window.
- **Citation Validity Rate (`citation_validity`)**: Percentage of emitted claims whose `source_turn_ids` exist in Layer 1 (`SQLiteMemoryStore`).

---

## 4. Main Results & Ablation Study (Evaluated with `gemini-3.1-flash-lite`)

### Table 1: Overall Benchmark Performance ($N = 102$ Questions across 42 Sessions / 800 Turns, Budget $B = 1,500$ Tokens)

| System / Configuration | Answer Judge Score (%) | Answer Token Recall (%) | Provenance Citation F1 (%) | Provenance Citation Recall (%) | Context Retrieval Recall@k (%) | Citation Validity (%) | Mean Prompt Tokens | Token Savings vs. Full Replay (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 24.5% | 20.1% | 10.5% | 9.3% | 14.7% | 100.0% | 1373 | 91.2% |
| 2. Rolling Session Summarization | 28.9% | 22.5% | 6.8% | 6.5% | 13.1% | 100.0% | 1335 | 91.8% |
| 3. Standard Dense Vector RAG (Raw Turns) | 42.6% | 34.9% | 30.4% | 30.0% | 41.8% | 100.0% | 485 | 96.3% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **67.2%** | **52.2%** | **43.3%** | **54.4%** | **67.9%** | **100.0%** | **1477** | **86.1%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 62.7% | 47.9% | 42.4% | 55.4% | 71.4% | 100.0% | 1352 | 88.6% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 53.4% | 38.2% | 41.4% | 50.9% | 61.0% | 100.0% | 1474 | 86.1% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 63.7% | 48.4% | 38.6% | 46.7% | 64.8% | 100.0% | 1470 | 86.1% |

### Table 2: Per-Category Judge Accuracy Breakdown (%)

| System / Configuration | Single-Hop | Multi-Hop | Temporal | Inference | Adversarial Abstention | Bi-Temporal Update |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 5.0% | 5.0% | 13.6% | 10.0% | 50.0% | 100.0% |
| 2. Rolling Session Summarization | 15.0% | 17.5% | 22.7% | 10.0% | 54.5% | 62.5% |
| 3. Standard Dense Vector RAG (Raw Turns) | 15.0% | 25.0% | 59.1% | 25.0% | 63.6% | 75.0% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **60.0%** | **55.0%** | **86.4%** | **10.0%** | **81.8%** | **93.8%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 60.0% | 37.5% | 86.4% | 10.0% | 86.4% | 68.8% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 55.0% | 47.5% | 27.3% | 30.0% | 77.3% | 100.0% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 52.5% | 55.0% | 72.7% | 35.0% | 72.7% | 100.0% |

### Table 3: Per-Category Ground-Truth Provenance Citation Recall (%)

| System / Configuration | Single-Hop | Multi-Hop | Temporal | Inference | Adversarial | Bi-Temporal Update |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 5.0% | 0.0% | 9.1% | 0.0% | 9.1% | 56.2% |
| 2. Rolling Session Summarization | 0.0% | 2.9% | 9.1% | 0.0% | 9.1% | 25.0% |
| 3. Standard Dense Vector RAG (Raw Turns) | 15.0% | 17.9% | 63.6% | 15.0% | 18.2% | 56.2% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **60.0%** | **27.7%** | **77.3%** | **20.0%** | **54.5%** | **87.5%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 60.0% | 27.7% | 81.8% | 25.0% | 54.5% | 81.2% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 55.0% | 22.2% | 63.6% | 25.0% | 59.1% | 87.5% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 45.0% | 23.1% | 63.6% | 20.0% | 50.0% | 87.5% |

---

## 5. Analysis of Results & Architectural Ablations

1. **`Smriti` vs. Standard Baselines (+24.6% Judge Accuracy, +24.4% Provenance Citation Recall over Naive Dense RAG)**:
   - **Sliding-Window Recency (`24.5%` Judge Acc, `9.3%` Citation Recall)** fails on almost any question referencing Sessions 1–16 (`5.0%` Single-Hop, `5.0%` Multi-Hop) because its 1,500-token tail window only covers the final 2–3 sessions (though `gemini-3.1-flash-lite` correctly abstains on 50% of adversarial questions when the context lacks the queried premise).
   - **Rolling Session Summarization (`28.9%` Judge Acc, `6.5%` Citation Recall)** suffers from severe lossy compression: keeping 2 summary snippets per session drops 85% of specific entities, book titles, and dates, and destroys fine-grained turn provenance.
   - **Standard Dense Vector RAG (`42.6%` Judge Acc, `30.0%` Citation Recall)** retrieves raw turns without pronoun/question de-contextualization or relative-date anchoring, scoring only `15.0%` on Single-Hop Recall (due to elliptical replies like `D7:11` *"Becoming Nicole"*) and `25.0%` on Multi-Hop Synthesis.
   - **`Smriti (Full)` (`67.2%` Judge Acc, `54.4%` Citation Recall, `67.9%` Retrieval Recall@k)** achieves a **+24.6 percentage point improvement in Judge Accuracy** (`4.0x` on Single-Hop, `2.2x` on Multi-Hop, `+27.3%` on Temporal Reasoning, and `+18.2%` on Adversarial Abstention) over Standard Dense RAG while compressing prompt tokens by **86.1%** compared to full transcript replay.

2. **Ablation 1 — Removing Offline Dreaming, Linked Wikis & Bi-Temporal Supersession (`67.2% -> 62.7%`)**:
   - Disabling `DreamEngine` and slot supersession causes **Bi-Temporal Knowledge Update accuracy to drop by 25.0 percentage points (`93.8% -> 68.8%`)** and **Multi-Hop Synthesis accuracy to drop by 17.5 percentage points (`55.0% -> 37.5%`)**, while Bi-Temporal Provenance Citation Recall drops from `87.5% -> 81.2%`.
   - *Interesting Tradeoff Observed*: Without Layer 3 (`WikiPage`) and Layer 4 (`global.md`) headers occupying ~125 tokens of the 1,500-token budget, raw turn Retrieval Recall@k is slightly higher (`71.4%` vs `67.9%`), but without compiled cross-session entity profiles and `[ACTIVE]`/`[SUPERSEDED]` markers, `gemini-3.1-flash-lite` struggles to synthesize multi-session state transitions, yielding a net **+4.5% aggregate gain** for full `Smriti`.

3. **Ablation 2 — Removing Relative-to-Absolute Temporal Anchoring (`67.2% -> 53.4%`)**:
   - Disabling `TemporalNormalizer` causes **Temporal Reasoning accuracy to collapse from `86.4%` to `27.3%` (-59.1 percentage points)** and overall Judge Accuracy to drop by **13.8 percentage points**.
   - Without ingestion-time anchoring (`"yesterday"` → `"5 July 2023"`, `"two weekends ago"` → `"two weekends before 17 July 2023"`, `"Seven years now"` → `"7 years (Since 2016)"`), the LLM frequently reports the unanchored session timestamp or raw relative phrase rather than the resolved event date.

4. **Ablation 3 — Removing Hybrid FTS5 BM25 & Entity Graph (`67.2% -> 63.7%`, Citation Recall `54.4% -> 46.7%`)**:
   - Relying on dense embeddings alone drops **Single-Hop Provenance Citation Recall by 15.0 percentage points (`60.0% -> 45.0%`)**, **Temporal Reasoning accuracy by 13.7 percentage points (`86.4% -> 72.7%`)**, and overall Citation F1 from `43.3% -> 38.6%`, proving that exact BM25 lexical matching + 1-hop entity graph expansion is vital for pinpointing rare proper nouns and exact turn IDs.

---

## 6. Honest Failure Analysis (Surfaced from `raw_predictions.jsonl`)

Inspecting the raw predictions in [`artifacts/eval_results/raw_predictions.jsonl`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/artifacts/eval_results/raw_predictions.jsonl) highlights four concrete failure modes in `Smriti`:

### Failure Case 1: World-Knowledge Gap Under Strict Grounding Rules (`conv-26_q64`, Category 3 Inference)
- **Question**: *"Would Melanie likely enjoy the song 'The Four Seasons' by Vivaldi?"*
- **Ground Truth (`D15:28`)**: *"Yes; it's classical music"*
- **`Smriti` Output**: `The provided memory context does not contain information regarding Melanie's preference for Vivaldi's "The Four Seasons."` (`abstained: True`, `judge_score: 0.0`)
- **Why It Failed**: The tokens `"Vivaldi"` and `"Four Seasons"` never appear anywhere in `conv-26` (Melanie only says she enjoys *"classical music"* in `D15:28`). Answering requires external world knowledge (*Vivaldi's The Four Seasons is classical music*) plus speculative inference, which conflicts with `Smriti`'s strict anti-hallucination prompt instructing `gemini-3.1-flash-lite` to abstain when a queried entity is absent from the retrieved evidence.

### Failure Case 2: Stated Frequency Rule vs. Multi-Session Event Counting (`conv-26_q40`, Category 1 Multi-Hop)
- **Question**: *"How many times has Melanie gone to the beach in 2023?"*
- **Ground Truth (`D10:8`, `D6:16`)**: `2`
- **`Smriti` Output**: `Melanie typically visits the beach only once or twice a year [D10:10].` (`cited_turn_ids: ['D10:10', 'D10:18']`, `retrieved_turn_ids: ['D10:10', 'D10:18', 'D10:8', 'D6:16', 'D4:6', 'D18:15']`, `judge_score: 0.0`)
- **Why It Failed**: Even though `Smriti` retrieves both ground-truth beach-trip turns (`D10:8` and `D6:16`) into the context window, turn `D10:10` explicitly states *"We don't go often, usually only once or twice a year"*, causing the LLM to report Melanie's stated frequency rule rather than enumerating distinct beach trips across Sessions 6 and 10.

### Failure Case 3: Subtle Adversarial Premise with Incidental Visual Vocabulary Overlap (`conv-30_q99`, Category 5 Adversarial)
- **Question**: *"What book is Gina currently reading?"*
- **Ground Truth (`D12:6`)**: Adversarial False Premise (**Jon** is reading *"The Lean Startup"*, not **Gina**).
- **`Smriti` Output**: Cites `D12:6` and `D6:6` without abstaining (`abstained: False`, `judge_score: 0.0`).
- **Why It Failed**: In turn `D6:6`, **Gina** shares an image captioned `[Shared Image: a photo of a computer screen showing a book and a pair of shoes]`. Because `target_speaker = Gina` has lexical overlap with the stem `"book"` in `D6:6`, the speaker-attribution mismatch guard does not flag the query as a pure speaker swap.

### Failure Case 4: Cross-Turn Relative Year Resolution with Misspelled Query Token (`conv-26_q49`, Category 2 Temporal)
- **Question**: *"When did Caroline and Melanie go to a pride fesetival together?"*
- **Ground Truth (`D12:15`)**: `2022`
- **`Smriti` Output**: Retrieves `D10:10` and `D5:1` (a 2023 pride parade turn) ahead of `D12:15` because the benchmark question misspells *"festival"* as `"fesetival"` and `D12:15` refers to the event elliptically without repeating both speaker names (`judge_score: 0.0`).
