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
- **Scope Limitation**: We evaluate on 3 complete multi-session histories (800 turns, 102 questions $\times$ 7 systems = 714 evaluations) rather than all 10 LoCoMo conversations so the full benchmark + 3 ablations executes in **< 25 seconds** on a single CPU laptop. Passing `--num-convs 10 --per-category 50` to `eval/download_data.py` scales to the full dataset.

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

## 4. Main Results & Ablation Study

### Table 1: Overall Benchmark Performance ($N = 102$ Questions across 42 Sessions / 800 Turns, Budget $B = 1,500$ Tokens)

| System / Configuration | Answer Judge Score (%) | Answer Token Recall (%) | Provenance Citation F1 (%) | Provenance Citation Recall (%) | Context Retrieval Recall@k (%) | Citation Validity (%) | Mean Prompt Tokens | Token Savings vs. Full Replay (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 6.9% | 14.8% | 10.0% | 9.8% | 14.7% | 100.0% | 1373 | 91.2% |
| 2. Rolling Session Summarization | 12.2% | 19.7% | 6.2% | 5.7% | 13.1% | 100.0% | 1335 | 91.8% |
| 3. Standard Dense Vector RAG (Raw Turns) | 24.0% | 35.0% | 30.0% | 33.4% | 41.8% | 100.0% | 485 | 96.3% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **62.7%** | **57.5%** | **43.3%** | **56.1%** | **67.9%** | **100.0%** | **1477** | **86.1%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 58.3% | 53.6% | 42.3% | 54.4% | 71.4% | 100.0% | 1345 | 88.8% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 45.6% | 41.0% | 42.7% | 54.4% | 61.0% | 100.0% | 1474 | 86.1% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 60.3% | 53.6% | 39.9% | 50.1% | 64.8% | 100.0% | 1470 | 86.1% |

### Table 2: Per-Category Judge Accuracy Breakdown (%)

| System / Configuration | Single-Hop | Multi-Hop | Temporal | Inference | Adversarial Abstention | Bi-Temporal Update |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 5.0% | 0.0% | 4.5% | 0.0% | 4.5% | 50.0% |
| 2. Rolling Session Summarization | 15.0% | 2.5% | 18.2% | 20.0% | 4.5% | 25.0% |
| 3. Standard Dense Vector RAG (Raw Turns) | 25.0% | 17.5% | 45.5% | 30.0% | 0.0% | 37.5% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **65.0%** | **60.0%** | **72.7%** | **40.0%** | **50.0%** | **100.0%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 65.0% | 47.5% | 81.8% | 40.0% | 50.0% | 50.0% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 55.0% | 47.5% | 13.6% | 40.0% | 50.0% | 100.0% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 57.5% | 57.5% | 72.7% | 35.0% | 50.0% | 100.0% |

### Table 3: Per-Category Ground-Truth Provenance Citation Recall (%)

| System / Configuration | Single-Hop | Multi-Hop | Temporal | Inference | Adversarial | Bi-Temporal Update |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 0.0% | 0.0% | 9.1% | 0.0% | 13.6% | 62.5% |
| 2. Rolling Session Summarization | 0.0% | 1.7% | 4.5% | 5.0% | 9.1% | 25.0% |
| 3. Standard Dense Vector RAG (Raw Turns) | 15.0% | 10.4% | 63.6% | 25.0% | 36.4% | 56.2% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **65.0%** | **23.5%** | **68.2%** | **25.0%** | **68.2%** | **87.5%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 65.0% | 17.7% | 72.7% | 25.0% | 68.2% | 68.8% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 60.0% | 24.8% | 63.6% | 25.0% | 68.2% | 87.5% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 50.0% | 25.6% | 63.6% | 20.0% | 59.1% | 87.5% |

---

## 5. Analysis of Results & Architectural Ablations

1. **`Smriti` vs. Standard Baselines (+38.7% Judge Accuracy, +22.7% Provenance Citation Recall over Naive RAG)**:
   - **Sliding-Window Recency (`6.9%`)** fails on any question referencing Sessions 1–16 because its 1,500-token tail window only covers the final 2–3 sessions.
   - **Rolling Session Summarization (`12.2%`, `5.7%` Citation Recall)** suffers from severe lossy compression: keeping 2 summary snippets per session drops 85% of specific entities, book titles, and dates, and destroys fine-grained turn provenance.
   - **Standard Dense Vector RAG (`24.0%` Judge Acc, `33.4%` Citation Recall)** retrieves raw turns without pronoun/question de-contextualization or relative-date anchoring, failing on elliptical replies (e.g., `D7:11` *"Becoming Nicole"*), state updates (`37.5%`), and adversarial speaker swaps (`0.0%`).
   - **`Smriti (Full)` (`62.7%` Judge Acc, `56.1%` Citation Recall, `67.9%` Retrieval Recall@k)** achieves a **2.6x improvement in Judge Accuracy** over Naive Dense RAG while compressing prompt tokens by **86.1%** compared to full transcript replay.

2. **Ablation 1 — Removing Offline Dreaming, Linked Wikis & Bi-Temporal Supersession (`62.7% -> 58.3%`)**:
   - Disabling `DreamEngine` causes **Bi-Temporal Knowledge Update accuracy to drop by 50.0 percentage points (`100.0% -> 50.0%`)** and **Multi-Hop Synthesis accuracy to drop by 12.5 percentage points (`60.0% -> 47.5%`)**, while Multi-Hop Provenance Citation Recall drops from `23.5% -> 17.7%`.
   - *Interesting Tradeoff Observed*: Without Dreaming, Temporal Reasoning is slightly higher (`81.8%` vs. `72.7%`, a difference of 2 questions out of 22) and raw turn Retrieval Recall@k is `71.4%` vs `67.9%`, because Layer 4 (`global.md`) and Layer 3 (`WikiPage` snippets) consume ~300 tokens of the 1,500-token budget and `drm_link_` claims occasionally rank ahead of a single-hop temporal claim. Overall, Dreaming yields a net **+4.4% aggregate gain** and is essential for multi-hop and state-mutation consistency.

3. **Ablation 2 — Removing Relative-to-Absolute Temporal Anchoring (`62.7% -> 45.6%`)**:
   - Disabling `TemporalNormalizer` causes **Temporal Reasoning accuracy to collapse from `72.7%` to `13.6%` (-59.1 percentage points)** and overall Judge Accuracy to drop by **17.1 points**.
   - Without ingestion-time anchoring (`"yesterday"` $\rightarrow$ `"5 July 2023"`, `"two weekends ago"` $\rightarrow$ `"two weekends before 17 July 2023"`, `"Seven years now"` $\rightarrow$ `"7 years (Since 2016)"`), the agent can only report the session timestamp rather than the true event date.

4. **Ablation 3 — Removing Hybrid FTS5 BM25 & Entity Graph (`62.7% -> 60.3%`, Citation Recall `56.1% -> 50.1%`)**:
   - Relying on dense embeddings alone drops **Single-Hop Provenance Citation Recall by 15.0 percentage points (`65.0% -> 50.0%`)** and overall Citation F1 from `43.3% -> 39.9%`, proving that exact BM25 lexical matching + 1-hop entity graph expansion is vital for pinpointing rare proper nouns and exact turn IDs.

---

## 6. Honest Failure Analysis (Surfaced from `raw_predictions.jsonl`)

Inspecting the raw predictions in [`artifacts/eval_results/raw_predictions.jsonl`](file:///Users/palakbansal/.gemini/antigravity/scratch/smriti-memory/artifacts/eval_results/raw_predictions.jsonl) highlights four concrete failure modes in `Smriti`:

### Failure Case 1: World-Knowledge Gap on Commonsense Inference (`conv-26_q61`, Category 3)
- **Question**: *"Would Caroline likely have Dr. Seuss books on her bookshelf?"*
- **Ground Truth (`D6:9`)**: *"Yes, since she collects classic children's books"*
- **`Smriti` Output**: `Abstained: The requested topic or detail was not mentioned in any past interaction.` (`judge_score: 0.0`)
- **Why It Failed**: The token `"Seuss"` never appears anywhere in the 19 sessions of `conv-26`. Answering requires external world knowledge (*Dr. Seuss is an author of classic children's books*) to bridge the query term `"Dr. Seuss"` to `D6:9` (*"classic children's books"*), causing our strict unmentioned-topic guard to abstain rather than guess.

### Failure Case 2: Multi-Session Aggregative Counting (`conv-26_q42`, Category 1 Multi-Hop)
- **Question**: *"How many times has Melanie gone to the beach in 2023?"*
- **Ground Truth (`D10:8`, `D6:16`)**: `2`
- **`Smriti` Output**: Retrieves both `D10:8` and `D6:16` in `retrieved_turn_ids` (`['D10:10', 'D4:6', 'D10:8', 'D6:16']`), and cites `['D10:10', 'D10:8']`, quoting Melanie saying *"usually only once or twice a year"*, but does not perform arithmetic aggregation over distinct beach trips across Sessions 6 and 10 (`judge_score: 0.0`).

### Failure Case 3: Subtle Adversarial Premise with Shared Speaker Vocabulary (`conv-26_q119`, Category 5 Adversarial)
- **Question**: *"What instrument does Melanie's brother play?"*
- **Ground Truth**: Adversarial False Premise (Melanie plays the clarinet/violin and talks about her kids/sister; she never mentions a brother playing an instrument).
- **`Smriti` Output**: Because **Melanie** herself discusses playing instruments across multiple turns (`D2:5`, `D6:14`), the speaker-attribution guard sees `target_speaker = Melanie` asserting instrument-related stems (`target_assertions > 0`) and fails to recognize that the relationship modifier (*"Melanie's brother"*) is the fabricated part of the premise (`abstained: False`, `judge_score: 0.0`).

### Failure Case 4: Cross-Turn Relative Year Resolution (`conv-26_q49`, Category 2 Temporal)
- **Question**: *"When did Caroline and Melanie go to a pride festival together?"*
- **Ground Truth (`D12:15`)**: `2022`
- **`Smriti` Output**: Retrieves `D5:1` (a 2023 pride parade turn) and `D10:10` ahead of `D12:15` because `D12:15` refers to the event elliptically as *"last year's festival"* without repeating both speaker names (`judge_score: 0.0`).
