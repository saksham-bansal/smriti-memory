# Smriti: Bi-Temporal 4-Layer Agent Memory Across Sessions

**`Smriti`** is a persistent cross-session memory system for AI agents—built and evaluated using **Google Gemini (`gemini-3.1-flash-lite`)**—that combines **Anthropic’s 4-Layer Memory Hierarchy & Offline "Dreaming" Consolidation** (*Transcripts → Atomic Memory Notes → Linked Entity/Topic Wikis → Always-Loaded `global.md` Index*) with **Graphiti-style Bi-Temporal State Supersession** (`ACTIVE` vs. `SUPERSEDED` validity intervals) and **Mandatory Claim-Level Provenance** back to source dialogue turns.

- **Architecture & Design**: See [**`DESIGN.md`**](./DESIGN.md)
- **Benchmark Evaluation, Baselines, Ablations & Error Analysis**: See [**`EVAL.md`**](./EVAL.md)
- **LLM Backbone**: Powered by **Google Gemini (`gemini-3.1-flash-lite`)** via `GEMINI_API_KEY`, with all 714 benchmark predictions cached in [`artifacts/llm_cache.db`](./artifacts/llm_cache.db) for instant zero-cost reproducibility.

---

## 1. Key Capabilities & Hard Requirements Checklist

| Requirement | How `Smriti` Satisfies It | Verification Command |
| :--- | :--- | :--- |
| **1. Persistent Memory Across Restarts** | All 4 memory layers (Episodic turns, Bi-temporal claims, Compiled Wikis/Entity Graph, and `global.md` header) + `float32` dense vectors persist in a single ACID-compliant SQLite WAL database. | `make demo` / `make test` |
| **2. Claim-Level Provenance** | Every extracted claim, Wiki bullet, and agent answer claim carries foreign-key `source_turn_ids`, `source_session_ids`, and verbatim `supporting_quotes` verified against Layer 1 by [`ProvenanceVerifier`](./src/smriti/agent/verifier.py). | `PYTHONPATH=src .venv/bin/pytest -k provenance` |
| **3. Open Benchmark & Measured Gains** | Evaluated with live `gemini-3.1-flash-lite` calls on **LoCoMo (`snap-research/locomo`, ACL 2024)** + a **Bi-Temporal Knowledge-Update Slice** (42 sessions, 800 turns, 102 questions). Improves Answer Judge Accuracy from **42.6% (Standard Dense RAG)** / **28.9% (Rolling Summary)** / **24.5% (Recency Buffer)** to **67.2% (`Smriti`)** and Provenance Citation Recall from **30.0%** to **54.4%** at **86.1% token savings**. | `make eval` |
| **4. One-Command Reproducibility** | A single command reruns all 7 systems (3 baselines, `Smriti`, 3 ablations = 714 predictions) and rebuilds summary tables from `artifacts/eval_results/raw_predictions.jsonl`. | `make eval && make report` |

---

## 2. Quickstart & Setup

### Dependencies & Requirements
- **Python**: `>= 3.9` (with standard `sqlite3` FTS5 enabled)
- **Core Packages**: `numpy>=1.21.0`, `requests>=2.28.0`, `pytest>=7.0.0`
- **External Daemons Required**: **None** (100% embedded SQLite WAL + FTS5 + NumPy vector index)

```bash
# 1. Create virtual environment and install package + dev dependencies
make setup
# Or manually:
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

# 2. Run the automated unit & integration test suite (5 tests, < 0.5s)
make test
```

### LLM API Configuration & Caching
`Smriti` uses a unified LLM API client (`GEMINI_API_KEY` for `gemini-3.1-flash-lite` or `OPENAI_API_KEY`) backed by a persistent SQLite prompt/response cache (`artifacts/llm_cache.db`, committed to the repo) plus a deterministic Extractive-Symbolic fallback when no API key or cached entry is available.

```bash
cp .env.example .env
# Set GEMINI_API_KEY=... in .env for uncached live LLM calls (cached benchmark calls replay from artifacts/llm_cache.db)
```

---

## 3. CLI Usage: Ingestion, Dreaming, Querying & Inspection

### 3.1 Automated Multi-Process Persistence Demo (`make demo`)
Spawns **4 sequential OS subprocesses** to ingest Session 1, ingest Session 2 (with state updates and relative dates), run an offline Dream consolidation pass, and query the persisted database across process restarts:

```bash
make demo
# Or: PYTHONPATH=src .venv/bin/python -m smriti.cli demo-persistence --db artifacts/demo_persistence.db
```

### 3.2 Step-by-Step CLI Commands

```bash
# Inspect the 4-layer memory store (Layer 1 turns, Layer 2 active/superseded claims, Layer 3 Wikis, Layer 4 global.md)
PYTHONPATH=src .venv/bin/python -m smriti.cli inspect \
  --db artifacts/demo_persistence.db \
  --conversation-id sarvam_demo_user

# Run an offline 'Dream' consolidation pass (synthesizes state updates, builds Entity/Topic Wikis & global.md)
PYTHONPATH=src .venv/bin/python -m smriti.cli dream \
  --db artifacts/demo_persistence.db \
  --conversation-id sarvam_demo_user

# Query across sessions with claim-level provenance (returns structured JSON with verified source_turn_ids & quotes)
PYTHONPATH=src .venv/bin/python -m smriti.cli ask \
  --db artifacts/demo_persistence.db \
  --conversation-id sarvam_demo_user \
  --query "Where does Aarav live currently and what diet does he follow?" \
  --budget 1500
```

Example JSON output from `smriti ask`:
```json
{
  "query": "Where does Aarav live currently and what diet does he follow?",
  "answer": "Aarav's dietary preference is currently 'vegan' (updated on 3:30 pm on 17 August; previously 'vegetarian') [S1:3, S2:1]. Aarav's lives in is currently 'Hyderabad' (updated on 3:30 pm on 17 August; previously 'Bangalore') [S1:1, S2:1].",
  "abstained": false,
  "claims": [
    {
      "claim_text": "Aarav's dietary preference is currently 'vegan' (updated on 3:30 pm on 17 August; previously 'vegetarian') [S1:3, S2:1].",
      "source_turn_ids": ["S1:3", "S2:1"],
      "source_session_ids": ["session_1", "session_2"],
      "supporting_quotes": [
        "S1:3 (10:15 am on 8 May, 2023, Aarav): I follow a vegetarian diet right now, and we use PostgreSQL for our user-profile service.",
        "S2:1 (3:30 pm on 17 August, 2023, Aarav): Big update! I moved to Hyderabad last Friday and I switched to a vegan diet."
      ],
      "verified": true
    },
    {
      "claim_text": "Aarav's lives in is currently 'Hyderabad' (updated on 3:30 pm on 17 August; previously 'Bangalore') [S1:1, S2:1].",
      "source_turn_ids": ["S1:1", "S2:1"],
      "source_session_ids": ["session_1", "session_2"],
      "supporting_quotes": [
        "S1:1 (10:15 am on 8 May, 2023, Aarav): Hey Meera! I currently live in Bangalore and I work as a Backend Engineer at Flipkart.",
        "S2:1 (3:30 pm on 17 August, 2023, Aarav): Big update! I moved to Hyderabad last Friday and I switched to a vegan diet."
      ],
      "verified": true
    }
  ],
  "prompt_tokens": 1248,
  "all_cited_turn_ids": ["S1:3", "S2:1", "S1:1"]
}
```

---

## 4. Benchmark Results Summary (Evaluated with `gemini-3.1-flash-lite`)

All 7 configurations (3 standard baselines, `Smriti`, and 3 architectural ablations) were evaluated with live **`gemini-3.1-flash-lite`** calls under an identical **1,500-token budget** across **42 sessions, 800 dialogue turns, and 102 stratified questions** (714 total predictions) from **LoCoMo (`snap-research/locomo`, ACL 2024)** + a **Bi-Temporal Knowledge-Update Slice**:

| System / Configuration (`gemini-3.1-flash-lite`) | Answer Judge Score (%) | Answer Token Recall (%) | Provenance Citation F1 (%) | Provenance Citation Recall (%) | Context Retrieval Recall@k (%) | Citation Validity (%) | Mean Prompt Tokens | Token Savings vs. Full Replay (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 24.5% | 20.1% | 10.5% | 9.3% | 14.7% | 100.0% | 1373 | 91.2% |
| 2. Rolling Session Summarization | 28.9% | 22.5% | 6.8% | 6.5% | 13.1% | 100.0% | 1335 | 91.8% |
| 3. Standard Dense Vector RAG (Raw Turns) | 42.6% | 34.9% | 30.4% | 30.0% | 41.8% | 100.0% | 485 | 96.3% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **67.2%** | **52.2%** | **43.3%** | **54.4%** | **67.9%** | **100.0%** | **1477** | **86.1%** |
| 5. Ablation: w/o Offline Dreaming, Wikis & Bi-Temporal Supersession | 62.7% | 47.9% | 42.4% | 55.4% | 71.4% | 100.0% | 1352 | 88.6% |
| 6. Ablation: w/o Relative-to-Absolute Temporal Anchoring | 53.4% | 38.2% | 41.4% | 50.9% | 61.0% | 100.0% | 1474 | 86.1% |
| 7. Ablation: w/o Hybrid BM25 & Entity Graph (Dense Only) | 63.7% | 48.4% | 38.6% | 46.7% | 64.8% | 100.0% | 1470 | 86.1% |

### Key Category & Ablation Highlights (See [**`EVAL.md`**](./EVAL.md) for Full Breakdown)
- **Single-Hop & Multi-Hop Recall**: `Smriti` achieves **60.0% Single-Hop** and **55.0% Multi-Hop** accuracy vs. **15.0% Single-Hop** and **25.0% Multi-Hop** for Standard Dense Vector RAG.
- **Temporal Reasoning (`86.4%`)**: Anchoring relative expressions (*"yesterday"*, *"last Friday"*, *"two weekends ago"*) at ingestion time lifts Temporal accuracy to **86.4%** (vs. **59.1%** Dense RAG, **22.7%** Rolling Summary, and **27.3%** when `TemporalNormalizer` is ablated).
- **Bi-Temporal State Updates (`93.8%`) & Dream Consolidation**: Disabling offline Dreaming & Bi-Temporal Supersession drops State-Update accuracy from **93.8% → 68.8%** (`-25.0 pts`) and Multi-Hop accuracy from **55.0% → 37.5%** (`-17.5 pts`).
- **Adversarial Speaker-Swap Abstention (`81.8%`)**: `Smriti`'s speaker-ownership verifier reaches **81.8%** accuracy on false-premise traps vs. **63.6%** for Dense RAG and **50.0%** for Recency Buffer.

---

## 5. Reproducing the Evaluation Pipeline

```bash
# Rerun all 3 baselines, Smriti (Full), and 3 ablations from scratch and rebuild Markdown tables:
make eval && make report

# Optional: Re-download raw LoCoMo (locomo10.json) from snap-research/locomo and regenerate subset manifest:
PYTHONPATH=src:. .venv/bin/python eval/download_data.py \
  --raw-path data/locomo10.json \
  --manifest-out artifacts/eval_subset_manifest.json
```

### Expected Runtime & Cost
- **Runtime**: **~15–20 seconds** on a single CPU when replaying cached `gemini-3.1-flash-lite` responses from `artifacts/llm_cache.db` (or **~6–8 minutes** for a cold uncached run across all 7 configurations × 102 questions = 714 predictions).
- **API Cost**: **`~$0.05`** for a cold uncached run with `gemini-3.1-flash-lite` (`385` unique LLM prompts across all 7 systems); **`$0.00`** on cached reruns via `artifacts/llm_cache.db`.

