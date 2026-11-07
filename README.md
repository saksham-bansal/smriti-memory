# Smriti: Bi-Temporal 4-Layer Agent Memory Across Sessions

**`Smriti`** is a persistent cross-session memory system for AI agents that combines **Anthropic’s 4-Layer Memory Hierarchy & Offline "Dreaming" Consolidation** (*Transcripts $\rightarrow$ Atomic Memory Notes $\rightarrow$ Linked Entity/Topic Wikis $\rightarrow$ Always-Loaded `global.md` Index*) with **Graphiti-style Bi-Temporal State Supersession** (`ACTIVE` vs. `SUPERSEDED` validity intervals) and **Mandatory Claim-Level Provenance** back to source dialogue turns.

- **Architecture & Design**: See [**`DESIGN.md`**](./DESIGN.md)
- **Benchmark Evaluation, Baselines, Ablations & Error Analysis**: See [**`EVAL.md`**](./EVAL.md)

---

## 1. Key Capabilities & Hard Requirements Checklist

| Requirement | How `Smriti` Satisfies It | Verification Command |
| :--- | :--- | :--- |
| **1. Persistent Memory Across Restarts** | All 4 memory layers (Episodic turns, Bi-temporal claims, Compiled Wikis/Entity Graph, and `global.md` header) + `float32` dense vectors persist in a single ACID-compliant SQLite WAL database. | `make demo` / `make test` |
| **2. Claim-Level Provenance** | Every extracted claim, Wiki bullet, and agent answer claim carries foreign-key `source_turn_ids`, `source_session_ids`, and verbatim `supporting_quotes` verified against Layer 1 by [`ProvenanceVerifier`](./src/smriti/agent/verifier.py). | `PYTHONPATH=src .venv/bin/pytest -k provenance` |
| **3. Open Benchmark & Measured Gains** | Evaluated on **LoCoMo (`snap-research/locomo`, ACL 2024)** + a **Bi-Temporal Knowledge-Update Slice** (42 sessions, 800 turns, 102 questions). Improves Answer Judge Accuracy from **24.0% (Naive Dense RAG)** to **62.7% (`Smriti`)** and Provenance Citation Recall from **33.4%** to **56.1%** at **86.1% token savings**. | `make eval` |
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

### Optional LLM API Configuration
`Smriti` includes both a unified LLM API client (`GEMINI_API_KEY` for `gemini-2.5-flash` or `OPENAI_API_KEY`) with persistent SQLite caching (`artifacts/llm_cache.db`) **and** a deterministic Extractive-Symbolic Reasoner so that ingestion, tests, and the full evaluation pipeline run reproducibly out-of-the-box with zero API cost.

```bash
cp .env.example .env
# Optional: set GEMINI_API_KEY=... in .env
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

## 4. Reproducing the Evaluation Pipeline

```bash
# Rerun all 3 baselines, Smriti (Full), and 3 ablations from scratch and rebuild Markdown tables:
make eval && make report

# Optional: Re-download raw LoCoMo (locomo10.json) from snap-research/locomo and regenerate subset manifest:
PYTHONPATH=src:. .venv/bin/python eval/download_data.py \
  --raw-path data/locomo10.json \
  --manifest-out artifacts/eval_subset_manifest.json
```

### Expected Runtime & Cost
- **Runtime**: **~15–20 seconds** on a single CPU for all 7 configurations $\times$ 102 questions (714 evaluations across 800 ingested turns).
- **API Cost**: **$0.00** in default deterministic mode (or ~$0.08 with `gemini-2.5-flash` uncached; $0.00 on cached reruns via `artifacts/llm_cache.db`).

---

## 5. 5–10 Minute Walkthrough Guide (Live or Recorded Demo)

Use the following 4-step sequence for a 5–10 minute walkthrough covering all required evaluation rubric points:

### Step 1 (Minutes 0:00–2:00): Architecture & 4-Layer + Dreaming Design
1. Open [`DESIGN.md`](./DESIGN.md) and walk through the Mermaid diagram:
   - Explain why naive RAG fails on **conversational pronouns/ellipsis**, **relative time markers** (*"yesterday"*, *"last Friday"*), **contradictory state updates** (*moving cities / changing diets*), and **speaker-swap false premises**.
   - Show how `Smriti` synthesizes Anthropic’s **4-Layer Stack & Offline Dreaming** with **Bi-Temporal Validity (`ACTIVE` vs. `SUPERSEDED`)** while preserving **strict `source_turn_ids` foreign keys** at every layer.

### Step 2 (Minutes 2:00–5:00): Live Multi-Process Persistence & Provenance Demo
Run:
```bash
PYTHONPATH=src .venv/bin/python -m smriti.cli demo-persistence --db artifacts/demo_persistence.db
PYTHONPATH=src .venv/bin/python -m smriti.cli inspect --db artifacts/demo_persistence.db --conversation-id sarvam_demo_user
```
Highlight:
- **Process 1** ingests Session 1 (`8 May 2023`) and exits.
- **Process 2** opens the same SQLite DB in a new OS process, ingests Session 2 (`17 August 2023`), marks `Bangalore` and `vegetarian` as `SUPERSEDED`, and exits.
- **Process 3** runs `smriti dream`, synthesizing `drm_trans_` state-evolution claims, compiling `[[Aarav]]` and `[[Meera]]` Wiki pages, and promoting core facts into Layer 4 (`global.md`).
- **Process 4** queries the persisted DB across restarts:
  1. *Current state*: Reports `Hyderabad` and `vegan` while noting prior states (`[S1:1, S2:1]`, `[S1:3, S2:1]`).
  2. *Historical state*: Recalls `Bangalore` (`[S1:1, S2:1]`).
  3. *Relative-to-absolute temporal reasoning*: Resolves *"adopted a puppy yesterday"* in Session 1 (`8 May 2023`) to **`7 May 2023`** (`[S1:2]`).
  4. *Adversarial Speaker-Swap Guard*: When asked *"When did Aarav adopt a golden retriever puppy?"*, abstains (`"abstained": true`) and cites `[S1:2]` showing *Meera* adopted the puppy.

### Step 3 (Minutes 5:00–7:30): Reproducible Benchmark Evaluation & Ablations
Run:
```bash
make eval && make report
```
Open [`EVAL.md`](./EVAL.md) and highlight:
- **62.7% Judge Accuracy** vs. **24.0% Standard Dense RAG**, **12.2% Rolling Summary**, and **6.9% Recency Buffer** under an identical 1,500-token budget (**86.1% token reduction** vs. full replay).
- **Ablation 1 (w/o Dreaming & Bi-Temporal Supersession)**: Bi-Temporal Update accuracy drops from **100.0% $\rightarrow$ 50.0%** and Multi-Hop drops from **60.0% $\rightarrow$ 47.5%**.
- **Ablation 2 (w/o Temporal Normalization)**: Temporal Reasoning collapses from **72.7% $\rightarrow$ 13.6%**.
- **Ablation 3 (w/o Hybrid BM25 & Entity Graph)**: Single-Hop Provenance Citation Recall drops from **65.0% $\rightarrow$ 50.0%**.

### Step 4 (Minutes 7:30–9:30): Inspecting a Real Failure / Edge Case Surfaced by Evaluation
Run the following command to inspect a real failure case from `artifacts/eval_results/raw_predictions.jsonl` (Documented in Section 6 of [`EVAL.md`](./EVAL.md)):
```bash
.venv/bin/python -c "
import json
for line in open('artifacts/eval_results/raw_predictions.jsonl'):
    r = json.loads(line)
    if r['system_id'] == 'smriti_full' and r['qa_id'] in ('conv-26_q61', 'conv-26_q42', 'conv-26_q119'):
        print(json.dumps(r, indent=2))
"
```
Explain the two surfaced edge cases:
1. **`conv-26_q61` (World-Knowledge Commonsense Gap)**: Asking whether Caroline has *Dr. Seuss* books when `D6:9` only mentions *"classic children's books"* causes `Smriti`’s strict unmentioned-topic guard to abstain because `"Seuss"` never appears in the conversation or concept clusters.
2. **`conv-26_q42` (Multi-Session Counting Aggregation)**: `Smriti` retrieves both ground-truth beach-trip turns (`D10:8`, `D6:16`), proving retrieval succeeded, but quotes the turns rather than counting the distinct episodes (`2`).
