## Rebuilt Evaluation Results (Generated from `artifacts/eval_results/raw_predictions.jsonl`)

### Table 1: Overall Benchmark Performance Across Baselines, `Smriti`, and Ablations

| System / Configuration | Answer Judge Score (%) | Answer Token Recall (%) | Provenance Citation F1 (%) | Provenance Citation Recall (%) | Context Retrieval Recall@k (%) | Citation Validity (%) | Mean Prompt Tokens | Token Savings vs. Full Replay (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 6.9% | 14.8% | 10.0% | 9.8% | 14.7% | 100.0% | 1373 | 91.2% |
| 2. Rolling Session Summarization | 12.2% | 19.7% | 6.2% | 5.7% | 13.1% | 100.0% | 1335 | 91.8% |
| 3. Standard Dense Vector RAG (Raw Turns) | 24.0% | 35.0% | 30.0% | 33.4% | 41.8% | 100.0% | 485 | 96.3% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **62.7%** | **57.5%** | **43.3%** | **56.1%** | **67.9%** | **100.0%** | 1477 | 86.1% |
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
