## Rebuilt Evaluation Results (Generated from `artifacts/eval_results/raw_predictions.jsonl`)

### Table 1: Overall Benchmark Performance Across Baselines, `Smriti`, and Ablations

| System / Configuration | Answer Judge Score (%) | Answer Token Recall (%) | Provenance Citation F1 (%) | Provenance Citation Recall (%) | Context Retrieval Recall@k (%) | Citation Validity (%) | Mean Prompt Tokens | Token Savings vs. Full Replay (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Sliding-Window Recency Buffer | 24.5% | 20.1% | 10.5% | 9.3% | 14.7% | 100.0% | 1373 | 91.2% |
| 2. Rolling Session Summarization | 28.9% | 22.5% | 6.8% | 6.5% | 13.1% | 100.0% | 1335 | 91.8% |
| 3. Standard Dense Vector RAG (Raw Turns) | 42.6% | 34.9% | 30.4% | 30.0% | 41.8% | 100.0% | 485 | 96.3% |
| **4. Smriti (Full 4-Layer + Dreaming + Bi-Temporal + Hybrid RRF)** | **67.2%** | **52.2%** | **43.3%** | **54.4%** | **67.9%** | **100.0%** | 1477 | 86.1% |
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
