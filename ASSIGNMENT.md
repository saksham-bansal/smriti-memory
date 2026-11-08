# Take-Home Assignment: Agent Memory Across Sessions

Build a persistent memory system that helps an agent use past interactions, and an evaluation pipeline that demonstrates an improvement over standard approaches.

---

## 1. Context

An agent’s usefulness across sessions depends on its ability to use past interactions. Agent memory is the persistent layer that makes those interactions available as context for future requests.

As interaction histories grow, replaying them becomes expensive and can exceed the context budget. Build a memory system that enables the agent to use its history effectively within those constraints.

You are responsible for researching what agent memory should do, where it breaks, and how to build and evaluate it. We want to see how you identify the important problems and turn your judgment into a working architecture with convincing evidence. A complete UI is not expected; runnable code is required.

---

## 2. What to Build

Build one integrated system covering:

1. **Ingestion and memory architecture.** Ingest interactions across sessions and design the memory layer. Decide what it should retain and how it should work; explain and justify your choices.
2. **Persistence and use.** Persist memory across runs and integrate it into the agent’s responses to future requests.
3. **Provenance.** Make the use of past interactions inspectable, with claim-level references to the supporting source interactions.
4. **Evaluation pipeline.** Research and choose an appropriate open benchmark, credible standard baselines, and suitable metrics. Decide how the system can and should be evaluated, then demonstrate that your methodology improves on those baselines.

Complete the assignment within the timeframe communicated by your Sarvam point of contact. A well-chosen, reproducible benchmark subset is acceptable if the full evaluation is impractical within that timeframe. Explain your scope and its limitations.

---

## 3. Hard Requirements

These are pass or fail. A smaller submission that respects them is stronger than a feature-complete one that breaks them.

1. **Working, persistent memory.** The system must retain memory across process restarts and demonstrate its use across sessions.
2. **Provenance is required.** Every factual answer claim about past interactions must trace to specific supporting source interactions.
3. **Open benchmark and measured improvement.** Choose and justify a publicly available benchmark that meaningfully tests agent memory. Select appropriate metrics and baselines, explain why they are suitable, and demonstrate improvement through fair comparisons. Choosing a convincing evaluation is part of the assignment.
4. **Reproducible evaluation.** One command must rerun the evaluation and rebuild the results from raw outputs. Provide the data selection, configurations, and artifacts needed to reproduce your claims. If using a subset, document its selection and make it reproducible.

---

## 4. Deliverables

1. **Source code in a private Git repo.** Commit throughout the assignment so the Git history shows your implementation, experiments, and iteration. A single large final commit is not sufficient.
2. **[`DESIGN.md`](./DESIGN.md)**, 2–4 pages: architecture and data flow, the reasoning behind your choices, provenance, tradeoffs, failure modes, and next steps.
3. **[`EVAL.md`](./EVAL.md)**: benchmark choice and version, subset selection if applicable, chosen metrics and baselines with justification, results, at least one ablation, and an honest failure analysis. Include raw outputs and evaluation scripts.
4. **A runnable repo and walkthrough: [`README.md`](./README.md)** with exact setup, ingestion, example API/CLI calls, and evaluation commands, dependencies, and expected runtime/cost. Provide a 5–10 minute recorded or live walkthrough showing successful use of memory across sessions, persistence across a restart, and at least one failure or edge case surfaced by your evaluation.

---

## 5. How We Evaluate

| Dimension | What strong looks like |
| :--- | :--- |
| **Memory architecture and judgment** | A well-reasoned memory architecture that delivers measured gains over credible standard approaches. |
| **Provenance and grounding** | Claims about past interactions trace to real supporting evidence. The system’s behavior supports the claims made about it. |
| **Evaluation rigor** | Thoughtful benchmark, baseline, and metric choices; fair, reproducible comparisons; a useful ablation and honest interpretation of results. |
| **Research and independence** | Identifies the important challenges, investigates them, and makes defensible choices without a prescribed solution. |
| **Engineering and communication** | An integrated, persistent, runnable system, meaningful commits, and clear design and results documents. |

---

## 6. Submission

Share the private repo and walkthrough link with your point of contact at Sarvam by the agreed deadline. Put [`DESIGN.md`](./DESIGN.md), [`EVAL.md`](./EVAL.md), and [`README.md`](./README.md) in the repo root. Include everything needed to reproduce the evaluation.

Clarifying questions are welcome. We expect you to go above and beyond what is considered standard. The depth of your research, the judgment behind your design, and the evidence for your improvements are how you will differentiate yourself.

*Confidential — for the named candidate only.*
