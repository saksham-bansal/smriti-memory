"""
Evaluation Metrics for Agent Memory Across Sessions (`eval/metrics.py`).

Computes three orthogonal families of metrics:
  1. Answer Quality:
     - `judge_score` (0.0 / 0.5 / 1.0): Semantic + Temporal + Adversarial Abstention grading
     - `answer_recall`: Stemmed content-token recall of the ground-truth answer
     - `token_f1`: SQuAD-style normalized token F1
  2. Provenance & Grounding Quality (Hard Requirement 2):
     - `citation_recall`: Recall of human-annotated ground-truth `evidence_turn_ids` in agent's cited `claims`
     - `citation_precision`: Precision of agent's cited `source_turn_ids` against ground-truth `evidence_turn_ids`
     - `citation_f1`: Harmonic mean of `citation_precision` and `citation_recall`
     - `retrieval_recall`: Recall of ground-truth `evidence_turn_ids` in the packed context window
     - `session_provenance_hit`: Whether the cited sessions contain the ground-truth evidence session
     - `citation_validity`: Fraction of emitted claims that pass deterministic Layer-1 existence verification
  3. Efficiency & Context Budget:
     - `prompt_tokens`: Context tokens consumed per query
     - `token_savings_pct`: Percentage reduction vs. Full-Context Transcript Replay
     - `latency_ms`: End-to-end query latency in milliseconds
"""
import re
from collections import Counter
from typing import Any, Dict, List, Set

from smriti.models import AgentResponse
from smriti.storage.vector_index import CONCEPT_CLUSTERS, _WORD_TO_CONCEPTS, _light_stem

_METRIC_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and", "or",
    "as", "that", "this", "it", "he", "she", "they", "them", "his", "her",
    "their", "would", "could", "should", "likely", "probably", "since", "because",
}


def _normalize_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"\[[A-Za-z0-9_:\-\s,]+\]", " ", str(text).lower())
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    return [t for t in tokens if t not in _METRIC_STOPWORDS]


def compute_token_f1_and_recall(pred: str, gold: str) -> Dict[str, float]:
    pred_toks = _normalize_tokens(pred)
    gold_toks = _normalize_tokens(gold)
    if not gold_toks:
        return {"token_f1": 1.0 if not pred_toks else 0.0, "answer_recall": 1.0}
    if not pred_toks:
        return {"token_f1": 0.0, "answer_recall": 0.0}

    pred_stems = [_light_stem(t) for t in pred_toks]
    gold_stems = [_light_stem(t) for t in gold_toks]

    common = Counter(pred_stems) & Counter(gold_stems)
    num_same = sum(common.values())
    if num_same == 0:
        return {"token_f1": 0.0, "answer_recall": 0.0}

    precision = num_same / len(pred_stems)
    recall = num_same / len(gold_stems)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"token_f1": round(f1, 4), "answer_recall": round(recall, 4)}


def compute_judge_score(
    qa_item: Dict[str, Any],
    resp: AgentResponse,
) -> float:
    """
    Grades answer accuracy on [0.0, 0.5, 1.0] across all 6 categories:
      - Category 5 (Adversarial / False Premise):
        Requires `resp.abstained == True` or explicit rejection of the false premise.
        Systems that naively output the trap fact without abstaining receive 0.0.
      - Category 6 (Bi-Temporal Knowledge Update):
        Requires the target state (`gold_answer`) to appear as the active/historical answer
        matching the question's temporal polarity.
      - Categories 1–4 (Multi-Hop, Temporal, Inference, Single-Hop):
        Evaluates stemmed content recall, concept-cluster equivalence, and date/year matching.
    """
    cat = int(qa_item.get("category", 1))
    gold = str(qa_item.get("answer", "")).strip()
    pred = resp.answer.strip()

    # Category 5: Adversarial / False-Premise Refusal
    if cat == 5 or qa_item.get("is_adversarial", False):
        if resp.abstained or re.search(
            r"\b(abstain|abstained|not mentioned|no record|false premise|rather|instead)\b",
            pred.lower(),
        ):
            return 1.0
        return 0.0

    # Non-adversarial questions: if system abstained, score is 0.0
    if resp.abstained:
        return 0.0

    # Category 6: Bi-Temporal Knowledge Updates
    if cat == 6:
        gold_low = gold.lower()
        pred_low = pred.lower()
        # Check the main clause before "(previously ...)" or "SUPERSEDED"
        primary_clause = pred_low.split("previously")[0].split("superseded")[0]
        if gold_low in primary_clause:
            return 1.0
        # For historical queries ("before moving"), check full answer
        if any(w in qa_item["question"].lower() for w in ("previously", "originally", "before")):
            if gold_low in pred_low:
                return 1.0
        overlap = compute_token_f1_and_recall(pred, gold)
        if overlap["answer_recall"] >= 0.75:
            return 1.0
        elif overlap["answer_recall"] >= 0.40:
            return 0.5
        return 0.0

    # Categories 1, 2, 3, 4: Standard + Concept-Cluster Entailment Grading
    # Strip bracketed [turn_id] citations so turn numbers like [D5:5] don't collide with date numbers
    eval_pred = re.sub(r"\[[A-Za-z0-9_:\-\s,]+\]", " ", pred)
    if cat == 2:
        # Expand number words in eval_pred for fair comparison ("four years" <-> "4 years", "ten" <-> "10")
        num_map = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        }
        for w_num, d_num in num_map.items():
            if re.search(rf"\b{w_num}\b", eval_pred.lower()):
                eval_pred += f" {d_num}"

    overlap = compute_token_f1_and_recall(eval_pred, gold)
    recall = overlap["answer_recall"]

    if cat == 2:
        # Strict Temporal Verification: all numeric tokens in gold (e.g. day '5' and year '2023')
        # and any explicit month names in gold must appear in eval_pred
        gold_digits = set(re.findall(r"\d+", gold))
        # Exclude clock time digits like "1:56" from session timestamps when checking date digits
        pred_no_clock = re.sub(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", " ", eval_pred, flags=re.IGNORECASE)
        pred_digits = set(re.findall(r"\d+", pred_no_clock))
        months_set = {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }
        gold_months = set(re.findall(r"[a-z]+", gold.lower())) & months_set
        pred_months = set(re.findall(r"[a-z]+", pred_no_clock.lower())) & months_set
        if gold_months and not gold_months.issubset(pred_months):
            return 0.0
        if gold_digits and not gold_digits.issubset(pred_digits):
            return 0.0
        if recall >= 0.60:
            return 1.0
        if recall >= 0.35:
            return 0.5
        return 0.0

    # Check concept-level synonym equivalence (e.g., "counseling" <-> "mental health", "kids" <-> "children")
    pred_toks = _normalize_tokens(pred)
    gold_toks = _normalize_tokens(gold)
    pred_concepts: Set[str] = set()
    for t in pred_toks:
        pred_concepts.update(_WORD_TO_CONCEPTS.get(t, []))
        pred_concepts.update(_WORD_TO_CONCEPTS.get(_light_stem(t), []))
    gold_concepts: Set[str] = set()
    for t in gold_toks:
        gold_concepts.update(_WORD_TO_CONCEPTS.get(t, []))
        gold_concepts.update(_WORD_TO_CONCEPTS.get(_light_stem(t), []))

    concept_overlap = (
        len(pred_concepts & gold_concepts) / len(gold_concepts) if gold_concepts else 0.0
    )

    # Check if the exact ground-truth evidence turn was cited & quoted in the response
    ev_ids = set(qa_item.get("evidence_turn_ids", []))
    cited_ids = set(resp.all_cited_turn_ids)
    has_exact_evidence_citation = bool(ev_ids & cited_ids)

    if recall >= 0.65:
        return 1.0
    if recall >= 0.35 or (concept_overlap >= 0.60 and has_exact_evidence_citation):
        # For Category 2 (Temporal), require the year or month/day number to match if gold has digits
        gold_digits = set(re.findall(r"\d+", gold))
        pred_digits = set(re.findall(r"\d+", re.sub(r"\[[A-Za-z0-9_:\-\s,]+\]", "", pred)))
        if cat == 2 and gold_digits:
            if gold_digits & pred_digits:
                return 1.0
            return 0.5 if recall >= 0.40 else 0.0
        return 1.0 if (recall >= 0.50 or (has_exact_evidence_citation and recall >= 0.30)) else 0.5
    if has_exact_evidence_citation and cat == 3:
        # Category 3 (open-ended inference like "Yes, since she collects classic children's books")
        # where the model grounds directly in the exact evidence turn D6:9
        return 0.5
    return 0.0


def _turn_to_session_num(turn_id: str) -> str:
    m = re.match(r"^([A-Za-z]+\d+):", turn_id.strip())
    return m.group(1) if m else turn_id


def evaluate_single_prediction(
    qa_item: Dict[str, Any],
    resp: AgentResponse,
    full_context_tokens: int = 8500,
) -> Dict[str, Any]:
    gold_ans = str(qa_item.get("answer", ""))
    ev_tids = list(qa_item.get("evidence_turn_ids", []))
    ev_set = set(ev_tids)

    cited_tids = list(resp.all_cited_turn_ids)
    cited_set = set(cited_tids)
    retrieved_set = set(resp.retrieved_turn_ids)

    # 1. Answer Metrics
    lex_metrics = compute_token_f1_and_recall(resp.answer, gold_ans)
    j_score = compute_judge_score(qa_item, resp)

    # 2. Provenance Metrics (Hard Requirement 2)
    if ev_set:
        cit_hits = len(ev_set & cited_set)
        cit_recall = cit_hits / len(ev_set)
        cit_precision = (cit_hits / len(cited_set)) if cited_set else (1.0 if resp.abstained else 0.0)
        cit_f1 = (
            (2 * cit_precision * cit_recall) / (cit_precision + cit_recall)
            if (cit_precision + cit_recall) > 0
            else 0.0
        )
        ret_recall = len(ev_set & retrieved_set) / len(ev_set)

        ev_sessions = {_turn_to_session_num(t) for t in ev_set}
        cited_sessions = {_turn_to_session_num(t) for t in cited_set}
        session_hit = 1.0 if (ev_sessions & cited_sessions) else 0.0
    else:
        cit_recall = 1.0
        cit_precision = 1.0
        cit_f1 = 1.0
        ret_recall = 1.0
        session_hit = 1.0

    # Citation Validity Rate: did every emitted claim pass Layer-1 turn verification?
    if resp.claims:
        validity = sum(1.0 for c in resp.claims if c.verified and c.source_turn_ids) / len(
            resp.claims
        )
    else:
        validity = 1.0 if resp.abstained else 0.0

    token_savings = max(
        0.0, (1.0 - (resp.prompt_tokens / max(full_context_tokens, 1))) * 100.0
    )

    return {
        "qa_id": qa_item["qa_id"],
        "conversation_id": qa_item["conversation_id"],
        "category": qa_item["category"],
        "category_name": qa_item["category_name"],
        "question": qa_item["question"],
        "gold_answer": gold_ans,
        "pred_answer": resp.answer,
        "abstained": resp.abstained,
        "gold_evidence_turn_ids": ev_tids,
        "cited_turn_ids": cited_tids,
        "retrieved_turn_ids": resp.retrieved_turn_ids[:10],
        "judge_score": round(j_score, 4),
        "answer_recall": round(lex_metrics["answer_recall"], 4),
        "token_f1": round(lex_metrics["token_f1"], 4),
        "citation_recall": round(cit_recall, 4),
        "citation_precision": round(cit_precision, 4),
        "citation_f1": round(cit_f1, 4),
        "retrieval_recall": round(ret_recall, 4),
        "session_provenance_hit": round(session_hit, 4),
        "citation_validity": round(validity, 4),
        "prompt_tokens": resp.prompt_tokens,
        "token_savings_pct": round(token_savings, 2),
        "latency_ms": round(resp.latency_ms, 2),
    }
