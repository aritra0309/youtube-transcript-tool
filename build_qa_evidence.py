#!/usr/bin/env python3
"""Build automated QA evidence and scoring from transcript files."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TIME_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")
WORD_RE = re.compile(r"[a-z0-9]+")

PROFILE_NAME = "balanced_strict_v1"


DOMAIN_SYNONYMS = {
    "argmax": {"highest", "brightest", "maximum", "max", "choice"},
    "activation": {"activations", "score", "scores", "confidence", "brightest"},
    "class": {"digit", "label", "output", "classification"},
    "autoregressive": {"repeated", "repeat", "iterative", "loop", "again"},
    "token": {"tokens", "word", "words", "chunk", "chunks"},
    "sequence": {"text", "passage", "context"},
    "generation": {"generate", "prediction", "predict", "completion"},
    "weather": {"rain", "humidity", "temperature", "precipitation", "wind"},
    "supervised": {"training", "data", "algorithm", "labeled"},
    "inference": {"predict", "prediction", "output"},
    "umbrella": {"broad", "field", "overall"},
    "statistical": {"statistics", "technique", "techniques"},
    "neural": {"network", "networks", "brain", "layer"},
    "attention": {"context", "relevant", "update", "talk"},
    "feedforward": {"mlp", "perceptron", "parallel", "independent"},
    "positionwise": {"parallel", "independently", "same"},
    "distribution": {"probability", "sample", "sampling"},
}

STOP_TERMS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "this",
    "that",
    "it",
    "as",
    "by",
    "from",
    "be",
    "using",
    "use",
    "what",
    "which",
    "how",
    "into",
    "then",
    "than",
    "where",
    "when",
    "their",
    "between",
    "through",
    "does",
    "your",
    "about",
    "these",
    "those",
    "very",
    "just",
    "also",
}

HIGH_SIGNAL_TERMS = {
    "predict",
    "prediction",
    "probability",
    "distribution",
    "attention",
    "layer",
    "neural",
    "network",
    "activation",
    "output",
    "input",
    "token",
    "transformer",
    "mlp",
    "feedforward",
    "algorithm",
    "learning",
    "model",
    "data",
}


CONCEPT_GROUPS = {
    "qa1": [
        {"output", "layer", "neuron", "digit"},
        {"activation", "score", "confidence"},
        {"argmax", "highest", "brightest", "choice", "prediction"},
    ],
    "qa2": [
        {"probability", "distribution"},
        {"sample", "sampling"},
        {"append", "added", "text"},
        {"repeat", "again", "iterative", "autoregressive"},
    ],
    "qa3": [
        {"precipitation", "humidity", "temperature", "wind"},
        {"rain", "predict", "prediction"},
        {"data", "training", "algorithm"},
    ],
    "qa4": [
        {"artificial", "intelligence", "ai"},
        {"machine", "learning"},
        {"deep", "learning", "neural", "network"},
        {"statistical", "technique"},
    ],
    "qa5": [
        {"attention", "context", "relevant", "update"},
        {"mlp", "feedforward", "perceptron"},
        {"parallel", "independent", "same", "operation"},
    ],
}


@dataclass
class TranscriptLine:
    line_no: int
    timestamp: str
    seconds: int
    text: str


def to_seconds(ts: str) -> int:
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def normalize_token(token: str) -> str:
    t = token.lower().strip()
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [normalize_token(t) for t in WORD_RE.findall(text.lower())]


def expand_token_set(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for tok in list(tokens):
        expanded.update({normalize_token(v) for v in DOMAIN_SYNONYMS.get(tok, set())})
        for key, values in DOMAIN_SYNONYMS.items():
            normalized_values = {normalize_token(v) for v in values}
            if tok in normalized_values:
                expanded.add(normalize_token(key))
                expanded.update(normalized_values)
    return expanded


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_ngrams(text: str, n: int = 4) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def char_ngram_similarity(a_text: str, b_text: str) -> float:
    a = char_ngrams(a_text, n=4)
    b = char_ngrams(b_text, n=4)
    return jaccard(a, b)


def read_timestamped_transcript(path: Path) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for idx, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = TIME_RE.match(raw_line)
        if not m:
            continue
        ts, text = m.groups()
        lines.append(
            TranscriptLine(
                line_no=idx,
                timestamp=ts,
                seconds=to_seconds(ts),
                text=text.strip(),
            )
        )
    return lines


def claim_density_score(text: str) -> float:
    lower = text.lower()
    keyword_hits = sum(1 for k in HIGH_SIGNAL_TERMS if k in lower)
    words = len(tokenize(text))
    if words == 0:
        return 0.0
    return keyword_hits / words


def high_signal_hits(text: str) -> int:
    tokens = set(tokenize(text))
    expanded = expand_token_set(tokens)
    return sum(1 for t in HIGH_SIGNAL_TERMS if t in expanded)


def window_lines(lines: list[TranscriptLine], start_s: int, end_s: int) -> list[TranscriptLine]:
    return [line for line in lines if start_s <= line.seconds <= end_s]


def spans_from_lines(lines: list[TranscriptLine], min_len: int = 2, max_len: int = 5):
    spans: list[list[TranscriptLine]] = []
    for i in range(len(lines)):
        for span_len in range(min_len, max_len + 1):
            j = i + span_len
            if j <= len(lines):
                spans.append(lines[i:j])
    return spans


def line_relevance_score(line: TranscriptLine, query_tokens: set[str]) -> float:
    line_tokens = expand_token_set(set(tokenize(line.text)))
    overlap = jaccard(query_tokens, line_tokens)
    density = claim_density_score(line.text)
    char_sim = char_ngram_similarity(" ".join(sorted(query_tokens)), line.text)
    signal_bonus = min(1.0, high_signal_hits(line.text) / 5)

    words = len(tokenize(line.text))
    length_penalty = 0.0
    if words < 4:
        length_penalty = 0.05

    return (0.45 * overlap) + (0.20 * density) + (0.20 * char_sim) + (0.15 * signal_bonus) - length_penalty


def best_overlap_lines(lines: list[TranscriptLine], query_text: str, top_k: int = 8) -> list[TranscriptLine]:
    query_tokens = expand_token_set(set(tokenize(query_text)))
    if not lines:
        return []

    # Span-level ranking for better contextual evidence
    span_candidates = spans_from_lines(lines, min_len=2, max_len=5)
    span_scores: list[tuple[float, list[TranscriptLine]]] = []
    for span in span_candidates:
        line_scores = [line_relevance_score(line, query_tokens) for line in span]
        span_text = " ".join(line.text for line in span)
        span_tokens = expand_token_set(set(tokenize(span_text)))
        span_overlap = jaccard(query_tokens, span_tokens)
        span_char = char_ngram_similarity(query_text, span_text)
        score = (0.50 * (sum(line_scores) / len(line_scores))) + (0.30 * span_overlap) + (0.20 * span_char)
        span_scores.append((score, span))

    span_scores.sort(key=lambda item: item[0], reverse=True)

    # take top spans, then dedupe lines preserving order
    picked_lines: list[TranscriptLine] = []
    seen = set()
    for _, span in span_scores[:4]:
        for line in span:
            if line.line_no not in seen:
                seen.add(line.line_no)
                picked_lines.append(line)

    picked_lines.sort(key=lambda line: line.seconds)
    return picked_lines[:top_k]


def key_terms_for_coverage(answer: str, question: str) -> set[str]:
    answer_tokens = set(tokenize(answer))
    question_tokens = set(tokenize(question))
    return {
        t
        for t in (answer_tokens | question_tokens)
        if len(t) >= 5 and t not in STOP_TERMS
    }


def concept_coverage(qa_id: str, evidence_tokens: set[str]) -> float:
    groups = CONCEPT_GROUPS.get(qa_id, [])
    if not groups:
        return 0.0

    covered = 0
    for group in groups:
        norm_group = {normalize_token(g) for g in group}
        expanded_group = expand_token_set(norm_group)
        if evidence_tokens & expanded_group:
            covered += 1
    return covered / len(groups)


def contains_required_terms(answer: str, question: str, evidence_text: str) -> dict[str, float | list[str]]:
    ev_tokens = expand_token_set(set(tokenize(evidence_text)))
    key_terms = key_terms_for_coverage(answer, question)

    if not key_terms:
        return {"coverage": 1.0, "missing_terms": []}

    present: list[str] = []
    missing: list[str] = []
    for term in sorted(key_terms):
        variants = {term}
        variants.update({normalize_token(v) for v in DOMAIN_SYNONYMS.get(term, set())})
        if variants & ev_tokens:
            present.append(term)
        else:
            missing.append(term)

    coverage = len(present) / len(key_terms)
    return {"coverage": round(coverage, 4), "missing_terms": missing}


def score_pair(
    qa_id: str,
    question: str,
    answer: str,
    evidence_lines: list[TranscriptLine],
    time_window_seconds: int,
) -> dict:
    evidence_text = " ".join(line.text for line in evidence_lines)
    answer_tokens = expand_token_set(set(tokenize(answer)))
    question_tokens = expand_token_set(set(tokenize(question)))
    evidence_tokens = expand_token_set(set(tokenize(evidence_text)))

    lexical_overlap = jaccard(answer_tokens, evidence_tokens)
    question_alignment = jaccard(question_tokens, evidence_tokens)
    char_sim = char_ngram_similarity(answer + " " + question, evidence_text)

    density = 0.0
    if evidence_lines:
        density = sum(claim_density_score(line.text) for line in evidence_lines) / len(evidence_lines)

    term_check = contains_required_terms(answer, question, evidence_text)
    concept_cov = concept_coverage(qa_id, evidence_tokens)

    relevance = (0.55 * lexical_overlap) + (0.25 * char_sim) + (0.20 * question_alignment)

    final = (
        0.33 * relevance
        + 0.28 * ((0.65 * term_check["coverage"]) + (0.35 * concept_cov))
        + 0.20 * question_alignment
        + 0.14 * density
        + 0.05 * min(1.0, time_window_seconds / 180)
    )

    high_signal_line_count = sum(1 for line in evidence_lines if high_signal_hits(line.text) >= 2)

    return {
        "final_score": round(final, 4),
        "relevance_score": round(relevance, 4),
        "lexical_overlap": round(lexical_overlap, 4),
        "char_similarity": round(char_sim, 4),
        "question_alignment": round(question_alignment, 4),
        "claim_density": round(density, 4),
        "term_coverage": term_check["coverage"],
        "concept_coverage": round(concept_cov, 4),
        "missing_terms": term_check["missing_terms"],
        "high_signal_line_count": high_signal_line_count,
    }


def status_from_score(score: dict) -> tuple[str, str]:
    final = score["final_score"]

    if final >= 0.47 and score["concept_coverage"] >= 0.30 and score["high_signal_line_count"] >= 2:
        return "pass", "meets score + concept + evidence quality gates"

    if final >= 0.37:
        return "review", "mid score band; useful but requires human check"

    if score["concept_coverage"] < 0.30:
        return "fail", "low concept coverage"

    return "fail", "score below threshold"


def main() -> None:
    root = Path(__file__).resolve().parent
    qa_path = root / "qa_pairs.json"
    out_json = root / "qa_evidence.json"
    out_csv = root / "retrieval_checks.csv"
    out_log = root / "method_log.txt"

    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))

    evidence_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "build_qa_evidence.py",
        "evaluator_profile": PROFILE_NAME,
        "pairs": [],
    }

    csv_rows = [
        [
            "id",
            "video_id",
            "source_file",
            "start",
            "end",
            "window_seconds",
            "score",
            "status",
            "reason",
            "relevance_score",
            "lexical_overlap",
            "char_similarity",
            "question_alignment",
            "term_coverage",
            "concept_coverage",
            "claim_density",
            "high_signal_line_count",
        ]
    ]

    pass_count = 0
    review_count = 0
    fail_count = 0

    for pair in qa_pairs:
        source = pair["source"]
        transcript_path = root / source["transcript_file"]
        lines = read_timestamped_transcript(transcript_path)

        start_s = to_seconds(source["start"])
        end_s = to_seconds(source["end"])
        window_s = max(0, end_s - start_s)

        in_window = window_lines(lines, start_s, end_s)
        query_text = f"{pair['question']} {pair['answer']}"
        evidence_lines = best_overlap_lines(in_window, query_text, top_k=10)
        score = score_pair(
            qa_id=pair["id"],
            question=pair["question"],
            answer=pair["answer"],
            evidence_lines=evidence_lines,
            time_window_seconds=window_s,
        )
        status, reason = status_from_score(score)

        if status == "pass":
            pass_count += 1
        elif status == "review":
            review_count += 1
        else:
            fail_count += 1

        evidence_report["pairs"].append(
            {
                "id": pair["id"],
                "question": pair["question"],
                "answer": pair["answer"],
                "source": source,
                "window_seconds": window_s,
                "score": score,
                "status": status,
                "status_reason": reason,
                "evidence_lines": [
                    {
                        "line_no": line.line_no,
                        "timestamp": line.timestamp,
                        "text": line.text,
                    }
                    for line in evidence_lines
                ],
            }
        )

        csv_rows.append(
            [
                pair["id"],
                source["video_id"],
                source["transcript_file"],
                source["start"],
                source["end"],
                str(window_s),
                str(score["final_score"]),
                status,
                reason,
                str(score["relevance_score"]),
                str(score["lexical_overlap"]),
                str(score["char_similarity"]),
                str(score["question_alignment"]),
                str(score["term_coverage"]),
                str(score["concept_coverage"]),
                str(score["claim_density"]),
                str(score["high_signal_line_count"]),
            ]
        )

    out_json.write_text(json.dumps(evidence_report, indent=2, ensure_ascii=True), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_rows)

    log_lines = [
        f"Generated at (UTC): {evidence_report['generated_at_utc']}",
        f"Method: automated transcript-window evidence extraction + scoring ({PROFILE_NAME})",
        "Inputs:",
        f"- QA file: {qa_path.name}",
        "Outputs:",
        f"- {out_json.name}",
        f"- {out_csv.name}",
        "Scoring rubric (Balanced-Strict Hybrid):",
        "- 33% evidence relevance",
        "- 28% blended term+concept coverage",
        "- 20% question-evidence alignment",
        "- 14% claim-density average",
        "- 5% timestamp-window quality",
        "Hard quality gates for pass:",
        "- final score >= 0.47",
        "- concept coverage >= 0.30",
        "- at least 2 high-signal evidence lines",
        "Thresholds:",
        "- pass: >= 0.47 with gates",
        "- review: 0.37 to 0.4699",
        "- fail: < 0.37 or below hard gates",
        "Results:",
        f"- pass: {pass_count}",
        f"- review: {review_count}",
        f"- fail: {fail_count}",
    ]
    out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_log}")
    print(f"Summary => pass={pass_count}, review={review_count}, fail={fail_count}")


if __name__ == "__main__":
    main()
