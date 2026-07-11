#!/usr/bin/env python3
"""O2 probe step 2 (CPU): symbolic decoder facing free-phrased questions.

The deployed symbolic decoder never parses question text: intent
(question_type, target_class[, target_class_b, threshold_n]) is metadata that
the template generator supplies. Under free phrasing that front end must be an
actual question parser. We evaluate two parser variants:

  * strict   -- exact regexes of the five generator templates (the parser the
                template-aligned system implicitly has). Parse failure => the
                decoder cannot form a query => scored incorrect.
  * fallback -- cheap keyword/cue-based intent classifier + class-name spotting
                (the "cheap remediation" of probe item 4).

If the recovered intent equals the true intent, the downstream decode is
byte-identical to the original template run, so the outcome is the original
s1 `correct`. If the intent is recovered wrongly, we re-decode from the
transmitted token stream (detector_counts_by_class in evidence_repr) under the
wrong intent and score against the true ground truth (uncalibrated counts for
wrong-class counting; calibration ratios are class-specific and unavailable
for a misparsed class -- noted caveat, such cases are rare).

Output: outputs/vlm/o2_symbolic_results.csv (two rows per paraphrase item:
parser=strict|fallback).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqa_semcom.vlm.answer import check_answer  # noqa: E402

IN_CSV = REPO_ROOT / "outputs/vlm/o2_paraphrases.csv"
OUT_CSV = REPO_ROOT / "outputs/vlm/o2_symbolic_results.csv"
TOLERANCE = 0.10  # == vlm.count_tolerance_ratio used everywhere in the chain

CLASSES = ["awning-tricycle", "pedestrian", "bicycle", "tricycle", "people",
           "truck", "motor", "car", "van", "bus"]  # longest-first matters

TEMPLATE_RES = [
    ("threshold", re.compile(r"^Are there at least (\d+) (.+?) objects in this area\?$")),
    ("comparison", re.compile(r"^Are there more (.+?) than (.+?) objects in this area\?$")),
    ("co_presence", re.compile(r"^Are there both (.+?) and (.+?) objects in this area\?$")),
    ("counting", re.compile(r"^How many (.+?) objects are in this area\?$")),
    ("presence", re.compile(r"^Are there (.+?) objects in this area\?$")),
]

THRESHOLD_CUES = ["at least", "or more", "or higher", "no fewer", "reach"]
COMPARISON_CUES = ["outnumber", "more numerous", "more frequent", "higher",
                   "larger", "compared", "more"]
CO_PRESENCE_CUES = ["both", "as well as", "together with", "and also"]
COUNTING_CUES = ["how many", "count", "number of", "total", "tally"]


def parse_strict(question: str):
    """Exact template regexes; class names must be valid VisDrone classes."""
    for qtype, rx in TEMPLATE_RES:
        m = rx.match(question)
        if not m:
            continue
        if qtype == "threshold":
            n, a = m.group(1), m.group(2)
            if a in CLASSES:
                return qtype, a, "", n
        elif qtype in ("comparison", "co_presence"):
            a, b = m.group(1), m.group(2)
            if a in CLASSES and b in CLASSES:
                return qtype, a, b, ""
        else:
            a = m.group(1)
            if a in CLASSES:
                return qtype, a, "", ""
    return None


def _find_classes(text: str) -> list[str]:
    """Class mentions in order of appearance; longest-first, spans masked so
    'tricycle' does not re-match inside 'awning-tricycle'."""
    hits: list[tuple[int, str]] = []
    masked = text
    for cls in CLASSES:
        for m in re.finditer(r"(?<![\w-])" + re.escape(cls) + r"(?![\w-])", masked):
            hits.append((m.start(), cls))
            masked = masked[:m.start()] + "#" * len(cls) + masked[m.end():]
    hits.sort()
    return [c for _, c in hits]


def parse_fallback(question: str):
    """Cheap keyword/cue intent parser (remediation candidate)."""
    low = question.lower()
    classes = _find_classes(low)
    nums = re.findall(r"\d+", low)
    if not classes:
        return None
    if nums and any(c in low for c in THRESHOLD_CUES):
        return "threshold", classes[0], "", nums[0]
    if len(classes) >= 2 and any(c in low for c in CO_PRESENCE_CUES):
        return "co_presence", classes[0], classes[1], ""
    if len(classes) >= 2 and any(c in low for c in COMPARISON_CUES):
        m = re.search(r"compared (?:with|to) ([\w-]+)", low)
        if m and m.group(1) in classes:
            b = m.group(1)
            a = next(c for c in classes if c != b)
            return "comparison", a, b, ""
        return "comparison", classes[0], classes[1], ""
    if any(c in low for c in COUNTING_CUES):
        return "counting", classes[0], "", ""
    return "presence", classes[0], "", ""


def counts_from_evidence(evidence: str) -> dict[str, int]:
    m = re.search(r"detector_counts_by_class: (.+)", evidence)
    if not m or m.group(1).strip() == "none":
        return {}
    out = {}
    for part in m.group(1).split(","):
        name, _, cnt = part.strip().rpartition(":")
        if name:
            out[name] = int(cnt)
    return out


def decode_with_intent(intent, evidence: str) -> str:
    qtype, a, b, n = intent
    counts = counts_from_evidence(evidence)
    ca, cb = counts.get(a, 0), counts.get(b, 0)
    if qtype == "presence":
        return "yes" if ca > 0 else "no"
    if qtype == "counting":
        return str(ca)
    if qtype == "comparison":
        return "yes" if ca > cb else "no"
    if qtype == "co_presence":
        return "yes" if ca > 0 and cb > 0 else "no"
    if qtype == "threshold":
        return "yes" if ca >= int(n) else "no"
    raise ValueError(qtype)


def norm_intent(qtype, a, b, n):
    n = str(int(n)) if str(n).strip() not in ("", "nan") else ""
    if qtype == "co_presence":  # decoder is symmetric in (a, b)
        a, b = sorted([a, b])
    return (qtype, a, b, n)


def main() -> int:
    df = pd.read_csv(IN_CSV, dtype={"image_id": str})
    rows = []
    for _, r in df.iterrows():
        truth = norm_intent(r.question_type, r.target_class,
                            r.target_class_b if isinstance(r.target_class_b, str) else "",
                            r.threshold_n if not pd.isna(r.threshold_n) else "")
        for parser_name, parser in (("strict", parse_strict), ("fallback", parse_fallback)):
            parsed = parser(str(r.question_para))
            if parsed is None:
                parse_ok, intent_match, predicted, correct = False, False, "", False
                pq = pa = pb = pn = ""
            else:
                pq, pa, pb, pn = parsed
                parse_ok = True
                intent = norm_intent(pq, pa, pb, pn)
                intent_match = intent == truth
                if intent_match:
                    predicted = "(same-as-original-decode)"
                    correct = bool(r.orig_symbolic_correct)
                else:
                    predicted = decode_with_intent((pq, pa, pb, pn), str(r.evidence_repr))
                    chk = check_answer(r.question_type, predicted,
                                       str(r.ground_truth_answer), tolerance_ratio=TOLERANCE)
                    correct = bool(chk.correct)
            rows.append({
                "image_id": r.image_id, "question_type": r.question_type,
                "question_orig": r.question_orig, "paraphrase_id": r.paraphrase_id,
                "template_id": r.template_id, "question_para": r.question_para,
                "parser": parser_name, "parse_ok": parse_ok,
                "parsed_qtype": pq, "parsed_class_a": pa, "parsed_class_b": pb,
                "parsed_n": pn, "intent_match": intent_match,
                "predicted": predicted, "correct": correct,
                "orig_symbolic_correct": bool(r.orig_symbolic_correct),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    for parser_name in ("strict", "fallback"):
        sub = out[out.parser == parser_name]
        print(f"[{parser_name}] parse_ok={sub.parse_ok.mean():.3f} "
              f"intent_match={sub.intent_match.mean():.3f} "
              f"acc={sub.correct.mean():.3f} (orig acc={sub.orig_symbolic_correct.mean():.3f})")
    # sanity: fallback parser must reproduce intent on the ORIGINAL template questions
    n_bad = 0
    for _, r in df.drop_duplicates(["image_id", "question_orig"]).iterrows():
        truth = norm_intent(r.question_type, r.target_class,
                            r.target_class_b if isinstance(r.target_class_b, str) else "",
                            r.threshold_n if not pd.isna(r.threshold_n) else "")
        parsed = parse_fallback(str(r.question_orig))
        if parsed is None or norm_intent(*parsed) != truth:
            n_bad += 1
    print(f"fallback on ORIGINAL templates: {n_bad} intent mismatches / "
          f"{df.drop_duplicates(['image_id', 'question_orig']).shape[0]}")
    print(f"wrote {len(out)} rows -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
