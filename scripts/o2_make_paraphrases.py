#!/usr/bin/env python3
"""O2 probe step 1: sample held-out test questions and generate free-phrasing
paraphrases from handwritten per-type template pools.

Reviewer concern: questions are deterministic template instantiations, so the
symbolic decoder (and its implicit question->intent alignment) may benefit from
template alignment. This probe rephrases each sampled question (2 paraphrases,
answer semantics unchanged, class names kept verbatim) and re-evaluates the
three receive channels at Rician 5 dB.

Sampling: presence 100 / counting 100 / comparison 80 / co_presence 80 /
threshold 80 = 440 questions from the 936-question test split
(crc32(image_id)%100 < 20) of outputs/vlm/v3_0_rician_predictions.csv.

Output: outputs/vlm/o2_paraphrases.csv  (one row per paraphrase item, with the
original s1 evidence_repr and s2 degraded-image path attached so downstream
runners need no other inputs).
"""
from __future__ import annotations

import random
import re
import zlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SEED = 20260710
SNR_BIN = "5dB"
PRED_CSV = REPO_ROOT / "outputs/vlm/v3_0_rician_predictions.csv"
OUT_CSV = REPO_ROOT / "outputs/vlm/o2_paraphrases.csv"

N_PER_TYPE = {"presence": 100, "counting": 100, "comparison": 80,
              "co_presence": 80, "threshold": 80}

# Handwritten paraphrase pools. {A}/{B} are VisDrone class names kept verbatim
# (never pluralised: "people", "awning-tricycle" do not take -s). {N} is the
# threshold integer. Pools mix synonym phrasing, colloquial wording, inversion
# and implicit question words.
POOLS: dict[str, list[tuple[str, str]]] = {
    "presence": [
        ("P1", "Is there any {A} in this scene?"),
        ("P2", "Can you spot any {A} here?"),
        ("P3", "Does this area contain any {A} objects?"),
        ("P4", "Any sign of {A} in this view?"),
        ("P5", "Tell me whether {A} objects appear in this area."),
        ("P6", "{A} -- present in this scene or not?"),
    ],
    "counting": [
        ("C1", "Count the {A} objects you can see."),
        ("C2", "What's the total number of {A} in view?"),
        ("C3", "Give me the {A} count for this scene."),
        ("C4", "How many instances of {A} appear here?"),
        ("C5", "Report the number of {A} objects detected in this area."),
        ("C6", "{A} in this area -- what's the tally?"),
    ],
    "comparison": [
        ("M1", "Do {A} objects outnumber {B} objects here?"),
        ("M2", "Compared with {B}, are {A} more numerous in this scene?"),
        ("M3", "Is the {A} count higher than the {B} count in this area?"),
        ("M4", "Between {A} and {B}, does {A} have the larger count in view?"),
        ("M5", "Are {A} detections more frequent than {B} detections here?"),
    ],
    "co_presence": [
        ("B1", "Do both {A} and {B} appear in this scene?"),
        ("B2", "Can you find {A} as well as {B} here?"),
        ("B3", "Are {A} and {B} both present in this view?"),
        ("B4", "Does the scene contain {A} together with {B}?"),
        ("B5", "Is it true that this area has {A} and also {B}?"),
    ],
    "threshold": [
        ("T1", "Are there {N} or more {A} in this scene?"),
        ("T2", "Does the {A} count reach {N} in this area?"),
        ("T3", "Can you confirm at least {N} {A} objects in view?"),
        ("T4", "Is the number of {A} here {N} or higher?"),
        ("T5", "Would you say there are no fewer than {N} {A} in this scene?"),
    ],
}

# strict template regexes of the original generators (order matters)
TEMPLATE_RES = [
    ("threshold", re.compile(r"^Are there at least (\d+) (.+?) objects in this area\?$")),
    ("comparison", re.compile(r"^Are there more (.+?) than (.+?) objects in this area\?$")),
    ("co_presence", re.compile(r"^Are there both (.+?) and (.+?) objects in this area\?$")),
    ("counting", re.compile(r"^How many (.+?) objects are in this area\?$")),
    ("presence", re.compile(r"^Are there (.+?) objects in this area\?$")),
]


def parse_original(question: str) -> tuple[str, str, str, str]:
    """Return (qtype, class_a, class_b, threshold_n) from a template question."""
    for qtype, rx in TEMPLATE_RES:
        m = rx.match(question)
        if not m:
            continue
        if qtype == "threshold":
            return qtype, m.group(2), "", m.group(1)
        if qtype in ("comparison", "co_presence"):
            return qtype, m.group(1), m.group(2), ""
        return qtype, m.group(1), "", ""
    raise ValueError(f"original question does not match any template: {question!r}")


def is_test(image_id: str, frac: float = 0.2) -> bool:
    return (zlib.crc32(str(image_id).encode()) % 100) < int(frac * 100)


def main() -> int:
    rng = random.Random(SEED)
    df = pd.read_csv(PRED_CSV, dtype={"image_id": str}, low_memory=False)
    df = df[df.snr_bin == SNR_BIN]
    df = df[df.image_id.map(is_test)]
    s1 = (df[df.service_level == 1]
          .drop_duplicates(subset=["image_id", "question"])
          .set_index(["image_id", "question"]))
    s2 = (df[df.service_level == 2]
          .drop_duplicates(subset=["image_id", "question"])
          .set_index(["image_id", "question"]))
    print(f"test split @{SNR_BIN}: s1={len(s1)} s2={len(s2)}")

    rows = []
    for qtype, n_take in N_PER_TYPE.items():
        keys = sorted(k for k in s1.index if s1.loc[k, "question_type"] == qtype)
        take = rng.sample(keys, n_take)
        for image_id, question in sorted(take):
            r1 = s1.loc[(image_id, question)]
            r2 = s2.loc[(image_id, question)]
            _, class_a, class_b, thr_n = parse_original(question)
            pool = POOLS[qtype]
            picks = rng.sample(pool, 2)
            for pidx, (tid, tmpl) in enumerate(picks, start=1):
                para = tmpl.format(A=class_a, B=class_b, N=thr_n)
                rows.append({
                    "image_id": image_id,
                    "question_type": qtype,
                    "question_orig": question,
                    "paraphrase_id": pidx,
                    "template_id": tid,
                    "question_para": para,
                    "ground_truth_answer": str(r1["ground_truth_answer"]),
                    "target_class": class_a,
                    "target_class_b": class_b,
                    "threshold_n": thr_n,
                    "snr_bin": SNR_BIN,
                    "channel_bin": r1["channel_bin"],
                    "payload_bytes": r1["payload_bytes"],
                    "evidence_repr": r1["evidence_repr"],
                    "degraded_image_path": r2["image_path"],
                    "orig_symbolic_correct": bool(r1["correct"]),
                    "orig_raw_decoder_correct": r1["raw_decoder_correct"],
                    "orig_decoder_mode": r1["decoder_mode"],
                    "orig_vlm_image_correct": bool(r2["correct"]),
                })
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} paraphrase items "
          f"({out.question_type.value_counts().to_dict()}) -> {OUT_CSV}")
    # sanity: paraphrases keep class token verbatim
    bad = [(q, a) for q, a in zip(out.question_para, out.target_class) if a not in q]
    assert not bad, bad[:3]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
