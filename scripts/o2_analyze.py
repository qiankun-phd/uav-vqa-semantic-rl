#!/usr/bin/env python3
"""O2 probe step 4 (CPU): aggregate the free-phrasing probe.

Channels
  a_strict   symbolic decoder, strict template parser front end
  a_fallback symbolic decoder, cheap keyword-parser front end
  b          VLM reads token text
  c          VLM reads degraded image

Original-phrasing controls come from existing logs, restricted to the sampled
440 questions: s1 `correct` (symbolic), p1_vlm_reads_tokens (b), s2 `correct`
(c). All numbers are Rician @5dB on the held-out test split.

Outputs: outputs/vlm/o2_probe_summary.csv + outputs/vlm/o2_probe_report.md
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PARA_CSV = REPO_ROOT / "outputs/vlm/o2_paraphrases.csv"
SYM_CSV = REPO_ROOT / "outputs/vlm/o2_symbolic_results.csv"
TOK_CSV = REPO_ROOT / "outputs/vlm/o2_vlm_tokens_predictions.csv"
IMG_CSV = REPO_ROOT / "outputs/vlm/o2_vlm_image_predictions.csv"
P1_CSV = REPO_ROOT / "outputs/vlm/p1_vlm_reads_tokens_rician_predictions.csv"
OUT_SUMMARY = REPO_ROOT / "outputs/vlm/o2_probe_summary.csv"
OUT_REPORT = REPO_ROOT / "outputs/vlm/o2_probe_report.md"

QTYPES = ["presence", "counting", "comparison", "co_presence", "threshold"]


def mcnemar_exact(x: pd.Series, y: pd.Series) -> tuple[int, int, float]:
    """Paired exact McNemar (two-sided binomial on discordant pairs)."""
    x = x.astype(bool).to_numpy()
    y = y.astype(bool).to_numpy()
    b = int(((x) & (~y)).sum())   # x correct, y wrong
    c = int(((~x) & (y)).sum())   # x wrong, y correct
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def main() -> int:
    para = pd.read_csv(PARA_CSV, dtype={"image_id": str})
    sym = pd.read_csv(SYM_CSV, dtype={"image_id": str})
    tok = pd.read_csv(TOK_CSV, dtype={"image_id": str})
    img = pd.read_csv(IMG_CSV, dtype={"image_id": str})
    p1 = pd.read_csv(P1_CSV, dtype={"image_id": str})
    p1 = p1[p1.snr_bin == "5dB"][["image_id", "question", "correct"]].rename(
        columns={"question": "question_orig", "correct": "orig_vlm_tokens_correct"})

    key = ["image_id", "question_orig", "paraphrase_id"]
    m = para.merge(
        sym[sym.parser == "strict"][key + ["parse_ok", "intent_match", "correct"]].rename(
            columns={"parse_ok": "strict_parse_ok", "intent_match": "strict_intent_match",
                     "correct": "a_strict"}), on=key, validate="1:1")
    m = m.merge(
        sym[sym.parser == "fallback"][key + ["parse_ok", "intent_match", "correct"]].rename(
            columns={"parse_ok": "fb_parse_ok", "intent_match": "fb_intent_match",
                     "correct": "a_fallback"}), on=key, validate="1:1")
    m = m.merge(tok[key + ["correct"]].rename(columns={"correct": "b_tokens"}),
                on=key, validate="1:1")
    m = m.merge(img[key + ["correct"]].rename(columns={"correct": "c_image"}),
                on=key, validate="1:1")
    m = m.merge(p1, on=["image_id", "question_orig"], how="left", validate="m:1")
    assert m.orig_vlm_tokens_correct.notna().all(), "missing p1 control rows"
    for col in ["a_strict", "a_fallback", "b_tokens", "c_image",
                "orig_symbolic_correct", "orig_vlm_tokens_correct", "orig_vlm_image_correct",
                "strict_parse_ok", "strict_intent_match", "fb_parse_ok", "fb_intent_match"]:
        m[col] = m[col].astype(bool)
    print(f"merged items: {len(m)}")

    # ---- per-qtype summary table ------------------------------------------
    recs = []
    for qt in QTYPES + ["ALL"]:
        g = m if qt == "ALL" else m[m.question_type == qt]
        gq = g.drop_duplicates(["image_id", "question_orig"])  # question-level controls
        recs.append({
            "question_type": qt,
            "n_questions": len(gq), "n_para_items": len(g),
            # original-phrasing controls (sampled questions only)
            "orig_symbolic": gq.orig_symbolic_correct.mean(),
            "orig_vlm_tokens": gq.orig_vlm_tokens_correct.mean(),
            "orig_vlm_image": gq.orig_vlm_image_correct.mean(),
            "orig_adv_sym_minus_vlmtok": gq.orig_symbolic_correct.mean() - gq.orig_vlm_tokens_correct.mean(),
            # paraphrase results
            "para_a_strict": g.a_strict.mean(),
            "para_a_fallback": g.a_fallback.mean(),
            "para_b_vlm_tokens": g.b_tokens.mean(),
            "para_c_vlm_image": g.c_image.mean(),
            "para_adv_fallback_minus_vlmtok": g.a_fallback.mean() - g.b_tokens.mean(),
            "para_adv_strict_minus_vlmtok": g.a_strict.mean() - g.b_tokens.mean(),
            # parser diagnostics
            "strict_parse_fail_rate": 1.0 - g.strict_parse_ok.mean(),
            "fallback_parse_fail_rate": 1.0 - g.fb_parse_ok.mean(),
            "fallback_intent_match_rate": g.fb_intent_match.mean(),
        })
    summary = pd.DataFrame(recs)
    summary.to_csv(OUT_SUMMARY, index=False)

    # ---- McNemar tests ------------------------------------------------------
    tests = []
    for qt in QTYPES + ["ALL"]:
        g = m if qt == "ALL" else m[m.question_type == qt]
        for name, x, y in [
            ("a_fallback vs b_tokens (para)", g.a_fallback, g.b_tokens),
            ("a_strict vs b_tokens (para)", g.a_strict, g.b_tokens),
            ("b_tokens vs c_image (para)", g.b_tokens, g.c_image),
            ("b_tokens: orig vs para", g.orig_vlm_tokens_correct, g.b_tokens),
            ("c_image: orig vs para", g.orig_vlm_image_correct, g.c_image),
            ("a_fallback: orig vs para", g.orig_symbolic_correct, g.a_fallback),
        ]:
            b, c, p = mcnemar_exact(x, y)
            tests.append({"question_type": qt, "test": name,
                          "acc_x": x.mean(), "acc_y": y.mean(),
                          "b_x_only": b, "c_y_only": c, "p_exact": p})
    tests = pd.DataFrame(tests)

    # ---- verdict ------------------------------------------------------------
    all_row = summary[summary.question_type == "ALL"].iloc[0]
    adv_orig = all_row.orig_adv_sym_minus_vlmtok
    adv_para = all_row.para_adv_fallback_minus_vlmtok
    p_adv = tests[(tests.question_type == "ALL") &
                  (tests.test == "a_fallback vs b_tokens (para)")].p_exact.iloc[0]
    if adv_para >= adv_orig - 0.02:
        verdict = "(i) advantage preserved under free phrasing (template-alignment effect negligible)"
    elif adv_para > 0 and p_adv < 0.05:
        verdict = (f"(ii) advantage shrinks ({adv_orig:+.3f} -> {adv_para:+.3f}) "
                   f"but remains significant (McNemar p={p_adv:.2e})")
    else:
        verdict = (f"(iii) advantage is mostly template alignment "
                   f"({adv_orig:+.3f} -> {adv_para:+.3f}, p={p_adv:.2e})")

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        f.write("# O2 free-phrasing probe (Rician @5dB, held-out test split)\n\n")
        f.write(f"- sampled questions: {m.drop_duplicates(['image_id','question_orig']).shape[0]}"
                f", paraphrase items: {len(m)} (2 per question)\n")
        f.write(f"- verdict (symbolic-with-fallback vs VLM-reads-tokens): {verdict}\n")
        f.write(f"- strict template parser on paraphrases: parse-fail rate "
                f"{1 - m.strict_parse_ok.mean():.3f}\n")
        f.write(f"- keyword fallback parser: parse-fail {1 - m.fb_parse_ok.mean():.3f}, "
                f"intent-match {m.fb_intent_match.mean():.3f}\n\n")
        f.write("## Per-type accuracy\n\n")
        f.write(summary.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## McNemar exact tests\n\n")
        f.write(tests.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(tests[tests.question_type == "ALL"].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT_SUMMARY} and {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
