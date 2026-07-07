#!/usr/bin/env python3
"""Build presence+counting VQA tasks for the DroneVehicle val split.

Input : data/raw/dronevehicle/val/vallabel/*.xml   (VOC-style; objects carry a
        <bndbox> OR a <polygon> -- we derive an axis-aligned bbox from whichever
        is present, though only the per-class COUNT is used downstream).
Images: data/raw/dronevehicle/val/valimg/<id>.jpg   (840x712, ~100px white border
        that we deliberately DO NOT crop -- the detector/VLM see the raw frame,
        matching how a real UAV link would deliver it).

Class mapping (DroneVehicle -> VisDrone-countable family that our YOLOv8n knows):
    car            -> car
    truck          -> truck
    bus            -> bus
    van            -> van
    feright car    -> truck   (also the "feright_car" spelling variant; DroneVehicle's
    feright_car    -> truck    "freight car" == small covered lorry, no VisDrone class
                               exists for it and its appearance is closest to truck)

Output schema is COLUMN-IDENTICAL to outputs/tasks/v1_7_tasks.csv:
    image_id,question_type,question,answer,target_class,object_count,risk_level,
    epsilon_k,tau_k,view_quality_bin,scale_proxy,occlusion_score,truncation_score,
    density_score,presence_polarity

Quality-metadata columns are NOT available for DroneVehicle (no per-object
occlusion/truncation labels, single nadir view) so they are filled with FIXED
DEFAULTS -- they only matter for the mock evaluator; the real Qwen2-VL pass
ignores them:
    view_quality_bin = "medium"
    scale_proxy = occlusion_score = truncation_score = "0.0"
    density_score = total object count in the image (informational only)

risk_level / epsilon_k / tau_k follow the v1_7 cyclic template: task rows
alternate risk normal<->critical, with epsilon 0.65/0.82 and tau 5.0/3.0 tied to
the risk level.

Deterministic sub-sampling: sort image ids, take every STEP-th so that ~500
images remain (STEP = round(N/500)). Per image emit <=2 presence rows (one
positive present class, one negative absent class from {car,truck,bus,van}) and
<=2 counting rows (the two present classes with the largest counts). Presence
yes/no is balanced by construction (each image contributes one yes + one no).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

FIELDS = [
    "image_id", "question_type", "question", "answer", "target_class", "object_count",
    "risk_level", "epsilon_k", "tau_k", "view_quality_bin", "scale_proxy",
    "occlusion_score", "truncation_score", "density_score", "presence_polarity",
]

CLASS_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "feright car": "truck",
    "feright_car": "truck",
    "freight car": "truck",
    "freight_car": "truck",
}
PRESENCE_CLASSES = ["car", "truck", "bus", "van"]  # fixed order for absent-class picks
TARGET_IMAGES = 500

VQ = "medium"
SCALE = "0.0"
OCC = "0.0"
TRUNC = "0.0"


def parse_counts(xml_path: str) -> Counter:
    """Return canonical-class -> count for one XML (unknown names skipped)."""
    counts: Counter = Counter()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return counts
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            continue
        canon = CLASS_MAP.get(name_el.text.strip())
        if canon is None:
            continue
        counts[canon] += 1
    return counts


def risk_fields(idx: int):
    """Cyclic normal/critical template tied to epsilon/tau."""
    if idx % 2 == 0:
        return "normal", "0.65", "5.0"
    return "critical", "0.82", "3.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", default="data/raw/dronevehicle/val/vallabel")
    ap.add_argument("--out", default="outputs/tasks/dv_tasks.csv")
    ap.add_argument("--target-images", type=int, default=TARGET_IMAGES)
    args = ap.parse_args()

    xmls = sorted(glob.glob(os.path.join(args.label_dir, "*.xml")))
    if not xmls:
        raise SystemExit(f"no XML found under {args.label_dir}")
    step = max(1, round(len(xmls) / args.target_images))
    picked = xmls[::step]

    rows = []
    risk_ctr = 0
    for xml_path in picked:
        image_id = os.path.splitext(os.path.basename(xml_path))[0]
        counts = parse_counts(xml_path)
        total = sum(counts.values())
        density = f"{float(total):.1f}"
        present = [(c, n) for c, n in counts.items() if n > 0]
        present.sort(key=lambda kv: (-kv[1], kv[0]))
        absent = [c for c in PRESENCE_CLASSES if counts.get(c, 0) == 0]

        def base(qtype, question, answer, tclass, ocount, polarity):
            nonlocal risk_ctr
            rl, eps, tau = risk_fields(risk_ctr)
            risk_ctr += 1
            return {
                "image_id": image_id, "question_type": qtype, "question": question,
                "answer": answer, "target_class": tclass, "object_count": str(ocount),
                "risk_level": rl, "epsilon_k": eps, "tau_k": tau, "view_quality_bin": VQ,
                "scale_proxy": SCALE, "occlusion_score": OCC, "truncation_score": TRUNC,
                "density_score": density, "presence_polarity": polarity,
            }

        # ---- presence: one positive (present) + one negative (absent) ----
        if present:
            pc, pn = present[0]
            rows.append(base("presence", f"Are there {pc} objects in this area?",
                             "yes", pc, pn, "positive"))
        if absent:
            nc = absent[0]
            rows.append(base("presence", f"Are there {nc} objects in this area?",
                             "no", nc, 0, "negative"))

        # ---- counting: two largest present classes ----
        for pc, pn in present[:2]:
            rows.append(base("counting", f"How many {pc} objects are in this area?",
                             pn, pc, pn, ""))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    qt = Counter(r["question_type"] for r in rows)
    pres_ans = Counter(r["answer"] for r in rows if r["question_type"] == "presence")
    n_imgs = len({r["image_id"] for r in rows})
    print(f"images total={len(xmls)} step={step} picked={len(picked)} with_tasks={n_imgs}")
    print(f"rows={len(rows)} by_type={dict(qt)}")
    print(f"presence answer balance yes={pres_ans['yes']} no={pres_ans['no']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
