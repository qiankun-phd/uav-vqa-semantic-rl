"""W3a tests: per-sample predictor round-trip, prototype parity, calibration."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vqa_semcom.quality.persample_predictor import (  # noqa: E402
    CLASSES,
    QTYPES,
    RISKS,
    VIEWS,
    PersamplePredictor,
    featurize,
    fit_logistic,
)


def _load_prototype():
    path = ROOT / "scripts" / "build_persample_policy.py"
    spec = importlib.util.spec_from_file_location("build_persample_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synth(n: int, seed: int = 7):
    rng = random.Random(seed)
    truth1 = [rng.uniform(-1.5, 1.5) for _ in range(26)]
    truth2 = [rng.uniform(-1.5, 1.5) for _ in range(26)]
    records, y1, y2 = [], [], []
    for _ in range(n):
        r = {
            "question_type": rng.choice(QTYPES),
            "target_class": rng.choice(CLASSES),
            "view_quality_bin": rng.choice(VIEWS),
            "risk_level": rng.choice(RISKS),
            "snr_bin": f"{rng.choice([-5, 0, 5, 10, 15, 20])}dB",
            "raw_detector_count": str(rng.randrange(0, 60)),
            "density_score": f"{rng.uniform(0.0, 200.0):.2f}",
            "presence_polarity": rng.choice(["positive", "negative"]),
        }
        x = featurize(r)
        z1 = sum(a * b for a, b in zip(truth1, x))
        z2 = sum(a * b for a, b in zip(truth2, x))
        y1.append(rng.random() < 1 / (1 + np.exp(-z1)))
        y2.append(rng.random() < 1 / (1 + np.exp(-z2)))
        records.append(r)
    return records, y1, y2


@pytest.fixture(scope="module")
def synth():
    return _synth(1500)


def test_save_load_round_trip(tmp_path, synth):
    records, y1, y2 = synth
    calib = [i % 5 == 0 for i in range(len(records))]
    model = PersamplePredictor.fit(records, {"1": y1, "2": y2}, calib_mask=calib,
                                   meta={"origin": "unit-test"})
    path = tmp_path / "model.json"
    model.save(path)
    loaded = PersamplePredictor.load(path)
    assert loaded.feature_version == model.feature_version
    assert loaded.services == model.services
    assert loaded.meta["origin"] == "unit-test"
    for s in model.services:
        assert loaded.heads[s].temperature == pytest.approx(model.heads[s].temperature)
        np.testing.assert_allclose(loaded.heads[s].weights, model.heads[s].weights)
        np.testing.assert_allclose(
            loaded.predict_proba(records, s), model.predict_proba(records, s))
        np.testing.assert_allclose(
            loaded.uncertainty(records, s), model.uncertainty(records, s))
    assert loaded.select(records) == model.select(records)


def test_featurize_matches_prototype(synth):
    proto = _load_prototype()
    records, _, _ = synth
    for r in records[:200]:
        assert featurize(r) == proto.featurize(r)


def test_decisions_match_prototype(synth):
    proto = _load_prototype()
    records, y1, y2 = synth
    X = [featurize(r) for r in records]
    w1 = proto.fit_logistic(X, [float(v) for v in y1])
    w2 = proto.fit_logistic(X, [float(v) for v in y2])
    p1, p2 = proto.predict(w1, X), proto.predict(w2, X)
    proto_sel = ["1" if (b - a) <= 0.0 else "2" for a, b in zip(p1, p2)]

    # No calibration fold -> weights trained on the identical sample set.
    model = PersamplePredictor.fit(records, {"1": y1, "2": y2})
    np.testing.assert_allclose(model.heads["1"].weights, fit_logistic(X, [float(v) for v in y1]))
    sel = model.select(records, margin=0.0, calibrated=False)
    agreement = sum(int(a == b) for a, b in zip(proto_sel, sel)) / len(sel)
    assert agreement > 0.99


def test_calibration_monotone_and_bounded(synth):
    records, y1, y2 = synth
    calib = [i % 4 == 0 for i in range(len(records))]
    model = PersamplePredictor.fit(records, {"1": y1, "2": y2}, calib_mask=calib)
    for s in model.services:
        head = model.heads[s]
        assert head.temperature > 0.0
        z = model.predict_logit(records, s)
        p = model.predict_proba(records, s, calibrated=True)
        assert np.all((p >= 0.0) & (p <= 1.0))
        order = np.argsort(z)
        assert np.all(np.diff(p[order]) >= -1e-12), "temperature scaling must keep ranking"
        u = model.uncertainty(records, s)
        assert np.all((u >= 0.0) & (u <= 1.0))


def test_rl_style_query_features():
    # The RL backend queries with the simulator vocabulary ("medium" view,
    # no detector-side features); featurize must stay total and map the alias.
    query = {"question_type": "counting", "view_quality_bin": "medium",
             "risk_level": "critical", "snr_bin": "10dB"}
    x = featurize(query)
    assert len(x) == 26
    view_slice = x[len(QTYPES) + len(CLASSES):len(QTYPES) + len(CLASSES) + len(VIEWS)]
    assert view_slice == [0.0, 1.0, 0.0]  # medium -> fair
