"""Per-sample service-quality predictor (W3a).

Pure-numpy, JSON-serialisable quality source that replaces coarse LUT cells
with per-sample calibrated correctness probabilities:

    P(correct | features, service)  for service in {"1" (token), "2" (image)}

The feature map (``feature_version="v1"``) is copied verbatim from the offline
prototype ``scripts/build_persample_policy.py`` so decisions stay comparable.
Each service head is a logistic regression trained with full-batch gradient
descent (identical hyper-parameters to the prototype) plus post-hoc
temperature scaling fitted on a held-out calibration fold.  A per-service
reliability table (10 probability bins) is stored alongside the weights and
drives the ``uncertainty`` estimate used by the RL quality backend
(uncertainty replaces the Wilson-LCB gap of the LUT path).

Extension hook: ``featurize`` accepts an optional ``service`` argument.  For
``feature_version="v1"`` it is ignored (one weight vector per service); a
future ``"v2"`` may fold the service level into the feature vector so a
single head covers all services.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FEATURE_VERSION = "v1"

QTYPES = ["presence", "counting", "comparison", "co_presence", "threshold"]
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor"]
VIEWS = ["poor", "fair", "good"]
RISKS = ["normal", "elevated", "critical"]

# The RL simulator uses "medium" for the middle view bin while the VLM
# prediction logs use "fair"; normalise before one-hot encoding.
_VIEW_ALIASES = {"medium": "fair"}

_DEFAULT_L2 = 1e-3
_DEFAULT_ITERS = 400
_DEFAULT_LR = 0.5
_DEFAULT_BINS = 10


def snr_value(record: Mapping[str, Any]) -> float:
    """SNR in dB from ``snr_bin`` (e.g. ``"5dB"``) or numeric fallbacks."""
    raw = record.get("snr_bin")
    if raw not in (None, ""):
        return float(str(raw).lower().replace("db", ""))
    for key in ("snr_db", "sensed_snr_db"):
        if record.get(key) not in (None, ""):
            return float(record[key])
    return 0.0


def featurize(record: Mapping[str, Any], service: int | str | None = None,
              feature_version: str = FEATURE_VERSION) -> list[float]:
    """Prototype-identical v1 feature map (see build_persample_policy.py).

    ``service`` is a reserved extension slot: ignored under v1.
    """
    if feature_version != "v1":
        raise ValueError(f"unknown feature_version: {feature_version}")
    view = record.get("view_quality_bin")
    view = _VIEW_ALIASES.get(str(view), view)
    f: list[float] = []
    f += [1.0 if record.get("question_type") == q else 0.0 for q in QTYPES]
    f += [1.0 if record.get("target_class") == c else 0.0 for c in CLASSES]
    f += [1.0 if view == v else 0.0 for v in VIEWS]
    f += [1.0 if record.get("risk_level") == k else 0.0 for k in RISKS]
    f.append(snr_value(record) / 20.0)
    raw = float(record.get("raw_detector_count") or 0)
    f.append(min(raw, 60.0) / 60.0)
    f.append(1.0 if raw > 0 else 0.0)
    f.append(float(record.get("density_score") or 0) / 200.0 if record.get("density_score") else 0.0)
    f.append(1.0 if record.get("presence_polarity") == "negative" else 0.0)
    return f


def feature_names(feature_version: str = FEATURE_VERSION) -> list[str]:
    if feature_version != "v1":
        raise ValueError(f"unknown feature_version: {feature_version}")
    return (
        [f"qtype={q}" for q in QTYPES]
        + [f"class={c}" for c in CLASSES]
        + [f"view={v}" for v in VIEWS]
        + [f"risk={k}" for k in RISKS]
        + ["snr_norm", "raw_count_norm", "raw_count_nonzero", "density_norm", "presence_negative"]
    )


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = _DEFAULT_L2,
                 iters: int = _DEFAULT_ITERS, lr: float = _DEFAULT_LR) -> np.ndarray:
    """Full-batch GD logistic fit, numerically identical to the prototype."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """1-D NLL grid search for the temperature-scaling parameter T."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(logits) == 0 or len(np.unique(y)) < 2:
        return 1.0
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(20.0), 400)):
        p = np.clip(_sigmoid(logits / t), 1e-12, 1.0 - 1e-12)
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def _reliability_table(probs: np.ndarray, y: np.ndarray,
                       n_bins: int = _DEFAULT_BINS) -> list[list[float]]:
    """Rows of [bin_lo, bin_hi, mean_predicted, empirical_accuracy, count]."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    rows: list[list[float]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n:
            rows.append([lo, hi, float(probs[mask].mean()), float(y[mask].mean()), float(n)])
        else:
            rows.append([lo, hi, (lo + hi) / 2.0, (lo + hi) / 2.0, 0.0])
    return rows


@dataclass
class ServiceHead:
    weights: np.ndarray            # (d + 1,), bias last
    temperature: float
    reliability: list[list[float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "weights": [float(v) for v in self.weights],
            "temperature": float(self.temperature),
            "reliability": [[float(v) for v in row] for row in self.reliability],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ServiceHead":
        return cls(
            weights=np.asarray(data["weights"], dtype=float),
            temperature=float(data.get("temperature", 1.0)),
            reliability=[list(map(float, row)) for row in data.get("reliability", [])],
        )


class PersamplePredictor:
    """Calibrated per-sample correctness predictor with per-service heads."""

    def __init__(self, heads: Mapping[str, ServiceHead],
                 feature_version: str = FEATURE_VERSION,
                 meta: Mapping[str, Any] | None = None) -> None:
        if not heads:
            raise ValueError("PersamplePredictor needs at least one service head")
        self.heads: dict[str, ServiceHead] = {str(k): v for k, v in heads.items()}
        self.feature_version = str(feature_version)
        self.meta: dict[str, Any] = dict(meta or {})

    # ------------------------------------------------------------------ API
    @property
    def services(self) -> list[str]:
        return sorted(self.heads)

    def _head(self, service: int | str) -> ServiceHead:
        key = str(int(service)) if not isinstance(service, str) else service
        if key not in self.heads:
            raise KeyError(f"no head for service {service!r} (have {self.services})")
        return self.heads[key]

    def _design(self, records: Sequence[Mapping[str, Any]],
                service: int | str | None = None) -> np.ndarray:
        X = np.asarray([featurize(r, service=service, feature_version=self.feature_version)
                        for r in records], dtype=float)
        return np.hstack([X, np.ones((len(X), 1))])

    def predict_logit(self, records: Sequence[Mapping[str, Any]],
                      service: int | str) -> np.ndarray:
        head = self._head(service)
        return self._design(records, service) @ head.weights

    def predict_proba(self, records: Sequence[Mapping[str, Any]],
                      service: int | str, calibrated: bool = True) -> np.ndarray:
        head = self._head(service)
        z = self.predict_logit(records, service)
        return _sigmoid(z / head.temperature if calibrated else z)

    def uncertainty(self, records: Sequence[Mapping[str, Any]],
                    service: int | str) -> np.ndarray:
        """Reliability-table uncertainty for each record's calibrated proba.

        Per bin: |mean_predicted - empirical| calibration residual plus the
        binomial CI half-width of the bin's empirical accuracy.  Empty bins
        return 1.0 (fully uncertain) so sparse regions stay conservative.
        """
        head = self._head(service)
        probs = self.predict_proba(records, service, calibrated=True)
        out = np.ones_like(probs)
        for lo, hi, mean_pred, emp, n in head.reliability:
            mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
            if not mask.any():
                continue
            if n <= 0:
                out[mask] = 1.0
                continue
            resid = abs(mean_pred - emp)
            half = 1.96 * float(np.sqrt(max(emp * (1.0 - emp), 1e-12) / n))
            out[mask] = min(1.0, resid + half)
        return out

    def select(self, records: Sequence[Mapping[str, Any]], margin: float = 0.0,
               calibrated: bool = True,
               cheap: str = "1", rich: str = "2") -> list[str]:
        """Cost-aware selection: prefer the cheap service within ``margin``."""
        p_cheap = self.predict_proba(records, cheap, calibrated=calibrated)
        p_rich = self.predict_proba(records, rich, calibrated=calibrated)
        return [cheap if (pr - pc) <= margin else rich for pc, pr in zip(p_cheap, p_rich)]

    # ------------------------------------------------------------ train/io
    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]],
            labels: Mapping[str, Sequence[Any]],
            calib_mask: Sequence[bool] | None = None,
            l2: float = _DEFAULT_L2, iters: int = _DEFAULT_ITERS,
            lr: float = _DEFAULT_LR, n_bins: int = _DEFAULT_BINS,
            feature_version: str = FEATURE_VERSION,
            meta: Mapping[str, Any] | None = None) -> "PersamplePredictor":
        """Train per-service heads.

        ``labels`` maps service key -> per-record 0/1 labels.  Rows flagged in
        ``calib_mask`` are excluded from the weight fit and used to fit the
        temperature and the reliability table; without a mask the model is
        left uncalibrated (T=1, reliability from the training data itself).
        """
        if not records:
            raise ValueError("fit needs at least one record")
        X = np.asarray([featurize(r, feature_version=feature_version) for r in records],
                       dtype=float)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        mask = (np.asarray(calib_mask, dtype=bool) if calib_mask is not None
                else np.zeros(len(records), dtype=bool))
        if len(mask) != len(records):
            raise ValueError("calib_mask length mismatch")
        heads: dict[str, ServiceHead] = {}
        for service, y_raw in labels.items():
            y = np.asarray([1.0 if bool(v) else 0.0 for v in y_raw], dtype=float)
            if len(y) != len(records):
                raise ValueError(f"labels for service {service!r} length mismatch")
            w = fit_logistic(X[~mask], y[~mask], l2=l2, iters=iters, lr=lr)
            if mask.any():
                z_cal = Xb[mask] @ w
                t = _fit_temperature(z_cal, y[mask])
                rel = _reliability_table(_sigmoid(z_cal / t), y[mask], n_bins=n_bins)
            else:
                t = 1.0
                rel = _reliability_table(_sigmoid(Xb[~mask] @ w), y[~mask], n_bins=n_bins)
            heads[str(service)] = ServiceHead(weights=w, temperature=t, reliability=rel)
        info = dict(meta or {})
        info.setdefault("n_fit", int((~mask).sum()))
        info.setdefault("n_calib", int(mask.sum()))
        info.setdefault("l2", l2)
        info.setdefault("iters", iters)
        info.setdefault("lr", lr)
        return cls(heads, feature_version=feature_version, meta=info)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "persample_predictor",
            "feature_version": self.feature_version,
            "feature_names": feature_names(self.feature_version),
            "services": {k: head.to_json() for k, head in self.heads.items()},
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PersamplePredictor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "persample_predictor":
            raise ValueError(f"{path}: not a persample_predictor json")
        heads = {k: ServiceHead.from_json(v) for k, v in data["services"].items()}
        return cls(heads, feature_version=data.get("feature_version", FEATURE_VERSION),
                   meta=data.get("meta", {}))


def expected_calibration_error(probs: Iterable[float], y: Iterable[Any],
                               n_bins: int = _DEFAULT_BINS) -> float:
    probs = np.asarray(list(probs), dtype=float)
    yv = np.asarray([1.0 if bool(v) else 0.0 for v in y], dtype=float)
    if len(probs) == 0:
        return 0.0
    ece = 0.0
    for lo, hi, mean_pred, emp, n in _reliability_table(probs, yv, n_bins=n_bins):
        if n > 0:
            ece += (n / len(probs)) * abs(mean_pred - emp)
    return float(ece)
