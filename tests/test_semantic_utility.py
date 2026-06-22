from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqa_semcom.semantic.utility import (  # noqa: E402
    SemanticUtilityModel,
    build_semantic_utility_from_predictions,
    write_semantic_utility_csv,
)


def _row(
    *,
    qtype: str = "presence",
    service_level: int = 1,
    snr_bin: str = "0dB",
    correct: bool = True,
    payload_bytes: int = 1024,
    view: str = "good",
    freshness: str = "fresh",
    risk: str = "normal",
) -> dict[str, str]:
    return {
        "question_type": qtype,
        "service_level": str(service_level),
        "channel_bin": "medium",
        "snr_bin": snr_bin,
        "view_quality_bin": view,
        "freshness_bin": freshness,
        "risk_level": risk,
        "correct": str(correct),
        "payload_bytes": str(payload_bytes),
    }


class SemanticUtilityTest(unittest.TestCase):
    def test_builds_required_utility_fields_with_ci(self) -> None:
        rows = [_row(correct=True), _row(correct=False), _row(correct=True)]
        cells = build_semantic_utility_from_predictions(rows)
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell.sample_count, 3)
        self.assertGreaterEqual(cell.accuracy_mean, 0.0)
        self.assertLessEqual(cell.accuracy_mean, 1.0)
        self.assertGreaterEqual(cell.accuracy_ci_low, 0.0)
        self.assertLessEqual(cell.accuracy_ci_high, 1.0)
        self.assertEqual(cell.accuracy_lcb, cell.accuracy_ci_low)
        self.assertAlmostEqual(cell.payload_kb, 1.0)
        self.assertGreaterEqual(cell.uncertainty, 0.0)
        self.assertLessEqual(cell.uncertainty, 1.0)

    def test_snr_monotonic_calibration_prevents_higher_snr_drop(self) -> None:
        rows: list[dict[str, str]] = []
        rows.extend([_row(snr_bin="0dB", correct=True) for _ in range(8)])
        rows.extend([_row(snr_bin="20dB", correct=False) for _ in range(8)])
        cells = build_semantic_utility_from_predictions(rows)
        by_snr = {cell.snr_bin: cell for cell in cells}
        self.assertGreaterEqual(by_snr["20dB"].accuracy_mean, by_snr["0dB"].accuracy_mean)
        self.assertIn("snr_monotonic_adjusted", by_snr["20dB"].calibration_note)

    def test_cache_service_is_snr_invariant(self) -> None:
        rows: list[dict[str, str]] = []
        rows.extend([_row(service_level=0, snr_bin="0dB", correct=True, payload_bytes=0) for _ in range(4)])
        rows.extend([_row(service_level=0, snr_bin="20dB", correct=False, payload_bytes=0) for _ in range(4)])
        cells = build_semantic_utility_from_predictions(rows)
        values = {cell.accuracy_mean for cell in cells}
        self.assertEqual(len(values), 1)
        self.assertTrue(all("cache_snr_invariant" in cell.calibration_note for cell in cells))

    def test_query_api_returns_lcb_payload_uncertainty_and_count(self) -> None:
        rows = [_row(snr_bin="0dB", correct=True), _row(snr_bin="0dB", correct=False)]
        cells = build_semantic_utility_from_predictions(rows)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "utility.csv"
            write_semantic_utility_csv(cells, csv_path)
            model = SemanticUtilityModel.from_csv(csv_path)
            estimate = model.U_sem("presence", 1, "0dB", "good", "fresh", "normal")
        self.assertEqual(estimate.sample_count, 2)
        self.assertGreaterEqual(estimate.accuracy_mean, 0.0)
        self.assertGreaterEqual(estimate.accuracy_lcb, 0.0)
        self.assertAlmostEqual(estimate.payload_kb, 1.0)
        self.assertGreater(estimate.uncertainty, 0.0)

    def test_query_api_uses_nearest_snr_fallback(self) -> None:
        rows = [_row(snr_bin="0dB", correct=True), _row(snr_bin="20dB", correct=True)]
        model = SemanticUtilityModel(build_semantic_utility_from_predictions(rows))
        estimate = model.U_sem("presence", 1, "18dB", "good", "fresh", "normal")
        self.assertGreater(estimate.sample_count, 0)

    def test_sparse_cells_are_marked_with_high_uncertainty(self) -> None:
        sparse = build_semantic_utility_from_predictions([_row(correct=True)], min_samples=5)[0]
        dense = build_semantic_utility_from_predictions([_row(correct=True) for _ in range(25)], min_samples=5)[0]
        self.assertIn("sparse_cell", sparse.calibration_note)
        self.assertGreater(sparse.uncertainty, dense.uncertainty)


if __name__ == "__main__":
    unittest.main()

