from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QwenPrepareScriptTest(unittest.TestCase):
    def test_prepare_qwen_dry_run_writes_status_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = tmp_path / "cfg.json"
            report = tmp_path / "status.md"
            cfg.write_text(
                "{"
                '"paths": {"qwen_model_status_md": "' + str(report) + '"},'
                '"vlm": {"model_name": "NotAReal/Model", "model_local_path": ""}'
                "}",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "prepare_qwen_model.py"), "--config", str(cfg)],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(report.exists())
            self.assertIn("status: `missing`", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
