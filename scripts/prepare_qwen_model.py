#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path
from vqa_semcom.vlm.model_setup import (
    ModelSetupStatus,
    download_from_huggingface,
    download_from_modelscope,
    find_cached_snapshot,
    looks_like_model_dir,
    write_status_report,
)


def _write(status: ModelSetupStatus, report_path: Path) -> int:
    write_status_report(status, report_path)
    print(f"status={status.status}")
    print(f"model_path={status.model_path}")
    print(f"source={status.source}")
    print(f"report={report_path}")
    return 0 if status.status in {"ready", "downloaded", "missing"} else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_qwen.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-local-path", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--try-modelscope", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vlm_cfg = cfg.get("vlm", {})
    model_id = args.model_name or vlm_cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct")
    report_path = resolve_path(cfg["paths"].get("qwen_model_status_md", "outputs/reports/qwen_model_setup_status.md"))

    local_path = args.model_local_path or vlm_cfg.get("model_local_path", "")
    if local_path and looks_like_model_dir(Path(local_path).expanduser()):
        return _write(ModelSetupStatus("ready", model_id, str(Path(local_path).expanduser()), "local_path", "Local model directory is usable."), report_path)

    cached = find_cached_snapshot(model_id)
    if cached is not None:
        return _write(ModelSetupStatus("ready", model_id, str(cached), "huggingface_cache", "Found complete cached Hugging Face snapshot."), report_path)

    if not args.download:
        return _write(
            ModelSetupStatus(
                "missing",
                model_id,
                "",
                "none",
                "No complete local model snapshot found. Re-run with --download to try HF mirror and optional ModelScope.",
            ),
            report_path,
        )

    errors: list[str] = []
    for endpoint, source in [("https://hf-mirror.com", "hf_mirror"), (None, "huggingface")]:
        try:
            path = download_from_huggingface(model_id, endpoint=endpoint)
            return _write(ModelSetupStatus("downloaded", model_id, str(path), source, "Model downloaded successfully."), report_path)
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")

    if args.try_modelscope:
        try:
            path = download_from_modelscope(model_id)
            return _write(ModelSetupStatus("downloaded", model_id, str(path), "modelscope", "Model downloaded successfully."), report_path)
        except Exception as exc:
            errors.append(f"modelscope: {type(exc).__name__}: {exc}")

    return _write(
        ModelSetupStatus(
            "blocked",
            model_id,
            "",
            "download_failed",
            "Model download failed. " + " | ".join(errors),
        ),
        report_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
