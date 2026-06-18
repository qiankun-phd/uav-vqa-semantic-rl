from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path

from vqa_semcom.config import ensure_parent


@dataclass(frozen=True)
class ModelSetupStatus:
    status: str
    model_id: str
    model_path: str
    source: str
    message: str


def model_cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def default_hf_cache_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def looks_like_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(payload.get("weight_map", {}).values())
        except Exception:
            shard_names = set()
        has_weights = bool(shard_names) and all((path / name).exists() and (path / name).stat().st_size > 1024 for name in shard_names)
    else:
        weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        has_weights = bool(weight_files) and all(file.stat().st_size > 1024 for file in weight_files)
    return has_config and has_weights


def find_cached_snapshot(model_id: str, cache_dir: Path | None = None) -> Path | None:
    root = (cache_dir or default_hf_cache_dir()) / model_cache_dir_name(model_id) / "snapshots"
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if looks_like_model_dir(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_model_reference(vlm_cfg: dict) -> str:
    local_path = str(vlm_cfg.get("model_local_path", "") or "").strip()
    if local_path and looks_like_model_dir(Path(local_path).expanduser()):
        return str(Path(local_path).expanduser())
    model_id = str(vlm_cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"))
    cached = find_cached_snapshot(model_id)
    if cached is not None:
        return str(cached)
    return model_id


def write_status_report(status: ModelSetupStatus, path: Path) -> None:
    ensure_parent(path)
    lines = [
        "# Qwen Model Setup Status",
        "",
        f"- status: `{status.status}`",
        f"- model id: `{status.model_id}`",
        f"- model path: `{status.model_path}`",
        f"- source: `{status.source}`",
        f"- message: {status.message}",
        f"- generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def download_from_huggingface(model_id: str, endpoint: str | None = None) -> Path:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model_id, resume_download=True)
    return Path(path)


def download_from_modelscope(model_id: str) -> Path:
    from modelscope import snapshot_download

    # ModelScope commonly uses the same namespace for Qwen models.
    path = snapshot_download(model_id)
    return Path(path)
