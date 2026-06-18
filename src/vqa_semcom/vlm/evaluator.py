from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vqa_semcom.vlm.answer import check_answer
from vqa_semcom.vlm.model_setup import resolve_model_reference


@dataclass(frozen=True)
class Prediction:
    predicted_answer: str
    normalized_prediction: str
    correct: bool
    latency_sec: float
    model_name: str


class Evaluator(Protocol):
    model_name: str

    def predict(self, task: dict[str, str], prompt: str, image_path: Path | None = None) -> str:
        ...


def deterministic_unit(*parts: str) -> float:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockVQAEvaluator:
    model_name = "mock-vlm"

    def predict(self, task: dict[str, str], prompt: str, image_path: Path | None = None) -> str:
        service_level = prompt.split("service_level=")[-1].split()[0] if "service_level=" in prompt else "2"
        channel = prompt.split("channel=")[-1].split()[0] if "channel=" in prompt else "good"
        view = task.get("view_quality_bin", "medium")
        qtype = task["question_type"]
        score = 0.56
        score += {"0": 0.00, "1": 0.16, "2": 0.24}.get(service_level, 0.0)
        score += {"bad": -0.12, "medium": 0.02, "good": 0.10}.get(channel, 0.0)
        score += {"poor": -0.10, "medium": 0.02, "good": 0.08}.get(view, 0.0)
        score += {"presence": 0.08, "counting": -0.02}.get(qtype, 0.0)
        unit = deterministic_unit(task["image_id"], task["question"], service_level, channel, view)
        is_correct = unit < max(0.05, min(0.98, score))
        if is_correct:
            return task["answer"]
        if qtype == "presence":
            return "no" if task["answer"].strip().lower() == "yes" else "yes"
        try:
            count = int(float(task["answer"]))
        except ValueError:
            count = 0
        return str(max(0, count + (2 if count < 5 else -2)))


class QwenVLEvaluator:
    def __init__(
        self,
        model_name: str,
        fallback_model_name: str | None = None,
        device: str = "cuda",
        torch_dtype: str = "auto",
        processor_use_fast: bool = False,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 24,
    ) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._device = device
        self._torch_dtype = torch_dtype
        self._processor_use_fast = processor_use_fast
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._max_new_tokens = max_new_tokens
        self._load(model_name, fallback_model_name)

    def _load(self, model_name: str, fallback_model_name: str | None) -> None:
        try:
            self._load_one(model_name)
        except Exception:
            if not fallback_model_name:
                raise
            self.model_name = fallback_model_name
            self._load_one(fallback_model_name)

    def _load_one(self, model_name: str) -> None:
        import torch
        from transformers import AutoProcessor

        model_name_lower = model_name.lower()
        model_type = ""
        config_path = Path(model_name) / "config.json"
        if config_path.exists():
            try:
                model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")).lower()
            except Exception:
                model_type = ""
        if "qwen2.5-vl" in model_name_lower or "qwen2_5" in model_name_lower or model_type == "qwen2_5_vl":
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass
        elif "qwen2-vl" in model_name_lower or "qwen2_vl" in model_name_lower or model_type == "qwen2_vl":
            try:
                from transformers import Qwen2VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass
        else:
            from transformers import AutoModelForVision2Seq as ModelClass

        is_local_model_dir = Path(model_name).expanduser().exists()
        dtype = "auto"
        if self._torch_dtype != "auto":
            dtype = getattr(torch, self._torch_dtype)
        device_map = "auto" if self._device == "cuda" and torch.cuda.is_available() else None
        kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if device_map is not None:
            kwargs["device_map"] = device_map
        if is_local_model_dir:
            kwargs["local_files_only"] = True
        processor_kwargs = {"trust_remote_code": True, "use_fast": self._processor_use_fast}
        if is_local_model_dir:
            processor_kwargs["local_files_only"] = True
        if self._min_pixels is not None:
            processor_kwargs["min_pixels"] = self._min_pixels
        if self._max_pixels is not None:
            processor_kwargs["max_pixels"] = self._max_pixels
        self._processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self._model = ModelClass.from_pretrained(model_name, **kwargs)
        if device_map is None and self._device:
            self._model.to(self._device)
        self._model.eval()

    def predict(self, task: dict[str, str], prompt: str, image_path: Path | None = None) -> str:
        if self._processor is None or self._model is None:
            raise RuntimeError("Qwen evaluator is not loaded.")
        import torch

        content = []
        if image_path is not None:
            content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = None
        if image_path is not None:
            try:
                from qwen_vl_utils import process_vision_info

                images, _ = process_vision_info(messages)
            except Exception:
                from PIL import Image

                images = [Image.open(image_path).convert("RGB")]
        inputs = self._processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        answer = self._processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer.strip()


def make_evaluator(kind: str, cfg: dict) -> Evaluator:
    if kind == "mock":
        return MockVQAEvaluator()
    if kind == "qwen":
        vlm_cfg = cfg.get("vlm", {})
        model_reference = resolve_model_reference(vlm_cfg)
        fallback_model_name = vlm_cfg.get("fallback_model_name", "Qwen/Qwen2-VL-2B-Instruct")
        if Path(model_reference).expanduser().exists():
            fallback_model_name = None
        return QwenVLEvaluator(
            model_name=model_reference,
            fallback_model_name=fallback_model_name,
            device=vlm_cfg.get("device", "cuda"),
            torch_dtype=vlm_cfg.get("torch_dtype", "auto"),
            processor_use_fast=bool(vlm_cfg.get("processor_use_fast", False)),
            min_pixels=vlm_cfg.get("min_pixels"),
            max_pixels=vlm_cfg.get("max_pixels"),
            max_new_tokens=int(vlm_cfg.get("max_new_tokens", 24)),
        )
    raise ValueError(f"Unsupported evaluator kind: {kind}")


def evaluate_prediction(evaluator: Evaluator, task: dict[str, str], prompt: str, image_path: Path | None, cfg: dict) -> Prediction:
    start = time.perf_counter()
    predicted = evaluator.predict(task, prompt, image_path=image_path)
    latency = time.perf_counter() - start
    check = check_answer(
        task["question_type"],
        predicted,
        task["answer"],
        tolerance_ratio=float(cfg.get("vlm", {}).get("count_tolerance_ratio", 0.10)),
    )
    return Prediction(
        predicted_answer=predicted,
        normalized_prediction=check.normalized_prediction,
        correct=check.correct,
        latency_sec=latency,
        model_name=evaluator.model_name,
    )
