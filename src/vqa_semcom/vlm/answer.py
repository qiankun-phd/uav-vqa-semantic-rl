from __future__ import annotations

import re
from dataclasses import dataclass


YES_WORDS = {"yes", "true", "present", "there are", "there is"}
NO_WORDS = {"no", "false", "none", "not present", "there are no", "there is no"}


@dataclass(frozen=True)
class AnswerCheck:
    normalized_prediction: str
    correct: bool


def normalize_yes_no(text: str) -> str:
    value = " ".join(text.strip().lower().replace(".", " ").replace(",", " ").split())
    if not value:
        return "unknown"
    if any(value == word or value.startswith(word + " ") for word in NO_WORDS):
        return "no"
    if any(value == word or value.startswith(word + " ") for word in YES_WORDS):
        return "yes"
    if re.search(r"\b(no|none|false)\b", value):
        return "no"
    if re.search(r"\b(yes|true)\b", value):
        return "yes"
    return "unknown"


def extract_count(text: str) -> int | None:
    value = text.strip().lower()
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(r"-?\d+", value)
    if match:
        return max(0, int(match.group(0)))
    for word, number in number_words.items():
        if re.search(rf"\b{word}\b", value):
            return number
    return None


def check_answer(question_type: str, prediction: str, ground_truth: str, tolerance_ratio: float = 0.10) -> AnswerCheck:
    if question_type in {"presence", "risk", "comparison", "co_presence", "threshold"}:
        normalized = normalize_yes_no(prediction)
        return AnswerCheck(normalized_prediction=normalized, correct=normalized == ground_truth.strip().lower())
    if question_type == "counting":
        pred_count = extract_count(prediction)
        try:
            gt_count = int(float(ground_truth))
        except ValueError:
            gt_count = 0
        if pred_count is None:
            return AnswerCheck(normalized_prediction="unknown", correct=False)
        tolerance = max(1, round(tolerance_ratio * gt_count))
        return AnswerCheck(normalized_prediction=str(pred_count), correct=abs(pred_count - gt_count) <= tolerance)
    normalized = prediction.strip().lower()
    return AnswerCheck(normalized_prediction=normalized, correct=normalized == ground_truth.strip().lower())

