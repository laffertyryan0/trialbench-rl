"""Reward functions for the 8 TrialBench tasks, one per reward "shape"
(continuous / binary / multi-class / structured), modeled on
tinker_cookbook.recipes.search_tool.tools.TextAnswerReward's contract:

    async def __call__(history: list[Message]) -> tuple[float, dict[str, float]]

All rewards look for "Answer:" as the final-answer marker (same convention
as TextAnswerReward, for consistency with the tool's own instructions) and
apply a small format_coef penalty when that marker is missing, so the model
gets partial signal even on malformed outputs rather than a flat zero.

Note on AUROC tasks (mortality_yn, adverse_yn, dropout_yn, outcome): the
RL reward here is plain correctness, NOT AUROC — AUROC is only meaningful
as an eval-time metric computed from logprobs, decoupled from what GRPO
optimizes during training. See design notes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message


def _final_assistant_text(history: list[Message]) -> str | None:
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return get_text_content(msg)
    return None


def _extract_answer(text: str) -> str | None:
    if "Answer:" not in text:
        return None
    return text.split("Answer:")[-1].strip()


@dataclass
class ContinuousReward:
    """Graded reward for a continuous target: exp(-|pred - true| / scale).

    `scale` should be set to roughly the target's typical spread (e.g. the
    IQR) so the reward isn't saturated at 0 or 1 everywhere — this is what
    keeps GRPO's within-group advantages non-degenerate for a continuous
    target (see design notes on reward-shape choice).
    """

    true_value: float
    scale: float
    format_coef: float = 0.1
    # "abs": exp(-|pred-true|/scale). Fine for roughly-symmetric targets like
    #   duration-in-days.
    # "log": exp(-|log10(pred+eps) - log10(true+eps)|/scale). Use for
    #   zero-skewed rates in [0,1] (mortality/adverse/dropout: median ~3e-4,
    #   mean ~0.1-0.2). Under "abs" with a small scale, answering "0.0"
    #   unconditionally scored ~0.75 on mortality_rate — a free-riding floor
    #   that GRPO can't learn past. Log-space keeps "0 when truth is ~0"
    #   rewarded while penalizing "0 when truth is 0.1" (~0.14).
    mode: str = "abs"
    log_eps: float = 1e-3

    def _graded(self, pred: float) -> float:
        if self.mode == "log":
            d = abs(math.log10(max(pred, 0.0) + self.log_eps) - math.log10(max(self.true_value, 0.0) + self.log_eps))
            return math.exp(-d / max(self.scale, 1e-6))
        return math.exp(-abs(pred - self.true_value) / max(self.scale, 1e-6))

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        text = _final_assistant_text(history)
        if text is None:
            return 0.0, {"format": 0.0, "graded": 0.0}
        answer = _extract_answer(text)
        correct_format = float(answer is not None)
        graded = 0.0
        if answer is not None:
            match = re.search(r"-?\d+\.?\d*", answer)
            if match:
                try:
                    pred = float(match.group())
                    graded = self._graded(pred)
                except ValueError:
                    pass
        reward = self.format_coef * (correct_format - 1) + graded
        return reward, {"format": correct_format, "graded": graded}


@dataclass
class BinaryReward:
    """Correctness reward for a binary (Y/N) target.

    Accepts yes/no, true/false, 0/1 in the answer text. This is the RL
    training signal; AUROC against this same task is computed separately
    at eval time from token logprobs, not from this reward.
    """

    true_value: int  # 0 or 1
    format_coef: float = 0.1
    _POSITIVE = {"yes", "y", "true", "1"}
    _NEGATIVE = {"no", "n", "false", "0"}

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        text = _final_assistant_text(history)
        if text is None:
            return 0.0, {"format": 0.0, "correct": 0.0}
        answer = _extract_answer(text)
        correct_format = float(answer is not None)
        correct = 0.0
        if answer is not None:
            norm = answer.strip().lower().rstrip(".")
            pred: int | None = None
            if norm in self._POSITIVE:
                pred = 1
            elif norm in self._NEGATIVE:
                pred = 0
            if pred is not None:
                correct = float(pred == self.true_value)
        reward = self.format_coef * (correct_format - 1) + correct
        return reward, {"format": correct_format, "correct": correct}


@dataclass
class MultiClassReward:
    """Exact-match reward over a fixed label set (failure_reason, dose_cls)."""

    true_label: str
    valid_labels: list[str]
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        text = _final_assistant_text(history)
        if text is None:
            return 0.0, {"format": 0.0, "correct": 0.0}
        answer = _extract_answer(text)
        correct_format = float(answer is not None)
        correct = 0.0
        if answer is not None:
            norm = answer.strip().lower().rstrip(".")
            valid_norm = {v.lower(): v for v in self.valid_labels}
            if norm in valid_norm:
                correct = float(valid_norm[norm] == self.true_label)
        reward = self.format_coef * (correct_format - 1) + correct
        return reward, {"format": correct_format, "correct": correct}


@dataclass
class EligibilityReward:
    """Partial-credit reward for eligibility-criteria-design: predict
    gender / min_age / max_age / healthy_volunteers as structured fields.
    (Free-text criteria generation is intentionally out of scope for this
    first pass — no good automatic scorer for it without an
    embedding-similarity judge, which is a separate piece of work.)

    Expects the model to answer in the form:
        Answer: gender=<All|Male|Female>; min_age=<years>; max_age=<years>; healthy_volunteers=<Yes|No>
    """

    true_gender: str
    true_min_age: float | None
    true_max_age: float | None
    true_healthy_volunteers: str
    format_coef: float = 0.1
    age_tolerance_years: float = 5.0

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        text = _final_assistant_text(history)
        if text is None:
            return 0.0, {"format": 0.0, "correct": 0.0}
        answer = _extract_answer(text)
        correct_format = float(answer is not None)
        if answer is None:
            return self.format_coef * (correct_format - 1), {"format": 0.0, "correct": 0.0}

        fields = {}
        for part in answer.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip().lower()] = v.strip()

        scores = []
        if "gender" in fields:
            scores.append(float(fields["gender"].lower() == self.true_gender.lower()))
        if "healthy_volunteers" in fields:
            scores.append(
                float(fields["healthy_volunteers"].lower() == self.true_healthy_volunteers.lower())
            )
        for key, true_age in (("min_age", self.true_min_age), ("max_age", self.true_max_age)):
            if key in fields and true_age is not None:
                m = re.search(r"-?\d+\.?\d*", fields[key])
                if m:
                    pred_age = float(m.group())
                    scores.append(float(abs(pred_age - true_age) <= self.age_tolerance_years))

        correct = sum(scores) / len(scores) if scores else 0.0
        reward = self.format_coef * (correct_format - 1) + correct
        return reward, {"format": correct_format, "correct": correct}
