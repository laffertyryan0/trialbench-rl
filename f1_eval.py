#!/usr/bin/env python3
"""Compute F1 (the paper's actual metric) for the trained checkpoint on
held-out test data, for direct comparison against TrialBench's Table 6
baselines. Binary F1 for mortality_yn/adverse_yn/dropout_yn/outcome,
macro-F1 for failure_reason/dose_cls (multi-class).

Reward functions give a training signal, not this metric — this is the
first time we've computed the actual number the paper reports.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict

import pandas as pd
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.tool_use import build_agent_tool_env

import loader
from pubchem_tool import PubChemTool
from rewards import _extract_answer
from trialbench_env import DOSE_CLS_LABELS, FAILURE_REASON_LABELS

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
CHECKPOINT_PATH = "tinker://895baa05-7593-5dd1-9a53-b8252959dac9:train:0/sampler_weights/final"  # batch 150 (final)
N_PER_TASK = 40
MAX_TOKENS = 2048
MAX_TURNS = 4

BINARY_TASKS = {
    "mortality_yn": ("mortality", "Y/N"),
    "adverse_yn": ("adverse_event", "Y/N"),
    "dropout_yn": ("dropout", "Y/N"),
    "outcome": ("outcome", "outcome"),
}
MULTICLASS_TASKS = {
    "failure_reason": ("failure_reason", "failure_reason", FAILURE_REASON_LABELS),
    "dose_cls": ("dose", "Avg", DOSE_CLS_LABELS),
}

from trialbench_env import _initial_messages, build_datum, TASK_INSTRUCTIONS  # noqa: E402


def sample_test_rows(underlying: str, label_col: str, n: int, seed: int = 0, is_phased: bool = True):
    if is_phased:
        xs, ys = [], []
        for p in ("1", "2", "3", "4"):
            x, y = loader.load_raw(underlying, phase=p, split="test")
            xs.append(x)
            ys.append(y)
        x, y = pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)
    else:
        x, y = loader.load_raw(underlying, phase=None, split="test")
    y_indexed = y.set_index(y.columns[0])
    rng = random.Random(seed)
    idx = list(range(len(x)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        x_row = x.iloc[i]
        nctid = x_row.iloc[0]
        if nctid not in y_indexed.index:
            continue
        y_row = y_indexed.loc[nctid]
        if hasattr(y_row, "iloc") and y_row.ndim > 1:
            y_row = y_row.iloc[0]
        if pd.isna(y_row.get(label_col)):
            continue
        rows.append((x_row, y_row))
        if len(rows) >= n:
            break
    return rows


def extract_binary_pred(text: str) -> int | None:
    answer = _extract_answer(text)
    if answer is None:
        return None
    norm = answer.strip().lower().rstrip(".")
    if norm in {"yes", "y", "true", "1"}:
        return 1
    if norm in {"no", "n", "false", "0"}:
        return 0
    return None


def extract_multiclass_pred(text: str, valid_labels: list[str]) -> str | None:
    answer = _extract_answer(text)
    if answer is None:
        return None
    norm = answer.strip().lower().rstrip(".")
    valid_norm = {v.lower(): v for v in valid_labels}
    return valid_norm.get(norm)


def binary_f1(preds: list[int | None], trues: list[int]) -> tuple[float, float, float]:
    tp = fp = fn = tn = 0
    for p, t in zip(preds, trues):
        if p is None:
            fn += 1 if t == 1 else 0
            tn += 1 if t == 0 else 0
            continue
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def macro_f1(preds: list[str | None], trues: list[str], labels: list[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(1 for p, t in zip(preds, trues) if p == label and t == label)
        fp = sum(1 for p, t in zip(preds, trues) if p == label and t != label)
        fn = sum(1 for p, t in zip(preds, trues) if p != label and t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


async def run_one(sampling_client, renderer, pubchem_tool, task, x_row, y_row):
    datum = build_datum(task, x_row, y_row)
    if datum is None:
        return None
    initial_messages = _initial_messages(datum, renderer, pubchem_tool)
    env = build_agent_tool_env(
        renderer=renderer,
        tools=[pubchem_tool.lookup_drug],
        initial_messages=initial_messages,
        reward_fn=datum.reward_fn,
        max_turns=MAX_TURNS,
        max_trajectory_tokens=16 * 1024,
        max_generation_tokens=MAX_TOKENS,
    )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=MAX_TOKENS)
    await run_rollout(policy, env)
    final_text = None
    for msg in reversed(env.message_env.history):
        if msg.get("role") == "assistant":
            final_text = get_text_content(msg)
            break
    return final_text


async def main():
    service_client = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)

    print(f"Loading trained checkpoint: {CHECKPOINT_PATH}")
    trained_client = service_client.create_sampling_client(model_path=CHECKPOINT_PATH)
    pubchem_tool = PubChemTool()

    print("\n" + "=" * 60)
    print("BINARY TASKS (F1)")
    print("=" * 60)
    for task, (underlying, label_col) in BINARY_TASKS.items():
        rows = sample_test_rows(underlying, label_col, N_PER_TASK)
        preds, trues = [], []
        for x_row, y_row in rows:
            text = await run_one(trained_client, renderer, pubchem_tool, task, x_row, y_row)
            pred = extract_binary_pred(text) if text else None
            preds.append(pred)
            trues.append(int(y_row[label_col]))
        f1, prec, rec = binary_f1(preds, trues)
        print(f"{task:15s} n={len(rows):3d}  F1={f1:.3f}  precision={prec:.3f}  recall={rec:.3f}  (paper baseline in report)")

    print("\n" + "=" * 60)
    print("MULTI-CLASS TASKS (macro-F1)")
    print("=" * 60)
    for task, (underlying, label_col, labels) in MULTICLASS_TASKS.items():
        is_phased = task != "dose_cls"
        rows = sample_test_rows(underlying, label_col, N_PER_TASK, is_phased=is_phased)
        preds, trues = [], []
        for x_row, y_row in rows:
            text = await run_one(trained_client, renderer, pubchem_tool, task, x_row, y_row)
            pred = extract_multiclass_pred(text, labels) if text else None
            preds.append(pred)
            true_val = y_row[label_col]
            trues.append(str(int(true_val)) if task == "dose_cls" else str(true_val))
        f1 = macro_f1(preds, trues, labels)
        print(f"{task:15s} n={len(rows):3d}  macro-F1={f1:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
