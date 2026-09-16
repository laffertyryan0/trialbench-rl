#!/usr/bin/env python3
"""Evaluate a trained checkpoint on genuinely contamination-controlled data:
trials whose RESULTS were first posted after Qwen3.6's release, fetched live
from ClinicalTrials.gov (see live_data.py for why this is a real guarantee,
not just "recent").

Covers the 4 task families cleanly derivable from live results data:
mortality, adverse events, dropout (each rate + yn), and duration. Does NOT
cover outcome/failure_reason/dose_cls — see live_data.py docstring for why.
"""

from __future__ import annotations

import asyncio
import sys

import tinker

ROLLOUT_TIMEOUT_SECONDS = 120  # a single stuck call must not hang the whole run

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.tool_use import build_agent_tool_env

import verbalizer
from live_data import build_live_test_set
from pubchem_tool import PubChemTool
from rewards import BinaryReward, ContinuousReward
from trialbench_env import TASK_INSTRUCTIONS, _question_for_task
from f1_eval import extract_binary_pred, binary_f1

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
# Update to whichever checkpoint you want to validate — set before running.
CHECKPOINT_PATH = "tinker://895baa05-7593-5dd1-9a53-b8252959dac9:train:0/sampler_weights/final"  # batch 150 (final)
N_TRIALS = 60
MAX_TOKENS = 2048
MAX_TURNS = 4

RATE_TASKS = {
    "mortality_rate": ("mortality_rate", 1.0),
    "adverse_rate": ("serious_adverse_rate", 1.0),
    "dropout_rate": ("droupout_rate", 1.0),
}
BINARY_TASKS = {
    "mortality_yn": "mortality_yn",
    "adverse_yn": "adverse_yn",
    "dropout_yn": "dropout_yn",
}


async def run_one(sampling_client, renderer, pubchem_tool, task, row, reward_fn):
    prompt = verbalizer.verbalize_trial(row)
    question = _question_for_task(task)
    tool_schemas = [pubchem_tool.lookup_drug.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=tool_schemas, system_prompt=TASK_INSTRUCTIONS)
    initial_messages = prefix + [{"role": "user", "content": f"{prompt}\n\nQuestion: {question}"}]

    env = build_agent_tool_env(
        renderer=renderer,
        tools=[pubchem_tool.lookup_drug],
        initial_messages=initial_messages,
        reward_fn=reward_fn,
        max_turns=MAX_TURNS,
        max_trajectory_tokens=16 * 1024,
        max_generation_tokens=MAX_TOKENS,
    )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=MAX_TOKENS)
    try:
        await asyncio.wait_for(run_rollout(policy, env), timeout=ROLLOUT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        print(f"  [TIMEOUT after {ROLLOUT_TIMEOUT_SECONDS}s on {task}, skipping this rollout]", flush=True)
        return None
    for msg in reversed(env.message_env.history):
        if msg.get("role") == "assistant":
            return get_text_content(msg)
    return None


async def main():
    service_client = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)
    pubchem_tool = PubChemTool()

    print(f"Fetching {N_TRIALS} live trials (results posted after Qwen3.6's release)...", flush=True)
    data = build_live_test_set(n=N_TRIALS)
    print(f"{len(data)} usable trials fetched.\n", flush=True)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}", flush=True)
    trained_client = service_client.create_sampling_client(model_path=CHECKPOINT_PATH)
    base_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    print("\n" + "=" * 70)
    print("CONTINUOUS TASKS (graded reward, log-space)")
    print("=" * 70)
    for task, (label_key, scale) in RATE_TASKS.items():
        base_rewards, trained_rewards, n = [], [], 0
        for i, (row, labels) in enumerate(data):
            if label_key not in labels:
                continue
            n += 1
            print(f"[{task}] example {i+1}/{len(data)} (usable #{n})...", flush=True)
            reward_fn = ContinuousReward(true_value=labels[label_key], scale=scale, mode="log")
            for name, client, bucket in [("base", base_client, base_rewards), ("trained", trained_client, trained_rewards)]:
                text = await run_one(client, renderer, pubchem_tool, task, row, reward_fn)
                r, _ = await reward_fn([{"role": "assistant", "content": text or ""}])
                bucket.append(r)
                print(f"    {name}: reward={r:.3f}", flush=True)
        if n:
            print(f"{task:15s} n={n:3d}  base={sum(base_rewards)/n:.3f}  trained={sum(trained_rewards)/n:.3f}  delta={(sum(trained_rewards)-sum(base_rewards))/n:+.3f}")

    print("\n" + "=" * 70)
    print("BINARY TASKS (F1, on trained checkpoint)")
    print("=" * 70)
    for task, label_key in BINARY_TASKS.items():
        preds, trues = [], []
        for i, (row, labels) in enumerate(data):
            if label_key not in labels:
                continue
            true_val = int(labels[label_key])
            print(f"[{task}] example {i+1}/{len(data)}...", flush=True)
            reward_fn = BinaryReward(true_value=true_val)
            text = await run_one(trained_client, renderer, pubchem_tool, task, row, reward_fn)
            pred = extract_binary_pred(text) if text else None
            print(f"    pred={pred}  true={true_val}", flush=True)
            preds.append(pred)
            trues.append(true_val)
        if trues:
            f1, prec, rec = binary_f1(preds, trues)
            print(f"{task:15s} n={len(trues):3d}  F1={f1:.3f}  precision={prec:.3f}  recall={rec:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
