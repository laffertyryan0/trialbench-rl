#!/usr/bin/env python3
"""Zero-shot (base model) vs trained-checkpoint comparison, on HELD-OUT TEST
data the training run has never seen (training only ever sampled split="train").

Answers: is the reward increase we've watched real learning, or format
compliance + easy-task exploitation? If base ≈ trained, it's the latter.
"""

from __future__ import annotations

import asyncio
import random

import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.tool_use import build_agent_tool_env

import loader
from pubchem_tool import PubChemTool
from trialbench_env import _initial_messages, build_datum

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
CHECKPOINT_PATH = "tinker://c3d2f38a-638b-5a13-b65e-492a7688717e:train:0/sampler_weights/000010"
N_PER_TASK = 8
MAX_TOKENS = 2048
MAX_TURNS = 4

# The 2 suspect rate tasks, the 2 "honest" (no-shortcut) tasks, and outcome
# (memorization suspect) — not the full 11, to keep this cheap and targeted.
TASKS_TO_CHECK = ["mortality_rate", "adverse_rate", "duration", "failure_reason", "outcome"]

TASK_TO_UNDERLYING = {
    "mortality_rate": "mortality",
    "adverse_rate": "adverse_event",
    "duration": "duration",
    "failure_reason": "failure_reason",
    "outcome": "outcome",
}


def sample_test_data(task: str, n: int, seed: int = 0):
    underlying = TASK_TO_UNDERLYING[task]
    xs, ys = [], []
    for p in ("1", "2", "3", "4"):
        x, y = loader.load_raw(underlying, phase=p, split="test")
        xs.append(x)
        ys.append(y)
    import pandas as pd

    x = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    y_indexed = y.set_index(y.columns[0])

    rng = random.Random(seed)
    idx = list(range(len(x)))
    rng.shuffle(idx)

    data = []
    for i in idx:
        x_row = x.iloc[i]
        nctid = x_row.iloc[0]
        if nctid not in y_indexed.index:
            continue
        y_row = y_indexed.loc[nctid]
        if hasattr(y_row, "iloc") and y_row.ndim > 1:
            y_row = y_row.iloc[0]
        datum = build_datum(task, x_row, y_row)
        if datum is None:
            continue
        data.append(datum)
        if len(data) >= n:
            break
    return data


async def eval_policy(sampling_client, renderer, tokenizer, pubchem_tool, task, datum):
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
    traj = await run_rollout(policy, env)
    return sum(t.reward for t in traj.transitions)


async def main():
    service_client = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)

    print("Creating sampling clients...")
    base_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    trained_client = service_client.create_sampling_client(model_path=CHECKPOINT_PATH)

    results = {}
    for task in TASKS_TO_CHECK:
        print(f"\n=== {task} ===")
        data = sample_test_data(task, N_PER_TASK)
        pubchem_tool = PubChemTool()

        base_rewards, trained_rewards = [], []
        for datum in data:
            base_r = await eval_policy(base_client, renderer, tokenizer, pubchem_tool, task, datum)
            trained_r = await eval_policy(trained_client, renderer, tokenizer, pubchem_tool, task, datum)
            base_rewards.append(base_r)
            trained_rewards.append(trained_r)

        base_avg = sum(base_rewards) / len(base_rewards) if base_rewards else float("nan")
        trained_avg = sum(trained_rewards) / len(trained_rewards) if trained_rewards else float("nan")
        results[task] = (base_avg, trained_avg, len(data))
        print(f"  n={len(data)}  base={base_avg:.3f}  trained(batch10)={trained_avg:.3f}  delta={trained_avg-base_avg:+.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY (base -> trained@batch10, held-out test data)")
    print("=" * 60)
    for task, (base_avg, trained_avg, n) in results.items():
        verdict = "REAL SIGNAL" if trained_avg - base_avg > 0.05 else "NO CLEAR IMPROVEMENT"
        print(f"{task:18s} n={n:2d}  base={base_avg:.3f}  trained={trained_avg:.3f}  delta={trained_avg-base_avg:+.3f}  [{verdict}]")


if __name__ == "__main__":
    asyncio.run(main())
