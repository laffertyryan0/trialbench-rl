#!/usr/bin/env python3
"""RL training on TrialBench tasks with Qwen3.6-35B-A3B (MoE, 3B active),
reasoning enabled by default, PubChem tool-use for drug-related questions.

Usage (smoke test, tiny/cheap):
    export TINKER_API_KEY=...
    export WANDB_API_KEY=...
    python train_trialbench.py --smoke-test

Usage (real run, all 8 tasks interleaved):
    python train_trialbench.py
"""

from __future__ import annotations

import argparse
import asyncio
import os

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder

from trialbench_env import TrialBenchRLDatasetBuilder

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

ALL_TASKS = [
    "mortality_rate",
    "mortality_yn",
    "adverse_rate",
    "adverse_yn",
    "dropout_rate",
    "dropout_yn",
    "duration",
    "outcome",
    "failure_reason",
    "dose_cls",
    "eligibility",
]

# Equal weight by default; downweight the 3 "yn" binary variants slightly
# since they're the same underlying event as their "_rate" sibling task,
# so weighting both at 1.0 would double-count that event's signal.
DEFAULT_WEIGHTS = {t: (0.6 if t.endswith("_yn") else 1.0) for t in ALL_TASKS}


def build_config(*, smoke_test: bool, tasks: list[str], log_path: str) -> train.Config:
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    print(f"Model: {MODEL_NAME}")
    print(f"Renderer (reasoning-enabled by default): {renderer_name}")
    print(f"Tasks: {tasks}")

    if smoke_test:
        batch_size, group_size, max_examples, max_turns = 2, 2, 4, 2
        max_tokens = 1536
    else:
        # max_examples is capped, not unlimited: this sandbox has only ~3.8GB
        # RAM, and eagerly building one EnvGroupBuilder per row across all 11
        # tasks (eligibility alone has 109K rows) OOM-killed the process
        # before any Tinker call happened. 500/task is generous headroom
        # against the ~2200 total group draws this run actually needs
        # (50 batches x 44 groups/batch, interleaved across 11 tasks).
        batch_size, group_size, max_examples, max_turns = 4, 8, 500, 4
        max_tokens = 2048

    sources = [
        TrialBenchRLDatasetBuilder(
            task=t,
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name=renderer_name,
            batch_size=batch_size,
            group_size=group_size,
            max_turns=max_turns,
            max_examples=max_examples,
        )
        for t in tasks
    ]
    weights = [DEFAULT_WEIGHTS[t] for t in tasks]

    dataset_builder = InterleavedRLDatasetBuilder(
        sources=sources,
        weights=weights,
        groups_per_batch=batch_size * len(tasks) if not smoke_test else batch_size,
        # 100, not 50: extending the completed 50-batch run. Schedule slots are
        # resolved by (seed, batch_index) independent of total_batches, so
        # batches 50-99 are genuinely new data, not a repeat of 0-49 — and
        # resume (behavior_if_exists="resume" below) picks up at batch 50
        # automatically via the last checkpoint.
        total_batches=2 if smoke_test else 150,
    )

    return train.Config(
        model_name=MODEL_NAME,
        renderer_name=renderer_name,
        recipe_name="trialbench_rl_smoke" if smoke_test else "trialbench_rl",
        dataset_builder=dataset_builder,
        learning_rate=4e-5,
        max_tokens=max_tokens,
        log_path=log_path,
        # 10, not 20: two harness restarts killed earlier runs before batch 20
        # ever saved. Paired with behavior_if_exists="resume" below so a killed
        # run picks up from its last checkpoint instead of restarting at 0.
        save_every=0 if smoke_test else 10,
        eval_every=0,
        wandb_project=None if smoke_test else "trialbench-qwen3.6-rl",
        wandb_name=None if smoke_test else "trialbench-all-tasks",
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tasks", default=None, help="comma-separated subset of tasks")
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS
    log_path = "./logs/smoke" if args.smoke_test else "./logs/trialbench_rl"

    config = build_config(smoke_test=args.smoke_test, tasks=tasks, log_path=log_path)
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists="delete" if args.smoke_test else "resume"
    )
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
