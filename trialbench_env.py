"""EnvGroupBuilder + RLDatasetBuilder for the 8 TrialBench tasks, modeled
directly on tinker_cookbook.recipes.search_tool.search_env's
SearchEnvGroupBuilder / SearchR1DatasetBuilder, but using build_agent_tool_env
with a shared PubChemTool instead of ChromaTool.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import chz
import pandas as pd

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

import loader
import verbalizer
from pubchem_tool import PubChemTool
from rewards import BinaryReward, ContinuousReward, EligibilityReward, MultiClassReward

TASK_INSTRUCTIONS = """You are an expert clinical trial analyst.

You will be given a description of a clinical trial (design, condition, intervention, \
eligibility, and for drug-related questions its SMILES chemical structure). Answer the \
question about this trial.

Instructions:
1. Think step by step about the trial's design and what it implies for the question asked.
2. If the question involves a drug and you want to know more about its chemical properties \
(molecular weight, formula, LogP, etc.) to help form a judgment, call the lookup_drug tool \
with its SMILES string.
3. Give your final answer after the "Answer:" prefix, in exactly the format requested in \
the question. Do not include anything after the answer.
"""

FAILURE_REASON_LABELS = ["poor enrollment", "efficacy", "safety", "Others"]
DOSE_CLS_LABELS = ["0", "1", "2", "3"]


@dataclass
class TrialBenchDatum:
    prompt: str
    question: str
    reward_fn: object
    task: str


def _question_for_task(task: str) -> str:
    return {
        "mortality_rate": "What fraction of enrolled patients died during this trial? Answer with a number between 0 and 1 (e.g. \"Answer: 0.05\").",
        "mortality_yn": "Did any patient death occur during this trial? Answer \"Answer: Yes\" or \"Answer: No\".",
        "adverse_rate": "What fraction of enrolled patients experienced a serious adverse event during this trial? Answer with a number between 0 and 1 (e.g. \"Answer: 0.10\").",
        "adverse_yn": "Did a serious adverse event occur during this trial? Answer \"Answer: Yes\" or \"Answer: No\".",
        "dropout_rate": "What fraction of enrolled patients dropped out of this trial before completion? Answer with a number between 0 and 1 (e.g. \"Answer: 0.15\").",
        "dropout_yn": "Did patient dropout occur during this trial? Answer \"Answer: Yes\" or \"Answer: No\".",
        "duration": "How many days did this trial take from start to completion? Answer with a number of days (e.g. \"Answer: 730\").",
        "outcome": "Was this trial's intervention approved/did the trial succeed? Answer \"Answer: Yes\" or \"Answer: No\".",
        "failure_reason": f"If this trial failed, what was the primary reason? Answer with exactly one of: {', '.join(FAILURE_REASON_LABELS)} (e.g. \"Answer: safety\").",
        "dose_cls": f"Classify the typical dose category for this drug. Answer with exactly one of: {', '.join(DOSE_CLS_LABELS)} (e.g. \"Answer: 2\").",
        "eligibility": "Design this trial's eligibility criteria. Answer in the form: \"Answer: gender=<All|Male|Female>; min_age=<years>; max_age=<years>; healthy_volunteers=<Yes|No>\"",
    }[task]


def build_datum(task: str, x_row: pd.Series, y_row: pd.Series) -> TrialBenchDatum | None:
    """Build one (prompt, reward_fn) pair for a task, or None if the row's
    label is missing/unusable (skipped rather than silently zero-rewarded).
    """
    question = _question_for_task(task)

    if task in ("mortality_rate", "mortality_yn"):
        prompt = verbalizer.verbalize_trial(x_row)
        if task == "mortality_rate":
            val = y_row.get("mortality_rate")
            if pd.isna(val):
                return None
            reward = ContinuousReward(true_value=float(val), scale=1.0, mode="log")
        else:
            val = y_row.get("Y/N")
            if pd.isna(val):
                return None
            reward = BinaryReward(true_value=int(val))

    elif task in ("adverse_rate", "adverse_yn"):
        prompt = verbalizer.verbalize_trial(x_row)
        if task == "adverse_rate":
            val = y_row.get("serious_adverse_rate")
            if pd.isna(val):
                return None
            reward = ContinuousReward(true_value=float(val), scale=1.0, mode="log")
        else:
            val = y_row.get("Y/N")
            if pd.isna(val):
                return None
            reward = BinaryReward(true_value=int(val))

    elif task in ("dropout_rate", "dropout_yn"):
        prompt = verbalizer.verbalize_trial(x_row)
        if task == "dropout_rate":
            val = y_row.get("droupout_rate")  # sic — typo in source data
            if pd.isna(val):
                return None
            reward = ContinuousReward(true_value=float(val), scale=1.0, mode="log")
        else:
            val = y_row.get("Y/N")
            if pd.isna(val):
                return None
            reward = BinaryReward(true_value=int(val))

    elif task == "duration":
        prompt = verbalizer.verbalize_trial(x_row)
        val = y_row.get("time_day")
        if pd.isna(val):
            return None
        reward = ContinuousReward(true_value=float(val), scale=365.0)

    elif task == "outcome":
        prompt = verbalizer.verbalize_trial(x_row)
        val = y_row.get("outcome")
        if pd.isna(val):
            return None
        reward = BinaryReward(true_value=int(val))

    elif task == "failure_reason":
        prompt = verbalizer.verbalize_trial(x_row)
        val = y_row.get("failure_reason")
        if pd.isna(val) or val not in FAILURE_REASON_LABELS:
            return None
        reward = MultiClassReward(true_label=val, valid_labels=FAILURE_REASON_LABELS)

    elif task == "dose_cls":
        prompt = verbalizer.verbalize_drug_only(x_row)
        val = y_row.get("Avg")
        if pd.isna(val):
            return None
        reward = MultiClassReward(true_label=str(int(val)), valid_labels=DOSE_CLS_LABELS)

    elif task == "eligibility":
        prompt = verbalizer.verbalize_eligibility_context(x_row)
        gender = y_row.get("eligibility/gender")
        hv = y_row.get("eligibility/healthy_volunteers")
        if pd.isna(gender) or pd.isna(hv):
            return None

        def _age(v):
            if pd.isna(v):
                return None
            import re

            m = re.search(r"\d+\.?\d*", str(v))
            return float(m.group()) if m else None

        reward = EligibilityReward(
            true_gender=str(gender),
            true_min_age=_age(y_row.get("eligibility/minimum_age")),
            true_max_age=_age(y_row.get("eligibility/maximum_age")),
            true_healthy_volunteers=str(hv),
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return TrialBenchDatum(prompt=prompt, question=question, reward_fn=reward, task=task)


def _initial_messages(datum: TrialBenchDatum, renderer: Renderer, pubchem_tool: PubChemTool) -> list[Message]:
    tool_schemas = [pubchem_tool.lookup_drug.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=TASK_INSTRUCTIONS,
    )
    user_content = f"{datum.prompt}\n\nQuestion: {datum.question}"
    return prefix + [{"role": "user", "content": user_content}]


class TrialBenchEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        datum: TrialBenchDatum,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        pubchem_tool: PubChemTool,
        max_trajectory_tokens: int = 16 * 1024,
        max_generation_tokens: int | None = None,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.pubchem_tool = pubchem_tool
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        initial_messages = _initial_messages(self.datum, renderer, self.pubchem_tool)

        return [
            build_agent_tool_env(
                renderer=renderer,
                tools=[self.pubchem_tool.lookup_drug],
                initial_messages=initial_messages,
                reward_fn=self.datum.reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.datum.task]


class TrialBenchRLDataset(RLDataset):
    def __init__(self, env_group_builders: list[TrialBenchEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.env_group_builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return len(self.env_group_builders) // self.batch_size


@chz.chz
class TrialBenchRLDatasetBuilder(RLDatasetBuilder):
    task: str  # one of the keys handled in build_datum()
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 4
    max_examples: int | None = None
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        pubchem_tool = PubChemTool()

        if self.task == "dose_cls":
            x, y = loader.load_raw("dose", phase=None, split="train")
        elif self.task == "eligibility":
            x, y = loader.load_raw("eligibility", phase=None, split="train")
        else:
            underlying = {
                "mortality_rate": "mortality",
                "mortality_yn": "mortality",
                "adverse_rate": "adverse_event",
                "adverse_yn": "adverse_event",
                "dropout_rate": "dropout",
                "dropout_yn": "dropout",
                "duration": "duration",
                "outcome": "outcome",
                "failure_reason": "failure_reason",
            }[self.task]
            x, y = loader.load_all_phases(underlying, split="train")

        y_indexed = y.set_index(y.columns[0])
        x_indexed = x.set_index(x.columns[0]) if x.columns[0] not in ("smiless",) else x

        rng = random.Random(self.seed)
        row_ids = list(range(len(x)))
        rng.shuffle(row_ids)
        if self.max_examples:
            row_ids = row_ids[: self.max_examples]

        env_builders = []
        for i in row_ids:
            x_row = x.iloc[i]
            nctid = x_row.iloc[0]
            try:
                y_row = y_indexed.loc[nctid] if nctid in y_indexed.index else None
            except Exception:
                y_row = None
            if y_row is None:
                continue
            if isinstance(y_row, pd.DataFrame):
                y_row = y_row.iloc[0]
            datum = build_datum(self.task, x_row, y_row)
            if datum is None:
                continue
            env_builders.append(
                TrialBenchEnvGroupBuilder(
                    datum=datum,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    max_turns=self.max_turns,
                    group_size=self.group_size,
                    pubchem_tool=pubchem_tool,
                )
            )

        dataset = TrialBenchRLDataset(env_group_builders=env_builders, batch_size=self.batch_size)
        return dataset, None
