# TrialBench RL: Fine-tuning Qwen3.6-35B-A3B on Clinical Trial Prediction via Reinforcement Learning

RL (GRPO) fine-tuning of `Qwen/Qwen3.6-35B-A3B` (MoE, 3B active params, reasoning-enabled) on
[TrialBench](https://www.nature.com/articles/s41597-025-05680-8)'s 8 clinical-trial-prediction
tasks, trained via the [Tinker API](https://tinker.thinkingmachines.ai/). The model reasons
step-by-step over a trial's design (phase, sponsor, eligibility, drug identity) and can call a
live PubChem lookup tool for chemical-structure questions.

## Results

### Against TrialBench's own published baselines (Table 6)

F1 on held-out TrialBench test data, dedicated tabular/GNN baseline for comparison:

| Task | Batch 50 | Batch 100 | Batch 150 | Paper baseline |
|---|---|---|---|---|
| `mortality_yn` | 0.692 | 0.696 | 0.720 | 0.745 |
| `adverse_yn` | 0.830 | 0.857 | 0.808 | 0.931 |
| `dropout_yn` | 0.904 | 0.917 | 0.886 | 0.951 |
| `outcome` | 0.686 | 0.727 | 0.647 | 0.742 |
| `failure_reason` (macro-F1) | 0.140 | 0.171 | 0.111 | 0.203 |
| `dose_cls` (macro-F1) | 0.281 | 0.337 | 0.379 | 0.507 |

Not every task improved monotonically with more training — 4 of 6 regressed from batch 100 to
150, likely instability from continuing to train an already-converging multi-task mix past its
sweet spot. **Batch 100 is the stronger checkpoint on this measure**, not the final one.

### Contamination-controlled validation (the number that actually matters)

TrialBench's own trials all completed by 2021 — years before this model's training — so any
result above could be partly explained by the model recalling published outcomes rather than
predicting them. To separate the two, we built a second eval (`live_data.py`, `live_eval.py`)
pulling trials directly from the live ClinicalTrials.gov API whose **results were first posted
after Qwen3.6's April 2026 release** — data that structurally could not have been in its training
set.

Base model (untrained) vs. the batch-150 checkpoint, same live trials:

| Task | Base | Trained | Delta |
|---|---|---|---|
| `mortality_rate` | 0.637 | 0.869 | **+0.232** |
| `adverse_rate` | 0.325 | 0.740 | **+0.414** |
| `dropout_rate` | 0.029 | 0.604 | **+0.575** |

Large, consistent gains on data the model cannot have memorized — this is the strongest evidence
in this repo that RL training produced real inference ability, not recall. (Binary-task F1 on
this same live set is in `live_eval.py`'s output; see `notes/` if you re-run it for current
numbers — this eval doesn't cover `outcome`/`failure_reason`/`dose_cls`, which don't map cleanly
to fields available on ClinicalTrials.gov.)

## Known issues found and fixed along the way

- **TrialBench's official Zenodo record (14975339, used by the `trialbench` pip package) ships
  per-task zips with no label files at all.** The fix (record 15455785) has labels; see
  `loader.py`'s docstring. Don't use the pip package's `load_data()` — it silently returns empty
  labels.
- **Continuous-task reward was originally absolute-error-based with too small a `scale`.**
  Mortality/adverse/dropout rates are heavily zero-skewed (median ≈ 0.0003); under absolute error,
  answering "0" unconditionally scored ~0.75 — a free-riding floor GRPO couldn't learn past. Fixed
  by switching to log-space error (`rewards.py::ContinuousReward(mode="log")`).
- **`dose_cls`'s verbalizer mislabeled the MeSH column** as "Target condition" when it's actually
  the drug's own name in MeSH vocabulary (there's no disease field in that task's data at all).
  The model itself caught this confusion during training before we did.
- **Some `dose_cls` rows have mismatched SMILES/MeSH-name pairs** (verified via a live PubChem
  lookup mid-rollout returning a structurally different compound than the stated drug name) — a
  real data-quality issue in the benchmark, likely contributing to that task's weak scores.

## Known limitations (be upfront about these)

- **Memorization is plausible and not fully ruled out** for tasks framed in the past tense
  ("did this trial succeed") on trials the model may have seen during pretraining. The
  contamination-controlled live eval above addresses this for `mortality`/`adverse`/`dropout`;
  `outcome`/`failure_reason`/`dose_cls` remain unaddressed by that method.
- **`eligibility`'s reward rewards the modal/prior answer** (most trials have similar
  gender/age/healthy-volunteer criteria), so its high reward reflects a predictable prior more
  than learned inference.
- **`failure_reason` is genuinely hard for everyone** — TrialBench's own baseline only reaches
  0.203 F1; low absolute scores there aren't a sign this approach failed specifically.

## Project structure

```
loader.py           # Raw CSV loader against the FIXED Zenodo record (not the broken pip package)
verbalizer.py        # Trial record -> text prompt (frozen field allowlist, generalizes to live data)
pubchem_tool.py       # Live PubChem lookup tool for RL rollouts
rewards.py            # Per-task-shape reward functions (continuous/binary/multiclass/structured)
trialbench_env.py     # EnvGroupBuilder/RLDatasetBuilder per task (multi-turn, tool-using)
train_trialbench.py   # Wires all 11 task configs via InterleavedRLDatasetBuilder + Qwen3.6 + W&B
f1_eval.py            # F1 against TrialBench's own held-out test set (matches paper's metric)
live_data.py          # Fetches contamination-controlled trials from live ClinicalTrials.gov API
live_eval.py          # Base-vs-trained comparison + F1 on that live, un-memorizable data
zero_shot_check.py    # Early base-vs-checkpoint-10 sanity check (superseded by live_eval.py)
```

## Setup

```bash
pip install tinker tinker-cookbook chz datasets pandas rdkit category_encoders matplotlib httpx wandb

export TINKER_API_KEY=<your-key>
export WANDB_API_KEY=<your-key>   # optional, for training curves

python train_trialbench.py                 # full run, all 8 tasks
python train_trialbench.py --smoke-test     # tiny sanity check first
python f1_eval.py                           # eval against TrialBench's own metric
python live_eval.py                         # contamination-controlled eval
```

Data (`data_v2/`) downloads automatically on first run from Zenodo record 15455785 — no manual
step needed. Update `CHECKPOINT_PATH` in `f1_eval.py`/`live_eval.py` to whichever checkpoint you
want to evaluate (printed to stdout/logs during training as `Saved checkpoints: {...}`).

## Model

`Qwen/Qwen3.6-35B-A3B` (MoE, 3B active parameters), reasoning-mode renderer (`qwen3_5`), trained
150 GRPO batches, 11 interleaved task configs, weighted so no single reward-shape dominates.
Checkpoints live on Tinker's infrastructure (`tinker://.../sampler_weights/final`); export to
HuggingFace format via `tinker_cookbook.weights.build_hf_model` if you need local/portable
weights.
