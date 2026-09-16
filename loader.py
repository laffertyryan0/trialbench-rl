"""Raw data loader for TrialBench, reading directly from the fixed Zenodo
record (15455785) rather than the trialbench pip package's loader — that
package still points at the broken v1 record (14975339) which ships
train_x.csv/test_x.csv with no labels at all. See notes/data-issue.md.

This loader returns RAW, pre-encoding dataframes (real column names, real
values) so the verbalizer can build human-readable text prompts. It does
NOT use trialbench's LeaveOneOutEncoder/tabular-flattening pipeline.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data_v2"

# task -> (zip filename, dir prefix inside zip, has_phase_subdirs, phase_dir_name)
TASK_FILES = {
    "mortality": ("mortality-event-prediction.zip", "mortality-event-prediction", True),
    "adverse_event": (
        "serious-adverse-event-forecasting.zip",
        "serious-adverse-event-forecasting",
        True,
    ),
    "dropout": (
        "patient-dropout-event-forecasting.zip",
        "patient-dropout-event-forecasting",
        True,
    ),
    "duration": ("trial-duration-forecasting.zip", "trial-duration-forecasting", True),
    "outcome": ("trial-approval-forecasting.zip", "trial-approval-forecasting", True),
    "failure_reason": (
        "trial-failure-reason-identification.zip",
        "trial-failure-reason-identification",
        True,
    ),
    "dose": ("drug-dose-prediction.zip", "drug-dose-prediction", "All"),  # single "All" dir
    "eligibility": ("eligibility-criteria-design.zip", "eligibility-criteria-design", False),
}


def load_raw(task: str, phase: str | None = None, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw (X, y) dataframes for a task/phase/split, straight from the zip.

    Args:
        task: one of TASK_FILES keys.
        phase: "1"/"2"/"3"/"4" for phase-split tasks; ignored for "dose"/"eligibility".
        split: "train" or "test".
    """
    zip_name, prefix, phase_mode = TASK_FILES[task]
    zpath = DATA_DIR / zip_name
    if not zpath.exists():
        raise FileNotFoundError(f"{zpath} not found — run the download step first.")

    if phase_mode is True:
        assert phase in ("1", "2", "3", "4"), f"task {task} requires phase in 1..4, got {phase}"
        subdir = f"Phase{phase}"
    elif phase_mode == "All":
        subdir = "All"
    else:
        subdir = None

    z = zipfile.ZipFile(zpath)
    x_path = f"{prefix}/{subdir}/{split}_x.csv" if subdir else f"{prefix}/{split}_x.csv"
    y_path = f"{prefix}/{subdir}/{split}_y.csv" if subdir else f"{prefix}/{split}_y.csv"
    # dose uses train_y_cls.csv, not train_y.csv
    names = z.namelist()
    if y_path not in names:
        alt = y_path.replace(f"{split}_y.csv", f"{split}_y_cls.csv")
        if alt in names:
            y_path = alt

    import io

    x = pd.read_csv(io.BytesIO(z.read(x_path)))
    y = pd.read_csv(io.BytesIO(z.read(y_path)))
    return x, y


def load_all_phases(task: str, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate all 4 phases for phase-split tasks. No-op for dose/eligibility."""
    zip_name, prefix, phase_mode = TASK_FILES[task]
    if phase_mode is not True:
        return load_raw(task, phase=None, split=split)
    xs, ys = [], []
    for p in ("1", "2", "3", "4"):
        x, y = load_raw(task, phase=p, split=split)
        xs.append(x)
        ys.append(y)
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)
