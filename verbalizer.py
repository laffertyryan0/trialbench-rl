"""Trial record -> text prompt. This is the ONE artifact that needs to be
shared between training and any future deployment/live-prediction use —
see the pipeline-parity discussion in the design notes. It operates on
raw ClinicalTrials.gov-style fields (the same fields load_raw() returns),
so it generalizes beyond TrialBench's frozen snapshot.

Field list is a frozen allowlist (not "verbalize everything") to avoid
re-introducing outcome-leaking columns. Missing fields are represented
explicitly ("not reported") rather than silently omitted, so the model
learns to handle the partial-information records it'll see on live,
in-progress trials at deployment time.
"""

from __future__ import annotations

import pandas as pd

# Fields describing the trial's design (known at registration time, not
# retrospective outcome fields). Order matters for readability, not
# correctness.
TRIAL_TEXT_FIELDS = [
    ("brief_title", "Title"),
    ("condition", "Condition(s)"),
    ("brief_summary/textblock", "Summary"),
    ("keyword", "Keywords"),
]
TRIAL_CATEGORICAL_FIELDS = [
    ("phase", "Phase"),
    ("study_type", "Study type"),
    ("study_design_info/allocation", "Allocation"),
    ("study_design_info/intervention_model", "Intervention model"),
    ("study_design_info/primary_purpose", "Primary purpose"),
    ("study_design_info/masking", "Masking"),
    ("sponsors/lead_sponsor/agency_class", "Lead sponsor type"),
    ("responsible_party/responsible_party_type", "Responsible party type"),
    ("eligibility/gender", "Eligible gender"),
    ("eligibility/healthy_volunteers", "Accepts healthy volunteers"),
    ("has_expanded_access", "Has expanded access"),
]
TRIAL_NUMERIC_FIELDS = [
    ("number_of_arms", "Number of arms"),
    ("eligibility/minimum_age", "Minimum age"),
    ("eligibility/maximum_age", "Maximum age"),
]
TRIAL_STRUCTURED_FIELDS = [
    ("intervention/intervention_name", "Intervention(s)/drug(s)"),
    ("smiless", "Drug SMILES structure(s)"),
    ("condition_browse/mesh_term", "Condition MeSH terms"),
    ("icdcode", "ICD-10 codes"),
]
TRIAL_CRITERIA_FIELD = ("eligibility/criteria/textblock", "Eligibility criteria")

ALL_TRIAL_FIELDS = (
    TRIAL_TEXT_FIELDS
    + TRIAL_CATEGORICAL_FIELDS
    + TRIAL_NUMERIC_FIELDS
    + TRIAL_STRUCTURED_FIELDS
    + [TRIAL_CRITERIA_FIELD]
)


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "not reported"
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return "not reported"
    return s


def verbalize_trial(row: pd.Series, max_criteria_chars: int = 1500) -> str:
    """Build a text description of a trial for the 6 trial-level tasks
    (mortality, adverse event, dropout, duration, outcome, failure_reason).
    Only fields actually present in this dataset's export are included —
    field *absence* (structural, this task's CSV never had the column) is
    silent; field *missingness* (present column, NaN value) is explicit.
    """
    lines = []
    for col, label in TRIAL_TEXT_FIELDS + TRIAL_CATEGORICAL_FIELDS + TRIAL_NUMERIC_FIELDS + TRIAL_STRUCTURED_FIELDS:
        if col in row.index:
            lines.append(f"{label}: {_fmt(row[col])}")
    if TRIAL_CRITERIA_FIELD[0] in row.index:
        crit = _fmt(row[TRIAL_CRITERIA_FIELD[0]])
        if len(crit) > max_criteria_chars:
            crit = crit[:max_criteria_chars] + " ... [truncated]"
        lines.append(f"{TRIAL_CRITERIA_FIELD[1]}: {crit}")
    return "\n".join(lines)


def verbalize_drug_only(row: pd.Series) -> str:
    """For dose/dose_cls: only SMILES + the drug's own MeSH term are
    available in the baseline data (column is literally
    `intervention_browse/mesh_term` — the drug's name in MeSH vocabulary,
    NOT a target condition/disease; there is no disease field in this
    export at all) — no trial-design context either. This is exactly
    where the PubChem tool adds value: the model can look up the drug's
    identity/properties by SMILES before answering.
    """
    lines = []
    smiles = row.get("smiless")
    mesh = row.get("intervention_browse/mesh_term")
    lines.append(f"Drug SMILES structure(s): {_fmt(smiles)}")
    lines.append(f"Drug name (MeSH term): {_fmt(mesh)}")
    return "\n".join(lines)


def verbalize_eligibility_context(row: pd.Series) -> str:
    """For eligibility-criteria-design: predict eligibility fields from
    the rest of the trial's design (title, condition, phase, intervention,
    etc.) — everything EXCEPT the eligibility fields themselves.
    """
    lines = []
    for col, label in TRIAL_TEXT_FIELDS + TRIAL_CATEGORICAL_FIELDS + TRIAL_NUMERIC_FIELDS + TRIAL_STRUCTURED_FIELDS:
        if col in row.index and not col.startswith("eligibility/"):
            lines.append(f"{label}: {_fmt(row[col])}")
    return "\n".join(lines)
