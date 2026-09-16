"""Fetch trials from the live ClinicalTrials.gov API v2 whose RESULTS were
first posted after Qwen3.6's release (April 2026) — meaning the specific
adverse-event/dropout numbers used as ground truth here were not publicly
postable data at any point the model could plausibly have been trained on.
This is a genuinely contamination-controlled validation set, unlike
TrialBench itself (whose trials all completed by 2021 — see loader.py's
docstring/design notes).

Adapts each study's JSON into the SAME field names verbalizer.py already
expects (a pandas Series with keys like 'brief_title', 'phase',
'study_design_info/allocation', etc.), so verbalize_trial() is reused
unchanged rather than duplicated.

Derivable tasks (from resultsSection, cleanly structured):
    mortality_rate/yn  <- adverseEventsModule.eventGroups[*].deathsNumAffected/AtRisk
    adverse_rate/yn    <- adverseEventsModule.eventGroups[*].seriousNumAffected/AtRisk
    dropout_rate/yn    <- participantFlowModule.periods[0] STARTED/NOT COMPLETED
    duration           <- statusModule start/completion dates

NOT derivable from live data (scoped out, see conversation notes):
    outcome (regulatory approval) — not a CT.gov field, TrialBench's label
        likely came from external FDA-database curation.
    failure_reason — whyStopped free text exists but needs its own
        classification step, not built here.
    dose_cls — not tied to a specific trial's results at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import requests

API_BASE = "https://clinicaltrials.gov/api/v2/studies"

# Safety margin past Qwen3.6's April 2026 release, so training-data
# collection (which precedes release) can't plausibly have captured these.
SAFE_RESULTS_POSTED_AFTER = "2026-06-01"


def fetch_recent_trials(n: int, results_after: str = SAFE_RESULTS_POSTED_AFTER, seed: int = 0) -> list[dict]:
    """Fetch up to n full study records with results posted after the cutoff."""
    params = {
        "query.term": f"AREA[ResultsFirstPostDate]RANGE[{results_after},MAX]",
        "pageSize": min(n * 2, 200),  # overfetch a bit; some will lack usable AE/flow data
        "fields": "NCTId",
    }
    r = requests.get(API_BASE, params=params, timeout=30)
    r.raise_for_status()
    nctids = [s["protocolSection"]["identificationModule"]["nctId"] for s in r.json().get("studies", [])]

    import random

    random.Random(seed).shuffle(nctids)

    studies = []
    for nctid in nctids:
        if len(studies) >= n:
            break
        resp = requests.get(f"{API_BASE}/{nctid}", timeout=30)
        if resp.status_code != 200:
            continue
        studies.append(resp.json())
    return studies


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def adapt_to_verbalizer_row(study: dict) -> pd.Series | None:
    """Map live API JSON -> the field names verbalize_trial() expects."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    desc = proto.get("descriptionModule", {})
    cond = proto.get("conditionsModule", {})
    design = proto.get("designModule", {})
    arms = proto.get("armsInterventionsModule", {})
    elig = proto.get("eligibilityModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})
    resp_party = proto.get("oversightModule", {})

    interventions = arms.get("interventions", [])
    intervention_names = [i.get("name", "") for i in interventions]

    row = {
        "brief_title": ident.get("briefTitle"),
        "condition": ", ".join(cond.get("conditions", [])) or None,
        "brief_summary/textblock": desc.get("briefSummary"),
        "keyword": ", ".join(proto.get("conditionsModule", {}).get("keywords", [])) or None,
        "phase": ", ".join(design.get("phases", [])) or None,
        "study_type": design.get("studyType"),
        "study_design_info/allocation": _get(design, "designInfo", "allocation"),
        "study_design_info/intervention_model": _get(design, "designInfo", "interventionModel"),
        "study_design_info/primary_purpose": _get(design, "designInfo", "primaryPurpose"),
        "study_design_info/masking": _get(design, "designInfo", "maskingInfo", "masking"),
        "sponsors/lead_sponsor/agency_class": _get(sponsor, "leadSponsor", "class"),
        "responsible_party/responsible_party_type": _get(resp_party, "responsibleParty", "type"),
        "eligibility/gender": _get(elig, "sex"),
        "eligibility/healthy_volunteers": _get(elig, "healthyVolunteers"),
        "has_expanded_access": _get(status, "expandedAccessInfo", "hasExpandedAccess"),
        "number_of_arms": len(arms.get("armGroups", [])) or None,
        "eligibility/minimum_age": elig.get("minimumAge"),
        "eligibility/maximum_age": elig.get("maximumAge"),
        "intervention/intervention_name": str(intervention_names) if intervention_names else None,
        "smiless": None,  # not available from CT.gov; not needed for the 4 tasks we validate here
        "condition_browse/mesh_term": None,
        "icdcode": None,
        "eligibility/criteria/textblock": elig.get("eligibilityCriteria"),
    }
    return pd.Series(row), status, study.get("resultsSection", {})


def extract_labels(status: dict, results: dict) -> dict:
    """Pull ground-truth labels for the 4 derivable task families."""
    labels = {}

    # --- duration ---
    start = _get(status, "startDateStruct", "date")
    completion = _get(status, "completionDateStruct", "date")
    if start and completion:
        try:
            d1, d2 = pd.to_datetime(start), pd.to_datetime(completion)
            labels["time_day"] = (d2 - d1).days
        except Exception:
            pass

    # --- mortality / adverse (aggregate across all eventGroups) ---
    ae = results.get("adverseEventsModule")
    if ae and ae.get("eventGroups"):
        deaths_affected = sum(int(g.get("deathsNumAffected", 0) or 0) for g in ae["eventGroups"])
        deaths_at_risk = sum(int(g.get("deathsNumAtRisk", 0) or 0) for g in ae["eventGroups"])
        serious_affected = sum(int(g.get("seriousNumAffected", 0) or 0) for g in ae["eventGroups"])
        serious_at_risk = sum(int(g.get("seriousNumAtRisk", 0) or 0) for g in ae["eventGroups"])
        if deaths_at_risk > 0:
            labels["mortality_rate"] = deaths_affected / deaths_at_risk
            labels["mortality_yn"] = int(deaths_affected > 0)
        if serious_at_risk > 0:
            labels["serious_adverse_rate"] = serious_affected / serious_at_risk
            labels["adverse_yn"] = int(serious_affected > 0)

    # --- dropout (Overall Study period, STARTED vs NOT COMPLETED) ---
    pf = results.get("participantFlowModule")
    if pf and pf.get("periods"):
        period = pf["periods"][0]
        started = not_completed = 0
        for ms in period.get("milestones", []):
            total = sum(int(a.get("numSubjects", 0) or 0) for a in ms.get("achievements", []))
            if ms.get("type") == "STARTED":
                started = total
            elif ms.get("type") == "NOT COMPLETED":
                not_completed = total
        if started > 0:
            labels["droupout_rate"] = not_completed / started  # match loader.py's field-name typo for reuse
            labels["dropout_yn"] = int(not_completed > 0)

    return labels


def build_live_test_set(n: int, results_after: str = SAFE_RESULTS_POSTED_AFTER, seed: int = 0):
    """Returns list of (verbalizer_row, labels_dict) for usable trials."""
    studies = fetch_recent_trials(n, results_after=results_after, seed=seed)
    out = []
    for s in studies:
        row, status, results = adapt_to_verbalizer_row(s)
        labels = extract_labels(status, results)
        if labels:  # skip studies with no usable ground truth at all
            out.append((row, labels))
    return out
