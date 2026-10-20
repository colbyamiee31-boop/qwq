"""Audit the archived C8 run without invoking GreenLight or a model solver."""

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "reaudit-inputs"
OUT = ROOT / "evidence" / "R5" / "C8_REAUDIT"
ORIGINAL_RUN_ID = "39364637096"
REPO = os.environ.get("GITHUB_REPOSITORY", "colbyamiee31-boop/qwq")
RUN_ID = os.environ.get("GITHUB_RUN_ID", "unknown")
RUN_SHA = os.environ.get("GITHUB_SHA", "unknown")
RUN_URL = (
    os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    + "/"
    + REPO
    + "/actions/runs/"
    + RUN_ID
)
TOLERANCE = 1e-6

ARTIFACTS = {
    "r4-m1": {
        "id": 11289405542,
        "run_id": "37164458429",
        "digest": "8a9867cfd9c1935668a4123990595b8085bed03daeb5d9a4e49e07f1e06fd47c",
    },
    "r4-final": {
        "id": 11289580354,
        "run_id": "37164458429",
        "digest": "9d9d16a0efd6ea8afb983a553b8860c1673ae64257692148924ae3949c219f7d",
    },
    "r4-prepared-inputs": {
        "id": 11289046331,
        "run_id": "37164458429",
        "digest": "ecff6bc7586f07aac393c26698488cd98164bee418f0a71fdae9f60fe4fafe0f",
    },
    "r3-m1": {
        "id": 11288765689,
        "run_id": "37163085468",
        "digest": "de6ae266728fed2ed535a5f591c5c1778b74ee75a750ddfb4f65ae32be1d0979",
    },
    "r5-c8-precheck-logs": {
        "id": 12239285093,
        "run_id": ORIGINAL_RUN_ID,
        "digest": "c6a69c47d5ca1d5847174a1201b3a879555886464d3d57af368ff99e875ed051",
    },
    "r5-c8-prepared-inputs": {
        "id": 12238467416,
        "run_id": ORIGINAL_RUN_ID,
        "digest": "bed01b46894246827a564194afea93a3b298e3999f29afeb845c93993727f8f2",
    },
    "r5-c8-model-output": {
        "id": 12238792702,
        "run_id": ORIGINAL_RUN_ID,
        "digest": "d829156d37bcec577fdc40efe7467eefa4b435f9c14013e8ef9fac44cadd62ff",
    },
    "r5-c8-reproduction": {
        "id": 12239022202,
        "run_id": ORIGINAL_RUN_ID,
        "digest": "be5d18b446bbf0afe8ae99422e4c8a187a2e8c996c346aad0d556284ffb8a19f",
    },
}

HANDOFF_HASHES = {
    "physbench/r5/field_flux.py": "b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd",
    "physbench/r5/field_ventilation_function.json": "4c7ae6aaab48097e522bb305b8353bae23ac59ffb0775508c077d6d1b130fc3b",
    "physbench/r5/R5_event_list_97_v2.csv": "5721220e0ebb13750ae7b89950f7998e80836d3c52d0c8ee9309380ef82167b5",
}

checks = []
artifact_metadata = []
failures = []


def add(name, passed, detail):
    record = {"check": name, "passed": bool(passed), "detail": str(detail)}
    checks.append(record)
    if not passed:
        failures.append(name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def github_json(url):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is unavailable")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def artifact_identity_gate():
    for name, expected in ARTIFACTS.items():
        detail = {
            "artifact": name,
            "artifact_id_expected": expected["id"],
            "run_id_expected": expected["run_id"],
            "digest_expected_sha256": expected["digest"],
        }
        passed = False
        try:
            payload = github_json(
                f"https://api.github.com/repos/{REPO}/actions/artifacts/{expected['id']}"
            )
            raw_digest = str(payload.get("digest") or "")
            actual_digest = (
                raw_digest[len("sha256:") :]
                if raw_digest.startswith("sha256:")
                else raw_digest
            )
            run = payload.get("workflow_run") or {}
            actual_run = str(run.get("id", ""))
            actual_name = str(payload.get("name", ""))
            expired = payload.get("expired")
            passed = (
                actual_digest == expected["digest"]
                and actual_run == expected["run_id"]
                and expired is False
                and actual_name == name
            )
            detail.update(
                {
                    "artifact_name_actual": actual_name,
                    "artifact_id_actual": payload.get("id"),
                    "run_id_actual": actual_run,
                    "digest_actual_api": raw_digest,
                    "expired_actual": expired,
                    "archive_download_url_present": bool(payload.get("archive_download_url")),
                }
            )
        except Exception as exc:  # retain all remaining audit checks on API failure
            detail["api_error"] = repr(exc)
        artifact_metadata.append(detail)
        add("artifact_identity_" + name.replace("-", "_"), passed, json.dumps(detail, sort_keys=True))


def find_unique(base, filename, check_name):
    base = Path(base)
    matches = sorted(p for p in base.rglob(filename) if p.is_file()) if base.exists() else []
    if len(matches) != 1:
        add(check_name, False, f"expected exactly one {filename} under {base}; found {[str(p) for p in matches]}")
        return None
    return matches[0]


def exact_file(base, relative_path, check_name):
    path = Path(base) / relative_path
    if not path.is_file():
        add(check_name, False, f"missing: {path}")
        return None
    return path


def read_json(path, check_name):
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        add(check_name, False, repr(exc))
        return {}


def read_csv(path, check_name):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        add(check_name, False, repr(exc))
        return None


def bool_series(series):
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif pd.isna(value):
            values.append(None)
        else:
            values.append(mapping.get(str(value).strip().lower()))
    return values


def sign_model(delta, high, low):
    scale = max(1.0, abs(float(high)), abs(float(low)))
    epsilon = 1e-8 * scale
    return 1 if delta > epsilon else (-1 if delta < -epsilon else 0)


def sign_obs(delta):
    return 1 if delta > 1e-12 else (-1 if delta < -1e-12 else 0)


def kappa(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0:
        return float("nan")
    observed = float(np.mean(a == b))
    pa = float(np.mean(a == 1))
    pb = float(np.mean(b == 1))
    expected = pa * pb + (1.0 - pa) * (1.0 - pb)
    return float("nan") if abs(1.0 - expected) < 1e-15 else (observed - expected) / (1.0 - expected)


def r1_metric(frame, variable):
    model_signs = np.asarray(
        [
            sign_model(delta, high, low)
            for delta, high, low in zip(
                frame[f"model_{variable}_delta"],
                frame[f"post_{variable}"],
                frame[f"pre_{variable}"],
            )
        ],
        dtype=int,
    )
    obs_col = f"obs_{variable}_delta"
    obs_signs = np.asarray([sign_obs(value) for value in frame[obs_col]], dtype=int)
    keep = (model_signs != 0) & (obs_signs != 0)
    model_signs = model_signs[keep]
    obs_signs = obs_signs[keep]
    resolved = frame.loc[keep].copy()
    concordance = float(np.mean(model_signs == obs_signs)) if len(model_signs) else float("nan")
    kappa_value = kappa(model_signs, obs_signs)
    gradient_col = "T_gradient_C" if variable == "T" else "AH_gradient_g_m3"
    denominator = np.abs(resolved[gradient_col].to_numpy(float)) * np.abs(resolved.delta_u.to_numpy(float))
    good = denominator > 1e-12
    if int(good.sum()) >= 3:
        rho = float(
            spearmanr(
                resolved.loc[good, f"model_{variable}_delta"].to_numpy(float) / denominator[good],
                resolved.loc[good, obs_col].to_numpy(float) / denominator[good],
            ).statistic
        )
    else:
        rho = float("nan")
    return {
        "n_resolved": int(len(model_signs)),
        "concordance": concordance,
        "kappa": kappa_value,
        "spearman_norm": rho,
        "unresolved": int(len(frame) - len(model_signs)),
    }


def finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def get_log(base, filename, check_name):
    base = Path(base)
    corrected = base / filename
    legacy = base / "evidence" / "R5" / "C8" / filename
    matches = [path for path in (corrected, legacy) if path.is_file()]
    if len(matches) != 1:
        add(check_name, False, f"log must exist at exactly one corrected or legacy path; found {[str(p) for p in matches]}")
        return None
    return matches[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=["event_id", "frozen_R4_CF0_T_delta", "frozen_R4_CF0_AH_delta", "C8_model_T_delta", "C8_model_AH_delta", "_merge", "abs_T_difference", "abs_AH_difference"]
    ).to_csv(OUT / "C8_event_comparison.csv", index=False)
    pd.DataFrame(columns=["path", "sha256", "bytes"]).to_csv(
        OUT / "C8_reaudit_file_hashes.csv", index=False
    )
    artifact_identity_gate()

    # The original report is copied byte-for-byte before any content is inspected.
    original_source = INPUTS / "r5-c8-reproduction"
    original_archive = OUT / "original_run_39364637096"
    if original_source.is_dir() and not original_archive.exists():
        shutil.copytree(original_source, original_archive)
        add("original_reproduction_artifact_archived_unmodified", True, str(original_archive))
    else:
        add(
            "original_reproduction_artifact_archived_unmodified",
            False,
            f"source missing or destination already exists: {original_source} -> {original_archive}",
        )

    # Compute hashes for every extracted artifact member, then include the archived copy.
    hash_rows = []
    if INPUTS.exists():
        for path in sorted(p for p in INPUTS.rglob("*") if p.is_file()):
            hash_rows.append(
                {
                    "path": "downloaded/" + path.relative_to(INPUTS).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    if original_archive.exists():
        for path in sorted(p for p in original_archive.rglob("*") if p.is_file()):
            hash_rows.append(
                {
                    "path": "evidence/R5/C8_REAUDIT/" + path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    for relative, expected_hash in HANDOFF_HASHES.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else ""
        if path.is_file():
            hash_rows.append(
                {
                    "path": "repository/" + relative,
                    "sha256": actual,
                    "bytes": path.stat().st_size,
                }
            )
        add("handoff_hash_" + Path(relative).stem, path.is_file() and actual == expected_hash, f"path={path}; expected={expected_hash}; actual={actual or 'missing'}")

    pd.DataFrame(hash_rows, columns=["path", "sha256", "bytes"]).to_csv(
        OUT / "C8_reaudit_file_hashes.csv", index=False
    )

    # Locate and validate the original precheck logs using the corrected artifact path.
    precheck = INPUTS / "r5-c8-precheck-logs"
    field_log = get_log(precheck, "field_flux_unit_tests.log", "field_flux_log_path")
    input_log = get_log(precheck, "protocol_input_checks.log", "protocol_input_log_path")
    field_text = field_log.read_text(errors="replace") if field_log else ""
    input_text = input_log.read_text(errors="replace") if input_log else ""
    add("field_flux_reference_tests", field_log is not None and "field_flux unit tests PASS" in field_text, f"path={field_log}; required marker present={'field_flux unit tests PASS' in field_text}")
    add("frozen_input_join_and_40_event_tests", input_log is not None and "protocol_input_gate" in input_text and "PASS" in input_text, f"path={input_log}; protocol_input_gate={'protocol_input_gate' in input_text}; PASS={'PASS' in input_text}")

    prepared = INPUTS / "r5-c8-prepared-inputs"
    r4_prepared = INPUTS / "r4-prepared-inputs"
    model_output = INPUTS / "r5-c8-model-output"
    r4_m1 = INPUTS / "r4-m1"
    r4_final = INPUTS / "r4-final"
    r3_m1 = INPUTS / "r3-m1"

    prep_path = find_unique(prepared, "PRIMARY_97_R2_INPUT.csv", "C8_prepared_input_present")
    r4_prep_path = find_unique(r4_prepared, "PRIMARY_97_R2_INPUT.csv", "frozen_R4_prepared_input_present")
    prep = read_csv(prep_path, "C8_prepared_input_readable")
    r4_prep = read_csv(r4_prep_path, "frozen_R4_prepared_input_readable")
    if prep_path and r4_prep_path:
        add("C8_prepared_input_matches_frozen_R4_input", prep_path.read_bytes() == r4_prep_path.read_bytes(), f"C8={sha256(prep_path)}; R4={sha256(r4_prep_path)}")
    else:
        add("C8_prepared_input_matches_frozen_R4_input", False, "one or both input files are missing")

    responses_path = find_unique(model_output, "c8_m1_responses.csv", "C8_response_file_present")
    audit_path = find_unique(model_output, "c8_m1_runtime_audit.csv", "C8_runtime_audit_present")
    model_summary_path = find_unique(model_output, "c8_m1_summary.json", "C8_model_summary_present")
    model_pip_path = find_unique(model_output, "evidence_R5_M1_PIP_FREEZE.txt", "C8_dependency_freeze_present")
    r4_responses_path = find_unique(r4_m1, "ctifl_common_flux_responses.csv", "frozen_R4_CF0_response_file_present")
    r4_audit_path = find_unique(r4_m1, "runtime_audit.csv", "frozen_R4_runtime_audit_present")
    r4_summary_path = find_unique(r4_m1, "summary.json", "frozen_R4_summary_present")
    r4_pip_path = find_unique(r4_m1, "evidence_R4_M1_PIP_FREEZE.txt", "frozen_R4_dependency_freeze_present")
    r3_responses_path = find_unique(r3_m1, "ctifl_event_responses.csv", "frozen_R3_HV_reference_present")
    r4_final_summary_path = exact_file(r4_final, Path("evidence") / "R4" / "M1" / "summary.json", "frozen_R4_final_M1_summary_present")
    r4_validation_path = find_unique(r4_final, "01_common_flux_empirical_validation.csv", "frozen_R4_validation_metrics_present")

    model_summary = read_json(model_summary_path, "C8_model_summary_readable")
    r4_summary = read_json(r4_summary_path, "frozen_R4_summary_readable")
    current = read_csv(responses_path, "C8_response_file_readable")
    audit = read_csv(audit_path, "C8_runtime_audit_readable")
    frozen = read_csv(r4_responses_path, "frozen_R4_response_file_readable")
    r4_audit = read_csv(r4_audit_path, "frozen_R4_runtime_audit_readable")

    # Read the original run's actual job results from the GitHub API.
    prepare_ok = False
    c8_ok = False
    original_job_detail = {"run_id": ORIGINAL_RUN_ID}
    try:
        jobs_payload = github_json(
            f"https://api.github.com/repos/{REPO}/actions/runs/{ORIGINAL_RUN_ID}/jobs?per_page=100"
        )
        jobs = jobs_payload.get("jobs", [])
        named = {}
        for job in jobs:
            named.setdefault(job.get("name"), []).append(job)
        prepare_ok = len(named.get("prepare", [])) == 1 and named["prepare"][0].get("conclusion") == "success"
        c8_ok = len(named.get("c8", [])) == 1 and named["c8"][0].get("conclusion") == "success"
        original_job_detail.update(
            {
                "prepare": [
                    {"id": item.get("id"), "status": item.get("status"), "conclusion": item.get("conclusion")}
                    for item in named.get("prepare", [])
                ],
                "c8": [
                    {"id": item.get("id"), "status": item.get("status"), "conclusion": item.get("conclusion")}
                    for item in named.get("c8", [])
                ],
            }
        )
    except Exception as exc:
        original_job_detail["api_error"] = repr(exc)
    add("original_prepare_job_succeeded", prepare_ok, json.dumps(original_job_detail, sort_keys=True))
    add("original_C8_model_job_succeeded", c8_ok, json.dumps(original_job_detail, sort_keys=True))

    add("C8_runtime_gate", model_summary.get("runtime_gate_pass") is True, json.dumps({key: model_summary.get(key) for key in ["status", "event_count", "all_finite", "all_prescribed_fluxes_finite_nonnegative", "max_init_error", "max_flux_identity_error", "max_CF0_DV_identity_error", "max_paired_initial_state_error", "max_paired_forcing_error"]}, sort_keys=True))
    add("frozen_R4_runtime_gate", r4_summary.get("gate_pass") is True, json.dumps({key: r4_summary.get(key) for key in ["gate_pass", "all_finite", "max_init_error", "max_flux_identity_error", "CF0_max_common_DV_identity_error"]}, sort_keys=True))

    if current is not None:
        response_schema = {"event_id", "model_T_delta", "model_AH_delta"}.issubset(current.columns)
        add("C8_response_schema", response_schema, ",".join(current.columns))
        ids = current.event_id.astype(str) if "event_id" in current.columns else pd.Series([], dtype=str)
        add("C8_40_events_complete", len(current) == 40 and ids.nunique() == 40, f"rows={len(current)} unique={ids.nunique()}")
    if prep is not None and {"anchor_primary_3_6", "event_id"}.issubset(prep.columns):
        flags = bool_series(prep["anchor_primary_3_6"])
        valid_flags = all(value is not None for value in flags)
        selected = prep.loc[pd.Series(flags, index=prep.index).fillna(False), "event_id"].astype(str)
        expected_ids = set(selected)
        add("frozen_anchor_primary_3_6_is_40", valid_flags and len(expected_ids) == 40 and selected.nunique() == 40, f"valid_flags={valid_flags}; rows={len(selected)}; unique anchor IDs={len(expected_ids)}")
        if current is not None and "event_id" in current.columns:
            actual_ids = set(current.event_id.astype(str))
            add("C8_event_set_matches_frozen_anchor_primary_3_6", actual_ids == expected_ids, f"missing={sorted(expected_ids - actual_ids)}; extra={sorted(actual_ids - expected_ids)}")
        else:
            add("C8_event_set_matches_frozen_anchor_primary_3_6", False, "C8 response event IDs unavailable")
    else:
        add("frozen_anchor_primary_3_6_is_40", False, "required frozen input columns unavailable")
        add("C8_event_set_matches_frozen_anchor_primary_3_6", False, "required frozen input columns unavailable")

    if audit is not None:
        required_audit = {"init_err", "flux_identity_error", "q_common", "q_top", "q_main", "finite", "paired_initial_state_error", "paired_forcing_error"}
        if not required_audit.issubset(audit.columns):
            add("C8_R4_runtime_audits", False, f"missing columns={sorted(required_audit - set(audit.columns))}")
        else:
            q = audit[["q_common", "q_top", "q_main"]].to_numpy(float) if len(audit) else np.array([[np.nan]])
            finite_flags = bool_series(audit.finite)
            all_finite = bool(len(audit) and all(value is True for value in finite_flags) and np.isfinite(q).all())
            q_nonnegative = bool((q >= 0).all())
            max_init = float(audit.init_err.max()) if len(audit) else float("inf")
            max_flux = float(audit.flux_identity_error.max()) if len(audit) else float("inf")
            pair_x = float(audit.paired_initial_state_error.max()) if len(audit) else float("inf")
            pair_d = float(audit.paired_forcing_error.max()) if len(audit) else float("inf")
            ok = len(audit) == 80 and all_finite and q_nonnegative and max_init <= 1e-8 and max_flux <= 1e-12 and pair_x == 0.0 and pair_d == 0.0
            add("C8_R4_runtime_audits", ok, json.dumps({"audit_rows": len(audit), "all_finite": all_finite, "q_nonnegative": q_nonnegative, "max_init_error": max_init, "max_flux_identity_error": max_flux, "max_paired_initial_state_error": pair_x, "max_paired_forcing_error": pair_d}, sort_keys=True))

    if r4_audit is not None and {"stage", "cohort", "finite", "init_err", "flux_identity_error"}.issubset(r4_audit.columns):
        base = r4_audit[(r4_audit.stage == "CF0") & (r4_audit.cohort == "primary")]
        flags = bool_series(base.finite)
        ok = len(base) == 80 and all(value is True for value in flags) and float(base.init_err.max()) <= 1e-8 and float(base.flux_identity_error.max()) <= 1e-12
        add("frozen_R4_CF0_primary_runtime_audit", ok, f"rows={len(base)}; all_finite={all(value is True for value in flags)}; max_init={float(base.init_err.max()) if len(base) else 'unavailable'}; max_flux={float(base.flux_identity_error.max()) if len(base) else 'unavailable'}")
    else:
        add("frozen_R4_CF0_primary_runtime_audit", False, "frozen R4 audit unavailable or missing required columns")

    # Byte identity ties the R4 model archive to the separately frozen final archive.
    same_summary = False
    if r4_summary_path and r4_final_summary_path:
        same_summary = r4_summary_path.read_bytes() == r4_final_summary_path.read_bytes()
    add("r4_m1_summary_byte_identical_to_r4_final", same_summary, f"r4-m1={r4_summary_path}; r4-final={r4_final_summary_path}; byte_identical={same_summary}")

    # Recompute the published R1.2 metrics from the exact response table used by C8.
    metric_checks = []
    metrics_ok = True
    if frozen is not None and r4_validation_path:
        subset = frozen[(frozen.stage == "CF0") & (frozen.cohort == "primary") & (frozen.horizon_min == 15) & (frozen.model == "M1")].copy()
        final_metrics = read_csv(r4_validation_path, "frozen_R4_validation_metrics_readable")
        if final_metrics is None:
            metrics_ok = False
        else:
            add("frozen_R4_CF0_primary_15min_response_rows", len(subset) == 40 and subset.event_id.astype(str).nunique() == 40, f"rows={len(subset)}; unique={subset.event_id.astype(str).nunique()}")
            for variable in ("T", "AH"):
                try:
                    recomputed = r1_metric(subset, variable)
                    rows = final_metrics[(final_metrics.cohort == "primary") & (final_metrics.stage == "CF0") & (final_metrics.model == "M1") & (final_metrics.variable == variable)]
                    if len(rows) != 1:
                        raise ValueError(f"expected one frozen metric row for {variable}; got {len(rows)}")
                    reference = rows.iloc[0]
                    differences = {}
                    variable_ok = int(reference.n_resolved) == recomputed["n_resolved"]
                    for metric_name in ("concordance", "kappa", "spearman_norm"):
                        actual = float(recomputed[metric_name])
                        expected = float(reference[metric_name])
                        difference = abs(actual - expected)
                        differences[metric_name] = {"recomputed": actual, "frozen": expected, "abs_difference": difference}
                        variable_ok = variable_ok and math.isfinite(actual) and math.isfinite(expected) and difference <= 1e-12
                    if variable == "T":
                        variable_ok = variable_ok and recomputed["n_resolved"] == 38 and abs(recomputed["concordance"] - 0.447368421053) <= 1e-12
                    record = {"variable": variable, "n_resolved_recomputed": recomputed["n_resolved"], "n_resolved_frozen": int(reference.n_resolved), "unresolved_recomputed": recomputed["unresolved"], **differences}
                    metric_checks.append(record)
                    add("R4_CF0_R1_2_metric_recalculation_" + variable, variable_ok, json.dumps(record, sort_keys=True))
                    metrics_ok = metrics_ok and variable_ok
                except Exception as exc:
                    metrics_ok = False
                    add("R4_CF0_R1_2_metric_recalculation_" + variable, False, repr(exc))
    else:
        metrics_ok = False
        add("R4_CF0_R1_2_metric_recalculation_T", False, "required frozen response or final metrics are unavailable")
        add("R4_CF0_R1_2_metric_recalculation_AH", False, "required frozen response or final metrics are unavailable")
    add("R4_CF0_metrics_recomputed_against_frozen_final", metrics_ok, json.dumps(metric_checks, sort_keys=True))

    # Keep the original event-level tolerance and the exact comparison rules.
    max_dt = None
    max_dah = None
    completed = 0
    if current is not None and frozen is not None:
        required_model = {"event_id", "model_T_delta", "model_AH_delta"}
        required_frozen = required_model | {"stage", "cohort", "horizon_min", "model"}
        if not required_model.issubset(current.columns) or not required_frozen.issubset(frozen.columns):
            add("C8_response_schema", False, f"current_missing={sorted(required_model - set(current.columns))}; frozen_missing={sorted(required_frozen - set(frozen.columns))}")
            add("C8_per_event_comparison", False, "required response columns are missing")
        else:
            ref = frozen[(frozen.stage == "CF0") & (frozen.cohort == "primary") & (frozen.horizon_min == 15) & (frozen.model == "M1")].copy()
            add("frozen_R4_CF0_primary_15min_40_events", len(ref) == 40 and ref.event_id.astype(str).nunique() == 40, f"rows={len(ref)} unique={ref.event_id.astype(str).nunique()}")
            c = current[["event_id", "model_T_delta", "model_AH_delta"]].copy()
            c["event_id"] = c.event_id.astype(str)
            f = ref[["event_id", "model_T_delta", "model_AH_delta"]].copy()
            f["event_id"] = f.event_id.astype(str)
            c = c.rename(columns={"model_T_delta": "C8_model_T_delta", "model_AH_delta": "C8_model_AH_delta"})
            f = f.rename(columns={"model_T_delta": "frozen_R4_CF0_T_delta", "model_AH_delta": "frozen_R4_CF0_AH_delta"})
            try:
                joined = f.merge(c, on="event_id", how="outer", indicator=True, validate="one_to_one")
                joined["abs_T_difference"] = np.abs(joined.C8_model_T_delta - joined.frozen_R4_CF0_T_delta)
                joined["abs_AH_difference"] = np.abs(joined.C8_model_AH_delta - joined.frozen_R4_CF0_AH_delta)
                completed = int((joined["_merge"] == "both").sum())
                max_dt = float(joined.abs_T_difference.max()) if completed else float("inf")
                max_dah = float(joined.abs_AH_difference.max()) if completed else float("inf")
                joined.to_csv(OUT / "C8_event_comparison.csv", index=False, float_format="%.12g")
                add("C8_event_id_alignment", len(joined) == 40 and completed == 40 and bool((joined["_merge"] == "both").all()), f"matched={completed}; comparison_rows={len(joined)}")
                add("C8_T_difference_within_1e-6", max_dt <= TOLERANCE, f"max_abs_difference_C={max_dt}; threshold={TOLERANCE}")
                add("C8_AH_difference_within_1e-6", max_dah <= TOLERANCE, f"max_abs_difference_g_m3={max_dah}; threshold={TOLERANCE}")
            except Exception as exc:
                add("C8_per_event_comparison", False, repr(exc))
    else:
        add("C8_per_event_comparison", False, "model or frozen response table unavailable")

    # Preserve the exact dependency comparison from the original audit.
    dep_match = False
    if r4_pip_path and model_pip_path:
        dep_match = r4_pip_path.read_text(encoding="utf-8").strip() == model_pip_path.read_text(encoding="utf-8").strip()
    add("frozen_R4_dependency_freeze_match", dep_match, f"R4={r4_pip_path}; C8={model_pip_path}; exact sorted pip-freeze text comparison")

    # Confirm the original failure report remains in the archived package unchanged.
    archived_reports = sorted(original_archive.rglob("C8_reproduction_report.md")) if original_archive.exists() else []
    original_fail_found = False
    original_report_path = None
    for path in archived_reports:
        text = path.read_text(errors="replace")
        if "FAIL" in text:
            original_fail_found = True
            original_report_path = path
            break
    add("original_C8_FAIL_report_retained", original_fail_found, str(original_report_path or archived_reports))

    passed = not failures
    status = "PASS" if passed else "FAIL"
    trigger_audit = os.environ.get("TRIGGER_AUDIT_OUTPUT", "Trigger audit output unavailable.")
    summary = {
        "status": status,
        "passed": passed,
        "audit_scope": "audit layer only; no GreenLight model execution",
        "original_run_id": ORIGINAL_RUN_ID,
        "original_run_status": "FAIL",
        "original_run_model_layer_status": "accepted; prepare and c8 jobs succeeded",
        "reaudit_run_id": RUN_ID,
        "reaudit_run_url": RUN_URL,
        "repository": REPO,
        "r5_branch": "r5-field-ventilation-flux",
        "reaudit_workflow_sha": RUN_SHA,
        "greenlight_commit_reference_only": "2d3febb1ea002b24b452e32293e990beb78d3ce1",
        "events_completed": completed,
        "events_expected": 40,
        "max_abs_T_difference_C": max_dt,
        "max_abs_AH_difference_g_m3": max_dah,
        "tolerance": TOLERANCE,
        "r4_metric_recalculation": metric_checks,
        "artifact_identity": artifact_metadata,
        "trigger_audit_output": trigger_audit,
        "all_checks_passed": passed,
        "failed_checks": failures,
        "checks": checks,
    }
    (OUT / "C8_reaudit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    report = [
        "# C8 audit-layer re-audit report",
        "",
        f"- Final status: **{status}**",
        f"- Original C8 run: {ORIGINAL_RUN_ID} (original aggregate status: FAIL)",
        f"- Re-audit Actions run: {RUN_URL}",
        f"- Branch: r5-field-ventilation-flux",
        f"- Re-audit workflow SHA: {RUN_SHA}",
        "- Scope: archived C8 audit only; no GreenLight model was executed.",
        "- Events compared: " + (f"{completed} / 40"),
        "- Maximum absolute temperature difference: " + (str(max_dt) if max_dt is not None else "unavailable") + " °C",
        "- Maximum absolute absolute-humidity difference: " + (str(max_dah) if max_dah is not None else "unavailable") + " g m-3",
        f"- Fixed tolerance: {TOLERANCE} for both variables",
        "- All checks passed: " + str(passed),
        "",
        "## Trigger audit output",
        "",
        "```text",
        trigger_audit.rstrip(),
        "```",
        "",
        "## Artifact identity gates",
        "",
        "Each artifact was checked through the GitHub API for its exact ID, producer run, SHA256 digest, name, and non-expired state.",
        "",
        "| Artifact | ID | Producer run | Expected SHA256 | Actual SHA256 | Expired | Result |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for item in artifact_metadata:
        report.append(
            "| {artifact} | {artifact_id_expected} | {run_id_expected} | {digest_expected_sha256} | {actual} | {expired} | {result} |".format(
                artifact=item["artifact"],
                artifact_id_expected=item["artifact_id_expected"],
                run_id_expected=item["run_id_expected"],
                digest_expected_sha256=item["digest_expected_sha256"],
                actual=item.get("digest_actual_api", "unavailable").replace("|", "\\|"),
                expired=item.get("expired_actual", "unavailable"),
                result="PASS" if any(c["check"] == "artifact_identity_" + item["artifact"].replace("-", "_") and c["passed"] for c in checks) else "FAIL",
            )
        )
    report += ["", "## R4 archive chain and metric recomputation", ""]
    report += ["- R4 M1 summary byte-identical to R4 final M1 summary: " + str(same_summary)]
    report += ["- R1.2 metrics recomputed from the archived R4 CF0 response file:"]
    report += ["", "| Variable | n resolved | Concordance | κ | Normalised Spearman ρ |", "|---|---:|---:|---:|---:|"]
    for item in metric_checks:
        report.append(
            f"| {item['variable']} | {item['n_resolved_recomputed']} | {item['concordance']['recomputed']:.12g} | {item['kappa']['recomputed']:.12g} | {item['spearman_norm']['recomputed']:.12g} |"
        )
    report += ["", "## Checks", "", "| Check | Result | Detail |", "|---|---:|---|"]
    for check in checks:
        report.append("| " + check["check"] + " | " + ("PASS" if check["passed"] else "FAIL") + " | " + check["detail"].replace("|", "\\|").replace("\n", " ") + " |")
    report += [
        "",
        "## Output files",
        "",
        "- `C8_reaudit_summary.json`: every check, artifact identity detail, and final status.",
        "- `C8_reaudit_report.md`: human-readable audit report and trigger-audit record.",
        "- `C8_event_comparison.csv`: regenerated event-by-event comparison.",
        "- `C8_reaudit_file_hashes.csv`: SHA256 and byte count for extracted artifact members and protocol handoff files.",
        "- `original_run_39364637096/`: unmodified copy of the original reproduction artifact.",
        "",
        "Original C8 artifacts and frozen R3/R4 artifacts were read only. No model, solver, or formal R5 condition was run.",
    ]
    (OUT / "C8_reaudit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit("C8 re-audit failed; see the saved summary and report.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        # The workflow uploads the complete evidence directory even on failure.
        OUT.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "FAIL",
            "passed": False,
            "audit_scope": "audit layer only; no GreenLight model execution",
            "original_run_id": ORIGINAL_RUN_ID,
            "original_run_status": "FAIL",
            "reaudit_run_id": RUN_ID,
            "reaudit_run_url": RUN_URL,
            "fatal_error": repr(exc),
            "checks": checks,
            "failed_checks": failures + ["fatal_error"],
        }
        (OUT / "C8_reaudit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (OUT / "C8_reaudit_report.md").write_text(
            "# C8 audit-layer re-audit report\n\n- Final status: **FAIL**\n- Fatal error: `" + repr(exc) + "`\n- No model was run.\n",
            encoding="utf-8",
        )
        print("Fatal audit error:", repr(exc), file=sys.stderr)
        raise SystemExit(1)
