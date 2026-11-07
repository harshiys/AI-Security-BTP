#!/usr/bin/env python3
"""
SentinelMesh Checkpoint 2 - Cyber-ML Integration Adapters.

Reads the Cyber teammate's behavioral fingerprint, communication graph,
message provenance, and dynamic trust outputs, mapping them cleanly to ML-B inputs.
"""
import json
import os
import pandas as pd


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def read_fingerprint(path):
    """Reads cyber1_fingerprint_drift.jsonl -> {(run_id, agent): drift}"""
    if not os.path.exists(path):
        return {}
    return {(r["run_id"], r["agent"]): float(r["drift"]) for r in _lines(path)}


def read_provenance(path):
    """Reads provenance_real.csv or .jsonl -> {run_id: {"signed": bool, "source": str, ...}}"""
    if not os.path.exists(path):
        return {}
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            s = r.get("signed")
            signed_val = bool(s) if pd.notna(s) and s != "" else False
            out[str(r["run_id"])] = {
                "signed": signed_val,
                "source": str(r.get("source", "unavailable")),
                "verification": str(r.get("verification", "not_recorded")),
                "reason": str(r.get("reason", ""))
            }
        return out
    else:
        return {r["run_id"]: {"signed": bool(r.get("signed", False)), "source": str(r.get("source", "unavailable"))} for r in _lines(path)}


def read_dynamic_trust(path):
    """Reads cyber_dynamic_trust.csv -> {(run_id, agent): dict of cyber metrics}"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[(str(r["run_id"]), str(r["agent"]))] = {
            "behavioral_drift": float(r.get("behavioral_drift", 0.0)),
            "graph_anomaly": float(r.get("graph_anomaly", 0.0)),
            "provenance_risk": float(r.get("provenance_risk", 0.0)),
            "cyber_risk": float(r.get("cyber_risk", 0.0)),
            "dynamic_trust": float(r.get("dynamic_trust", 1.0)),
            "trust_verdict": str(r.get("trust_verdict", "pass")),
            "primary_reason": str(r.get("primary_reason", ""))
        }
    return out


def cyber_for_run(run_id, fingerprint=None, provenance=None, dynamic_trust=None):
    """Bundle all Cyber signals known about one run for the ML-B scorer."""
    out = {}
    if fingerprint:
        out["fingerprint"] = {a: d for (rid, a), d in fingerprint.items() if rid == run_id}
    if provenance and run_id in provenance:
        out["provenance"] = provenance[run_id]
    if dynamic_trust:
        out["dynamic_trust"] = {a: v for (rid, a), v in dynamic_trust.items() if rid == run_id}
    return out
