#!/usr/bin/env python3
"""
ML-B scorer: ONE function turns a run prefix into a schema-valid record + a sidecar.

    from scorer import load_bundle, score_run_prefix
    bundle = load_bundle("out_sim/mlb_bundle.joblib")
    record, sidecar = score_run_prefix(events_prefix, run_id, bundle, cyber=None)

events_prefix: DataFrame with the ML-A events columns (acting_agent, target_agent, event_type, event_idx, n_calls,
is_error, msg_len, tool_name), agent names already normalised like ML-A. Only events with event_idx < K are read.
Verdict: block (hard rule or untrusted provenance) > flag (risk above validated threshold, or high fingerprint drift) > pass.
"""
import json, os
import joblib
import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
import cp2_contract as C
import mla_pipeline as P
import graph_features as G

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR_SCHEMA = json.load(open(os.path.join(HERE, "schema", "sidecar.schema.json"), encoding="utf-8"))
FINGERPRINT_FLAG = 0.7        # ASSUMED unit of cybersec's drift score; agree it with them


def load_bundle(path):
    return joblib.load(path)


def score_run_prefix(events_prefix, run_id, bundle, roster=None, task_id=None, cyber=None, now=None):
    K = bundle["K"]
    ev = events_prefix.copy()
    ev["run_id"] = run_id
    ev["task_id"] = task_id if task_id is not None else "unknown_task"
    ev = ev[ev["event_idx"] < K]
    if roster is None:
        names = set(ev["acting_agent"]) | set(t for t in ev["target_agent"].dropna() if t)
        roster = sorted(names - {"user", ""})
    ros = pd.DataFrame({"run_id": run_id, "agent": list(roster)})

    tab = P.build_features(ev, ros, K)
    gr = G.graph_features(ev, ros, K, edge_index=bundle["edge_index"], allowed_tasks=bundle["train_tasks"])
    d = tab.merge(gr, on=["run_id", "agent"])
    X = d[bundle["cols"]].to_numpy(float)
    risk = bundle["model"].predict_proba(X)[:, 1]
    d["risk"] = risk

    hard = []
    for c, lim in bundle["rule_limits"].items():
        for _, r in d.iterrows():
            if r[c] > lim:
                hard.append({"rule": f"{c} > {lim:g}", "agent_index": int(_), "value": float(r[c])})
    cyber = cyber or {}
    prov = cyber.get("provenance")
    if prov and (prov.get("source") == "untrusted" or (prov.get("signed") is False and prov.get("source") != "unavailable")):
        hard.append({"rule": "provenance untrusted or unsigned", "value": prov})
    drift = cyber.get("fingerprint", {})
    drift_hit = {a: v for a, v in drift.items() if v > FINGERPRINT_FLAG}

    run_risk = float(risk.max()) if len(risk) else 0.0
    if hard:
        verdict = "block"
    elif run_risk > bundle["threshold"] or drift_hit:
        verdict = "flag"
    else:
        verdict = "pass"
    score = 1.0 if verdict == "block" else float(min(1.0, max(run_risk, max(drift_hit.values(), default=0.0))))

    record = C.make_record(P.sha(str(run_id)), verdict, score, now)      # validated against the shared schema

    top = []
    if len(d):
        worst = d.iloc[int(np.argmax(risk))]
        z = {c: (float(worst[c]) - bundle["benign_mean"][c]) / bundle["benign_std"][c] for c in bundle["cols"]}
        for c in sorted(z, key=lambda k: -abs(z[k]))[:5]:
            top.append({"feature": c, "value": round(float(worst[c]), 4), "z_vs_benign": round(z[c], 2)})
    sidecar = {"item_id": record["item_id"], "decision_point_events": int(K), "feature_set": bundle["feature_set"],
               "threshold": round(float(bundle["threshold"]), 4),
               "component_scores": {"model_risk_run_max": round(run_risk, 4),
                                    "model_risk_by_agent": {P.sha(run_id + a): round(float(r), 4)
                                                            for a, r in zip(d["agent"], risk)},
                                    "hard_rules_fired": hard, "fingerprint_drift_flagged": drift_hit,
                                    "provenance": prov},
               "top_features": top, "n_events_seen": int(len(ev))}
    Draft202012Validator(SIDECAR_SCHEMA).validate(json.loads(json.dumps(sidecar, default=str)))
    return record, sidecar
