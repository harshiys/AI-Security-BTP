#!/usr/bin/env python3
"""
SentinelMesh Checkpoint 2: Integrated Master Pipeline
Unifies Cyber Telemetry (Graph, Drift, Provenance) + ML-B Multi-Agent Behavioral Model.
Dataset: Real TAMAS (Microsoft AutoGen Direct Prompt Injection Benchmark).
Outputs: Strictly adheres to schema/shared_schema_format.json (5 fields).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cp2_contract as C
import cybersec_adapters as CSA


def log_step(title):
    print(f"\n--- {title} ---")


def main():
    print("=" * 78)
    print("SENTINELMESH CHECKPOINT 2: RUNTIME MULTI-AGENT TRUST & BEHAVIORAL FABRIC")
    print("DATASET: Real TAMAS (Microsoft AutoGen DPI Benchmark)")
    print("=" * 78)

    # 1. Run Cyber Pipeline
    log_step("Phase 1: Running Cyber Telemetry & Dynamic Trust Pipeline")
    events_csv = str(ROOT / "data" / "events.csv")
    cyber_res = ROOT / "cyber_results"
    cyber_res.mkdir(parents=True, exist_ok=True)

    cyber_cmds = [
        [sys.executable, str(ROOT / "cyber_src" / "communication_graph.py"), "--events", events_csv, "--out", str(cyber_res / "communication_graph.csv")],
        [sys.executable, str(ROOT / "cyber_src" / "provenance.py"), "--events", events_csv, "--out", str(cyber_res / "provenance_real.csv")],
        [sys.executable, str(ROOT / "cyber_src" / "attack_scenarios.py"), "--events", events_csv, "--out", str(cyber_res / "cp2_attack_scenarios.json")],
        [sys.executable, str(ROOT / "cyber_src" / "cyber_fusion.py"), "--fingerprint", str(cyber_res / "cyber1_fingerprint_drift.jsonl"), "--graph", str(cyber_res / "communication_graph.csv"), "--provenance", str(cyber_res / "provenance_real.csv"), "--out", str(cyber_res / "cyber_cp2_fusion.csv")],
        [sys.executable, str(ROOT / "cyber_src" / "cyber_trust.py"), "--input", str(cyber_res / "cyber_cp2_fusion.csv"), "--out", str(cyber_res / "cyber_dynamic_trust.csv")]
    ]

    for cmd in cyber_cmds:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if res.returncode != 0:
            print(f"Error in command {' '.join(cmd)}:\n{res.stderr}")
            sys.exit(1)

    print("[PASS] Phase 1: Cyber Communication Graph, Provenance & Dynamic Trust Complete.")

    # 2. Run ML Trust Layer
    log_step("Phase 2: Running ML Multi-Agent Behavioral & Graph Evaluation")
    out_tamas = ROOT / "out_tamas_real"
    if not (out_tamas / "mlb_bundle.joblib").exists():
        print(f"Error: Trained model bundle not found in {out_tamas}")
        sys.exit(1)

    ml_cmd = [sys.executable, str(ROOT / "trust_layer.py"), "--out", str(out_tamas)]
    res = subprocess.run(ml_cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"Error running trust_layer.py:\n{res.stderr}")
        sys.exit(1)

    print("[PASS] Phase 2: ML-A & ML-B Multi-Agent Behavioral Scoring Complete.")

    # 3. Fuse Cyber Evidence with ML-B and Validate Against Shared Schema
    log_step("Phase 3: Fusing Cyber Evidence & Verifying Shared Schema Compliance")
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ingest Cyber evidence
    drift_map = CSA.read_fingerprint(str(cyber_res / "cyber1_fingerprint_drift.jsonl"))
    prov_map = CSA.read_provenance(str(cyber_res / "provenance_real.csv"))
    trust_map = CSA.read_dynamic_trust(str(cyber_res / "cyber_dynamic_trust.csv"))

    # Ingest ML-B verdicts and index
    item_index = pd.read_csv(out_tamas / "cp2_item_index_internal.csv")
    ml_verdicts = [json.loads(l) for l in open(out_tamas / "checkpoint2_verdicts.jsonl", encoding="utf-8") if l.strip()]
    ml_map = {r["item_id"]: r for r in ml_verdicts}

    fused_records = []
    summary_rows = []
    now = C.utc_now()

    for _, row in item_index.iterrows():
        item_id = str(row["item_id"])
        run_id = str(row["run_id"])
        agent = str(row["agent"])

        ml_rec = ml_map.get(item_id, {})
        ml_score = float(ml_rec.get("score", 0.0))
        ml_verdict = ml_rec.get("verdict", "pass")

        cyber_info = trust_map.get((run_id, agent), {})
        cyber_risk = float(cyber_info.get("cyber_risk", 0.0))
        dyn_trust = float(cyber_info.get("dynamic_trust", 1.0))
        drift = float(cyber_info.get("behavioral_drift", drift_map.get((run_id, agent), 0.0)))
        prov = prov_map.get(run_id, {})

        # Composite fused score (weighted average of ML-B interaction risk and Cyber telemetry)
        fused_score = round(max(ml_score, cyber_risk), 4)

        # Verdict logic: block on untrusted provenance -> flag on high risk or drift -> pass
        if prov.get("source") == "untrusted":
            verdict = "block"
            fused_score = 1.0
        elif ml_verdict == "flag" or cyber_risk >= 0.70 or drift >= 0.60:
            verdict = "flag"
        else:
            verdict = "pass"

        # Generate strictly-validated 5-field record
        contract_rec = C.make_record(item_id, verdict, fused_score, now)
        fused_records.append(contract_rec)

        summary_rows.append({
            "item_id": item_id,
            "run_id": run_id,
            "agent": agent,
            "ml_score": ml_score,
            "behavioral_drift": drift,
            "cyber_risk": cyber_risk,
            "dynamic_trust": dyn_trust,
            "verdict": verdict,
            "score": fused_score,
            "timestamp": now
        })

    # Write output files
    verdicts_jsonl = results_dir / "checkpoint2_verdicts.jsonl"
    with open(verdicts_jsonl, "w", encoding="utf-8") as fh:
        for r in fused_records:
            fh.write(json.dumps(r) + "\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "checkpoint2_fused_summary.csv", index=False)

    # Strict Schema Verification
    schema_path = ROOT / "schema" / "shared_schema_format.json"
    schema = json.load(open(schema_path, encoding="utf-8"))
    validator = Draft7Validator(schema)

    valid_count = 0
    for r in fused_records:
        validator.validate(r)
        assert len(r) == 5, f"Record has {len(r)} fields, expected exactly 5"
        assert set(r.keys()) == {"checkpoint_id", "item_id", "verdict", "score", "timestamp"}
        valid_count += 1

    print(f"[PASS] 100% Schema Validation: {valid_count}/{len(fused_records)} records strictly comply with Draft-7 (5 fields).")
    print(f"       Verdict Breakdown: pass={sum(1 for r in fused_records if r['verdict']=='pass')}, flag={sum(1 for r in fused_records if r['verdict']=='flag')}, block={sum(1 for r in fused_records if r['verdict']=='block')}")
    print(f"       Outputs saved to: {verdicts_jsonl.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print("ALL INTEGRATION STAGES COMPLETED SUCCESSFULLY ON REAL TAMAS DATA!")
    print("=" * 78)


if __name__ == "__main__":
    main()
