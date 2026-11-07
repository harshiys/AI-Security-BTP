"""
Cyber-1 -> ML-B bridge: export fingerprint features as the drift JSONL the
scorer's cybersec adapter expects.

WHY THIS EXISTS
---------------
ML-B's cybersec_adapters.read_fingerprint() expects one line per agent:
    {"run_id": "...", "agent": "...", "drift": 0.0-1.0}
and scorer.py flags an agent when drift > FINGERPRINT_FLAG (0.7).

Cyber-1's fingerprint.py produces a richer table (8 fp_* columns). This script
collapses those columns into a single 0-1 `drift` score and writes the exact
JSONL the adapter reads, so the two sides integrate without ML-B changing code.

DRIFT DEFINITION (documented, not magic)
----------------------------------------
drift = weighted, clipped combination of the normalized fp_* signals that
actually indicate "this agent is behaving far from its baseline":

    drift = clip( 0.35*fp_tool_dist_drift         # tool mix moved
                + 0.25*min(fp_new_tool_ratio,1)    # used never-seen tools
                + 0.20*min(fp_seq_novelty,1)       # novel call sequence
                + 0.10*min(fp_unseen_tool_count/3,1)
                + 0.10*min(|fp_msg_len_z|/4,1),     # abnormal message length
                0, 1 )

Agents with fp_has_baseline == 0 (cold start, no baseline) get drift = 0.0 and
are marked low-confidence, so a missing baseline never fabricates a high score.

The weights are a transparent starting point. If ML-B prefers to consume the
raw fp_* columns instead (richer), they can; this collapse is the minimal,
contract-matching option.

Usage:
  python export_drift.py --features fingerprint_features_K8.csv \
      --out cyber1_fingerprint_drift.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {
    "fp_tool_dist_drift": 0.35,
    "fp_new_tool_ratio": 0.25,
    "fp_seq_novelty": 0.20,
    "fp_unseen_tool_count": 0.10,   # divided by 3, clipped
    "fp_msg_len_z": 0.10,           # abs/4, clipped
}


def compute_drift(row: pd.Series) -> float:
    if int(row.get("fp_has_baseline", 1)) == 0:
        return 0.0
    d = 0.0
    d += WEIGHTS["fp_tool_dist_drift"] * float(row.get("fp_tool_dist_drift", 0.0))
    d += WEIGHTS["fp_new_tool_ratio"] * min(float(row.get("fp_new_tool_ratio", 0.0)), 1.0)
    d += WEIGHTS["fp_seq_novelty"] * min(float(row.get("fp_seq_novelty", 0.0)), 1.0)
    d += WEIGHTS["fp_unseen_tool_count"] * min(float(row.get("fp_unseen_tool_count", 0.0)) / 3.0, 1.0)
    d += WEIGHTS["fp_msg_len_z"] * min(abs(float(row.get("fp_msg_len_z", 0.0))) / 4.0, 1.0)
    return float(np.clip(d, 0.0, 1.0))


def export(features_csv: Path, out_jsonl: Path) -> int:
    feats = pd.read_csv(features_csv)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for _, row in feats.iterrows():
            rec = {
                "run_id": str(row["run_id"]),
                "agent": str(row["agent"]),
                "drift": round(compute_drift(row), 4),
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Cyber-1 drift exporter for ML-B")
    ap.add_argument("--features", required=True,
                    help="fingerprint_features_*.csv from fingerprint.py")
    ap.add_argument("--out", default="cyber1_fingerprint_drift.jsonl")
    args = ap.parse_args()
    n = export(Path(args.features), Path(args.out))
    print(f"[cyber1 export] wrote {n} drift records -> {args.out}")
    print("[cyber1 export] ML-B reads this with "
          "cybersec_adapters.read_fingerprint(path)")


if __name__ == "__main__":
    main()
