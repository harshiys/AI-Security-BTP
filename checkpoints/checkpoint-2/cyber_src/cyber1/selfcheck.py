"""
Cyber-1 — leakage self-check for the fingerprint features.

Mirrors ML-A's selfcheck philosophy (section 4 of ML-A's README): every
feature builder must obey the leak-free contract. This asserts two things for
fingerprint_features:

  1. Decision point K: events with event_idx >= K must not change any fp_*
     value. We append/scramble future events and require the output is
     identical.

  2. No future leak within baseline: baselines built on harmless runs only must
     not change when attack-run rows are added to the events table (features
     for a run must depend only on that run + the fixed baseline).

Run:  python3 selfcheck.py --events events.csv --labels labels.csv
Exit code 0 = all checks pass.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fingerprint import DEFAULT_K, build_baselines, fingerprint_features

FP_COLS = ["fp_tool_dist_drift", "fp_seq_novelty", "fp_new_tool_ratio",
           "fp_msg_len_z", "fp_tool_rate_dev", "fp_err_rate_dev",
           "fp_unseen_tool_count"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return (df.sort_values(["run_id", "agent"])
              .reset_index(drop=True)[["run_id", "agent"] + FP_COLS])


def check_decision_point(events: pd.DataFrame, baseline, K: int) -> bool:
    """Appending events after K must not move any fp_* value."""
    base = _sorted(fingerprint_features(events, baseline, K=K))

    # Build a tampered copy: duplicate every row with event_idx >= K, mutate it
    # heavily (new tool names, huge msg_len, errors), and append.
    future = events[events["event_idx"] >= K].copy()
    if future.empty:
        # If no events beyond K exist, synthesize some well past K.
        future = events.copy()
    future["event_idx"] = future["event_idx"] + 10_000
    future["tool_name"] = "ZZZ_injected_tool"
    future["tool_names"] = "ZZZ_injected_tool"
    future["msg_len"] = 99999
    future["is_error"] = True
    tampered = pd.concat([events, future], ignore_index=True)

    after = _sorted(fingerprint_features(tampered, baseline, K=K))
    ok = base.equals(after)
    print(f"  [check] decision-point K={K}: "
          f"{'PASS' if ok else 'FAIL'} "
          f"(future events do not change fp_*)")
    return ok


def check_scramble_future_tools(events: pd.DataFrame, baseline, K: int) -> bool:
    """Scrambling tool names AFTER K must not move any fp_* value."""
    base = _sorted(fingerprint_features(events, baseline, K=K))
    ev2 = events.copy()
    mask = ev2["event_idx"] >= K
    ev2.loc[mask, "tool_name"] = "SCRAMBLED"
    ev2.loc[mask, "tool_names"] = "SCRAMBLED"
    after = _sorted(fingerprint_features(ev2, baseline, K=K))
    ok = base.equals(after)
    print(f"  [check] scramble tools after K: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_baseline_isolation(events: pd.DataFrame, labels: pd.DataFrame,
                             K: int) -> bool:
    """Baseline must be identical whether or not attack runs are present.

    (Baselines are built from harmless runs only, so adding/removing attack
    runs from the events table must not change the learned normal profile.)
    """
    bl_full = build_baselines(events, labels, K=K)

    harmless_runs = set(
        labels.loc[labels["run_has_attack"] == 0, "run_id"].unique())
    ev_harmless = events[events["run_id"].isin(harmless_runs)]
    lb_harmless = labels[labels["run_id"].isin(harmless_runs)]
    bl_harmless_only = build_baselines(ev_harmless, lb_harmless, K=K)

    ok = set(bl_full.keys()) == set(bl_harmless_only.keys())
    if ok:
        for key in bl_full:
            a, b = bl_full[key], bl_harmless_only[key]
            if (abs(a.msg_len_mean - b.msg_len_mean) > 1e-9
                    or a.tool_dist != b.tool_dist):
                ok = False
                break
    print(f"  [check] baseline isolation (harmless-only): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    labels = pd.read_csv(args.labels)
    baseline = build_baselines(events, labels, K=args.k)

    print(f"[cyber1 selfcheck] K={args.k}, {len(baseline)} baselines")
    results = [
        check_decision_point(events, baseline, args.k),
        check_scramble_future_tools(events, baseline, args.k),
        check_baseline_isolation(events, labels, args.k),
    ]
    all_ok = all(results)
    print(f"[cyber1 selfcheck] {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
