"""
Cyber-1 — Incident forensics timeline.

For any flagged (run_id, agent), reconstruct a human-readable timeline of what
the agent did, event by event, with the fingerprint deviations annotated. This
is the artifact a human reviewer reads when the trust score flags an agent.

Reads the frozen events table + the fp_* features and prints/saves a timeline:
  - every event the agent took (within the decision window)
  - which fp_* features were abnormal
  - the first point an unseen/rogue tool appeared (the "turn" moment)

Usage:
  python3 forensics.py --events events.csv --features fingerprint_features_K8.csv \
      --run <run_id> --agent <agent> [--k 8] [--out timeline.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fingerprint import DEFAULT_K, agent_base

# fp thresholds above which a feature is called "abnormal" in the narrative.
FP_FLAGS = {
    "fp_tool_dist_drift": 0.5,
    "fp_seq_novelty": 0.5,
    "fp_new_tool_ratio": 0.2,
    "fp_msg_len_z": 2.0,
    "fp_tool_rate_dev": 0.3,
    "fp_err_rate_dev": 0.3,
    "fp_unseen_tool_count": 1,
}


def build_timeline(events: pd.DataFrame, features: pd.DataFrame,
                   run_id: str, agent: str, K: int = DEFAULT_K) -> str:
    sub = events[(events["run_id"] == run_id)
                 & (events["acting_agent"] == agent)
                 & (events["event_idx"] < K)].sort_values("event_idx")
    frow = features[(features["run_id"] == run_id)
                    & (features["agent"] == agent)]

    lines = [f"# Incident Forensics — agent `{agent}`",
             f"**run:** `{run_id}`  |  **role:** `{agent_base(agent)}`  "
             f"|  **decision window:** first {K} events", ""]

    if frow.empty:
        lines.append("_No fingerprint row for this (run, agent)._")
        return "\n".join(lines)
    fr = frow.iloc[0]

    # --- abnormal fingerprint summary ---
    abnormal = []
    for feat, thr in FP_FLAGS.items():
        val = fr.get(feat, 0)
        if feat == "fp_msg_len_z":
            if abs(val) >= thr:
                abnormal.append((feat, val))
        elif val >= thr:
            abnormal.append((feat, val))

    lines.append("## Fingerprint deviations")
    if fr.get("fp_has_baseline", 1) == 0:
        lines.append("- ⚠️ No baseline for this role (cold start) — "
                     "deviations unreliable.")
    if abnormal:
        for feat, val in abnormal:
            lines.append(f"- **{feat} = {val}** (abnormal)")
    else:
        lines.append("- No fp_* feature crossed its abnormal threshold.")
    lines.append("")

    # --- event-by-event timeline ---
    lines.append("## Timeline")
    turn_idx = None
    for _, e in sub.iterrows():
        tag = ""
        if e["event_type"] == "ToolCallRequestEvent" and isinstance(
                e["tool_name"], str) and e["tool_name"]:
            tag = f"  →  tool: `{e['tool_name']}`"
            if turn_idx is None and fr.get("fp_unseen_tool_count", 0) > 0:
                # mark the first tool call as candidate 'turn' moment
                turn_idx = int(e["event_idx"])
                tag += "   ⟵ first tool call (possible turn)"
        err = "  [ERROR]" if e["is_error"] else ""
        lines.append(
            f"- `#{int(e['event_idx']):>2}` {e['event_type']}"
            f" ({e['acting_agent']} → {e['target_agent']}){tag}{err}")
    lines.append("")

    # --- verdict hint ---
    lines.append("## Reviewer note")
    if abnormal:
        lines.append(
            "This agent drifted from its normal behavioral baseline on the "
            "features above before the decision point. Recommend **flag + "
            "isolate** pending review; cross-check with ML-B's graph signal "
            "and Cyber-2's provenance flags before any hard block.")
    else:
        lines.append(
            "No strong behavioral deviation detected by fingerprinting. If this "
            "agent was flagged, the signal likely came from another component "
            "(graph, provenance, or a hard rule), not Cyber-1.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cyber-1 forensics timeline")
    ap.add_argument("--events", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    features = pd.read_csv(args.features)
    md = build_timeline(events, features, args.run, args.agent, args.k)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"[forensics] wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
