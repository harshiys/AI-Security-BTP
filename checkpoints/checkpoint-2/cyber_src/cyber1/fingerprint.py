"""
Cyber-1 — Behavioral Fingerprinting for Checkpoint 2 (Runtime Agent Trust).

WHAT THIS IS
------------
ML-A's pipeline produces an events table and raw per-agent COUNT features
(a_n_toolreq, a_n_distinct_tools, a_mean_len, ...). Cyber-1 owns the part ML-A
deliberately does NOT build: **baseline-relative deviation**.

  ML-A says:   "this agent made 5 tool calls, 2 distinct tools, mean msg 140 chars"
  Cyber-1 says: "this agent's tool mix drifted 3.1 sigma from its NORMAL baseline,
                 its call sequence is 0.8 novel, its message length is +2.4 z"

Those deviations are the fp_* columns. ML-A consumes them as extra features and
never rebuilds them (contract from the work-division doc).

HOW IT WORKS
------------
1. build_baselines(events_df, labels_df)
   From HARMLESS runs only, learn each (scenario, agent-role)'s normal profile:
     - tool-usage distribution  (which tools, how often)
     - call-sequence bigrams    (what normally follows what)
     - message-length stats     (mean, std)
     - tool-call rate, error rate
   Baselines are keyed by (scenario, agent_base) so an agent is compared to its
   own role's norm, not a global average.

2. fingerprint_features(events_df, baseline, K=8)
   For each (run, agent) using ONLY the first K events (decision-point K, the
   same leak-free rule ML-A uses), emit one row of fp_* deviation features.

DESIGN RULES (inherited from ML-A's leakage contract)
-----------------------------------------------------
- Decision point K: never read event_idx >= K.
- No tool NAME is used as a feature value (names only used to compare
  distributions against the agent's own baseline, never emitted as a feature).
- Baselines are built from harmless runs only, so attack information never
  leaks into the normal profile.
- Pure function signature fingerprint_features(events_df, baseline) as the
  work-division doc specifies, with K and window optional.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Events-table schema (frozen by ML-A in phase 1). We read these columns.
# --------------------------------------------------------------------------
EVENT_COLS = [
    "run_id", "task_id", "scenario", "config", "event_idx",
    "acting_agent", "target_agent", "event_type", "tool_name",
    "tool_names", "n_calls", "is_error", "msg_len",
]
TOOLREQ = "ToolCallRequestEvent"
TEXTMSG = "TextMessage"
HANDOFF = "HandoffMessage"

# Default decision point (matches ML-A's primary_K = 8).
DEFAULT_K = 8


def agent_base(agent: str) -> str:
    """Collapse per-instance agent names to their role.

    Colluding/contradicting configs spawn variants like
    'assessment_agent_a', 'assessment_agent_b'. They share a role baseline,
    so we strip a trailing single-letter suffix.
    """
    if agent is None:
        return ""
    parts = agent.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[0]
    return agent


# --------------------------------------------------------------------------
# 1. Baselines (from harmless runs only)
# --------------------------------------------------------------------------
@dataclass
class AgentBaseline:
    """Normal behavioral profile for one (scenario, agent-role)."""
    scenario: str
    agent_base: str
    tool_dist: Dict[str, float] = field(default_factory=dict)   # tool -> prob
    bigrams: Dict[Tuple[str, str], float] = field(default_factory=dict)
    msg_len_mean: float = 0.0
    msg_len_std: float = 1.0
    tool_rate_mean: float = 0.0     # toolreq events / total events
    err_rate_mean: float = 0.0
    n_runs_seen: int = 0


def _tool_sequence(sub: pd.DataFrame) -> List[str]:
    """Ordered list of tool names requested by an agent in a run (within window)."""
    reqs = sub[sub["event_type"] == TOOLREQ]
    return [str(t) for t in reqs["tool_name"].tolist() if isinstance(t, str) and t]


def build_baselines(events_df: pd.DataFrame,
                    labels_df: pd.DataFrame,
                    K: int | None = None) -> Dict[Tuple[str, str], AgentBaseline]:
    """Learn per-(scenario, agent-role) normal profiles from HARMLESS runs.

    Parameters
    ----------
    events_df : the frozen events table.
    labels_df : ML-A's labels table (used only to pick harmless runs:
                run_has_attack == 0).
    K : if given, baselines are built on the first K events only, so the
        normal profile is measured on the same window the features use.
    """
    harmless_runs = set(
        labels_df.loc[labels_df["run_has_attack"] == 0, "run_id"].unique()
    )
    ev = events_df[events_df["run_id"].isin(harmless_runs)].copy()
    if K is not None:
        ev = ev[ev["event_idx"] < K]

    # accumulators keyed by (scenario, agent_base)
    tool_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    bigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    msg_lens: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    tool_rate: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    err_rate: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    runs_seen: Dict[Tuple[str, str], set] = defaultdict(set)

    for (run_id, scenario, agent), sub in ev.groupby(
            ["run_id", "scenario", "acting_agent"], sort=False):
        if agent == "user":
            continue
        key = (scenario, agent_base(agent))
        runs_seen[key].add(run_id)

        # tool distribution
        seq = _tool_sequence(sub)
        for t in seq:
            tool_counts[key][t] += 1
        # sequence bigrams (what follows what)
        for a, b in zip(seq, seq[1:]):
            bigram_counts[key][(a, b)] += 1

        # message lengths (text messages the agent produced)
        texts = sub[sub["event_type"] == TEXTMSG]["msg_len"].tolist()
        msg_lens[key].extend(float(x) for x in texts)

        # tool-call rate and error rate within this run-window
        n_ev = len(sub)
        if n_ev:
            tool_rate[key].append(
                (sub["event_type"] == TOOLREQ).sum() / n_ev)
            err_rate[key].append(float(sub["is_error"].mean()))

    baselines: Dict[Tuple[str, str], AgentBaseline] = {}
    for key in runs_seen:
        scenario, abase = key
        # normalize tool distribution to probabilities
        tc = tool_counts[key]
        total = sum(tc.values())
        tool_dist = {t: c / total for t, c in tc.items()} if total else {}
        # normalize bigrams
        bc = bigram_counts[key]
        btotal = sum(bc.values())
        bigrams = {k: c / btotal for k, c in bc.items()} if btotal else {}

        lens = msg_lens[key] or [0.0]
        baselines[key] = AgentBaseline(
            scenario=scenario,
            agent_base=abase,
            tool_dist=tool_dist,
            bigrams=bigrams,
            msg_len_mean=float(np.mean(lens)),
            msg_len_std=float(np.std(lens)) or 1.0,
            tool_rate_mean=float(np.mean(tool_rate[key])) if tool_rate[key] else 0.0,
            err_rate_mean=float(np.mean(err_rate[key])) if err_rate[key] else 0.0,
            n_runs_seen=len(runs_seen[key]),
        )
    return baselines


# --------------------------------------------------------------------------
# 2. Deviation features
# --------------------------------------------------------------------------
def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two categorical distributions (0..1)."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            if ak > 0 and b.get(k, 0.0) > 0:
                s += ak * math.log2(ak / b[k])
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def fingerprint_features(events_df: pd.DataFrame,
                         baseline: Dict[Tuple[str, str], AgentBaseline],
                         K: int = DEFAULT_K) -> pd.DataFrame:
    """Return one row per (run_id, agent) of fp_* deviation features.

    Uses ONLY events with event_idx < K (decision-point rule).

    fp_* columns
    ------------
    fp_tool_dist_drift   : JS-divergence of this agent's tool mix vs its baseline (0..1)
    fp_seq_novelty       : fraction of call-sequence bigrams never seen in baseline
    fp_new_tool_ratio    : fraction of tool calls to tools absent from baseline
    fp_msg_len_z         : z-score of mean message length vs baseline
    fp_tool_rate_dev     : tool-call rate minus baseline tool-rate
    fp_err_rate_dev      : error rate minus baseline error rate
    fp_unseen_tool_count : count of distinct tools never seen in baseline
    fp_has_baseline      : 1 if a baseline existed for this role, else 0
                           (0 => the deviations are cold-start, treat with care)
    """
    ev = events_df[events_df["event_idx"] < K].copy()
    rows: List[dict] = []

    for (run_id, scenario, agent), sub in ev.groupby(
            ["run_id", "scenario", "acting_agent"], sort=False):
        if agent == "user":
            continue
        key = (scenario, agent_base(agent))
        bl = baseline.get(key)

        # --- this agent's behaviour in-window ---
        seq = _tool_sequence(sub)
        tc = Counter(seq)
        total = sum(tc.values())
        cur_dist = {t: c / total for t, c in tc.items()} if total else {}
        bigrams = list(zip(seq, seq[1:]))
        texts = sub[sub["event_type"] == TEXTMSG]["msg_len"].tolist()
        cur_len = float(np.mean(texts)) if texts else 0.0
        n_ev = len(sub)
        cur_tool_rate = (sub["event_type"] == TOOLREQ).sum() / n_ev if n_ev else 0.0
        cur_err_rate = float(sub["is_error"].mean()) if n_ev else 0.0

        if bl is None:
            # cold start: no baseline for this role (e.g. new variant agent).
            rows.append({
                "run_id": run_id, "agent": agent,
                "fp_tool_dist_drift": 0.0,
                "fp_seq_novelty": 0.0,
                "fp_new_tool_ratio": 0.0,
                "fp_msg_len_z": 0.0,
                "fp_tool_rate_dev": 0.0,
                "fp_err_rate_dev": 0.0,
                "fp_unseen_tool_count": 0,
                "fp_has_baseline": 0,
            })
            continue

        # tool distribution drift
        drift = _js_divergence(cur_dist, bl.tool_dist)
        # sequence novelty: bigrams not in baseline
        if bigrams:
            novel = sum(1 for bg in bigrams if bg not in bl.bigrams)
            seq_novelty = novel / len(bigrams)
        else:
            seq_novelty = 0.0
        # new / unseen tools
        unseen = [t for t in tc if t not in bl.tool_dist]
        new_tool_ratio = (sum(tc[t] for t in unseen) / total) if total else 0.0
        # message length z-score
        msg_z = (cur_len - bl.msg_len_mean) / (bl.msg_len_std or 1.0)
        # rate deviations
        tool_rate_dev = cur_tool_rate - bl.tool_rate_mean
        err_rate_dev = cur_err_rate - bl.err_rate_mean

        rows.append({
            "run_id": run_id, "agent": agent,
            "fp_tool_dist_drift": round(drift, 4),
            "fp_seq_novelty": round(seq_novelty, 4),
            "fp_new_tool_ratio": round(new_tool_ratio, 4),
            "fp_msg_len_z": round(msg_z, 4),
            "fp_tool_rate_dev": round(tool_rate_dev, 4),
            "fp_err_rate_dev": round(err_rate_dev, 4),
            "fp_unseen_tool_count": len(unseen),
            "fp_has_baseline": 1,
        })

    cols = ["run_id", "agent", "fp_tool_dist_drift", "fp_seq_novelty",
            "fp_new_tool_ratio", "fp_msg_len_z", "fp_tool_rate_dev",
            "fp_err_rate_dev", "fp_unseen_tool_count", "fp_has_baseline"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Cyber-1 behavioral fingerprinting (fp_* features)")
    ap.add_argument("--events", required=True, help="events.csv from ML-A")
    ap.add_argument("--labels", required=True, help="labels.csv from ML-A")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="decision point K")
    ap.add_argument("--out", default="fingerprint_features.csv")
    args = ap.parse_args()

    events = pd.read_csv(args.events)
    labels = pd.read_csv(args.labels)
    bl = build_baselines(events, labels, K=args.k)
    print(f"[cyber1] built {len(bl)} role baselines from harmless runs")
    feats = fingerprint_features(events, bl, K=args.k)
    feats.to_csv(args.out, index=False)
    print(f"[cyber1] wrote {len(feats)} fp_* rows -> {args.out}")
    print(feats.describe(include='all').T[["mean", "min", "max"]].head(20))
