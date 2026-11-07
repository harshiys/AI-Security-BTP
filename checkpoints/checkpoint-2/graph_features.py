#!/usr/bin/env python3
"""
ML-B: agent communication graph + graph features (SentinelMesh Checkpoint 2).

One NetworkX DiGraph per run, built ONLY from events with event_idx < K.
Nodes: the user and the agents (tool nodes are off: tool names are never used, same rule as ML-A).
Edges: acting_agent -> target_agent. Edge kinds counted: message, handoff (delegation).
Orchestrator: TAMAS RoundRobin has none, so the agent the user addresses first is used as the entry point.
Features need no labels. 'Novel edge' = an edge never seen in HARMLESS runs of OTHER training tasks
(the caller supplies which tasks count as training, so test tasks never inform it).
"""
import networkx as nx
import numpy as np
import pandas as pd

GRAPH_RUN_COLS = ["g_n_nodes", "g_n_edges", "g_density", "g_max_out_deg", "g_orch_fanout", "g_max_depth",
                  "g_n_reciprocal", "g_pingpong", "g_n_novel_edges", "g_novel_frac", "g_agents_touched"]
GRAPH_AGENT_COLS = ["g_in_deg", "g_out_deg", "g_betweenness", "g_pagerank", "g_is_entry",
                    "g_depth", "g_novel_out", "g_novel_in"]
GRAPH_COLS = GRAPH_RUN_COLS + GRAPH_AGENT_COLS


def prefix_edges(w):
    """Edge list [(src, dst, kind)] from a prefix of events. Self-loops (tool calls) are not edges."""
    out = []
    for a, t, et in zip(w["acting_agent"], w["target_agent"], w["event_type"]):
        if isinstance(t, str) and t and a != t:
            out.append((a, t, "handoff" if et == "HandoffMessage" else "message"))
    return out


def build_graph(w):
    g = nx.DiGraph()
    for a, t, kind in prefix_edges(w):
        if g.has_edge(a, t):
            g[a][t]["n"] += 1
            g[a][t][kind] = g[a][t].get(kind, 0) + 1
        else:
            g.add_edge(a, t, n=1, **{kind: 1})
    return g


def pingpong(seq):
    """Number of A,B,A,B alternations (A != B) in the order agents acted."""
    return sum(1 for i in range(len(seq) - 3)
               if seq[i] == seq[i + 2] and seq[i + 1] == seq[i + 3] and seq[i] != seq[i + 1])


def benign_edge_index(events, harmless_runs):
    """edge -> set of task_ids where it appeared in a harmless run (whole run; used as a training reference)."""
    idx = {}
    h = events[events["run_id"].isin(harmless_runs)]
    for (rid, tid), g in h.groupby(["run_id", "task_id"]):
        for a, t, _ in prefix_edges(g):
            idx.setdefault((a, t), set()).add(tid)
    return idx


def _novel(edge, task_id, edge_index, allowed):
    seen = edge_index.get(edge, ())
    return not any((x in allowed and x != task_id) for x in seen)


def graph_features(events, roster, K, edge_index=None, allowed_tasks=None):
    """Per (run, agent) graph features from events with event_idx < K ONLY. No labels read."""
    edge_index = edge_index or {}
    allowed = set(allowed_tasks) if allowed_tasks is not None else set()
    task_of = events.groupby("run_id")["task_id"].first().to_dict() if "task_id" in events else {}
    by_run = {rid: g for rid, g in events.groupby("run_id", sort=False)}
    rows = []
    for rid, agents in roster.groupby("run_id", sort=False):
        ev = by_run.get(rid)
        if ev is None:
            continue
        w = ev[ev["event_idx"] < K].sort_values("event_idx")
        g = build_graph(w)
        tid = task_of.get(rid)
        edges = list(g.edges())
        novel = {e for e in edges if _novel(e, tid, edge_index, allowed)} if edge_index else set()
        first_user = w[w["acting_agent"] == "user"]
        entry = first_user["target_agent"].iloc[0] if len(first_user) else None
        depth = nx.single_source_shortest_path_length(g, "user") if "user" in g else {}
        seq = [a for a in w["acting_agent"] if a != "user"]
        n_nodes = g.number_of_nodes()
        bet = nx.betweenness_centrality(g) if n_nodes else {}
        try:
            pr = nx.pagerank(g) if n_nodes else {}
        except nx.PowerIterationFailedConvergence:
            pr = {}
        run_f = dict(
            g_n_nodes=n_nodes, g_n_edges=len(edges),
            g_density=nx.density(g) if n_nodes > 1 else 0.0,
            g_max_out_deg=max((d for _, d in g.out_degree()), default=0),
            g_orch_fanout=g.out_degree(entry) if entry in g else 0,
            g_max_depth=max(depth.values(), default=0),
            g_n_reciprocal=sum(1 for a, b in edges if a < b and g.has_edge(b, a)),
            g_pingpong=pingpong(seq),
            g_n_novel_edges=len(novel), g_novel_frac=len(novel) / max(1, len(edges)),
            g_agents_touched=len(set(seq)))
        for ag in agents["agent"]:
            rows.append(dict(
                run_id=rid, agent=ag, **run_f,
                g_in_deg=g.in_degree(ag) if ag in g else 0, g_out_deg=g.out_degree(ag) if ag in g else 0,
                g_betweenness=bet.get(ag, 0.0), g_pagerank=pr.get(ag, 0.0),
                g_is_entry=int(ag == entry), g_depth=depth.get(ag, n_nodes),
                g_novel_out=sum(1 for e in novel if e[0] == ag), g_novel_in=sum(1 for e in novel if e[1] == ag)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ self-checks
def selfcheck(events, roster, K=8, n_runs=25):
    """(1) wrecking every event at/after K must not move the features (no future leak);
       (2) hand-made log with a known ping-pong gives the expected numbers."""
    runs = roster["run_id"].drop_duplicates().head(n_runs)
    ros = roster[roster["run_id"].isin(runs)]
    ev = events[events["run_id"].isin(runs)]
    base = graph_features(ev, ros, K)
    wrecked = ev.copy()
    late = wrecked["event_idx"] >= K
    wrecked.loc[late, "target_agent"] = "zz_future_agent"
    wrecked.loc[late, "acting_agent"] = "zz_future_agent"
    assert graph_features(wrecked, ros, K).equals(base), "graph features read events at/after K"
    hand = pd.DataFrame({"run_id": "h", "task_id": "t", "event_idx": range(6),
                         "acting_agent": ["user", "a", "b", "a", "b", "a"],
                         "target_agent": ["a", "b", "a", "b", "a", "b"],
                         "event_type": ["TextMessage"] * 6})
    f = graph_features(hand, pd.DataFrame({"run_id": ["h"], "agent": ["a"]}), 6)
    assert f.g_pingpong.iloc[0] == 2 and f.g_n_reciprocal.iloc[0] == 1 and f.g_max_depth.iloc[0] == 2, f.T
    return True
