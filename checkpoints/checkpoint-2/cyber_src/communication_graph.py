#!/usr/bin/env python3
"""SentinelMesh CP2 Cyber: communication-graph telemetry.

Builds the observed agent-to-agent graph from real TAMAS-derived events. No
attack labels are used while constructing features. The graph is a telemetry
input for the ML-B/GNN layer and also provides interpretable graph anomalies.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import networkx as nx

AGENT_EVENTS = {"TextMessage", "HandoffMessage", "ToolCallSummaryMessage"}

def build_graph(events, K=8):
    ev=events[events.event_idx < K].sort_values("event_idx")
    g=nx.DiGraph()
    for _,r in ev.iterrows():
        a=str(r.acting_agent); t=r.target_agent
        if a=="user" or pd.isna(t) or str(t)=="" or a==str(t): continue
        t=str(t)
        kind="handoff" if r.event_type=="HandoffMessage" else "message"
        if g.has_edge(a,t):
            g[a][t]["count"] += 1
            g[a][t][kind] = g[a][t].get(kind,0)+1
        else:
            g.add_edge(a,t,count=1,message=int(kind=="message"),handoff=int(kind=="handoff"))
    return g

def graph_rows(events, K=8):
    rows=[]; graphs={}
    for rid, ev in events.groupby("run_id", sort=False):
        g=build_graph(ev,K); graphs[rid]=g
        entry=None
        u=ev[(ev.event_idx<K)&(ev.acting_agent=="user")]
        if len(u): entry=u.iloc[0].target_agent
        for ag in sorted(set(ev.acting_agent)-{"user"}):
            depth=nx.single_source_shortest_path_length(g,"user") if "user" in g else {}
            rows.append({"run_id":rid,"agent":ag,"graph_nodes":g.number_of_nodes(),"graph_edges":g.number_of_edges(),
                         "in_degree":g.in_degree(ag),"out_degree":g.out_degree(ag),
                         "pagerank":nx.pagerank(g).get(ag,0.0) if g.number_of_nodes() else 0.0,
                         "betweenness":nx.betweenness_centrality(g).get(ag,0.0) if g.number_of_nodes() else 0.0,
                         "is_entry":int(ag==entry),"depth":depth.get(ag,-1),
                         "touched_agents":len(set(ev[ev.event_idx<K].acting_agent)-{"user"})})
    return pd.DataFrame(rows), graphs

def make_edge_baseline(events,K=8):
    edges=set()
    for _,ev in events.groupby("run_id"):
        g=build_graph(ev,K); edges.update(g.edges())
    return edges

def add_edge_novelty(rows, graphs, baseline_edges):
    out=rows.copy();
    out["novel_out_edges"]=0; out["novel_in_edges"]=0
    for i,r in out.iterrows():
        g=graphs[r.run_id]; ag=r.agent
        out.loc[i,"novel_out_edges"]=sum((ag,b) not in baseline_edges for b in g.successors(ag)) if ag in g else 0
        out.loc[i,"novel_in_edges"]=sum((a,ag) not in baseline_edges for a in g.predecessors(ag)) if ag in g else 0
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--events",default="data/events.csv"); ap.add_argument("--out",default="results/communication_graph.csv"); ap.add_argument("--K",type=int,default=8); a=ap.parse_args()
    e=pd.read_csv(a.events); rows,graphs=graph_rows(e,a.K)
    # Leave-one-run-out novelty: each run is compared with edges from all other runs.
    vals=[]
    for rid,g in graphs.items():
        base=set()
        for other,og in graphs.items():
            if other!=rid: base.update(og.edges())
        for ag in set(e.loc[e.run_id==rid,"acting_agent"])-{"user"}:
            vals.append((rid,ag,sum((ag,b) not in base for b in g.successors(ag)) if ag in g else 0,
                         sum((x,ag) not in base for x in g.predecessors(ag)) if ag in g else 0))
    nov=pd.DataFrame(vals,columns=["run_id","agent","novel_out_edges_loo","novel_in_edges_loo"])
    rows=rows.merge(nov,on=["run_id","agent"],how="left")
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); rows.to_csv(a.out,index=False)
    payload={rid:{"nodes":list(g.nodes()),"edges":[[u,v,d] for u,v,d in g.edges(data=True)]} for rid,g in graphs.items()}
    Path(a.out).with_suffix('.json').write_text(json.dumps(payload,indent=2))

if __name__=='__main__': main()
