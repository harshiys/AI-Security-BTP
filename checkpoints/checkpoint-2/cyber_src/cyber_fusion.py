#!/usr/bin/env python3
"""Cyber-side fusion adapter for CP2.

Combines the three Cyber signals at the boundary to ML-B:
behavioral drift, communication-graph anomaly and message provenance.
This is an interpretable cyber score; the team's ML-B model may fuse it with
its learned risk score separately.
"""
import argparse, json
from pathlib import Path
import pandas as pd

def normalize_graph(row):
    # conservative interpretable graph anomaly score
    vals=[float(row.get('novel_out_edges_loo',0)),float(row.get('novel_in_edges_loo',0))]
    deg=float(row.get('out_degree',0))+float(row.get('in_degree',0))
    return min(1.0,(sum(vals)+0.5*(deg>4))/2.0)

def fuse(fp,graph,prov):
    d=fp.merge(graph,on=['run_id','agent'],how='outer')
    d['behavioral_drift']=d['drift'].fillna(0.0)
    d['graph_anomaly']=d.apply(normalize_graph,axis=1)
    p=prov.set_index('run_id') if len(prov) else None
    def pv(r):
        if p is None or r.run_id not in p.index: return 0.0
        s=p.loc[r.run_id,'signed']
        src=p.loc[r.run_id,'source']
        return 0.0 if (pd.isna(s) or src in ('trusted','unavailable')) else 1.0
    d['provenance_risk']=d.apply(pv,axis=1)
    d['cyber_risk']=(0.50*d.behavioral_drift+0.35*d.graph_anomaly+0.15*d.provenance_risk).clip(0,1)
    d['cyber_verdict']=d.cyber_risk.apply(lambda x:'flag' if x>=0.7 else 'pass')
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fingerprint',default='results/cyber1_fingerprint_drift.jsonl'); ap.add_argument('--graph',default='results/communication_graph.csv'); ap.add_argument('--provenance',default='results/provenance_real.csv'); ap.add_argument('--out',default='results/cyber_cp2_fusion.csv'); a=ap.parse_args()
    fp=pd.DataFrame([json.loads(x) for x in open(a.fingerprint,encoding='utf8') if x.strip()]); g=pd.read_csv(a.graph); p=pd.read_csv(a.provenance)
    out=fuse(fp,g,p); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
if __name__=='__main__': main()
