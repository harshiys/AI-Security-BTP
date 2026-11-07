#!/usr/bin/env python3
"""Interpretable Cyber evidence -> dynamic trust score for CP2.

This is deliberately a transparent cyber-side score, not a replacement for
ML-B. It converts the three cyber evidence channels into one 0..1 risk value
and exposes the reasons so the final project trust layer can fuse it.
"""
import argparse
from pathlib import Path
import pandas as pd

def score(df):
    x=df.copy()
    # Behaviour: strongest because it is directly observed per agent.
    b=x.get('behavioral_drift',pd.Series(0.0,index=x.index)).fillna(0).clip(0,1)
    # Graph: two novel edges are treated as a strong anomaly; degree alone is weak.
    no=x.get('novel_out_edges_loo',pd.Series(0,index=x.index)).fillna(0)
    ni=x.get('novel_in_edges_loo',pd.Series(0,index=x.index)).fillna(0)
    ga=((no+ni)/2).clip(0,1)
    # Provenance: only explicit failure is risky. 'unavailable' is not a failure.
    pr=x.get('provenance_risk',pd.Series(0.0,index=x.index)).fillna(0).clip(0,1)
    x['cyber_risk']=(0.55*b+0.30*ga+0.15*pr).clip(0,1)
    x['dynamic_trust']=1-x.cyber_risk
    x['trust_verdict']=x.cyber_risk.apply(lambda z:'flag' if z>=0.70 else 'pass')
    def reason(r):
        vals={'behavioral_drift':r.behavioral_drift,'graph_anomaly':r.get('graph_anomaly',0),'provenance_risk':r.provenance_risk}
        return max(vals,key=vals.get)
    x['primary_reason']=x.apply(reason,axis=1)
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='results/cyber_cp2_fusion.csv'); ap.add_argument('--out',default='results/cyber_dynamic_trust.csv'); a=ap.parse_args()
    x=pd.read_csv(a.input); x=score(x); Path(a.out).parent.mkdir(parents=True,exist_ok=True); x.to_csv(a.out,index=False)
    print(x[['run_id','agent','behavioral_drift','graph_anomaly','provenance_risk','cyber_risk','dynamic_trust','trust_verdict']].to_string(index=False))
if __name__=='__main__': main()
