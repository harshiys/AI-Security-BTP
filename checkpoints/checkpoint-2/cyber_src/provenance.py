#!/usr/bin/env python3
"""SentinelMesh CP2 Cyber: message-provenance implementation.

Provides a small dependency-free HMAC-SHA256 protocol for the runtime testbed.
Historical TAMAS logs do not contain cryptographic signatures, so the real-data
report marks provenance as UNAVAILABLE rather than fabricating signatures.
"""
import argparse, hashlib, hmac, json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def canonical_message(sender, receiver, operation, payload, nonce):
    obj={"sender":sender,"receiver":receiver,"operation":operation,"payload":payload,"nonce":nonce}
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sign_message(secret, sender, receiver, operation, payload, nonce):
    body=canonical_message(sender,receiver,operation,payload,nonce)
    sig=hmac.new(secret.encode(),body,hashlib.sha256).hexdigest()
    return {"message":json.loads(body),"signature":sig,"algorithm":"HMAC-SHA256"}

def verify_message(secret, signed):
    m=signed["message"]; expected=hmac.new(secret.encode(),canonical_message(m["sender"],m["receiver"],m["operation"],m["payload"],m["nonce"]),hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected,signed["signature"])

def real_provenance(events):
    rows=[]
    for rid,ev in events.groupby("run_id",sort=False):
        # TAMAS event schema has no signature, key id, nonce or cryptographic envelope.
        rows.append({"run_id":rid,"signed":None,"source":"unavailable","verification":"not_recorded","reason":"historical TAMAS event log contains no cryptographic message signature"})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--events',default='data/events.csv'); ap.add_argument('--out',default='results/provenance_real.csv'); a=ap.parse_args()
    e=pd.read_csv(a.events); df=real_provenance(e); Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
    demo=sign_message('demo-secret','news_gathering_agent','fact_checking_agent','delegate',{'task':'fact-check'},'nonce-001')
    Path(a.out).with_name('provenance_protocol_demo.json').write_text(json.dumps({'signed_message':demo,'verification':verify_message('demo-secret',demo)},indent=2))
if __name__=='__main__': main()
