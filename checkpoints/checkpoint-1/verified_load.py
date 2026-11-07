"""Drop-in replacement for joblib.load that refuses an unsealed or modified classifier.

joblib files are pickles: loading a swapped one can run attacker code, so the check
must happen BEFORE the load.
"""
import json
import sys
from datetime import timedelta

import joblib

from trustchain import load_public_key, key_id, verify_signed


def load_verified(model_path, record_path="classifier_record.json", pubkey="scanner_public.pem",
                  item_id="c1-classifier", pin=None, max_age=timedelta(days=30)):
    pub = load_public_key(pubkey)
    if pin and key_id(pub) != pin:
        raise SystemExit("BLOCKED: public key does not match pinned key")
    try:
        with open(record_path) as f:
            recs = json.load(f)
    except (FileNotFoundError, ValueError):
        raise SystemExit("BLOCKED: classifier not loaded (no readable signed record; run sign_model.py)")
    reason = "no record"
    for r in recs:
        ok, reason = verify_signed(r, model_path, pub, expected_item_id=item_id, max_age=max_age)
        if ok:
            return joblib.load(model_path)
    raise SystemExit(f"BLOCKED: classifier not loaded ({reason})")
