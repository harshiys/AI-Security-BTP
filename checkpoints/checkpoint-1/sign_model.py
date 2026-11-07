"""Seals the approved classifier file (e.g. work/model/c1_model.joblib).

  python sign_model.py work/model/c1_model.joblib --out classifier_record.json
Run this once, right after s3_train_eval.py, and again after every retrain.
"""
import argparse
import json

from trustchain import load_private_key, sign_entry

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--item-id", default="c1-classifier")
p.add_argument("--out", default="classifier_record.json")
p.add_argument("--key", default="scanner_private.pem")
a = p.parse_args()
# "pass" here means "approved version", not a malware verdict.
rec = sign_entry({"checkpoint_id": "CKPT1_STATIC", "item_id": a.item_id, "verdict": "pass", "score": 0.0},
                 a.model, load_private_key(a.key))
with open(a.out, "w") as f:
    json.dump([rec], f, indent=2)
print("sealed", a.model, "->", a.out, "| key_id", rec["record"]["key_id"])
