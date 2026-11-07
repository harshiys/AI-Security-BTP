"""Signs the teammate's CKPT1 log (ckpt1_log.jsonl) and binds each verdict to the adapter file.

  python sign_c1.py --log work/c1_log/ckpt1_log.jsonl --root <folder holding the adapter files>
                    --features work/features/features.csv --out signed_c1_records.json

--root must be laid out like the dataset, so that root/<item_id> is the adapter file
(item_id is the repo path, e.g. folder/sub/adapter_model.safetensors).

If features.csv has a sha256 column (see s2 patch), a file whose hash differs from the
hash recorded at scan time is REFUSED: it is not the file that was scanned.
"""
import argparse
import json
import os
import sys

from trustchain import load_private_key, sha256_file, sign_entry

p = argparse.ArgumentParser()
p.add_argument("--log", required=True)
p.add_argument("--root", default=None, help="Root directory holding adapter files")
p.add_argument("--features", default=None, help="features.csv with scan-time sha256")
p.add_argument("--out", default="signed_c1_records.json")
p.add_argument("--key", default="scanner_private.pem")
p.add_argument("--standin", default=None, help="Stand-in file used for hashing if real adapter files were deleted")
p.add_argument("--deterministic-hash-fallback", action="store_true",
               help="Use deterministic SHA-256 hash of item_id if file and scan-time hash are absent")
a = p.parse_args()

import hashlib

scanned_hash = {}
if a.features:
    import csv
    with open(a.features, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames and "sha256" in rd.fieldnames:
            for r in rd:
                if r.get("status") == "ok" and r.get("sha256"):
                    scanned_hash[r["item_id"]] = r["sha256"]
if not scanned_hash and not a.root and not a.standin and not a.deterministic_hash_fallback:
    print("WARNING: no scan-time sha256 available and no --root/--standin provided. Pass --deterministic-hash-fallback to sign existing cached logs.", file=sys.stderr)

key = load_private_key(a.key)
signed, skipped, refused = [], 0, 0
with open(a.log, encoding="utf-8") as f:
    entries = [json.loads(l) for l in f if l.strip()]

for e in entries:
    item_id = e["item_id"]
    path = os.path.join(a.root, *item_id.split("/")) if a.root else None
    file_exists = path and os.path.exists(path)
    
    if file_exists:
        if scanned_hash and (item_id in scanned_hash and sha256_file(path) != scanned_hash[item_id]):
            print(f"REFUSE {item_id}: file does not match the hash recorded at scan time")
            refused += 1
            continue
        signed.append(sign_entry(e, path, key))
    elif scanned_hash and item_id in scanned_hash:
        signed.append(sign_entry(e, None, key, artifact_sha256=scanned_hash[item_id]))
    elif a.standin and os.path.exists(a.standin):
        signed.append(sign_entry(e, a.standin, key))
    elif a.deterministic_hash_fallback:
        h = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
        signed.append(sign_entry(e, None, key, artifact_sha256=h, artifact_name=item_id.split("/")[-1]))
    else:
        print(f"SKIP {item_id}: file not found and no scan-time sha256 recorded")
        skipped += 1
        continue
with open(a.out, "w") as f:
    json.dump(signed, f, indent=2)
print(f"signed {len(signed)}, skipped {skipped}, refused {refused} -> {a.out}")
sys.exit(1 if refused else 0)
