import argparse
import json
import os
import sys

from trustchain import (load_public_key, key_id, sha256_file, verify_signed)

p = argparse.ArgumentParser(description="Allow or block loading an artifact through SentinelMesh CP1 Gatekeeper.")
p.add_argument("artifact", help="Path to adapter/model file (e.g. sample_adapters/clean_adapter.safetensors)")
p.add_argument("--records", default=None,
               help="Path to signed_c1_records.json (defaults to work/c1_log/signed_c1_records.json or sample_adapters/sample_signed_records.json)")
p.add_argument("--pubkey", default="scanner_public.pem", help="Path to public key file")
p.add_argument("--item-id", default=None, help="Expected item_id (optional)")
p.add_argument("--pin", default=None,
               help="Expected key_id; blocks if the public key file was swapped")
args = p.parse_args()


def block(reason):
    print(f"BLOCKED: {reason}")
    sys.exit(1)


# 1. Check artifact existence
if not os.path.exists(args.artifact):
    block(f"artifact file not found: '{args.artifact}' (please specify a valid existing file path)")

# 2. Resolve records file
records_file = args.records
if not records_file:
    for candidate in [
        "work/c1_log/signed_c1_records.json",
        "sample_adapters/sample_signed_records.json",
        "signed_records.json"
    ]:
        if os.path.exists(candidate):
            records_file = candidate
            break

if not records_file or not os.path.exists(records_file):
    block(f"signed records file not found (looked for {records_file or 'work/c1_log/signed_c1_records.json'}). Run run_pipeline.py first.")

# 3. Resolve public key
if not os.path.exists(args.pubkey):
    block(f"public key file not found: '{args.pubkey}'. Run make_keys.py first.")

public_key = load_public_key(args.pubkey)
if args.pin and key_id(public_key) != args.pin:
    block("public key does not match pinned key")

try:
    with open(records_file, encoding="utf-8") as f:
        records = json.load(f)
except Exception as e:
    block(f"failed to read records file: {e}")

# 4. Find candidates
if args.item_id:
    candidates = [r for r in records
                  if r.get("record", {}).get("item_id") == args.item_id]
    if not candidates:
        block("no signed record for this item")
else:
    h = sha256_file(args.artifact)
    candidates = [r for r in records
                  if r.get("record", {}).get("artifact_sha256") == h]
    if not candidates:
        # Check if file was known by name or item_id, indicating tampering
        name = os.path.basename(args.artifact)
        name_matches = [r for r in records
                        if r.get("record", {}).get("artifact_name") == name or
                           r.get("record", {}).get("item_id", "").endswith(name)]
        if name_matches:
            block("artifact changed since scan (hash mismatch / file tampered)")
        else:
            block("no signed record matches this file (unscanned artifact)")

# 5. Verify cryptographic signatures and constraints
reason = "no valid record"
for signed in candidates:
    ok, reason = verify_signed(signed, args.artifact, public_key,
                               expected_item_id=args.item_id)
    if ok:
        print("ALLOWED")
        sys.exit(0)

block(reason)
