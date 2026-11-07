import argparse
import json
import os

from trustchain import load_private_key, sign_entry

p = argparse.ArgumentParser(description="Sign every scanner verdict.")
p.add_argument("--results", default="c1_scan_results.json")
p.add_argument("--artifact", default="c1_scan_results.json",
               help="stand-in file used for every item until real adapters exist")
p.add_argument("--artifact-dir", default=None,
               help="folder of real adapters, each named <item_id>.<ext>")
p.add_argument("--out", default="signed_records.json")
p.add_argument("--key", default="scanner_private.pem")
args = p.parse_args()

key = load_private_key(args.key)
with open(args.results) as f:
    entries = json.load(f)


def find_artifact(item_id):
    if args.artifact_dir:
        for name in os.listdir(args.artifact_dir):
            if os.path.splitext(name)[0] == item_id:
                return os.path.join(args.artifact_dir, name)
        return None
    return args.artifact


signed, skipped = [], 0
for entry in entries:
    path = find_artifact(entry["item_id"])
    if path is None or not os.path.exists(path):
        print(f"SKIP {entry['item_id']}: artifact file not found")
        skipped += 1
        continue
    signed.append(sign_entry(entry, path, key))

with open(args.out, "w") as f:
    json.dump(signed, f, indent=2)

print(f"signed {len(signed)} records, skipped {skipped} -> {args.out}")
