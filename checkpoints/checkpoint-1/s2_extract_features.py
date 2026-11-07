#!/usr/bin/env python3
"""Stage 2 - static feature extraction with a resumable cache.

For every manifest row: fetch ONE adapter -> read only lora_* tensors safely -> compute features -> append to
work/features/*.csv -> delete the downloaded file (unless --keep-files). The base LLM is never loaded.

  --probe   print tensor names/shapes + features for one adapter per class and exit (no cache writes).
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from c1.common import is_real_tag, make_source
from c1.features import METRICS, extract_adapter, feature_names
from c1.safeload import load_lora_tensors
from trustchain import sha256_file

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["hf", "local"], default=None,
                help="default: the same source stage 1 used (recorded in manifest_report.json)")
ap.add_argument("--local-root", help="default: the root stage 1 used")
ap.add_argument("--work", default="work")
ap.add_argument("--limit", type=int, default=0, help="process at most N uncached adapters (0 = all)")
ap.add_argument("--max-gb", type=float, default=3.0, help="skip single adapters larger than this")
ap.add_argument("--keep-files", action="store_true")
ap.add_argument("--probe", action="store_true")
args = ap.parse_args()

work = Path(args.work)
man = pd.read_csv(work / "manifest.csv").fillna({"config_path": ""})
mrep = json.loads((work / "manifest_report.json").read_text(encoding="utf-8"))
tags = set(man["data_tag"])
if len(tags) != 1:
    sys.exit(f"manifest mixes data sources {sorted(tags)}; refusing (never mix synthetic and real data)")
data_tag = tags.pop()
# use the SAME source stage 1 used: same dataset commit (hf) or same local folder -> reproducible, traceable rows
args.source = args.source or ("local" if mrep.get("local_root_effective") else "hf")
local_root = args.local_root or mrep.get("local_root_effective")
src = make_source(args.source, local_root, work, mrep.get("revision"))
fdir = work / "features"; fdir.mkdir(parents=True, exist_ok=True)
info_p = fdir / "cache_info.json"
info = {"data_tag": data_tag, "revision": mrep.get("revision")}
if info_p.exists():
    old = json.loads(info_p.read_text(encoding="utf-8"))
    # real-from-HF and real-on-disk are the same data; what must never mix is real vs synthetic (or two commits)
    clash = (is_real_tag(old.get("data_tag", "")) != is_real_tag(data_tag)
             or (old.get("revision") and info["revision"] and old["revision"] != info["revision"]))
    if clash:
        sys.exit(f"{info_p} was written for a different data source ({old} vs {info}). "
                 "Use a fresh --work folder instead of mixing caches.")
info_p.write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"data_tag={data_tag}  revision={info['revision']}")


def fetch_pair(row):
    w = src.fetch(row.item_id)
    c = src.fetch(row.config_path) if row.config_path else None
    return w, c


if args.probe:
    for lab in ("benign", "backdoor"):
        row = next(man[man.label == lab].itertuples())
        w, c = fetch_pair(row)
        t = load_lora_tensors(w)
        print(f"\n=== PROBE {lab}: {row.item_id}  ({row.size_bytes/1e6:.1f} MB) ===")
        print(f"{len(t)} lora tensors; sample:")
        for k in list(t)[:6]:
            print(f"   {k}  shape={t[k].shape}")
        meta, feats, rows = extract_adapter(w, c)
        print("meta:", meta)
        print("layer-level example (first q row):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[0].items()})
        print("a few aggregate features:", {k: round(feats[k], 4) for k in feature_names()[:4]})
        if not args.keep_files:
            src.release(row.item_id)
    sys.exit(0)

feat_csv, layer_csv = fdir / "features.csv", fdir / "layer_features.csv"
cols = ["item_id", "status", "error", "sha256", "seconds", "n_pairs_q", "n_pairs_v", "n_pairs_other",
        "rank_detected", "n_layers", "scale", "scale_source"] + feature_names()
done = set()
if feat_csv.exists():
    old = pd.read_csv(feat_csv, usecols=["item_id", "status"])
    done = set(old.loc[old.status == "ok", "item_id"])
todo = man[~man.item_id.isin(done)]
if args.limit:
    todo = todo.head(args.limit)
print(f"{len(man)} in manifest, {len(done)} cached ok, {len(todo)} to extract")

for i, row in enumerate(todo.itertuples(), 1):
    rec = {c: math.nan for c in cols}; rec.update(item_id=row.item_id, error="")
    lrows, t0 = [], time.time()
    try:
        if row.size_bytes > args.max_gb * 1e9:
            raise RuntimeError(f"skipped: {row.size_bytes/1e9:.1f} GB > --max-gb")
        w, c = fetch_pair(row)
        rec["sha256"] = sha256_file(w)  # hash of the exact bytes being scanned
        meta, feats, lrows = extract_adapter(w, c)
        rec.update(meta); rec.update(feats); rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001  - record and continue; never crash the batch
        rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"[:300]
    finally:
        if not args.keep_files:
            try: src.release(row.item_id)
            except Exception: pass  # noqa: BLE001
    rec["seconds"] = round(time.time() - t0, 2)
    pd.DataFrame([rec], columns=cols).to_csv(feat_csv, mode="a", header=not feat_csv.exists(), index=False)
    if lrows:
        ldf = pd.DataFrame(lrows); ldf.insert(0, "item_id", row.item_id)
        ldf[["item_id", "module", "layer", "rank", *METRICS]].to_csv(layer_csv, mode="a", header=not layer_csv.exists(), index=False)
    print(f"[{i}/{len(todo)}] {rec['status']:5s} {rec['seconds']:6.1f}s  {row.item_id}  {rec['error']}")

res = pd.read_csv(feat_csv)
print("\nfeature cache status:", res.drop_duplicates("item_id", keep="last").status.value_counts().to_dict())
